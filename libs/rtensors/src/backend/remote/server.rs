//! Remote tensor server. Each client connection gets its own session (buffer store + backend),
//! executed by one worker thread in request order and dropped when the client disconnects.

use std::{collections::HashMap, net::{IpAddr, TcpListener, TcpStream}, thread::{self, JoinHandle}};

use crate::{backend::{cpu::Cpu, ContiguityTypes, remote::protocol::{read_frame_limited, write_frame, BufId, Layout, Op, RemoteDevice, Reply, Request, Response, ScalarOp, Slice, TypelessBuf, UnaryOp, Value, PROTOCOL_VERSION}, Backend, BackendMatMul}, core::{primitives::DeviceType, tensor::TensorError, value::{types, DType, TensorValue}, Dim, MetaTensor}, ops::reduction::ReductionOpTypes};

/// Limits applied to every connection a [`RemoteServer`] accepts.
///
/// Worst-case memory per connection is roughly `max_session_bytes` of live buffers plus
/// `queue_depth` queued requests of up to the frame limit each (plus temporaries the backend
/// allocates while running an op, e.g. CUDA reduction scratch space, which are not counted).
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Upper bound on the bytes of live buffers one connection may hold. `None` means no limit,
    /// in which case an oversized allocation can exhaust (and abort) the server process.
    pub max_session_bytes: Option<usize>,
    /// Requests buffered per connection before the server stops reading from its socket, which
    /// in turn blocks the client's writes (backpressure).
    pub queue_depth: usize,
    /// Largest request frame accepted, in bytes. `None` derives it from `max_session_bytes`
    /// (that plus 1 MiB of slack), or 1 TiB without a session cap. Uploads larger than this
    /// are rejected and the connection is dropped, before the payload is buffered.
    pub max_frame_bytes: Option<u64>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { max_session_bytes: None, queue_depth: 1024, max_frame_bytes: None }
    }
}

impl ServerConfig {
    fn frame_limit(&self) -> u64 {
        self.max_frame_bytes.unwrap_or_else(|| match self.max_session_bytes {
            Some(max) => (max as u64).saturating_add(1 << 20),
            None => crate::backend::remote::protocol::MAX_FRAME_BYTES,
        })
    }
}

/// Serves tensor operations to [`super::client::RemoteBackend`] clients over TCP.
///
/// Every connection gets its own session: its buffers live on the backend the client asked for
/// in its handshake (CPU, or CUDA when built with the `cuda` feature) and are dropped when the
/// connection closes. There is no authentication or encryption, so only listen on trusted networks.
pub struct RemoteServer {
    address: IpAddr,
    port: u16,
    config: ServerConfig,
}

impl RemoteServer {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
            config: ServerConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Accepts connections until the listener fails. Blocks the calling thread.
    pub fn serve(&mut self) -> std::io::Result<()> {
        let listener = TcpListener::bind((self.address, self.port))?;
        self.serve_on(listener)
    }

    /// Like [`Self::serve`], on an already bound listener (e.g. one bound to port 0).
    pub fn serve_on(&mut self, listener: TcpListener) -> std::io::Result<()> {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let config = self.config.clone();
                    thread::spawn(move || handle_connection(stream, config));
                }
                Err(e) => {
                    tracing::warn!("remote server: connection failed: {e}");
                }
            }
        }
        Ok(())
    }
}

/// Launches a new server in a background thread listening on the given IP and port.
pub fn launch_server(ip: IpAddr, port: u16) -> Result<JoinHandle<()>, TensorError> {
    let mut server = RemoteServer::new(ip, port);
    let handle = thread::spawn(move || {
        if let Err(e) = server.serve() {
            tracing::error!("remote server on {ip}:{port} stopped: {e}");
        }
    });
    Ok(handle)
}

/// Starts a background server on `ip:port` unless something is already listening there.
/// The listener is bound before returning, so callers can connect immediately.
#[cfg(test)]
pub(crate) fn ensure_test_server(ip: IpAddr, port: u16) {
    if let Ok(listener) = TcpListener::bind((ip, port)) {
        thread::spawn(move || {
            let _ = RemoteServer::new(ip, port).serve_on(listener);
        });
    }
}

/// A backend the server can execute ops on, plus what it can actually do. Some backend methods
/// are `todo!()` or panic for certain inputs; the server reports those as unsupported instead
/// of calling them (a panic aborts a release-built server).
pub(crate) trait ServerBackend:
    Backend
    + BackendMatMul<u8> + BackendMatMul<u16> + BackendMatMul<u32> + BackendMatMul<u64> + BackendMatMul<u128>
    + BackendMatMul<i8> + BackendMatMul<i16> + BackendMatMul<i32> + BackendMatMul<i64> + BackendMatMul<i128>
    + BackendMatMul<f32> + BackendMatMul<f64> + BackendMatMul<types::boolean>
{
    /// Whether `apply_reduce_*` handles `op`.
    fn supports_reduction(op: &ReductionOpTypes) -> bool;
    /// Whether `apply_argmax_*` is implemented.
    const SUPPORTS_ARGMAX: bool;
    /// Longest flat reduction the kernels handle.
    const MAX_REDUCE_LEN: usize = usize::MAX;
}

impl ServerBackend for Cpu {
    fn supports_reduction(op: &ReductionOpTypes) -> bool {
        // The CPU accumulators only exist for these; others panic in `get_accumulator`.
        matches!(op, ReductionOpTypes::Sum | ReductionOpTypes::Prod | ReductionOpTypes::Max | ReductionOpTypes::Min | ReductionOpTypes::Mean)
    }
    const SUPPORTS_ARGMAX: bool = false;
}

#[cfg(feature = "cuda")]
impl ServerBackend for crate::backend::cuda::Cuda {
    fn supports_reduction(op: &ReductionOpTypes) -> bool {
        // The flat CUDA kernel silently skips LogSumExp, leaving the output untouched.
        !matches!(op, ReductionOpTypes::ArgMax | ReductionOpTypes::ArgMin | ReductionOpTypes::LogSumExp)
    }
    const SUPPORTS_ARGMAX: bool = true;
    // The CUB-based kernels take the element count as an `int`.
    const MAX_REDUCE_LEN: usize = i32::MAX as usize;
}

/// Generates the type-erased buffer enum and the `Elem` trait that moves typed buffers in and out of it.
macro_rules! any_buf {
    ($($variant:ident => $t:ty),+ $(,)?) => {
        /// A backend buffer of any dtype.
        pub(crate) enum AnyBuf<B: Backend> {
            $($variant(B::Buf<$t>)),+
        }

        /// Element types that can live in an [`AnyBuf`].
        pub(crate) trait Elem: TensorValue {
            fn wrap<B: Backend>(buf: B::Buf<Self>) -> AnyBuf<B>;
            fn get<B: Backend>(buf: &AnyBuf<B>) -> Option<&B::Buf<Self>>;
            fn get_mut<B: Backend>(buf: &mut AnyBuf<B>) -> Option<&mut B::Buf<Self>>;
        }

        $(
            impl Elem for $t {
                fn wrap<B: Backend>(buf: B::Buf<Self>) -> AnyBuf<B> { AnyBuf::$variant(buf) }
                #[allow(unreachable_patterns)]
                fn get<B: Backend>(buf: &AnyBuf<B>) -> Option<&B::Buf<Self>> {
                    match buf { AnyBuf::$variant(b) => Some(b), _ => None }
                }
                #[allow(unreachable_patterns)]
                fn get_mut<B: Backend>(buf: &mut AnyBuf<B>) -> Option<&mut B::Buf<Self>> {
                    match buf { AnyBuf::$variant(b) => Some(b), _ => None }
                }
            }
        )+
    };
}

any_buf!(
    U8 => u8, U16 => u16, U32 => u32, U64 => u64, U128 => u128,
    I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128,
    F32 => f32, F64 => f64, Bool => types::boolean,
);

/// Expands `$body` once per dtype in the chosen class, with `$T` bound to the element type.
/// Dtypes outside the class produce an `UnsupportedOperation` error naming `$what`.
macro_rules! dispatch {
    (@arms $dtype:expr, $what:expr, $T:ident => $body:expr; $($variant:ident => $t:ty),+) => {
        match $dtype {
            $(DType::$variant => { type $T = $t; $body })+
            #[allow(unreachable_patterns)]
            other => Err(TensorError::UnsupportedOperation(format!("{} is not supported for dtype {:?}", $what, other))),
        }
    };
    (any $dtype:expr, $what:expr, $T:ident => $body:expr) => {
        dispatch!(@arms $dtype, $what, $T => $body;
            U8 => u8, U16 => u16, U32 => u32, U64 => u64, U128 => u128,
            I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128,
            F32 => f32, F64 => f64, BOOL => types::boolean)
    };
    (signed $dtype:expr, $what:expr, $T:ident => $body:expr) => {
        dispatch!(@arms $dtype, $what, $T => $body;
            I8 => i8, I16 => i16, I32 => i32, I64 => i64, I128 => i128, F32 => f32, F64 => f64)
    };
    (float $dtype:expr, $what:expr, $T:ident => $body:expr) => {
        dispatch!(@arms $dtype, $what, $T => $body; F32 => f32, F64 => f64)
    };
}

/// Calls the `_contiguous` / `_1d_strided` / `_nd` variant of `$method` matching `$layout`.
/// `$extra` arguments (e.g. a scalar operand) go between the buffer and the layout arguments.
macro_rules! with_layout {
    ($backend:expr, $method:ident, $buf:expr, $layout:expr $(, $extra:expr)*) => {
        paste::paste! {{
            // Evaluate the buffer before destructuring the layout, since it may borrow the layout.
            let buf = $buf;
            match $layout {
                Layout::Contiguous { start, len } => $backend.[<$method _contiguous>](buf $(, $extra)*, start, len),
                Layout::Strided1d { offset, stride, len } => $backend.[<$method _1d_strided>](buf $(, $extra)*, offset, stride, len),
                Layout::Nd { offset, shape, stride } => $backend.[<$method _nd>](buf $(, $extra)*, offset, &shape, &stride),
            }
        }}
    };
}

fn missing(id: BufId) -> TensorError {
    TensorError::RemoteError(format!("buffer {id} does not exist"))
}

fn wrong_dtype(buf: TypelessBuf, actual: DType) -> TensorError {
    TensorError::RemoteError(format!("buffer {} has dtype {:?}, request expected {:?}", buf.id, actual, buf.dtype))
}

fn invalid(msg: String) -> TensorError {
    TensorError::RemoteError(format!("invalid request: {msg}"))
}

/// Checks that every element addressed by `offset + sum(i_d * stride_d)` (for `i_d < shape_d`)
/// lies inside a buffer of `len` elements. Backends index without (or with panicking) bounds
/// checks, and a panic aborts a release-built server, so requests are validated up front.
fn check_extent(what: &str, offset: usize, shape: &[usize], stride: &[isize], len: usize) -> Result<(), TensorError> {
    if shape.len() != stride.len() {
        return Err(invalid(format!("{what}: shape has {} dims but stride has {}", shape.len(), stride.len())));
    }
    if element_count(shape).is_none() {
        return Err(invalid(format!("{what}: shape {shape:?} has too many elements")));
    }
    if shape.iter().any(|&d| d == 0) {
        // Nothing is addressed, but kernels still slice `offset..offset`.
        if offset > len {
            return Err(invalid(format!("{what}: offset {offset} is past the end of a buffer with {len} elements")));
        }
        return Ok(());
    }
    let overflow = || invalid(format!("{what}: extent overflows"));
    let (mut lo, mut hi) = (offset as i128, offset as i128);
    for (&d, &s) in shape.iter().zip(stride) {
        // |span| < 2^64 * 2^63, so the product fits in i128; the running sums are checked.
        let span = (d as i128 - 1) * s as i128;
        if span < 0 {
            lo = lo.checked_add(span).ok_or_else(overflow)?;
        } else {
            hi = hi.checked_add(span).ok_or_else(overflow)?;
        }
    }
    if lo < 0 || hi >= len as i128 {
        return Err(invalid(format!("{what}: addresses elements {lo}..={hi} of a buffer with {len} elements")));
    }
    Ok(())
}

/// Product of `dims`, or `None` if it overflows `usize`.
fn element_count(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Same error the CPU backend reports for an out-of-range `read`/`write`.
fn check_index(offset: usize, len: usize) -> Result<(), TensorError> {
    if offset >= len {
        return Err(TensorError::IdxOutOfBounds(format!("Index {offset} out of bounds for buffer of length {len}")));
    }
    Ok(())
}

fn check_layout(layout: &Layout, len: usize) -> Result<(), TensorError> {
    match layout {
        Layout::Contiguous { start, len: n } => check_extent("layout", *start, &[*n], &[1], len),
        Layout::Strided1d { offset, stride, len: n } => check_extent("layout", *offset, &[*n], &[*stride], len),
        Layout::Nd { offset, shape, stride } => check_extent("layout", *offset, shape, stride, len),
    }
}

fn check_meta(what: &str, meta: &MetaTensor, len: usize) -> Result<(), TensorError> {
    check_extent(what, meta.offset, meta.shape.as_slice(), meta.strides.as_ref(), len)
}

/// Checks a matmul operand the way the CPU/CUDA kernels address it: `rows x cols` matrices with
/// one unit-stride dimension (per `contiguity`), `batches` of them `strides[rank - 3]` apart.
fn check_matmul_operand(
    what: &str, meta: &MetaTensor, contiguity: &ContiguityTypes, batches: usize, rows: usize, cols: usize, len: usize,
) -> Result<(), TensorError> {
    let rank = meta.rank();
    if rank < 2 {
        return Err(invalid(format!("{what}: matmul operands need rank >= 2")));
    }
    let shape = meta.shape.as_slice();
    if shape[rank - 2] != rows || shape[rank - 1] != cols {
        return Err(invalid(format!("{what}: expected a {rows}x{cols} matrix, got shape {shape:?}")));
    }
    if shape[..rank - 2].iter().product::<usize>() != batches {
        return Err(invalid(format!("{what}: batch dims {:?} do not multiply to {batches}", &shape[..rank - 2])));
    }
    let strides: &[isize] = meta.strides.as_ref();
    let (row_stride, col_stride) = (strides[rank - 2], strides[rank - 1]);
    let unit_ok = match contiguity {
        ContiguityTypes::RowMajor => col_stride == 1 || cols <= 1,
        ContiguityTypes::ColumnMajor => row_stride == 1 || rows <= 1,
        ContiguityTypes::None => false,
    };
    if !unit_ok || row_stride < 0 || col_stride < 0 {
        return Err(invalid(format!("{what}: strides {strides:?} do not match {contiguity:?} layout")));
    }
    let batch_stride = if rank > 2 { strides[rank - 3] } else { 0 };
    // Kernels step through batches by strides[rank - 3] alone, which is only right when the
    // batch dims collapse into one.
    let collapsible = (0..rank.saturating_sub(3))
        .all(|d| shape[d] <= 1 || strides[d] == strides[d + 1] * shape[d + 1] as isize);
    if batch_stride < 0 || !collapsible {
        return Err(invalid(format!("{what}: batch dims must be collapsible, got strides {strides:?}")));
    }
    let as_batched = [batches, rows, cols];
    let as_strides = [batch_stride, row_stride, col_stride];
    check_extent(what, meta.offset, &as_batched, &as_strides, len)
}

/// Checks a reduction along `dim` of a contiguous, row-major `src` into `dst_len` outputs.
/// The kernels index `src` as a dense row-major block of `size` elements (the CPU kernel from
/// index 0, ignoring the offset, which is a known CPU-backend bug; CUDA from the offset), so
/// require exactly that layout and that `offset + size` fits.
///
/// Returns `false` when there are no outputs, in which case the kernel must not run: with a
/// zero-size dim the kernels still compute at least one output (`inner_dimensions` clamps to 1).
fn check_reduce_nd(meta: &MetaTensor, dim: Dim, src_len: usize, dst_len: usize) -> Result<bool, TensorError> {
    let shape = meta.shape.as_slice();
    let strides: &[isize] = meta.strides.as_ref();
    if strides.len() != shape.len() {
        return Err(invalid(format!("reduction source has {} dims but {} strides", shape.len(), strides.len())));
    }
    if dim >= shape.len() {
        return Err(invalid(format!("reduction dim {dim} out of range for rank {}", shape.len())));
    }
    let too_big = || invalid(format!("reduction source shape {shape:?} has too many elements"));
    let size = element_count(shape).ok_or_else(too_big)?;
    let outputs = element_count(&shape[..dim]).zip(element_count(&shape[dim + 1..]))
        .and_then(|(outer, inner)| outer.checked_mul(inner))
        .ok_or_else(too_big)?;
    if outputs == 0 {
        return Ok(false);
    }
    let mut expected = 1isize;
    for d in (0..shape.len()).rev() {
        if shape[d] > 1 && strides[d] != expected {
            return Err(invalid(format!("reduction source must be row-major contiguous, got strides {strides:?} for shape {shape:?}")));
        }
        expected = expected.saturating_mul(shape[d] as isize);
    }
    if meta.offset.checked_add(size).is_none_or(|end| end > src_len) {
        return Err(invalid(format!("reduction source of {size} elements at offset {} exceeds buffer of {src_len}", meta.offset)));
    }
    if dst_len < outputs {
        return Err(invalid(format!("reduction output needs {outputs} elements, buffer has {dst_len}")));
    }
    Ok(true)
}

fn check_reduce_flat(start: usize, len: usize, src_len: usize, dst_len: usize) -> Result<(), TensorError> {
    check_extent("reduction source", start, &[len], &[1], src_len)?;
    if dst_len == 0 {
        return Err(invalid("reduction output buffer is empty".into()));
    }
    Ok(())
}

fn check_reduction_op<B: ServerBackend>(op: &ReductionOpTypes, arg: bool) -> Result<(), TensorError> {
    let is_arg = matches!(op, ReductionOpTypes::ArgMax | ReductionOpTypes::ArgMin);
    if arg != is_arg {
        return Err(invalid(format!("{op:?} is not valid for this reduction entry point")));
    }
    let supported = if arg { B::SUPPORTS_ARGMAX } else { B::supports_reduction(op) };
    if !supported {
        return Err(TensorError::UnsupportedOperation(format!("{op:?} reduction is not implemented by the server's backend")));
    }
    Ok(())
}

/// Buffers owned by one client connection.
pub(crate) struct Store<B: Backend> {
    bufs: HashMap<BufId, AnyBuf<B>>,
    /// Bytes held by live buffers, and the optional cap on it.
    bytes: usize,
    max_bytes: Option<usize>,
}

impl<B: Backend> Store<B> {
    fn new(max_bytes: Option<usize>) -> Self {
        Self { bufs: HashMap::new(), bytes: 0, max_bytes }
    }

    fn size_of<T: Elem>(len: usize) -> Result<usize, TensorError> {
        len.checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| invalid(format!("allocation of {len} {:?} elements overflows", T::DTYPE)))
    }

    /// Checks that `len` more elements of `T` fit within the session's budget. Called before
    /// the backend allocates, because a failed allocation aborts the process.
    fn reserve<T: Elem>(&self, dst: TypelessBuf, len: usize) -> Result<(), TensorError> {
        if dst.dtype != T::DTYPE {
            return Err(wrong_dtype(dst, T::DTYPE));
        }
        if self.bufs.contains_key(&dst.id) {
            return Err(TensorError::RemoteError(format!("buffer {} already exists", dst.id)));
        }
        let bytes = Self::size_of::<T>(len)?;
        let total = self.bytes.checked_add(bytes);
        if let Some(max) = self.max_bytes {
            if total.is_none_or(|t| t > max) {
                return Err(TensorError::RemoteError(format!(
                    "allocating {bytes} bytes would exceed this session's limit of {max} bytes ({} in use)", self.bytes
                )));
            }
        }
        if bytes > isize::MAX as usize {
            return Err(invalid(format!("allocation of {bytes} bytes is too large")));
        }
        Ok(())
    }

    /// Stores a buffer previously checked with [`Self::reserve`].
    fn insert<T: Elem>(&mut self, dst: TypelessBuf, len: usize, buf: B::Buf<T>) -> Result<(), TensorError> {
        self.reserve::<T>(dst, len)?;
        self.bytes += Self::size_of::<T>(len)?;
        self.bufs.insert(dst.id, T::wrap(buf));
        Ok(())
    }

    fn remove(&mut self, id: BufId, backend: &B) {
        if let Some(buf) = self.bufs.remove(&id) {
            let bytes = any_len(backend, &buf) * dtype_size(dtype_of(&buf));
            self.bytes = self.bytes.saturating_sub(bytes);
        }
    }

    fn get<T: Elem>(&self, buf: TypelessBuf) -> Result<&B::Buf<T>, TensorError> {
        let any = self.bufs.get(&buf.id).ok_or_else(|| missing(buf.id))?;
        T::get(any).filter(|_| buf.dtype == T::DTYPE).ok_or_else(|| wrong_dtype(buf, dtype_of(any)))
    }

    fn get_mut<T: Elem>(&mut self, buf: TypelessBuf) -> Result<&mut B::Buf<T>, TensorError> {
        let any = self.bufs.get_mut(&buf.id).ok_or_else(|| missing(buf.id))?;
        let actual = dtype_of(any);
        T::get_mut(any).filter(|_| buf.dtype == T::DTYPE).ok_or_else(|| wrong_dtype(buf, actual))
    }

    /// Raw pointers to several buffers that may alias one another. Each distinct id is looked up
    /// exactly once, since a second `get_mut` on the same slot would invalidate the first pointer.
    fn ptrs<T: Elem, const N: usize>(&mut self, bufs: [TypelessBuf; N]) -> Result<[*mut B::Buf<T>; N], TensorError> {
        let mut out: [*mut B::Buf<T>; N] = [std::ptr::null_mut(); N];
        for i in 0..N {
            out[i] = match (0..i).find(|&j| bufs[j].id == bufs[i].id) {
                Some(j) => out[j],
                None => self.get_mut::<T>(bufs[i])? as *mut _,
            };
        }
        Ok(out)
    }
}

fn any_len<B: Backend>(backend: &B, buf: &AnyBuf<B>) -> usize {
    match buf {
        AnyBuf::U8(b) => backend.len(b), AnyBuf::U16(b) => backend.len(b), AnyBuf::U32(b) => backend.len(b),
        AnyBuf::U64(b) => backend.len(b), AnyBuf::U128(b) => backend.len(b), AnyBuf::I8(b) => backend.len(b),
        AnyBuf::I16(b) => backend.len(b), AnyBuf::I32(b) => backend.len(b), AnyBuf::I64(b) => backend.len(b),
        AnyBuf::I128(b) => backend.len(b), AnyBuf::F32(b) => backend.len(b), AnyBuf::F64(b) => backend.len(b),
        AnyBuf::Bool(b) => backend.len(b),
    }
}

fn dtype_size(dtype: DType) -> usize {
    let size: Result<usize, TensorError> = dispatch!(any dtype, "size", T => Ok(std::mem::size_of::<T>()));
    size.unwrap_or(0)
}

fn dtype_of<B: Backend>(buf: &AnyBuf<B>) -> DType {
    match buf {
        AnyBuf::U8(_) => DType::U8, AnyBuf::U16(_) => DType::U16, AnyBuf::U32(_) => DType::U32,
        AnyBuf::U64(_) => DType::U64, AnyBuf::U128(_) => DType::U128, AnyBuf::I8(_) => DType::I8,
        AnyBuf::I16(_) => DType::I16, AnyBuf::I32(_) => DType::I32, AnyBuf::I64(_) => DType::I64,
        AnyBuf::I128(_) => DType::I128, AnyBuf::F32(_) => DType::F32, AnyBuf::F64(_) => DType::F64,
        AnyBuf::Bool(_) => DType::BOOL,
    }
}

/// One client's backend and buffers.
pub(crate) struct Session<B: ServerBackend> {
    backend: B,
    store: Store<B>,
    device: DeviceType,
}

impl<B: ServerBackend> Session<B> {
    fn new(backend: B, device: DeviceType, max_bytes: Option<usize>) -> Self {
        Self { backend, store: Store::new(max_bytes), device }
    }

    fn execute(&mut self, op: Op) -> Result<Reply, TensorError> {
        let Self { backend, store, device } = self;
        match op {
            Op::Hello { .. } => Err(TensorError::RemoteError("unexpected second handshake".into())),
            Op::Free { buf } => {
                store.remove(buf, backend);
                Ok(Reply::Ack)
            }
            Op::Sync => Ok(Reply::Ack),
            Op::DeviceType => Ok(Reply::DeviceType(device.clone())),
            Op::Alloc { dst, len } => dispatch!(any dst.dtype, "alloc", T => {
                store.reserve::<T>(dst, len)?;
                let buf = backend.alloc::<T>(len)?;
                store.insert::<T>(dst, len, buf).map(|_| Reply::Ack)
            }),
            Op::AllocFromSlice { dst, src } => dispatch!(any dst.dtype, "alloc_from_slice", T => {
                let data = src.to_boxed_slice::<T>()?;
                let len = data.len();
                store.reserve::<T>(dst, len)?;
                let buf = backend.alloc_from_slice::<T>(data)?;
                store.insert::<T>(dst, len, buf).map(|_| Reply::Ack)
            }),
            Op::CopyFromSlice { dst, src } => dispatch!(any dst.dtype, "copy_from_slice", T => {
                let data = src.to_boxed_slice::<T>()?;
                let buf = store.get_mut::<T>(dst)?;
                if backend.len(buf) != data.len() {
                    return Err(TensorError::SizeMismatch(format!(
                        "copy_from_slice: source has {} elements, destination has {}", data.len(), backend.len(buf)
                    )));
                }
                backend.copy_from_slice(buf, &data).map(|_| Reply::Ack)
            }),
            Op::CopyRangeWithin { dst, src, dst_offset, src_offset, len } => dispatch!(any dst.dtype, "copy_range_within", T => {
                if dst.id == src.id {
                    return Err(TensorError::RemoteError("copy_range_within source and destination must be different buffers".into()));
                }
                let [dst_ptr, src_ptr] = store.ptrs::<T, 2>([dst, src])?;
                // SAFETY (for the lengths below too): the pointers come from the store, which is
                // not touched while they are in use.
                let (dst_len, src_len) = unsafe { (backend.len(&*dst_ptr), backend.len(&*src_ptr)) };
                check_extent("copy_range_within dst", dst_offset, &[len], &[1], dst_len)?;
                check_extent("copy_range_within src", src_offset, &[len], &[1], src_len)?;
                // SAFETY: distinct ids, so the pointers refer to different buffers in the store,
                // which is not touched while they are in use.
                let (dst_buf, src_buf) = unsafe { (&mut *dst_ptr, &*src_ptr) };
                backend.copy_range_within(dst_buf, src_buf, dst_offset, src_offset, len).map(|_| Reply::Ack)
            }),
            Op::Read { buf, offset } => dispatch!(any buf.dtype, "read", T => {
                let src = store.get::<T>(buf)?;
                check_index(offset, backend.len(src))?;
                let value = backend.read(src, offset)?;
                Ok(Reply::Value(Value::from_value(value)))
            }),
            Op::Write { buf, offset, value } => dispatch!(any buf.dtype, "write", T => {
                let value = value.to_value::<T>()?;
                let dst = store.get_mut::<T>(buf)?;
                check_index(offset, backend.len(dst))?;
                backend.write(dst, offset, value).map(|_| Reply::Ack)
            }),
            Op::Copy { src, dst } => dispatch!(any src.dtype, "copy", T => {
                let len = backend.len(store.get::<T>(src)?);
                store.reserve::<T>(dst, len)?;
                let copy = backend.copy(store.get::<T>(src)?)?;
                store.insert::<T>(dst, len, copy).map(|_| Reply::Ack)
            }),
            Op::Dump { src } => dispatch!(any src.dtype, "dump", T => {
                let data = backend.dump(store.get::<T>(src)?)?;
                Ok(Reply::Slice(Slice::from_boxed_slice(data)))
            }),
            Op::Broadcast { left, right, dst, op } => dispatch!(any dst.0.dtype, "broadcast", T => {
                if left.1.shape != dst.1.shape || right.1.shape != dst.1.shape {
                    return Err(invalid(format!(
                        "broadcast operands must be pre-broadcast to the output shape {:?}", dst.1.shape
                    )));
                }
                let [left_ptr, right_ptr, dst_ptr] = store.ptrs::<T, 3>([left.0, right.0, dst.0])?;
                // SAFETY: pointers into the store, which is not touched while they are in use.
                unsafe {
                    check_meta("broadcast left", &left.1, backend.len(&*left_ptr))?;
                    check_meta("broadcast right", &right.1, backend.len(&*right_ptr))?;
                    check_meta("broadcast dst", &dst.1, backend.len(&*dst_ptr))?;
                }
                // Broadcast explicitly allows dst to alias an input and works on raw pointers.
                backend.broadcast(
                    (left_ptr as *const _, &left.1),
                    (right_ptr as *const _, &right.1),
                    (dst_ptr, &dst.1),
                    op,
                ).map(|_| Reply::Ack)
            }),
            Op::Matmul { lhs, rhs, dst, b, m, k, n } => dispatch!(any dst.dtype, "matmul", T => {
                if dst.id == lhs.0.id || dst.id == rhs.0.id {
                    return Err(TensorError::RemoteError("matmul destination must not alias an input".into()));
                }
                let [lhs_ptr, rhs_ptr, dst_ptr] = store.ptrs::<T, 3>([lhs.0, rhs.0, dst])?;
                // SAFETY: one lookup per distinct id; dst is distinct from both inputs and the
                // inputs (which may be the same buffer, e.g. `x @ x`) are only read.
                let (lhs_buf, rhs_buf, dst_buf) = unsafe { (&*lhs_ptr, &*rhs_ptr, &mut *dst_ptr) };
                if lhs.1.rank() != rhs.1.rank() {
                    return Err(invalid("matmul operands must have the same rank".into()));
                }
                if lhs.2 != rhs.2 && !matches!(T::DTYPE, DType::F32 | DType::F64) {
                    // The generic (non-BLAS) kernels only handle matching layouts.
                    return Err(TensorError::UnsupportedOperation(format!(
                        "matmul with {:?} x {:?} operands for dtype {:?}", lhs.2, rhs.2, T::DTYPE
                    )));
                }
                check_matmul_operand("matmul lhs", &lhs.1, &lhs.2, b, m, k, backend.len(lhs_buf))?;
                check_matmul_operand("matmul rhs", &rhs.1, &rhs.2, b, k, n, backend.len(rhs_buf))?;
                let out = b.checked_mul(m).and_then(|x| x.checked_mul(n))
                    .ok_or_else(|| invalid("matmul output size overflows".into()))?;
                if backend.len(dst_buf) < out {
                    return Err(invalid(format!("matmul dst has {} elements, needs {out}", backend.len(dst_buf))));
                }
                <B as BackendMatMul<T>>::matmul(
                    backend,
                    (lhs_buf, &lhs.1, lhs.2),
                    (rhs_buf, &rhs.1, rhs.2),
                    dst_buf,
                    b, m, k, n,
                ).map(|_| Reply::Ack)
            }),
            Op::Fill { buf, value, layout } => dispatch!(any buf.dtype, "fill", T => {
                let value = value.to_value::<T>()?;
                with_layout!(backend, fill, target::<B, T>(backend, store, buf, &layout)?, layout, value).map(|_| Reply::Ack)
            }),
            Op::Convert { src, dst } => dispatch!(any src.dtype, "convert", T => dispatch!(any dst.dtype, "convert", N => {
                let (src_ptr, dst_ptr) = distinct_ptrs::<B, T, N>(store, src, dst)?;
                // SAFETY: distinct ids, one lookup each (distinct_ptrs); the store is not mutated
                // while these references are live.
                let (src_buf, dst_buf) = unsafe { (&*src_ptr, &mut *dst_ptr) };
                if backend.len(src_buf) != backend.len(dst_buf) {
                    return Err(TensorError::SizeMismatch(format!(
                        "Buffer size mismatch in convert: src size {}, dst size {}", backend.len(src_buf), backend.len(dst_buf)
                    )));
                }
                backend.convert::<T, N>(src_buf, dst_buf).map(|_| Reply::Ack)
            })),
            Op::ReduceFlat { src, dst, start, len, op } => dispatch!(float src.dtype, "reduce", T => {
                check_reduction_op::<B>(&op, false)?;
                let (src_ptr, dst_ptr) = distinct_ptrs::<B, T, T>(store, src, dst)?;
                // SAFETY: distinct ids, one lookup each (distinct_ptrs); the store is not mutated
                // while these references are live.
                let (src_buf, dst_buf) = unsafe { (&*src_ptr, &mut *dst_ptr) };
                check_reduce_flat(start, len, backend.len(src_buf), backend.len(dst_buf))?;
                if len > B::MAX_REDUCE_LEN {
                    return Err(TensorError::UnsupportedOperation(format!("reductions over more than {} elements", B::MAX_REDUCE_LEN)));
                }
                backend.apply_reduce_contiguous_flat(src_buf, dst_buf, start, len, op).map(|_| Reply::Ack)
            }),
            Op::ReduceNd { src, dst, dim, op } => dispatch!(float src.0.dtype, "reduce", T => {
                check_reduction_op::<B>(&op, false)?;
                let (src_ptr, dst_ptr) = distinct_ptrs::<B, T, T>(store, src.0, dst.0)?;
                // SAFETY: distinct ids, one lookup each (distinct_ptrs); the store is not mutated
                // while these references are live.
                let (src_buf, dst_buf) = unsafe { (&*src_ptr, &mut *dst_ptr) };
                if !check_reduce_nd(&src.1, dim, backend.len(src_buf), backend.len(dst_buf))? {
                    return Ok(Reply::Ack);
                }
                backend.apply_reduce_contiguous_nd((src_buf, &src.1), (dst_buf, &dst.1), dim, op).map(|_| Reply::Ack)
            }),
            Op::ArgFlat { src, dst, start, len, op } => dispatch!(float src.dtype, "argmax", T => {
                check_reduction_op::<B>(&op, true)?;
                let (src_ptr, dst_ptr) = distinct_ptrs::<B, T, u64>(store, src, dst)?;
                // SAFETY: distinct ids, one lookup each (distinct_ptrs); the store is not mutated
                // while these references are live.
                let (src_buf, dst_buf) = unsafe { (&*src_ptr, &mut *dst_ptr) };
                check_reduce_flat(start, len, backend.len(src_buf), backend.len(dst_buf))?;
                if len > B::MAX_REDUCE_LEN {
                    return Err(TensorError::UnsupportedOperation(format!("reductions over more than {} elements", B::MAX_REDUCE_LEN)));
                }
                backend.apply_argmax_contiguous_flat(src_buf, dst_buf, start, len, op).map(|_| Reply::Ack)
            }),
            Op::ArgNd { src, dst, dim, op } => dispatch!(float src.0.dtype, "argmax", T => {
                check_reduction_op::<B>(&op, true)?;
                let (src_ptr, dst_ptr) = distinct_ptrs::<B, T, u64>(store, src.0, dst.0)?;
                // SAFETY: distinct ids, one lookup each (distinct_ptrs); the store is not mutated
                // while these references are live.
                let (src_buf, dst_buf) = unsafe { (&*src_ptr, &mut *dst_ptr) };
                if !check_reduce_nd(&src.1, dim, backend.len(src_buf), backend.len(dst_buf))? {
                    return Ok(Reply::Ack);
                }
                backend.apply_argmax_contiguous_nd((src_buf, &src.1), (dst_buf, &dst.1), dim, op).map(|_| Reply::Ack)
            }),
            Op::Unary { buf, op, layout } => unary(backend, store, buf, op, layout).map(|_| Reply::Ack),
            Op::Scalar { buf, op, value, layout } => scalar(backend, store, buf, op, value, layout).map(|_| Reply::Ack),
        }
    }
}

/// Generates `float_unary`, which maps each listed float-only unary op to its backend method.
macro_rules! float_unary_ops {
    ($($op:ident => $method:ident),+ $(,)?) => {
        fn float_unary<B: ServerBackend, T: Elem + crate::core::value::WeightValue>(
            backend: &B, buf: &mut B::Buf<T>, op: UnaryOp, layout: Layout,
        ) -> Result<(), TensorError> {
            match op {
                $(UnaryOp::$op => with_layout!(backend, $method, buf, layout),)+
                other => Err(TensorError::UnsupportedOperation(format!("{} is not a float-only unary op", other.name()))),
            }
        }
    };
}

float_unary_ops!(
    Sigmoid => apply_sigmoid, Silu => apply_silu, Tanh => apply_tanh, Sqrt => apply_sqrt,
    Ln => apply_ln, Expm1 => apply_expm1, Ln1p => apply_ln1p, Floor => apply_floor,
    Ceil => apply_ceil, Round => apply_round, Trunc => apply_trunc, Sin => apply_sin,
    Cos => apply_cos, Tan => apply_tan, Asin => apply_asin, Acos => apply_acos,
    Atan => apply_atan, Sinh => apply_sinh, Cosh => apply_cosh, Asinh => apply_asinh,
    Acosh => apply_acosh, Atanh => apply_atanh, Rsqrt => apply_rsqrt,
    Reciprocal => apply_reciprocal, Square => apply_square, Cube => apply_cube,
    Exp => apply_exp, Sign => apply_sign,
);

/// Pointers to a source and a different destination buffer, possibly of different dtypes.
/// Each comes from one lookup, and the store must not be touched while they are in use.
#[allow(clippy::type_complexity)]
fn distinct_ptrs<B: ServerBackend, S: Elem, D: Elem>(store: &mut Store<B>, src: TypelessBuf, dst: TypelessBuf) -> Result<(*const B::Buf<S>, *mut B::Buf<D>), TensorError> {
    if src.id == dst.id {
        return Err(invalid("source and destination must be different buffers".into()));
    }
    let src_ptr = store.get_mut::<S>(src)? as *const B::Buf<S>;
    let dst_ptr = store.get_mut::<D>(dst)? as *mut B::Buf<D>;
    Ok((src_ptr, dst_ptr))
}

/// Looks up the buffer an elementwise op writes and checks the layout stays inside it.
fn target<'a, B: ServerBackend, T: Elem>(backend: &B, store: &'a mut Store<B>, buf: TypelessBuf, layout: &Layout) -> Result<&'a mut B::Buf<T>, TensorError> {
    let target = store.get_mut::<T>(buf)?;
    check_layout(layout, backend.len(target))?;
    Ok(target)
}

fn unary<B: ServerBackend>(backend: &B, store: &mut Store<B>, buf: TypelessBuf, op: UnaryOp, layout: Layout) -> Result<(), TensorError> {
    match op {
        UnaryOp::Neg => dispatch!(signed buf.dtype, op.name(), T => with_layout!(backend, apply_neg, target::<B, T>(backend, store, buf, &layout)?, layout)),
        UnaryOp::Relu => dispatch!(any buf.dtype, op.name(), T => with_layout!(backend, apply_relu, target::<B, T>(backend, store, buf, &layout)?, layout)),
        UnaryOp::Abs => dispatch!(any buf.dtype, op.name(), T => with_layout!(backend, apply_abs, target::<B, T>(backend, store, buf, &layout)?, layout)),
        _ => dispatch!(float buf.dtype, op.name(), T => float_unary::<B, T>(backend, target::<B, T>(backend, store, buf, &layout)?, op, layout)),
    }
}

fn scalar<B: ServerBackend>(backend: &B, store: &mut Store<B>, buf: TypelessBuf, op: ScalarOp, value: Value, layout: Layout) -> Result<(), TensorError> {
    macro_rules! run {
        ($class:ident, $method:ident) => {
            dispatch!($class buf.dtype, op.name(), T => {
                let value = value.to_value::<T>()?;
                with_layout!(backend, $method, target::<B, T>(backend, store, buf, &layout)?, layout, value)
            })
        };
    }
    match op {
        ScalarOp::Add => run!(any, scalar_apply_add),
        ScalarOp::Sub => run!(any, scalar_apply_sub),
        ScalarOp::Mul => run!(any, scalar_apply_mul),
        ScalarOp::Div => run!(any, scalar_apply_div),
        ScalarOp::LeakyRelu => run!(any, scalar_apply_leaky_relu),
        ScalarOp::Log => run!(float, scalar_apply_log),
        ScalarOp::Log1p => run!(float, scalar_apply_log1p),
        ScalarOp::Elu => run!(float, scalar_apply_elu),
    }
}

/// Backend selected for a connection.
enum DeviceSession {
    Cpu(Session<Cpu>),
    #[cfg(feature = "cuda")]
    Cuda(Session<crate::backend::cuda::Cuda>),
}

impl DeviceSession {
    fn open(device: RemoteDevice, config: &ServerConfig) -> Result<Self, TensorError> {
        match device {
            RemoteDevice::Cpu => Ok(DeviceSession::Cpu(Session::new(Cpu::new(), DeviceType::Cpu, config.max_session_bytes))),
            #[cfg(feature = "cuda")]
            RemoteDevice::Cuda(ordinal) => {
                let backend = crate::backend::cuda::Cuda::construct(ordinal)?;
                Ok(DeviceSession::Cuda(Session::new(backend, DeviceType::Cuda(ordinal), config.max_session_bytes)))
            }
            #[cfg(not(feature = "cuda"))]
            RemoteDevice::Cuda(_) => Err(TensorError::UnsupportedOperation(
                "this server was built without CUDA support".into(),
            )),
        }
    }

    fn device(&self) -> DeviceType {
        match self {
            DeviceSession::Cpu(session) => session.device.clone(),
            #[cfg(feature = "cuda")]
            DeviceSession::Cuda(session) => session.device.clone(),
        }
    }

    fn execute(&mut self, op: Op) -> Result<Reply, TensorError> {
        match self {
            DeviceSession::Cpu(session) => session.execute(op),
            #[cfg(feature = "cuda")]
            DeviceSession::Cuda(session) => session.execute(op),
        }
    }
}

fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> &str {
    panic.downcast_ref::<&str>().copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Reads requests on the calling thread and executes them in order on a worker thread, so slow
/// ops don't stop the socket from being drained. Returning drops the session and its buffers.
fn handle_connection(stream: TcpStream, config: ServerConfig) {
    let _ = stream.set_nodelay(true);
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("remote server: failed to clone stream: {e}");
            return;
        }
    };
    // Bounded, so a client that outpaces execution is slowed down instead of growing the queue.
    let (tx, rx) = flume::bounded::<Request>(config.queue_depth.max(1));

    let frame_limit = config.frame_limit();
    let worker = thread::spawn(move || {
        // Shut the socket down however this thread exits (including a panic), so the reader
        // loop and the client both see the connection close instead of hanging.
        struct ShutdownOnDrop(TcpStream);
        impl Drop for ShutdownOnDrop {
            fn drop(&mut self) {
                let _ = self.0.shutdown(std::net::Shutdown::Both);
            }
        }
        let mut writer = ShutdownOnDrop(writer);
        let writer = &mut writer.0;
        let mut session: Option<DeviceSession> = None;
        for request in rx.iter() {
            let id = request.id;
            let reply = request.reply;
            let silent = matches!(request.op, Op::Free { .. });
            let result = match (&mut session, request.op) {
                (None, Op::Hello { version, device }) if version == PROTOCOL_VERSION => {
                    // Opening a backend can panic (e.g. CUDA context creation); report it.
                    std::panic::catch_unwind(|| DeviceSession::open(device, &config))
                        .unwrap_or_else(|panic| Err(TensorError::RemoteError(format!(
                            "failed to open {device:?} session: {}", panic_message(&panic)
                        ))))
                        .map(|opened| {
                        let device = opened.device();
                        session = Some(opened);
                        Reply::DeviceType(device)
                    })
                }
                (None, Op::Hello { version, .. }) => Err(TensorError::RemoteError(format!(
                    "protocol version mismatch: client speaks {version}, server speaks {PROTOCOL_VERSION}"
                ))),
                (None, _) => Err(TensorError::RemoteError("handshake required before any other request".into())),
                // Requests are validated before they reach the backend, so a panic here is a backend
                // bug. With unwinding it becomes an error for this request; note that a server built
                // with `panic = "abort"` (this workspace's release profile) still aborts.
                (Some(session), op) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.execute(op)))
                    .unwrap_or_else(|panic| Err(TensorError::RemoteError(format!("server panicked: {}", panic_message(&panic))))),
            };
            let handshake_failed = session.is_none();
            // Fire-and-forget requests only hear back when they fail.
            if reply || (result.is_err() && !silent) {
                if let Err(e) = write_frame(writer, &Response { id, result }) {
                    tracing::warn!("remote server: failed to send response: {e}");
                    break;
                }
            }
            if handshake_failed {
                break;
            }
        }
    });

    let mut reader = stream;
    loop {
        match read_frame_limited::<_, Request>(&mut reader, frame_limit) {
            Ok(Some(request)) => {
                if tx.send(request).is_err() {
                    break; // worker stopped
                }
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("remote server: dropping connection: {e}");
                break;
            }
        }
    }
    drop(tx);
    let _ = worker.join();
}
