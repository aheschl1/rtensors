//! Remote tensor server. Each client connection gets its own session (buffer store + backend),
//! executed by one worker thread in request order and dropped when the client disconnects.

use std::{collections::HashMap, net::{IpAddr, TcpListener, TcpStream}, thread::{self, JoinHandle}};

use crate::{backend::{cpu::Cpu, ContiguityTypes, remote::protocol::{read_frame, write_frame, BufId, Layout, Op, Reply, Request, Response, ScalarOp, Slice, TypelessBuf, UnaryOp, Value, PROTOCOL_VERSION}, Backend, BackendMatMul}, core::{primitives::DeviceType, tensor::TensorError, value::{types, DType, TensorValue}, MetaTensor}};

pub(crate) struct RemoteServer {
    address: IpAddr,
    port: u16
}

impl RemoteServer {
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
        }
    }

    pub fn serve(&mut self) -> std::io::Result<()> {
        let listener = TcpListener::bind((self.address, self.port))?;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    thread::spawn(move || handle_connection(stream));
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

/// A backend the server can execute every op on.
pub(crate) trait ServerBackend:
    Backend
    + BackendMatMul<u8> + BackendMatMul<u16> + BackendMatMul<u32> + BackendMatMul<u64> + BackendMatMul<u128>
    + BackendMatMul<i8> + BackendMatMul<i16> + BackendMatMul<i32> + BackendMatMul<i64> + BackendMatMul<i128>
    + BackendMatMul<f32> + BackendMatMul<f64> + BackendMatMul<types::boolean>
{
}

impl<B> ServerBackend for B where
    B: Backend
    + BackendMatMul<u8> + BackendMatMul<u16> + BackendMatMul<u32> + BackendMatMul<u64> + BackendMatMul<u128>
    + BackendMatMul<i8> + BackendMatMul<i16> + BackendMatMul<i32> + BackendMatMul<i64> + BackendMatMul<i128>
    + BackendMatMul<f32> + BackendMatMul<f64> + BackendMatMul<types::boolean>
{
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
    if shape.iter().any(|&d| d == 0) {
        return Ok(());
    }
    let (mut lo, mut hi) = (offset as i128, offset as i128);
    for (&d, &s) in shape.iter().zip(stride) {
        let span = (d as i128 - 1) * s as i128;
        if span < 0 { lo += span } else { hi += span }
    }
    if lo < 0 || hi >= len as i128 {
        return Err(invalid(format!("{what}: addresses elements {lo}..={hi} of a buffer with {len} elements")));
    }
    Ok(())
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

/// Buffers owned by one client connection.
pub(crate) struct Store<B: Backend> {
    bufs: HashMap<BufId, AnyBuf<B>>,
}

impl<B: Backend> Store<B> {
    fn new() -> Self {
        Self { bufs: HashMap::new() }
    }

    fn insert<T: Elem>(&mut self, dst: TypelessBuf, buf: B::Buf<T>) -> Result<(), TensorError> {
        if dst.dtype != T::DTYPE {
            return Err(wrong_dtype(dst, T::DTYPE));
        }
        if self.bufs.contains_key(&dst.id) {
            return Err(TensorError::RemoteError(format!("buffer {} already exists", dst.id)));
        }
        self.bufs.insert(dst.id, T::wrap(buf));
        Ok(())
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
    fn new(backend: B, device: DeviceType) -> Self {
        Self { backend, store: Store::new(), device }
    }

    fn execute(&mut self, op: Op) -> Result<Reply, TensorError> {
        let Self { backend, store, device } = self;
        match op {
            Op::Hello { .. } => Err(TensorError::RemoteError("unexpected second handshake".into())),
            Op::Sync => Ok(Reply::Ack),
            Op::DeviceType => Ok(Reply::DeviceType(device.clone())),
            Op::Alloc { dst, len } => dispatch!(any dst.dtype, "alloc", T => {
                let buf = backend.alloc::<T>(len)?;
                store.insert::<T>(dst, buf).map(|_| Reply::Ack)
            }),
            Op::AllocFromSlice { dst, src } => dispatch!(any dst.dtype, "alloc_from_slice", T => {
                let buf = backend.alloc_from_slice::<T>(src.to_boxed_slice::<T>()?)?;
                store.insert::<T>(dst, buf).map(|_| Reply::Ack)
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
                let copy = backend.copy(store.get::<T>(src)?)?;
                store.insert::<T>(dst, copy).map(|_| Reply::Ack)
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
}

impl DeviceSession {
    fn execute(&mut self, op: Op) -> Result<Reply, TensorError> {
        match self {
            DeviceSession::Cpu(session) => session.execute(op),
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
fn handle_connection(stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let mut writer = match stream.try_clone() {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("remote server: failed to clone stream: {e}");
            return;
        }
    };
    let (tx, rx) = flume::unbounded::<Request>();

    let worker = thread::spawn(move || {
        let mut session: Option<DeviceSession> = None;
        for request in rx.iter() {
            let id = request.id;
            let reply = request.reply;
            let result = match (&mut session, request.op) {
                (None, Op::Hello { version }) if version == PROTOCOL_VERSION => {
                    session = Some(DeviceSession::Cpu(Session::new(Cpu::new(), DeviceType::Cpu)));
                    Ok(Reply::Ack)
                }
                (None, Op::Hello { version }) => Err(TensorError::RemoteError(format!(
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
            if reply || result.is_err() {
                if let Err(e) = write_frame(&mut writer, &Response { id, result }) {
                    tracing::warn!("remote server: failed to send response: {e}");
                    break;
                }
            }
            if handshake_failed {
                break;
            }
        }
        let _ = writer.shutdown(std::net::Shutdown::Both);
    });

    let mut reader = stream;
    loop {
        match read_frame::<_, Request>(&mut reader) {
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
