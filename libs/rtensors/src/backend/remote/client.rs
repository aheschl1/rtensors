use std::{collections::HashMap, fmt::Debug, net::{IpAddr, Shutdown, TcpStream}, sync::{atomic::{AtomicU64, Ordering}, Arc, Mutex, OnceLock}};

use crate::{backend::{remote::{get_backend_default, protocol::{read_frame, write_frame, BufId, Layout, Op, Reply, Request, Response, ScalarOp, Slice, TypelessBuf, UnaryOp, Value, PROTOCOL_VERSION}}, Backend, BackendMatMul}, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::{TensorValue, WeightValue}, Dim, MetaTensor}, ops::{base::BinaryOpType, linalg::ConvConfig2D, reduction::ReductionOpTypes}};

/// Source of process-unique connection ids, used to catch buffers from one connection being
/// passed to another (buffer ids are only meaningful within their own connection).
static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

/// Request ids carry the sending thread's tag in their high bits, so the failure of a pipelined
/// request is reported to the thread that sent it rather than to whichever thread calls next.
const THREAD_TAG_SHIFT: u32 = 40;
static NEXT_THREAD_TAG: AtomicU64 = AtomicU64::new(0);

fn thread_tag() -> u64 {
    thread_local! {
        static TAG: u64 = NEXT_THREAD_TAG.fetch_add(1, Ordering::Relaxed) & ((1 << (64 - THREAD_TAG_SHIFT)) - 1);
    }
    TAG.with(|tag| *tag)
}

/// Handle to a buffer living on a remote server.
#[derive(Debug, PartialEq, Eq)]
pub struct RemoteBuf<T: TensorValue> {
    pub(crate) id: BufId,
    pub(crate) connection: u64,
    pub(crate) len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

/// Connection state shared between callers and the reader thread. One mutex guards all of it,
/// so registering a waiter can never race with the reader tearing the connection down.
#[derive(Default)]
struct State {
    /// Callers waiting for the response to a request, keyed by request id.
    pending: HashMap<u64, flume::Sender<Result<Reply, TensorError>>>,
    /// Per sending thread (see [`thread_tag`]): the first error reported for one of its
    /// fire-and-forget requests that has not been surfaced yet.
    deferred: HashMap<u64, TensorError>,
    /// Set once the connection is unusable, with the reason.
    closed: Option<String>,
}

struct Inner {
    remote_addr: IpAddr,
    remote_port: u16,
    connection: u64,
    next_request: AtomicU64,
    next_buffer: AtomicU64,
    state: Arc<Mutex<State>>,
    /// Write half of the socket. Holding the lock while writing keeps request order equal to
    /// send order, which the server's in-order execution relies on.
    writer: OnceLock<Mutex<TcpStream>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Wakes the reader thread with EOF and tells the server to drop this session's buffers.
        if let Some(writer) = self.writer.get() {
            let stream = writer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

/// Backend that executes every operation on a remote [`super::server::RemoteServer`].
///
/// Operations that only mutate remote buffers are pipelined: they return as soon as the request
/// is written, and the server runs requests in order. If one of them fails, the error is
/// returned by the next call the same thread makes on this backend (or by [`RemoteBackend::sync`]).
/// Clones share one connection and may be used from several threads.
#[derive(Clone)]
pub struct RemoteBackend {
    inner: Arc<Inner>,
}

impl Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("remote_addr", &self.inner.remote_addr)
            .field("remote_port", &self.inner.remote_port)
            .field("connection", &self.inner.connection)
            .finish()
    }
}

#[inline]
fn closed_error(reason: &str) -> TensorError {
    TensorError::RemoteError(format!("remote connection closed: {reason}"))
}

#[inline]
fn unexpected_reply() -> TensorError {
    TensorError::RemoteError("received an unexpected reply from the remote server".to_string())
}

#[inline]
fn unsupported(op: &str) -> TensorError {
    TensorError::UnsupportedOperation(format!("{op} is not supported by the remote backend yet"))
}

impl RemoteBackend {
    pub fn new_with_address(remote_addr: IpAddr, remote_port: u16) -> Result<Self, std::io::Error> {
        Ok(Self {
            inner: Arc::new(Inner {
                remote_addr,
                remote_port,
                connection: NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed),
                next_request: AtomicU64::new(0),
                next_buffer: AtomicU64::new(0),
                state: Arc::new(Mutex::new(State::default())),
                writer: OnceLock::new(),
            }),
        })
    }

    /// Opens the TCP connection, starts the reader thread and performs the version handshake.
    pub fn connect(&mut self) -> Result<(), std::io::Error> {
        if self.inner.writer.get().is_some() {
            return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "remote backend is already connected"));
        }
        let stream = TcpStream::connect((self.inner.remote_addr, self.inner.remote_port))?;
        stream.set_nodelay(true)?;
        let read_stream = stream.try_clone()?;
        self.inner.writer.set(Mutex::new(stream))
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::AlreadyExists, "remote backend is already connected"))?;

        let state = self.inner.state.clone();
        std::thread::spawn(move || read_incoming(state, read_stream));

        match self.call(Op::Hello { version: PROTOCOL_VERSION }) {
            Ok(Reply::Ack) => Ok(()),
            Ok(_) => Err(std::io::Error::other("unexpected handshake reply")),
            Err(e) => Err(std::io::Error::other(e.to_string())),
        }
    }

    pub fn address(&self) -> (IpAddr, u16) {
        (self.inner.remote_addr, self.inner.remote_port)
    }

    /// Waits until the server has executed every request sent so far (by any thread), then
    /// returns the first unreported error among the requests this thread sent.
    pub fn sync(&self) -> Result<(), TensorError> {
        match self.call(Op::Sync)? {
            Reply::Ack => Ok(()),
            _ => Err(unexpected_reply()),
        }
    }

    /// Fails fast if the connection is gone or one of this thread's pipelined requests failed.
    fn check_state(state: &mut State) -> Result<(), TensorError> {
        if let Some(reason) = &state.closed {
            return Err(closed_error(reason));
        }
        if let Some(e) = state.deferred.remove(&thread_tag()) {
            return Err(e);
        }
        Ok(())
    }

    fn next_request_id(&self) -> u64 {
        let counter = self.inner.next_request.fetch_add(1, Ordering::Relaxed) & ((1 << THREAD_TAG_SHIFT) - 1);
        (thread_tag() << THREAD_TAG_SHIFT) | counter
    }

    fn write(&self, request: &Request) -> Result<(), TensorError> {
        let writer = self.inner.writer.get()
            .ok_or_else(|| TensorError::RemoteError("remote backend is not connected".to_string()))?;
        let mut stream = writer.lock().unwrap();
        write_frame(&mut *stream, request).inspect_err(|e| {
            let mut state = self.inner.state.lock().unwrap();
            state.closed.get_or_insert_with(|| e.to_string());
            state.pending.clear();
        })
    }

    /// Sends a request and waits for its reply.
    fn call(&self, op: Op) -> Result<Reply, TensorError> {
        let id = self.next_request_id();
        let (tx, rx) = flume::bounded(1);
        {
            let mut state = self.inner.state.lock().unwrap();
            Self::check_state(&mut state)?;
            state.pending.insert(id, tx);
        }
        if let Err(e) = self.write(&Request { id, reply: true, op }) {
            self.inner.state.lock().unwrap().pending.remove(&id);
            return Err(e);
        }
        let result = rx.recv().map_err(|_| {
            let state = self.inner.state.lock().unwrap();
            closed_error(state.closed.as_deref().unwrap_or("reader stopped"))
        })?;
        // The server answers in order, so any failure of a pipelined request this thread sent
        // earlier has been recorded by now. Surface it here rather than letting it go unnoticed.
        if let Some(e) = self.inner.state.lock().unwrap().deferred.remove(&thread_tag()) {
            return Err(e);
        }
        result
    }

    /// Sends a request without waiting. Failures are reported by a later call.
    fn submit(&self, op: Op) -> Result<(), TensorError> {
        let id = self.next_request_id();
        Self::check_state(&mut self.inner.state.lock().unwrap())?;
        self.write(&Request { id, reply: false, op })
    }

    /// Converts a buffer handle for the wire, rejecting handles from another connection.
    #[inline]
    fn wire<T: TensorValue>(&self, buf: &RemoteBuf<T>) -> Result<TypelessBuf, TensorError> {
        if buf.connection != self.inner.connection {
            return Err(TensorError::RemoteError(
                "buffer belongs to a different remote connection".to_string(),
            ));
        }
        Ok(TypelessBuf { id: buf.id, dtype: T::DTYPE })
    }

    /// Reserves a new buffer handle. The server creates the buffer when it executes the request.
    #[inline]
    fn new_buf<T: TensorValue>(&self, len: usize) -> RemoteBuf<T> {
        RemoteBuf {
            id: self.inner.next_buffer.fetch_add(1, Ordering::Relaxed),
            connection: self.inner.connection,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    fn unary<T: TensorValue>(&self, buf: &RemoteBuf<T>, op: UnaryOp, layout: Layout) -> Result<(), TensorError> {
        self.submit(Op::Unary { buf: self.wire(buf)?, op, layout })
    }

    fn fill<T: TensorValue>(&self, buf: &RemoteBuf<T>, value: T, layout: Layout) -> Result<(), TensorError> {
        self.submit(Op::Fill { buf: self.wire(buf)?, value: Value::from_value(value), layout })
    }

    fn scalar<T: TensorValue>(&self, buf: &RemoteBuf<T>, op: ScalarOp, value: T, layout: Layout) -> Result<(), TensorError> {
        self.submit(Op::Scalar { buf: self.wire(buf)?, op, value: Value::from_value(value), layout })
    }
}

/// Implements the `contiguous` / `1d_strided` / `nd` methods of each listed unary op by
/// forwarding them as `Op::Unary`. Trait bounds such as `T: WeightValue` are enforced by the
/// trait at the call site; the server re-checks the dtype.
macro_rules! remote_unary {
    ($($name:ident => $op:ident),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<apply_ $name _nd>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, shape: &[usize], stride: &[isize]) -> Result<(), TensorError> {
                    self.unary(buf, UnaryOp::$op, Layout::Nd { offset, shape: shape.to_vec(), stride: stride.to_vec() })
                }
                fn [<apply_ $name _1d_strided>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, stride: isize, len: usize) -> Result<(), TensorError> {
                    self.unary(buf, UnaryOp::$op, Layout::Strided1d { offset, stride, len })
                }
                fn [<apply_ $name _contiguous>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, start: usize, len: usize) -> Result<(), TensorError> {
                    self.unary(buf, UnaryOp::$op, Layout::Contiguous { start, len })
                }
            )+
        }
    };
}

/// Same as `remote_unary`, for ops with one scalar operand.
macro_rules! remote_scalar {
    ($($name:ident => $op:ident),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<scalar_apply_ $name _nd>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, shape: &[usize], stride: &[isize]) -> Result<(), TensorError> {
                    self.scalar(buf, ScalarOp::$op, value, Layout::Nd { offset, shape: shape.to_vec(), stride: stride.to_vec() })
                }
                fn [<scalar_apply_ $name _1d_strided>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, stride: isize, len: usize) -> Result<(), TensorError> {
                    self.scalar(buf, ScalarOp::$op, value, Layout::Strided1d { offset, stride, len })
                }
                fn [<scalar_apply_ $name _contiguous>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, start: usize, len: usize) -> Result<(), TensorError> {
                    self.scalar(buf, ScalarOp::$op, value, Layout::Contiguous { start, len })
                }
            )+
        }
    };
}

impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf<T>;

    fn new() -> Self {
        get_backend_default().expect("No default remote backend available")
    }

    fn device_type() -> DeviceType {
        DeviceType::Remote {
            ip: "127.0.0.1".parse().unwrap(),
            port: 7878,
            remote_type: DeviceType::Cpu.into(),
        }
    }

    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, TensorError> {
        let buf = self.new_buf(src.len());
        self.submit(Op::AllocFromSlice { dst: self.wire(&buf)?, src: Slice::from_boxed_slice(src) })?;
        Ok(buf)
    }

    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, TensorError> {
        let buf = self.new_buf(len);
        self.submit(Op::Alloc { dst: self.wire(&buf)?, len })?;
        Ok(buf)
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), TensorError> {
        if src.len() != dst.len {
            return Err(TensorError::SizeMismatch(format!(
                "copy_from_slice: source has {} elements, destination has {}", src.len(), dst.len
            )));
        }
        self.submit(Op::CopyFromSlice { dst: self.wire(dst)?, src: Slice::from_slice(src) })
    }

    fn copy_range_within<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &Self::Buf<T>, dst_offset: usize, src_offset: usize, len: usize) -> Result<(), TensorError> {
        self.submit(Op::CopyRangeWithin { dst: self.wire(dst)?, src: self.wire(src)?, dst_offset, src_offset, len })
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, TensorError> {
        match self.call(Op::Read { buf: self.wire(buf)?, offset })? {
            Reply::Value(value) => value.to_value::<T>(),
            _ => Err(unexpected_reply()),
        }
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), TensorError> {
        self.submit(Op::Write { buf: self.wire(buf)?, offset, value: Value::from_value(value) })
    }

    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        buf.len
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError> {
        let dst = self.new_buf(src.len);
        self.submit(Op::Copy { src: self.wire(src)?, dst: self.wire(&dst)? })?;
        Ok(dst)
    }

    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError> {
        match self.call(Op::Dump { src: self.wire(src)? })? {
            Reply::Slice(slice) => slice.to_boxed_slice::<T>(),
            _ => Err(unexpected_reply()),
        }
    }

    fn convert<T: TensorValue, N: TensorValue>(&self, src: &Self::Buf<T>, dst: &mut Self::Buf<N>) -> Result<(), TensorError> {
        if src.len != dst.len {
            return Err(TensorError::SizeMismatch(format!(
                "Buffer size mismatch in convert: src size {}, dst size {}", src.len, dst.len
            )));
        }
        self.submit(Op::Convert { src: self.wire(src)?, dst: self.wire(dst)? })
    }

    fn fill_nd<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, shape: &[usize], stride: &[isize]) -> Result<(), TensorError> {
        self.fill(buf, value, Layout::Nd { offset, shape: shape.to_vec(), stride: stride.to_vec() })
    }
    fn fill_1d_strided<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, stride: isize, len: usize) -> Result<(), TensorError> {
        self.fill(buf, value, Layout::Strided1d { offset, stride, len })
    }
    fn fill_contiguous<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, start: usize, len: usize) -> Result<(), TensorError> {
        self.fill(buf, value, Layout::Contiguous { start, len })
    }

    fn broadcast<T: TensorValue>(
        &self,
        left: (*const Self::Buf<T>, &MetaTensor),
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
    ) -> Result<(), TensorError> {
        // SAFETY: the trait requires the caller to pass valid buffer pointers. Only the handles
        // are read here; the data lives on the server.
        let (left_buf, right_buf, dst_buf) = unsafe { (&*left.0, &*right.0, &*dst.0) };
        self.submit(Op::Broadcast {
            left: (self.wire(left_buf)?, left.1.clone()),
            right: (self.wire(right_buf)?, right.1.clone()),
            dst: (self.wire(dst_buf)?, dst.1.clone()),
            op,
        })
    }

    remote_unary!(
        neg => Neg, relu => Relu, sigmoid => Sigmoid, silu => Silu, tanh => Tanh, abs => Abs,
        sqrt => Sqrt, ln => Ln, expm1 => Expm1, ln1p => Ln1p, floor => Floor, ceil => Ceil,
        round => Round, trunc => Trunc, sin => Sin, cos => Cos, tan => Tan, asin => Asin,
        acos => Acos, atan => Atan, sinh => Sinh, cosh => Cosh, asinh => Asinh, acosh => Acosh,
        atanh => Atanh, rsqrt => Rsqrt, reciprocal => Reciprocal, square => Square, cube => Cube,
        exp => Exp, sign => Sign,
    );

    remote_scalar!(
        add => Add, sub => Sub, mul => Mul, div => Div, log => Log, log1p => Log1p,
        leaky_relu => LeakyRelu, elu => Elu,
    );

    fn apply_reduce_contiguous_flat<T: WeightValue>(&self, src: &Self::Buf<T>, dst: &mut Self::Buf<T>, start: usize, len: usize, op: ReductionOpTypes) -> Result<(), TensorError> {
        self.submit(Op::ReduceFlat { src: self.wire(src)?, dst: self.wire(dst)?, start, len, op })
    }
    fn apply_reduce_contiguous_nd<T: WeightValue>(&self, src: (&Self::Buf<T>, &MetaTensor), dst: (&mut Self::Buf<T>, &MetaTensor), dim: Dim, op: ReductionOpTypes) -> Result<(), TensorError> {
        self.submit(Op::ReduceNd { src: (self.wire(src.0)?, src.1.clone()), dst: (self.wire(dst.0)?, dst.1.clone()), dim, op })
    }
    fn apply_argmax_contiguous_flat<T: WeightValue>(&self, src: &Self::Buf<T>, dst: &mut Self::Buf<u64>, start: usize, len: usize, op: ReductionOpTypes) -> Result<(), TensorError> {
        self.submit(Op::ArgFlat { src: self.wire(src)?, dst: self.wire(dst)?, start, len, op })
    }
    fn apply_argmax_contiguous_nd<T: WeightValue>(&self, src: (&Self::Buf<T>, &MetaTensor), dst: (&mut Self::Buf<u64>, &MetaTensor), dim: Dim, op: ReductionOpTypes) -> Result<(), TensorError> {
        self.submit(Op::ArgNd { src: (self.wire(src.0)?, src.1.clone()), dst: (self.wire(dst.0)?, dst.1.clone()), dim, op })
    }
    fn apply_conv_2d<T: WeightValue>(&self, _input: (&Self::Buf<T>, &MetaTensor), _kernel: (&Self::Buf<T>, &MetaTensor), _output: &mut Self::Buf<T>, _config: &ConvConfig2D) -> Result<(), TensorError> {
        // Neither the CPU nor the CUDA backend implements conv_2d yet, so there is nothing to forward to.
        Err(unsupported("conv_2d"))
    }
}

impl<T: TensorValue> BackendMatMul<T> for RemoteBackend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &MetaTensor, ContiguityTypes),
        rhs: (&Self::Buf<T>, &MetaTensor, ContiguityTypes),
        dst: &mut Self::Buf<T>,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), TensorError> {
        self.submit(Op::Matmul {
            lhs: (self.wire(lhs.0)?, lhs.1.clone(), lhs.2),
            rhs: (self.wire(rhs.0)?, rhs.1.clone(), rhs.2),
            dst: self.wire(dst)?,
            b, m, k, n,
        })
    }
}

/// Reader thread: routes replies to their waiting callers and records failures of
/// fire-and-forget requests. On disconnect it wakes every waiter with an error.
fn read_incoming(state: Arc<Mutex<State>>, mut stream: TcpStream) {
    let reason = loop {
        match read_frame::<_, Response>(&mut stream) {
            Ok(Some(response)) => {
                let mut state = state.lock().unwrap();
                match state.pending.remove(&response.id) {
                    // The waiter may have given up; nothing else to do then.
                    Some(waiter) => { let _ = waiter.send(response.result); }
                    None => {
                        if let Err(e) = response.result {
                            // Keep the sender's first failure; later ones are usually consequences of it.
                            state.deferred.entry(response.id >> THREAD_TAG_SHIFT).or_insert(e);
                        }
                    }
                }
            }
            Ok(None) => break "server closed the connection".to_string(),
            Err(e) => break e.to_string(),
        }
    };
    let mut state = state.lock().unwrap();
    state.closed.get_or_insert(reason);
    // Dropping the senders wakes every waiter with a receive error.
    state.pending.clear();
}
