//! Remote tensor server. Each client connection gets its own session (buffer store + backend),
//! executed by one worker thread in request order and dropped when the client disconnects.

use std::{collections::HashMap, net::{IpAddr, TcpListener, TcpStream}, thread::{self, JoinHandle}};

use crate::{backend::{cpu::Cpu, remote::protocol::{read_frame, write_frame, BufId, Layout, Op, Reply, Request, Response, ScalarOp, Slice, TypelessBuf, UnaryOp, Value, PROTOCOL_VERSION}, Backend, BackendMatMul}, core::{primitives::DeviceType, tensor::TensorError, value::{types, DType, TensorValue}}};

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
        paste::paste! {
            match $layout {
                Layout::Contiguous { start, len } => $backend.[<$method _contiguous>]($buf $(, $extra)*, start, len),
                Layout::Strided1d { offset, stride, len } => $backend.[<$method _1d_strided>]($buf $(, $extra)*, offset, stride, len),
                Layout::Nd { offset, shape, stride } => $backend.[<$method _nd>]($buf $(, $extra)*, offset, &shape, &stride),
            }
        }
    };
}

fn missing(id: BufId) -> TensorError {
    TensorError::RemoteError(format!("buffer {id} does not exist"))
}

fn wrong_dtype(buf: TypelessBuf, actual: DType) -> TensorError {
    TensorError::RemoteError(format!("buffer {} has dtype {:?}, request expected {:?}", buf.id, actual, buf.dtype))
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
                // SAFETY: distinct ids, so the pointers refer to different buffers in the store,
                // which is not touched while they are in use.
                let (dst_buf, src_buf) = unsafe { (&mut *dst_ptr, &*src_ptr) };
                backend.copy_range_within(dst_buf, src_buf, dst_offset, src_offset, len).map(|_| Reply::Ack)
            }),
            Op::Read { buf, offset } => dispatch!(any buf.dtype, "read", T => {
                let value = backend.read(store.get::<T>(buf)?, offset)?;
                Ok(Reply::Value(Value::from_value(value)))
            }),
            Op::Write { buf, offset, value } => dispatch!(any buf.dtype, "write", T => {
                let value = value.to_value::<T>()?;
                backend.write(store.get_mut::<T>(buf)?, offset, value).map(|_| Reply::Ack)
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
                let [left_ptr, right_ptr, dst_ptr] = store.ptrs::<T, 3>([left.0, right.0, dst.0])?;
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

fn unary<B: ServerBackend>(backend: &B, store: &mut Store<B>, buf: TypelessBuf, op: UnaryOp, layout: Layout) -> Result<(), TensorError> {
    match op {
        UnaryOp::Neg => dispatch!(signed buf.dtype, op.name(), T => with_layout!(backend, apply_neg, store.get_mut::<T>(buf)?, layout)),
        UnaryOp::Relu => dispatch!(any buf.dtype, op.name(), T => with_layout!(backend, apply_relu, store.get_mut::<T>(buf)?, layout)),
        UnaryOp::Abs => dispatch!(any buf.dtype, op.name(), T => with_layout!(backend, apply_abs, store.get_mut::<T>(buf)?, layout)),
        _ => dispatch!(float buf.dtype, op.name(), T => float_unary::<B, T>(backend, store.get_mut::<T>(buf)?, op, layout)),
    }
}

fn scalar<B: ServerBackend>(backend: &B, store: &mut Store<B>, buf: TypelessBuf, op: ScalarOp, value: Value, layout: Layout) -> Result<(), TensorError> {
    macro_rules! run {
        ($class:ident, $method:ident) => {
            dispatch!($class buf.dtype, op.name(), T => {
                let value = value.to_value::<T>()?;
                with_layout!(backend, $method, store.get_mut::<T>(buf)?, layout, value)
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
                // A backend panic (e.g. an out-of-range layout from a misbehaving client) becomes
                // an error for that request instead of killing the connection.
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
