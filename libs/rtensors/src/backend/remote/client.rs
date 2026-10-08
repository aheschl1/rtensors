use std::{collections::HashMap, fmt::Debug, io::{Read, Write}, net::IpAddr, sync::{atomic::{AtomicBool, AtomicU32, Ordering}, Arc, Condvar, Mutex, RwLock}};

use crate::{backend::{remote::{get_backend_default, protocol::{Messages, Request, Response, TypelessBuf}}, Backend, BackendMatMul}, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue, WeightValue}, Dim, MetaTensor}, ops::{base::BinaryOpType, linalg::ConvConfig2D, reduction::ReductionOpTypes}};
use flume;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteBuf<T: TensorValue> {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
    pub (crate) _marker: std::marker::PhantomData<T>,
}

impl<T: TensorValue> RemoteBuf<T> {
    #[inline(always)]
    fn to_typeless(&self) -> TypelessBuf {
        TypelessBuf {
            id: self.id,
            dtype: self.dtype,
        }
    }

    #[inline(always)]
    pub(crate) fn from_typeless(buf: TypelessBuf) -> Self {
        Self {
            id: buf.id,
            dtype: buf.dtype,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: TensorValue> From<&mut RemoteBuf<T>> for TypelessBuf {
    fn from(buf: &mut RemoteBuf<T>) -> Self {
        buf.to_typeless()
    }
}

impl<T: TensorValue> From<&RemoteBuf<T>> for TypelessBuf {
    fn from(buf: &RemoteBuf<T>) -> Self {
        buf.to_typeless()
    }
}

impl<T: TensorValue> From<*const RemoteBuf<T>> for TypelessBuf {
    fn from(buf: *const RemoteBuf<T>) -> Self {
        unsafe { (&*buf).to_typeless() }
    }
}

impl<T: TensorValue> From<*mut RemoteBuf<T>> for TypelessBuf {
    fn from(buf: *mut RemoteBuf<T>) -> Self {
        unsafe { (&*buf).to_typeless() }
    }
}

#[derive(Debug)]
struct PendingHandler {
    count: Arc<AtomicU32>,
    cv: Condvar,
    mutex: Mutex<()>,
}

impl PendingHandler {
    fn inc(&self) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
    fn dec(&self) {
        let prev = self.count.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            // Take the lock so the notify cannot land between a waiter's check and its wait.
            let _guard = self.mutex.lock().unwrap();
            self.cv.notify_all();
        }
    }
    fn sync(&self) {
        let mut guard = self.mutex.lock().unwrap();
        while self.count.load(Ordering::SeqCst) > 0 {
            guard = self.cv.wait(guard).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct RemoteBackend {
    remote_addr: IpAddr,
    remote_port: u16,
    message_id: Arc<AtomicU32>,
    pending: Arc<PendingHandler>,
    messages_outgoing_sender: flume::Sender<Request>,
    messages_outgoing_receiver: flume::Receiver<Request>,
    pending_response: Arc<RwLock<HashMap<u32, flume::Sender<Messages>>>>,
    poisoned: Arc<AtomicBool>,
}
    
impl Debug for RemoteBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteBackend")
            .field("remote_addr", &self.remote_addr)
            .field("remote_port", &self.remote_port)
            .field("message_id", &self.message_id)
            .field("pending", &self.pending)
            .field("pending_response", &self.pending_response)
            .finish()
    }
}

impl RemoteBackend {
    pub fn new_with_address(remote_addr: IpAddr, remote_port: u16) -> Result<Self, std::io::Error> {
        let pending = PendingHandler {
            count: Arc::new(AtomicU32::new(0)),
            cv: Condvar::new(),
            mutex: Mutex::new(()),
        };
        let (sender, receiver) = flume::unbounded();
        let res = Self {
            remote_addr,
            remote_port,
            pending: Arc::new(pending),
            message_id: Arc::new(AtomicU32::new(0)),
            messages_outgoing_sender: sender,
            messages_outgoing_receiver: receiver,
            pending_response: Arc::new(RwLock::new(HashMap::new())),
            poisoned: Arc::new(AtomicBool::new(false)),
        };
        Ok(res)
    }

    fn poison(&self) {
        self.poisoned.store(true, Ordering::SeqCst);
    }

    pub fn sync(&self) {
        self.pending.sync();
    }

    #[inline(always)]
    fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    fn send_message(&self, msg: Messages) -> flume::Receiver<Messages>{
        if self.is_poisoned() {
            panic!("Attempted to send message on poisoned RemoteBackend. Reasons for poison:
            1. An asynchronous operation reported an error from the remote backend.
            2. The RemoteBackend is in an inconsistent state and can no longer process messages safely.");
        }
        self.pending.inc();
        let (sender, receiver) = flume::bounded(1);
        let mid = self.next_message_id();
        {
            let mut pending = self.pending_response.write().unwrap();
            pending.insert(mid, sender);
        }
        let req = Request {
            task_id: mid,
            message: msg,
        };
        self.messages_outgoing_sender.send(req).expect("Failed to send message to outgoing channel");
        receiver
    }

    pub(crate) fn connect(&mut self) -> Result<(), std::io::Error> {
        let stream = std::net::TcpStream::connect((self.remote_addr, self.remote_port))?;
        stream.set_nodelay(true)?;

        let read_stream = stream.try_clone()?;
        let write_stream = stream;

        let remote = self.clone();
        std::thread::spawn(move || {
            drain_outgoing(remote, write_stream);
        });
        let remote = self.clone();
        std::thread::spawn(move || {
            read_incoming(remote, read_stream);
        });
        Ok(())
    }

    pub fn address(&self) -> (IpAddr, u16) {
        (self.remote_addr, self.remote_port)
    }

    #[inline(always)]
    fn next_message_id(&self) -> u32 {
        self.message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

}

// macro_rules! send_recv {
//     ($self:expr, $message:expr, $response_pattern:pat => $result:expr) => {{
//         let receiver = $self.send_message($message);
//         let response = receiver.recv()
//             .map_err(|_| TensorError::BackendError("Failed to receive response".to_string()))?;
//         match response {
//             $response_pattern => $result,
//             _ => Err(TensorError::BackendError("Unexpected response type".to_string())),
//         }
//     }};
// }

/// Generates `contiguous` / `1d_strided` / `nd` stubs for a unary op the remote protocol
/// does not support yet. They return `UnsupportedOperation` instead of panicking.
macro_rules! remote_unsupported_unary {
    ($($name:ident),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<apply_ $name _nd>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _offset: usize, _shape: &[usize], _stride: &[isize]) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<apply_ $name>])))
                }
                fn [<apply_ $name _1d_strided>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _offset: usize, _stride: isize, _len: usize) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<apply_ $name>])))
                }
                fn [<apply_ $name _contiguous>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _start: usize, _len: usize) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<apply_ $name>])))
                }
            )+
        }
    };
}

/// Same as `remote_unsupported_unary`, for scalar ops (`scalar_apply_*`).
macro_rules! remote_unsupported_scalar {
    ($($name:ident),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<scalar_apply_ $name _nd>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _offset: usize, _shape: &[usize], _stride: &[isize]) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<scalar_apply_ $name>])))
                }
                fn [<scalar_apply_ $name _1d_strided>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _offset: usize, _stride: isize, _len: usize) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<scalar_apply_ $name>])))
                }
                fn [<scalar_apply_ $name _contiguous>]<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _start: usize, _len: usize) -> Result<(), TensorError> {
                    Err(unsupported(stringify!([<scalar_apply_ $name>])))
                }
            )+
        }
    };
}

/// Forwards a `scalar_apply_*` op to the server's `ApplyElementwiseBinary*` messages.
macro_rules! remote_scalar_binary {
    ($($name:ident => $op:ident),+ $(,)?) => {
        paste::paste! {
            $(
                fn [<scalar_apply_ $name _nd>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, shape: &[usize], stride: &[isize]) -> Result<(), TensorError> {
                    self.ack(Messages::ApplyElementwiseBinaryNd {
                        buf: buf.into(),
                        op: (BinaryOpType::$op, value.into()),
                        offset,
                        shape: shape.to_vec(),
                        stride: stride.to_vec(),
                    })
                }
                fn [<scalar_apply_ $name _1d_strided>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, offset: usize, stride: isize, len: usize) -> Result<(), TensorError> {
                    self.ack(Messages::ApplyElementwiseBinary1dStrided {
                        buf: buf.into(),
                        op: (BinaryOpType::$op, value.into()),
                        offset,
                        stride,
                        len,
                    })
                }
                fn [<scalar_apply_ $name _contiguous>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, start: usize, len: usize) -> Result<(), TensorError> {
                    self.ack(Messages::ApplyElementwiseBinaryContiguous {
                        buf: buf.into(),
                        op: (BinaryOpType::$op, value.into()),
                        start,
                        len,
                    })
                }
            )+
        }
    };
}

#[inline]
fn unsupported(op: &str) -> TensorError {
    TensorError::UnsupportedOperation(format!("{op} is not supported by the remote backend yet"))
}

impl RemoteBackend {
    /// Sends a message whose response carries only a `Result<(), TensorError>`.
    fn ack(&self, message: Messages) -> Result<(), TensorError> {
        let receiver = self.send_message(message);
        match receiver.recv() {
            Ok(Messages::ApplyElementwiseBinaryNdResponse(result))
            | Ok(Messages::ApplyElementwiseBinary1dStridedResponse(result))
            | Ok(Messages::ApplyElementwiseBinaryContiguousResponse(result)) => result,
            Ok(_) => Err(TensorError::BackendError("Received unexpected RPC response message".to_string())),
            Err(e) => Err(TensorError::BackendError(format!("Failed to receive RPC response: {}", e))),
        }
    }
}

#[proc::routines(Messages)]
impl Backend for RemoteBackend {
    type Buf<T: TensorValue> = RemoteBuf<T>;
    #[rpc(skip)]
    fn new() -> Self {
        get_backend_default().expect("No default remote backend available")
    }
    #[rpc(skip)]
    fn device_type() -> crate::core::primitives::DeviceType {
        DeviceType::Remote {
            ip: "127.0.0.1".parse().unwrap(),
            port: 7878,
            remote_type: DeviceType::Cpu.into(),
        }
    }
    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError>{unreachable!("Macro implmented")}
    #[rpc(extra(dtype: DType = T::DTYPE))]
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, crate::core::tensor::TensorError>{unreachable!("Macro implmented")}
    
    #[rpc(sync)]
    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), crate::core::tensor::TensorError> {unreachable!("Macro implmented")}
    
    #[rpc(sync)]
    fn copy_range_within<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &Self::Buf<T>, dst_offset: usize, src_offset: usize, len: usize) -> Result<(), TensorError>{unreachable!("Macro implmented")}
    
    #[rpc(sync)]
    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, crate::core::tensor::TensorError> {unreachable!("Macro implmented")}
    
    #[rpc(sync)]
    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), crate::core::tensor::TensorError> {unreachable!("Macro implmented")}
    #[rpc(skip)] // skip because the pattern is different when not Result
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        let message = Messages::Len {
            buf: buf.into(),
        };
        let receiver = self.send_message(message);
        match receiver.recv() {
            Ok(Messages::LenResponse(len)) => {len},
            _ => panic!("Failed to get buffer length or unexpected response"),
        }
    }

    #[rpc(sync)]
    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, crate::core::tensor::TensorError>{unreachable!("Macro implmented")}
    
    #[rpc(sync)]
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, crate::core::tensor::TensorError>{unreachable!("Macro implmented")}

    #[rpc(skip)]
    fn convert<T: TensorValue, N: TensorValue>(&self, _src: &Self::Buf<T>, _dst: &mut Self::Buf<N>) -> Result<(), TensorError> {
        Err(unsupported("convert"))
    }

    #[rpc(skip)]
    fn fill_nd<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _offset: usize, _shape: &[usize], _stride: &[isize]) -> Result<(), TensorError> {
        Err(unsupported("fill"))
    }
    #[rpc(skip)]
    fn fill_1d_strided<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _offset: usize, _stride: isize, _len: usize) -> Result<(), TensorError> {
        Err(unsupported("fill"))
    }
    #[rpc(skip)]
    fn fill_contiguous<T: TensorValue>(&self, _buf: &mut Self::Buf<T>, _value: T, _start: usize, _len: usize) -> Result<(), TensorError> {
        Err(unsupported("fill"))
    }
    
    fn broadcast<T: TensorValue>(
        &self,
        left: (*const Self::Buf<T>, &crate::core::MetaTensor), 
        right: (*const Self::Buf<T>, &crate::core::MetaTensor),
        dst: (*mut Self::Buf<T>, &crate::core::MetaTensor),
        op: crate::ops::base::BinaryOpType
    ) -> Result<(), crate::core::tensor::TensorError>{unreachable!("Macro implmented")}
    fn apply_neg_contiguous<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        start: usize,
        len: usize
    ) -> Result<(), TensorError>{unreachable!("Macro implmented")}
    fn apply_neg_1d_strided<T: TensorValue>(
        &self, buf: &mut Self::Buf<T>, 
        offset: usize,
        stride: isize,
        len: usize
    ) -> Result<(), TensorError> {unreachable!("Macro implmented")}
    fn apply_neg_nd<T: TensorValue>(
        &self,
        buf: &mut Self::Buf<T>,
        offset: usize,
        shape: &[usize],
        stride: &[isize],
    ) -> Result<(), TensorError>{unreachable!("Macro implmented")}

    remote_scalar_binary!(add => Add, sub => Sub, mul => Mul, div => Div);
    remote_unsupported_scalar!(log, log1p, leaky_relu, elu);

    remote_unsupported_unary!(
        relu, sigmoid, silu, tanh, abs, sqrt, ln, expm1, ln1p, floor, ceil, round, trunc,
        sin, cos, tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh,
        rsqrt, reciprocal, square, cube, exp, sign,
    );

    #[rpc(skip)]
    fn apply_reduce_contiguous_flat<T: WeightValue>(&self, _src: &Self::Buf<T>, _dst: &mut Self::Buf<T>, _start: usize, _len: usize, _op: ReductionOpTypes) -> Result<(), TensorError> {
        Err(unsupported("reduce"))
    }
    #[rpc(skip)]
    fn apply_reduce_contiguous_nd<T: WeightValue>(&self, _src: (&Self::Buf<T>, &MetaTensor), _dst: (&mut Self::Buf<T>, &MetaTensor), _dim: Dim, _op: ReductionOpTypes) -> Result<(), TensorError> {
        Err(unsupported("reduce"))
    }
    #[rpc(skip)]
    fn apply_argmax_contiguous_flat<T: WeightValue>(&self, _src: &Self::Buf<T>, _dst: &mut Self::Buf<u64>, _start: usize, _len: usize, _op: ReductionOpTypes) -> Result<(), TensorError> {
        Err(unsupported("argmax"))
    }
    #[rpc(skip)]
    fn apply_argmax_contiguous_nd<T: WeightValue>(&self, _src: (&Self::Buf<T>, &MetaTensor), _dst: (&mut Self::Buf<u64>, &MetaTensor), _dim: Dim, _op: ReductionOpTypes) -> Result<(), TensorError> {
        Err(unsupported("argmax"))
    }
    #[rpc(skip)]
    fn apply_conv_2d<T: WeightValue>(&self, _input: (&Self::Buf<T>, &MetaTensor), _kernel: (&Self::Buf<T>, &MetaTensor), _output: &mut Self::Buf<T>, _config: &ConvConfig2D) -> Result<(), TensorError> {
        Err(unsupported("conv_2d"))
    }
}

#[proc::routines(Messages)]
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
    ) -> Result<(), TensorError> {unreachable!("Macro implmented")}
}

#[inline]
fn read_response(stream: &mut std::net::TcpStream, len_buf: &mut  [u8; 4]) -> Result<Response, Box<bincode::ErrorKind>> {
    stream.read_exact(len_buf).unwrap();
    let msg_len = u32::from_le_bytes(*len_buf) as usize;
    let mut msg_buf = vec![0u8; msg_len];
    stream.read_exact(&mut msg_buf).unwrap();
    Response::deserialize(&msg_buf)
}

#[inline]
fn send_message_to_channel(remote: &RemoteBackend, msg: Response) {
    let task_id = msg.task_id;
    let sender = {
        let mut pending = remote.pending_response.write().unwrap();
        pending.remove(&task_id)
    };
    if let Some(sender) = sender {
        sender.send(msg.message).unwrap();
    }
}

// todo, make this async

/// Thread function to drain outgoing messages and send them over the TCP stream
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to send messages over
fn drain_outgoing(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let receiver = remote.messages_outgoing_receiver.clone();
    loop {        
        if let Ok(req) = receiver.recv() {
            if remote.is_poisoned() {
                break;
            }
            let serialized = req.serialize().unwrap();
            let n = serialized.len();
            let n_bytes = (n as u32).to_le_bytes();
            stream.write_all(&n_bytes).unwrap();
            stream.write_all(&serialized).unwrap();
        } else {
            // Channel closed, exit thread
            break;
        }
    }
}

/// Thread function to read incoming messages from the TCP stream
/// 
/// Upon receiving a message, if it is marked as not asynchronous, it sends the message to the waiting channel
/// Otherwise, it handles asynchronous messages accordingly. If the asynchronous message is complete and does not
/// indicate an error, it simply decrements the pending count. If it indicates an error, it poisons the RemoteBackend to prevent further operations.
/// If the asynchronous message is not complete, it sends a follow-up message to the waiting channel, and does not decrement the pending count.
/// 
/// # Arguments
/// * `remote` - The RemoteBackend instance
/// * `stream` - The TCP stream to read messages from
fn read_incoming(remote: RemoteBackend, mut stream: std::net::TcpStream) {
    let mut len_buf = [0u8; 4];
    loop {
        let msg = read_response(&mut stream, &mut len_buf).unwrap();
        if !msg.asynchronous {
            debug_assert!(msg.complete);
            send_message_to_channel(&remote, msg);
            remote.pending.dec();
        }else if msg.complete {
            // no need to send follow up, just decrement pending. because, there was an incomplete one before
            // that sent the message. async followup is just a backend notification.
            // nobody waits on follow up of async
            if let Some(e) = msg.error {
                remote.poison();
                panic!("Inconsistent state detected. Received error in async message: {:?}", e);
            } 
            remote.pending.dec();
        }else{
            //send initial follow up to receiver, do not decrement pending yet
            send_message_to_channel(&remote, msg);
        }
    }
}
