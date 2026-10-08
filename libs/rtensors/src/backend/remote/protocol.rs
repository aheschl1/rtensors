//! Wire protocol shared by the remote client and server.
//!
//! Every frame is a little-endian `u64` byte length followed by a bincode payload. The client
//! sends [`Request`]s and the server answers with [`Response`]s. The server executes requests
//! strictly in the order they arrive, so a request never observes a buffer before every earlier
//! request has finished with it. That ordering is what lets most operations be fire-and-forget:
//! only requests with `reply: true` get a response on success, while a failing request always
//! gets one so the client can surface the error.

use std::io::{Read, Write};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue}, Dim, MetaTensor}, ops::{base::BinaryOpType, reduction::ReductionOpTypes}};

/// Bumped whenever the wire format changes; client and server must agree.
pub(crate) const PROTOCOL_VERSION: u32 = 3;

/// Upper bound on a single frame. Frames are read incrementally, so this only guards against
/// nonsensical length prefixes rather than reserving memory up front.
pub(crate) const MAX_FRAME_BYTES: u64 = 1 << 40;

/// Identifies a buffer within one connection. Chosen by the client so allocations can be
/// pipelined without waiting for the server.
pub(crate) type BufId = u64;

/// Serializes a `Vec<u8>` as one byte string instead of a sequence of `u8`s, which bincode
/// would otherwise encode (and decode) one element at a time.
mod bytes {
    use serde::{de::{SeqAccess, Visitor}, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(data: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(data)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        struct BytesVisitor;
        impl<'de> Visitor<'de> for BytesVisitor {
            type Value = Vec<u8>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a byte string")
            }
            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E> {
                Ok(v.to_vec())
            }
            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E> {
                Ok(v)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0).min(4096));
                while let Some(b) = seq.next_element()? {
                    out.push(b);
                }
                Ok(out)
            }
        }
        deserializer.deserialize_byte_buf(BytesVisitor)
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Slice {
    #[serde(with = "bytes")]
    pub(crate) data: Vec<u8>, // bytes
    pub(crate) dtype: DType,
}

/// Rejects byte patterns that are not valid values of `dtype`. Every numeric dtype accepts any
/// bit pattern, but a `bool` must be exactly 0 or 1, so untrusted bytes are checked first.
#[inline]
fn validate_bytes(dtype: DType, data: &[u8]) -> Result<(), TensorError> {
    if dtype == DType::BOOL && data.iter().any(|&b| b > 1) {
        return Err(TensorError::BackendError("Invalid boolean byte in payload".to_string()));
    }
    Ok(())
}

impl Slice {
    #[inline(always)]
    pub(crate) fn from_boxed_slice<T: TensorValue>(boxed: Box<[T]>) -> Self {
        // Copy rather than reinterpreting the allocation: a `Box<[T]>` cannot be freed as a
        // `Vec<u8>` because the allocation layouts (alignment) differ.
        Self::from_slice(&boxed)
    }

    #[inline(always)]
    pub(crate) fn from_slice<T: TensorValue>(slice: &[T]) -> Self {
        let dtype = T::DTYPE;
        let len = std::mem::size_of_val(slice);
        let mut data = Vec::<u8>::with_capacity(len);
        // SAFETY: `TensorValue` types are plain-old-data, `data` has room for `len` bytes,
        // and u8 has no alignment requirement.
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, data.as_mut_ptr(), len);
            data.set_len(len);
        }
        Self { data, dtype }
    }

    #[inline(always)]
    pub(crate) fn to_boxed_slice<T: TensorValue>(self) -> Result<Box<[T]>, TensorError> {
        if self.dtype != T::DTYPE {
            return Err(TensorError::BackendError(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::DTYPE, self.dtype
            )));
        }
        let size = std::mem::size_of::<T>();
        if size == 0 || self.data.len() % size != 0 {
            return Err(TensorError::BackendError(format!(
                "Slice of {} bytes is not a whole number of {:?} elements",
                self.data.len(), T::DTYPE
            )));
        }
        validate_bytes(self.dtype, &self.data)?;
        let len = self.data.len() / size;
        let mut out = Vec::<T>::with_capacity(len);
        // SAFETY: `out` is a properly aligned allocation for `len` elements of `T`, and the
        // byte buffer holds exactly `len * size_of::<T>()` bytes. The byte buffer may be
        // unaligned for `T`, which is why we copy instead of reinterpreting it.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.as_ptr(), out.as_mut_ptr() as *mut u8, self.data.len());
            out.set_len(len);
        }
        Ok(out.into_boxed_slice())
    }
}


#[derive(Serialize, Deserialize)]
pub(crate) struct Value {
    #[serde(with = "bytes")]
    data: Vec<u8>, // bytes
    dtype: DType,
}

impl Value {
    #[inline(always)]
    pub(crate) fn from_value<T: TensorValue>(value: T) -> Self {
        Slice::from_slice(std::slice::from_ref(&value)).into_value()
    }

    #[inline(always)]
    pub(crate) fn to_value<T: TensorValue>(self) -> Result<T, TensorError> {
        if self.dtype != T::DTYPE {
            return Err(TensorError::BackendError(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::DTYPE, self.dtype
            )));
        }
        if self.data.len() != std::mem::size_of::<T>() {
            return Err(TensorError::BackendError(format!(
                "Value of {} bytes does not match {:?}",
                self.data.len(), T::DTYPE
            )));
        }
        validate_bytes(self.dtype, &self.data)?;
        // SAFETY: the length matches `T` exactly and the bit pattern was validated;
        // the bytes may be unaligned, hence read_unaligned.
        Ok(unsafe { std::ptr::read_unaligned(self.data.as_ptr() as *const T) })
    }
}

impl Slice {
    #[inline(always)]
    fn into_value(self) -> Value {
        Value { data: self.data, dtype: self.dtype }
    }
}

/// Which of the server's backends a connection should execute on.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum RemoteDevice {
    #[default]
    Cpu,
    /// CUDA device ordinal on the server. Requires a server built with the `cuda` feature.
    Cuda(usize),
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TypelessBuf {
    pub(crate) id: BufId,
    pub(crate) dtype: DType,
}

/// Memory layout of the elements an elementwise op touches, mirroring the
/// `_contiguous` / `_1d_strided` / `_nd` method families of [`crate::backend::Backend`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) enum Layout {
    Contiguous { start: usize, len: usize },
    Strided1d { offset: usize, stride: isize, len: usize },
    Nd { offset: usize, shape: Vec<usize>, stride: Vec<isize> },
}

/// Defines an op enum together with its snake_case name, used for error messages.
macro_rules! op_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $method:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum $name {
            $($variant),+
        }

        impl $name {
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($method)),+
                }
            }
        }
    };
}

op_enum!(
    /// In-place unary ops (`Backend::apply_<op>_*`).
    UnaryOp {
        Neg => neg, Relu => relu, Sigmoid => sigmoid, Silu => silu, Tanh => tanh, Abs => abs,
        Sqrt => sqrt, Ln => ln, Expm1 => expm1, Ln1p => ln1p, Floor => floor, Ceil => ceil,
        Round => round, Trunc => trunc, Sin => sin, Cos => cos, Tan => tan, Asin => asin,
        Acos => acos, Atan => atan, Sinh => sinh, Cosh => cosh, Asinh => asinh, Acosh => acosh,
        Atanh => atanh, Rsqrt => rsqrt, Reciprocal => reciprocal, Square => square, Cube => cube,
        Exp => exp, Sign => sign,
    }
);

op_enum!(
    /// In-place ops taking one scalar operand (`Backend::scalar_apply_<op>_*`).
    ScalarOp {
        Add => add, Sub => sub, Mul => mul, Div => div, Log => log, Log1p => log1p,
        LeakyRelu => leaky_relu, Elu => elu,
    }
);

impl From<BinaryOpType> for ScalarOp {
    fn from(op: BinaryOpType) -> Self {
        match op {
            BinaryOpType::Add => ScalarOp::Add,
            BinaryOpType::Sub => ScalarOp::Sub,
            BinaryOpType::Mul => ScalarOp::Mul,
            BinaryOpType::Div => ScalarOp::Div,
        }
    }
}

/// Operations the server can execute. Buffer-creating ops carry the id the client chose.
#[derive(Serialize, Deserialize)]
pub(crate) enum Op {
    /// Must be the first request on a connection; answered with `Reply::DeviceType`.
    Hello { version: u32, device: RemoteDevice },
    /// Drops a buffer. Never answered, not even on failure (the handle is already gone).
    Free { buf: BufId },
    /// No-op that is always answered; flushes the pipeline and surfaces pending errors.
    Sync,
    DeviceType,
    Alloc { dst: TypelessBuf, len: usize },
    AllocFromSlice { dst: TypelessBuf, src: Slice },
    CopyFromSlice { dst: TypelessBuf, src: Slice },
    CopyRangeWithin { dst: TypelessBuf, src: TypelessBuf, dst_offset: usize, src_offset: usize, len: usize },
    Read { buf: TypelessBuf, offset: usize },
    Write { buf: TypelessBuf, offset: usize, value: Value },
    Copy { src: TypelessBuf, dst: TypelessBuf },
    Dump { src: TypelessBuf },
    Broadcast {
        left: (TypelessBuf, MetaTensor),
        right: (TypelessBuf, MetaTensor),
        dst: (TypelessBuf, MetaTensor),
        op: BinaryOpType,
    },
    Matmul {
        lhs: (TypelessBuf, MetaTensor, ContiguityTypes),
        rhs: (TypelessBuf, MetaTensor, ContiguityTypes),
        dst: TypelessBuf,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Unary { buf: TypelessBuf, op: UnaryOp, layout: Layout },
    Scalar { buf: TypelessBuf, op: ScalarOp, value: Value, layout: Layout },
    Fill { buf: TypelessBuf, value: Value, layout: Layout },
    /// Elementwise dtype conversion between two equally sized buffers.
    Convert { src: TypelessBuf, dst: TypelessBuf },
    /// Reduces `src[start..start + len]` into `dst[0]`.
    ReduceFlat { src: TypelessBuf, dst: TypelessBuf, start: usize, len: usize, op: ReductionOpTypes },
    /// Reduces a contiguous tensor along `dim`.
    ReduceNd { src: (TypelessBuf, MetaTensor), dst: (TypelessBuf, MetaTensor), dim: Dim, op: ReductionOpTypes },
    /// Like `ReduceFlat`, writing an index into a `u64` buffer.
    ArgFlat { src: TypelessBuf, dst: TypelessBuf, start: usize, len: usize, op: ReductionOpTypes },
    /// Like `ReduceNd`, writing indices into a `u64` buffer.
    ArgNd { src: (TypelessBuf, MetaTensor), dst: (TypelessBuf, MetaTensor), dim: Dim, op: ReductionOpTypes },
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) id: u64,
    /// Explicit async flag: when false the server only responds if the op fails.
    pub(crate) reply: bool,
    pub(crate) op: Op,
}

/// Successful results. Most ops only ever produce `Ack`.
#[derive(Serialize, Deserialize)]
pub(crate) enum Reply {
    Ack,
    Value(Value),
    Slice(Slice),
    DeviceType(DeviceType),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Response {
    pub(crate) id: u64,
    pub(crate) result: Result<Reply, TensorError>,
}

fn io_err(e: std::io::Error) -> TensorError {
    TensorError::RemoteError(format!("connection error: {e}"))
}

/// Serializes `msg` as one length-prefixed frame and writes it with a single `write_all`.
pub(crate) fn write_frame<W: Write, M: Serialize>(w: &mut W, msg: &M) -> Result<(), TensorError> {
    let mut frame = vec![0u8; 8];
    bincode::serialize_into(&mut frame, msg)
        .map_err(|e| TensorError::RemoteError(format!("failed to encode message: {e}")))?;
    let n = (frame.len() - 8) as u64;
    frame[..8].copy_from_slice(&n.to_le_bytes());
    w.write_all(&frame).map_err(io_err)?;
    w.flush().map_err(io_err)
}

/// Reads one frame. Returns `Ok(None)` if the peer closed the connection cleanly between frames.
pub(crate) fn read_frame<R: Read, M: DeserializeOwned>(r: &mut R) -> Result<Option<M>, TensorError> {
    read_frame_limited(r, MAX_FRAME_BYTES)
}

/// [`read_frame`] with a caller-chosen size limit.
pub(crate) fn read_frame_limited<R: Read, M: DeserializeOwned>(r: &mut R, max_bytes: u64) -> Result<Option<M>, TensorError> {
    let mut len = [0u8; 8];
    let mut filled = 0;
    while filled < len.len() {
        match r.read(&mut len[filled..]) {
            Ok(0) if filled == 0 => return Ok(None),
            Ok(0) => return Err(TensorError::RemoteError("connection closed mid-frame".into())),
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(io_err(e)),
        }
    }
    let n = u64::from_le_bytes(len);
    if n > max_bytes {
        return Err(TensorError::RemoteError(format!("frame of {n} bytes exceeds the {max_bytes} byte limit")));
    }
    // Grow the buffer as bytes arrive instead of trusting the prefix with one huge allocation.
    let mut payload = Vec::with_capacity(n.min(64 << 20) as usize);
    r.take(n).read_to_end(&mut payload).map_err(io_err)?;
    if payload.len() as u64 != n {
        return Err(TensorError::RemoteError("connection closed mid-frame".into()));
    }
    bincode::deserialize(&payload)
        .map(Some)
        .map_err(|e| TensorError::RemoteError(format!("failed to decode message: {e}")))
}
