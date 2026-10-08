use serde::{Deserialize, Serialize};

use crate::{backend::remote::client::RemoteBuf, core::{meta::ContiguityTypes, primitives::DeviceType, tensor::TensorError, value::{DType, TensorValue}, MetaTensor}, ops::base::BinaryOpType};


#[derive(Serialize, Deserialize)]
pub(crate) struct Slice {
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

impl<T: TensorValue> From<Slice> for Result<Box<[T]>, TensorError> {
    fn from(val: Slice) -> Self {
        val.to_boxed_slice::<T>()
    }
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


impl<T: TensorValue> From<Box<[T]>> for Slice {
    fn from(boxed: Box<[T]>) -> Self {
        Slice::from_boxed_slice(boxed)
    }
}


impl<T: TensorValue> From<&[T]> for Slice {
    fn from(slice: &[T]) -> Self {
        Slice::from_slice(slice)
    }
}


#[derive(Serialize, Deserialize)]
pub(crate) struct Value {
    data: Vec<u8>, // bytes
    dtype: DType,
}

impl<T: TensorValue> From<Value> for Result<T, TensorError> {
    fn from(val: Value) -> Self {
        val.to_value::<T>()
    }
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

impl<T: TensorValue> From<T> for Value {
    fn from(value: T) -> Self {
        Value::from_value(value)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub(crate) struct TypelessBuf {
    pub(crate) id: u32,
    pub(crate) dtype: DType,
}

impl<T: TensorValue> From<TypelessBuf> for Result<RemoteBuf<T>, TensorError> {
    fn from(val: TypelessBuf) -> Self {
        Ok(RemoteBuf::from_typeless(val))
    }
}


#[derive(Serialize, Deserialize)]
pub(crate) struct Response {
    pub(crate) asynchronous: bool,
    pub(crate) complete: bool,
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
    pub(crate) error: Option<TensorError>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) task_id: u32,
    pub(crate) message: Messages,
}

impl Request {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        debug_assert!(!self.message.is_response());
        bincode::serialize(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        let resp: Request = bincode::deserialize(data)?;
        debug_assert!(!resp.message.is_response());
        Ok(resp)
    }
}

impl Response {
    #[inline(always)]
    pub fn serialize(&self) -> Result<Vec<u8>, bincode::Error> {
        debug_assert!(self.message.is_response());
        bincode::serialize(self)
    }

    #[inline(always)]
    pub fn deserialize(data: &[u8]) -> Result<Self, bincode::Error> {
        let resp: Response = bincode::deserialize(data)?;
        debug_assert!(resp.message.is_response());
        Ok(resp)
    }
}

impl<T: TensorValue> From<RemoteBuf<T>> for TypelessBuf {
    fn from(buf: RemoteBuf<T>) -> Self {
        Self {
            id: buf.id,
            dtype: buf.dtype,
        }
    }
}

macro_rules! impl_typeless_buf_conversions {
    ($($type:ty),+ $(,)?) => {
        $(
            impl From<TypelessBuf> for RemoteBuf<$type> {
                fn from(buf: TypelessBuf) -> Self {
                    Self {
                        id: buf.id,
                        dtype: buf.dtype,
                        _marker: std::marker::PhantomData::<$type>,
                    }
                }
            }
        )+
    };
}

impl_typeless_buf_conversions!(
    f32, f64,
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
);

#[derive(Serialize, Deserialize)]
pub (crate) enum Messages {
    ErrorResponse {
        message: String,
    },
    DeviceType,
    DeviceTypeResponse {
        device_type: DeviceType
    },

    AllocFromSlice {
        src: Slice
    },
    AllocFromSliceResponse(Result<TypelessBuf, TensorError>),
    
    Alloc {
        len: usize,
        dtype: DType,
    },
    AllocResponse (Result<TypelessBuf, TensorError>),

    CopyFromSlice {
        dst: TypelessBuf,
        src: Slice
    },
    CopyFromSliceResponse (Result<(), TensorError>),

    Read {
        buf: TypelessBuf,
        offset: usize,
    },
    ReadResponse (Result<Value, TensorError>,),

    Write {
        buf: TypelessBuf,
        offset: usize,
        value: Value,
    },
    WriteResponse (Result<(), TensorError>),

    Len {
        buf: TypelessBuf,
    },
    LenResponse (usize),

    Copy {
        src: TypelessBuf,
    },
    CopyResponse(Result<TypelessBuf, TensorError>),

    Dump {
        src: TypelessBuf,
    },
    DumpResponse (Result<Slice, TensorError>),

    ApplyElementwiseBinary1dStrided {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyElementwiseBinary1dStridedResponse (Result<(), TensorError>),

    ApplyElementwiseBinaryContiguous {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        start: usize,
        len: usize,
    },
    ApplyElementwiseBinaryContiguousResponse (Result<(), TensorError>),

    ApplyElementwiseBinaryNd {
        buf: TypelessBuf,
        op: (BinaryOpType, Value),
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    ApplyElementwiseBinaryNdResponse (Result<(), TensorError>),

    Broadcast {
        left: (TypelessBuf, MetaTensor),
        right: (TypelessBuf, MetaTensor),
        dst: (TypelessBuf, MetaTensor),
        op: BinaryOpType,
    },
    BroadcastResponse (Result<(), TensorError>),

    ApplyNegContiguous {
        buf: TypelessBuf,
        start: usize,
        len: usize,
    },
    ApplyNegContiguousResponse (Result<(), TensorError>),

    ApplyNeg1dStrided {
        buf: TypelessBuf,
        offset: usize,
        stride: isize,
        len: usize,
    },
    ApplyNeg1dStridedResponse (Result<(), TensorError>),

    ApplyNegNd {
        buf: TypelessBuf,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<isize>,
    },
    ApplyNegNdResponse (Result<(), TensorError>),

    Matmul {
        lhs: (TypelessBuf, MetaTensor, ContiguityTypes),
        rhs: (TypelessBuf, MetaTensor, ContiguityTypes),
        dst: TypelessBuf,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    MatmulResponse (Result<(), TensorError>),










    CopyRangeWithin {
        dst: TypelessBuf,
        src: TypelessBuf,
        dst_offset: usize,
        src_offset: usize,
        len: usize,
    },
    CopyRangeWithinResponse(Result<(), TensorError>),

    ActionCompleted(u32)

}

impl Messages {
    #[inline(always)]
    pub fn is_response(&self) -> bool {
        match self {
            Messages::DeviceTypeResponse { .. } |
            Messages::AllocFromSliceResponse { .. } |
            Messages::AllocResponse { .. } |
            Messages::CopyFromSliceResponse { .. } |
            Messages::ReadResponse { .. } |
            Messages::WriteResponse { .. } |
            Messages::LenResponse { .. } |
            Messages::CopyResponse { .. } |
            Messages::DumpResponse { .. } |
            Messages::ApplyElementwiseBinary1dStridedResponse { .. } |
            Messages::ApplyElementwiseBinaryContiguousResponse { .. } |
            Messages::ApplyElementwiseBinaryNdResponse { .. } |
            Messages::BroadcastResponse { .. } |
            Messages::MatmulResponse { .. } |
            Messages::ApplyNeg1dStridedResponse { .. } |
            Messages::ApplyNegContiguousResponse { .. } |
            Messages::ApplyNegNdResponse { .. } |
            Messages::ErrorResponse { .. } |
            Messages::ActionCompleted { .. } |
            Messages::CopyRangeWithinResponse { .. } => true,
            _ => false,
        }
    }
}