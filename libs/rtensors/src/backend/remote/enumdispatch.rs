use crate::{backend::{remote::{client::RemoteBuf, protocol::{Slice, TypelessBuf}, server::ClientConnection}, Backend, BackendMatMul, ContiguityTypes}, core::{primitives::DeviceType, tensor::TensorError, value::{types, DType}, MetaTensor}};
use super::server::select_buffer;

#[inline(always)]
pub(crate) fn dispatch_alloc_from_slice(
    slice: Slice,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match slice.dtype {
        DType::U8 => alloc_from_slice_for_dtype!(slice, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_from_slice_for_dtype!(slice, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_from_slice_for_dtype!(slice, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_from_slice_for_dtype!(slice, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_from_slice_for_dtype!(slice, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_from_slice_for_dtype!(slice, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_from_slice_for_dtype!(slice, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_from_slice_for_dtype!(slice, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_from_slice_for_dtype!(slice, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_from_slice_for_dtype!(slice, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_from_slice_for_dtype!(slice, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_from_slice_for_dtype!(slice, connection, F64, f64, f64_buffers),
        DType::BOOL => alloc_from_slice_for_dtype!(slice, connection, BOOL, types::boolean, bool_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
pub(crate) fn dispatch_alloc(
    len: usize,
    dtype: DType,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let remote_buf = match dtype {
        DType::U8 => alloc_for_dtype!(len, connection, U8, u8, u8_buffers),
        DType::U16 => alloc_for_dtype!(len, connection, U16, u16, u16_buffers),
        DType::U32 => alloc_for_dtype!(len, connection, U32, u32, u32_buffers),
        DType::U64 => alloc_for_dtype!(len, connection, U64, u64, u64_buffers),
        DType::U128 => alloc_for_dtype!(len, connection, U128, u128, u128_buffers),
        DType::I8 => alloc_for_dtype!(len, connection, I8, i8, i8_buffers),
        DType::I16 => alloc_for_dtype!(len, connection, I16, i16, i16_buffers),
        DType::I32 => alloc_for_dtype!(len, connection, I32, i32, i32_buffers),
        DType::I64 => alloc_for_dtype!(len, connection, I64, i64, i64_buffers),
        DType::I128 => alloc_for_dtype!(len, connection, I128, i128, i128_buffers),
        DType::F32 => alloc_for_dtype!(len, connection, F32, f32, f32_buffers),
        DType::F64 => alloc_for_dtype!(len, connection, F64, f64, f64_buffers),
        DType::BOOL => alloc_for_dtype!(len, connection, BOOL, types::boolean, bool_buffers),
    };
    Ok(remote_buf)
}

#[inline(always)]
pub(crate) fn dispatch_copy_from_slice(
    dst: TypelessBuf,
    src: Slice,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match dst.dtype {
        DType::U8 => copy_from_slice_for_dtype!(dst.id, src, connection, u8, u8_buffers),
        DType::U16 => copy_from_slice_for_dtype!(dst.id, src, connection, u16, u16_buffers),
        DType::U32 => copy_from_slice_for_dtype!(dst.id, src, connection, u32, u32_buffers),
        DType::U64 => copy_from_slice_for_dtype!(dst.id, src, connection, u64, u64_buffers),
        DType::U128 => copy_from_slice_for_dtype!(dst.id, src, connection, u128, u128_buffers),
        DType::I8 => copy_from_slice_for_dtype!(dst.id, src, connection, i8, i8_buffers),
        DType::I16 => copy_from_slice_for_dtype!(dst.id, src, connection, i16, i16_buffers),
        DType::I32 => copy_from_slice_for_dtype!(dst.id, src, connection, i32, i32_buffers),
        DType::I64 => copy_from_slice_for_dtype!(dst.id, src, connection, i64, i64_buffers),
        DType::I128 => copy_from_slice_for_dtype!(dst.id, src, connection, i128, i128_buffers),
        DType::F32 => copy_from_slice_for_dtype!(dst.id, src, connection, f32, f32_buffers),
        DType::F64 => copy_from_slice_for_dtype!(dst.id, src, connection, f64, f64_buffers),
        DType::BOOL => copy_from_slice_for_dtype!(dst.id, src, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_read(
    buf: TypelessBuf,
    offset: usize,
    connection: &ClientConnection,
) -> Result<crate::backend::remote::protocol::Value, TensorError> {
    let value = match buf.dtype {
        DType::U8 => read_for_dtype!(buf.id, offset, connection, u8, u8_buffers),
        DType::U16 => read_for_dtype!(buf.id, offset, connection, u16, u16_buffers),
        DType::U32 => read_for_dtype!(buf.id, offset, connection, u32, u32_buffers),
        DType::U64 => read_for_dtype!(buf.id, offset, connection, u64, u64_buffers),
        DType::U128 => read_for_dtype!(buf.id, offset, connection, u128, u128_buffers),
        DType::I8 => read_for_dtype!(buf.id, offset, connection, i8, i8_buffers),
        DType::I16 => read_for_dtype!(buf.id, offset, connection, i16, i16_buffers),
        DType::I32 => read_for_dtype!(buf.id, offset, connection, i32, i32_buffers),
        DType::I64 => read_for_dtype!(buf.id, offset, connection, i64, i64_buffers),
        DType::I128 => read_for_dtype!(buf.id, offset, connection, i128, i128_buffers),
        DType::F32 => read_for_dtype!(buf.id, offset, connection, f32, f32_buffers),
        DType::F64 => read_for_dtype!(buf.id, offset, connection, f64, f64_buffers),
        DType::BOOL => read_for_dtype!(buf.id, offset, connection, types::boolean, bool_buffers),
    };
    Ok(value)
}

#[inline(always)]
pub(crate) fn dispatch_write(
    buf: TypelessBuf,
    offset: usize,
    value: crate::backend::remote::protocol::Value,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => write_for_dtype!(buf.id, offset, value, connection, u8, u8_buffers),
        DType::U16 => write_for_dtype!(buf.id, offset, value, connection, u16, u16_buffers),
        DType::U32 => write_for_dtype!(buf.id, offset, value, connection, u32, u32_buffers),
        DType::U64 => write_for_dtype!(buf.id, offset, value, connection, u64, u64_buffers),
        DType::U128 => write_for_dtype!(buf.id, offset, value, connection, u128, u128_buffers),
        DType::I8 => write_for_dtype!(buf.id, offset, value, connection, i8, i8_buffers),
        DType::I16 => write_for_dtype!(buf.id, offset, value, connection, i16, i16_buffers),
        DType::I32 => write_for_dtype!(buf.id, offset, value, connection, i32, i32_buffers),
        DType::I64 => write_for_dtype!(buf.id, offset, value, connection, i64, i64_buffers),
        DType::I128 => write_for_dtype!(buf.id, offset, value, connection, i128, i128_buffers),
        DType::F32 => write_for_dtype!(buf.id, offset, value, connection, f32, f32_buffers),
        DType::F64 => write_for_dtype!(buf.id, offset, value, connection, f64, f64_buffers),
        DType::BOOL => write_for_dtype!(buf.id, offset, value, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_len(
    buf: TypelessBuf,
    connection: &ClientConnection,
) -> Result<usize, TensorError> {
    let len = match buf.dtype {
        DType::U8 => len_for_dtype!(buf.id, connection, u8, u8_buffers),
        DType::U16 => len_for_dtype!(buf.id, connection, u16, u16_buffers),
        DType::U32 => len_for_dtype!(buf.id, connection, u32, u32_buffers),
        DType::U64 => len_for_dtype!(buf.id, connection, u64, u64_buffers),
        DType::U128 => len_for_dtype!(buf.id, connection, u128, u128_buffers),
        DType::I8 => len_for_dtype!(buf.id, connection, i8, i8_buffers),
        DType::I16 => len_for_dtype!(buf.id, connection, i16, i16_buffers),
        DType::I32 => len_for_dtype!(buf.id, connection, i32, i32_buffers),
        DType::I64 => len_for_dtype!(buf.id, connection, i64, i64_buffers),
        DType::I128 => len_for_dtype!(buf.id, connection, i128, i128_buffers),
        DType::F32 => len_for_dtype!(buf.id, connection, f32, f32_buffers),
        DType::F64 => len_for_dtype!(buf.id, connection, f64, f64_buffers),
        DType::BOOL => len_for_dtype!(buf.id, connection, types::boolean, bool_buffers),
    };
    Ok(len)
}

#[inline(always)]
pub(crate) fn dispatch_copy(
    src: TypelessBuf,
    connection: &ClientConnection,
) -> Result<TypelessBuf, TensorError> {
    let new_buf = match src.dtype {
        DType::U8 => copy_for_dtype!(src.id, connection, U8, u8, u8_buffers),
        DType::U16 => copy_for_dtype!(src.id, connection, U16, u16, u16_buffers),
        DType::U32 => copy_for_dtype!(src.id, connection, U32, u32, u32_buffers),
        DType::U64 => copy_for_dtype!(src.id, connection, U64, u64, u64_buffers),
        DType::U128 => copy_for_dtype!(src.id, connection, U128, u128, u128_buffers),
        DType::I8 => copy_for_dtype!(src.id, connection, I8, i8, i8_buffers),
        DType::I16 => copy_for_dtype!(src.id, connection, I16, i16, i16_buffers),
        DType::I32 => copy_for_dtype!(src.id, connection, I32, i32, i32_buffers),
        DType::I64 => copy_for_dtype!(src.id, connection, I64, i64, i64_buffers),
        DType::I128 => copy_for_dtype!(src.id, connection, I128, i128, i128_buffers),
        DType::F32 => copy_for_dtype!(src.id, connection, F32, f32, f32_buffers),
        DType::F64 => copy_for_dtype!(src.id, connection, F64, f64, f64_buffers),
        DType::BOOL => copy_for_dtype!(src.id, connection, BOOL, types::boolean, bool_buffers),
    };
    Ok(new_buf)
}

#[inline(always)]
pub(crate) fn dispatch_dump(
    src: TypelessBuf,
    connection: &ClientConnection,
) -> Result<Slice, TensorError> {
    let slice = match src.dtype {
        DType::U8 => dump_for_dtype!(src.id, connection, u8, u8_buffers),
        DType::U16 => dump_for_dtype!(src.id, connection, u16, u16_buffers),
        DType::U32 => dump_for_dtype!(src.id, connection, u32, u32_buffers),
        DType::U64 => dump_for_dtype!(src.id, connection, u64, u64_buffers),
        DType::U128 => dump_for_dtype!(src.id, connection, u128, u128_buffers),
        DType::I8 => dump_for_dtype!(src.id, connection, i8, i8_buffers),
        DType::I16 => dump_for_dtype!(src.id, connection, i16, i16_buffers),
        DType::I32 => dump_for_dtype!(src.id, connection, i32, i32_buffers),
        DType::I64 => dump_for_dtype!(src.id, connection, i64, i64_buffers),
        DType::I128 => dump_for_dtype!(src.id, connection, i128, i128_buffers),
        DType::F32 => dump_for_dtype!(src.id, connection, f32, f32_buffers),
        DType::F64 => dump_for_dtype!(src.id, connection, f64, f64_buffers),
        DType::BOOL => dump_for_dtype!(src.id, connection, types::boolean, bool_buffers),
    };
    Ok(slice)
}

#[inline(always)]
pub(crate) fn dispatch_apply_binary_elementwise_contiguous(
    buf: TypelessBuf,
    op: crate::ops::base::BinaryOpType,
    value: crate::backend::remote::protocol::Value,
    start: usize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, f64, f64_buffers),
        DType::BOOL => apply_elementwise_binary_contiguous_for_dtype!(buf.id, op, value, start, len, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_apply_binary_elementwise_1d_strided(
    buf: TypelessBuf,
    op: crate::ops::base::BinaryOpType,
    value: crate::backend::remote::protocol::Value,
    offset: usize,
    stride: isize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, f64, f64_buffers),
        DType::BOOL => apply_elementwise_binary_1d_strided_for_dtype!(buf.id, op, value, offset, stride, len, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_apply_binary_elementwise_nd(
    buf: TypelessBuf,
    op: crate::ops::base::BinaryOpType,
    value: crate::backend::remote::protocol::Value,
    offset: usize,
    shape: &[usize],
    stride: &[isize],
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::U8 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u8, u8_buffers),
        DType::U16 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u16, u16_buffers),
        DType::U32 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u32, u32_buffers),
        DType::U64 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u64, u64_buffers),
        DType::U128 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, u128, u128_buffers),
        DType::I8 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i8, i8_buffers),
        DType::I16 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i16, i16_buffers),
        DType::I32 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i32, i32_buffers),
        DType::I64 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i64, i64_buffers),
        DType::I128 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, i128, i128_buffers),
        DType::F32 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, f32, f32_buffers),
        DType::F64 => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, f64, f64_buffers),
        DType::BOOL => apply_elementwise_binary_nd_for_dtype!(buf.id, op, value, offset, shape, stride, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_apply_neg_contiguous(
    buf: TypelessBuf,
    start: usize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::I8 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, i8, i8_buffers),
        DType::I16 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, i16, i16_buffers),
        DType::I32 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, i32, i32_buffers),
        DType::I64 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, i64, i64_buffers),
        DType::I128 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, i128, i128_buffers),
        DType::F32 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, f32, f32_buffers),
        DType::F64 => apply_neg_contiguous_for_dtype!(buf.id, start, len, connection, f64, f64_buffers),
        _ => {
            // Negation is not defined for unsigned or boolean types
            Err(TensorError::UnsupportedOperation(format!(
                "Negation is not supported for dtype {:?}",
                buf.dtype
            )))
        }
    }
}

#[inline(always)]
pub(crate) fn dispatch_apply_neg_1d_strided(
    buf: TypelessBuf,
    offset: usize,
    stride: isize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::I8 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, i8, i8_buffers),
        DType::I16 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, i16, i16_buffers),
        DType::I32 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, i32, i32_buffers),
        DType::I64 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, i64, i64_buffers),
        DType::I128 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, i128, i128_buffers),
        DType::F32 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, f32, f32_buffers),
        DType::F64 => apply_neg_1d_strided_for_dtype!(buf.id, offset, stride, len, connection, f64, f64_buffers),
        _ => {
            // Negation is not defined for unsigned or boolean types
            Err(TensorError::UnsupportedOperation(format!(
                "Negation is not supported for dtype {:?}",
                buf.dtype
            )))
        }
    }
}

#[inline(always)]
pub(crate) fn dispatch_apply_neg_nd(
    buf: TypelessBuf,
    offset: usize,
    shape: &[usize],
    stride: &[isize],
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match buf.dtype {
        DType::I8 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, i8, i8_buffers),
        DType::I16 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, i16, i16_buffers),
        DType::I32 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, i32, i32_buffers),
        DType::I64 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, i64, i64_buffers),
        DType::I128 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, i128, i128_buffers),
        DType::F32 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, f32, f32_buffers),
        DType::F64 => apply_neg_nd_for_dtype!(buf.id, offset, shape, stride, connection, f64, f64_buffers),
        _ => {
            // Negation is not defined for unsigned or boolean types
            Err(TensorError::UnsupportedOperation(format!(
                "Negation is not supported for dtype {:?}",
                buf.dtype
            )))
        }
    }
}

#[inline(always)]
pub(crate) fn dispatch_broadcast(
    left: (TypelessBuf, MetaTensor),
    right: (TypelessBuf, MetaTensor),
    dst: (TypelessBuf, MetaTensor),
    op: crate::ops::base::BinaryOpType,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    let (left_buf, left_meta) = left;
    let (right_buf, right_meta) = right;
    let (dst_buf, dst_meta) = dst;
    
    match dst_buf.dtype {
        DType::U8 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u8, u8_buffers),
        DType::U16 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u16, u16_buffers),
        DType::U32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u32, u32_buffers),
        DType::U64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u64, u64_buffers),
        DType::U128 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, u128, u128_buffers),
        DType::I8 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i8, i8_buffers),
        DType::I16 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i16, i16_buffers),
        DType::I32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i32, i32_buffers),
        DType::I64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i64, i64_buffers),
        DType::I128 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, i128, i128_buffers),
        DType::F32 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, f32, f32_buffers),
        DType::F64 => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, f64, f64_buffers),
        DType::BOOL => broadcast_for_dtype!(left_buf.id, &left_meta, right_buf.id, &right_meta, dst_buf.id, &dst_meta, op, connection, types::boolean, bool_buffers),
    }
}

#[inline(always)]
pub(crate) fn dispatch_matmul(
    lhs: (TypelessBuf, MetaTensor, ContiguityTypes),
    rhs: (TypelessBuf, MetaTensor, ContiguityTypes),
    dst: TypelessBuf,
    b: usize,
    m: usize,
    k: usize,
    n: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    let (lhs_buf, lhs_meta, lhs_contiguity) = lhs;
    let (rhs_buf, rhs_meta, rhs_contiguity) = rhs;
    
    match lhs_buf.dtype {
        DType::U8 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, U8, u8, u8_buffers),
        DType::U16 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, U16, u16, u16_buffers),
        DType::U32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, U32, u32, u32_buffers),
        DType::U64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, U64, u64, u64_buffers),
        DType::U128 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, U128, u128, u128_buffers),
        DType::I8 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, I8, i8, i8_buffers),
        DType::I16 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, I16, i16, i16_buffers),
        DType::I32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, I32, i32, i32_buffers),
        DType::I64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, I64, i64, i64_buffers),
        DType::I128 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, I128, i128, i128_buffers),
        DType::F32 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, F32, f32, f32_buffers),
        DType::F64 => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, F64, f64, f64_buffers),
        DType::BOOL => matmul_for_dtype!(lhs_buf.id, &lhs_meta, lhs_contiguity, rhs_buf.id, &rhs_meta, rhs_contiguity, dst.id, b, m, k, n, connection, BOOL, types::boolean, bool_buffers),
    };
    
    Ok(())
}

pub(crate) fn dispath_copy_within(
    dst: TypelessBuf,
    src: TypelessBuf,
    dst_offset: usize,
    src_offset: usize,
    len: usize,
    connection: &ClientConnection,
) -> Result<(), TensorError> {
    match dst.dtype {
        DType::U8 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, u8, u8_buffers),
        DType::U16 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, u16, u16_buffers),
        DType::U32 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, u32, u32_buffers),
        DType::U64 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, u64, u64_buffers),
        DType::U128 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, u128, u128_buffers),
        DType::I8 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, i8, i8_buffers),
        DType::I16 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, i16, i16_buffers),
        DType::I32 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, i32, i32_buffers),
        DType::I64 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, i64, i64_buffers),
        DType::I128 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, i128, i128_buffers),
        DType::F32 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, f32, f32_buffers),
        DType::F64 => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, f64, f64_buffers),
        DType::BOOL => copy_within_for_dtype!(dst.id, src.id, dst_offset, src_offset, len, connection, types::boolean, bool_buffers),
    }
}