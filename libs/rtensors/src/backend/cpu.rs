

use crate::{backend::{Backend, BackendMatMul}, core::{meta::TensorOffsetIterator, primops::{Exp, InvExp, SquareRoot}, tensor::TensorError, value::{types, TensorValue, WeightValue}, MetaTensor}, openblas::{blasint, cblas_dgemm, cblas_sgemm, CBLAS_ORDER, CBLAS_TRANSPOSE}, ops::{base::BinaryOpType, reduction::Accumulator}};
use crate::backend::ContiguityTypes;
use crate::core::value::TypeConstants;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Cpu;

macro_rules! elemwise_contiguous_loop {
    ($buf:expr, $start:expr, $len:expr, |$x:ident| $body:expr) => {{
        let slice = &mut $buf[$start .. $start + $len];
        for $x in slice.iter_mut() {
            *$x = $body;
        }
    }};
}

macro_rules! elemwise_1d_strided_loop {
    ($buf:expr, $offset:expr, $stride:expr, $len:expr, |$x:ident| $body:expr) => {{
        let mut idx: isize = $offset as isize;
        for _ in 0..$len {
            let $x = &mut $buf[idx as usize];
            *$x = $body;
            idx += $stride;
        }
    }};
}

macro_rules! elemwise_nd_loop {
    ($buf:expr, $offset:expr, $shape:expr, $stride:expr, |$x:ident| $body:expr) => {{
        let iter = TensorOffsetIterator::new(
            $shape,
            $stride,
            $offset,
        );
        for idx in iter {
            let $x = &mut $buf[idx];
            *$x = $body;
        }
    }};
}

macro_rules! impl_cpu_unary {
    ($name:ident, $func:ident $( where $($extra:tt)+ )?) => {
        paste::paste! {
            fn [<apply_ $name _1d_strided>]<T: TensorValue>(
                &self, buf: &mut Self::Buf<T>, 
                    offset: usize,
                    stride: isize,
                    len: usize
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_1d_strided_loop!(bufptr, offset, stride, len, |x| $func(x));
                Ok(())
            }

            fn [<apply_ $name _contiguous>]<T: TensorValue>(
                &self, buf: &mut Self::Buf<T>, 
                    start: usize,
                    len: usize
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_contiguous_loop!(bufptr, start, len, |x| $func(x) );
                Ok(())
            }

            fn [<apply_ $name _nd>]<T: TensorValue>(
                    &self,
                    buf: &mut Self::Buf<T>,
                    offset: usize,
                    shape: &[usize],
                    stride: &[isize],
                ) -> Result<(), TensorError>
                $( where $($extra)+ )?
                {
                let bufptr = buf.as_mut();
                elemwise_nd_loop!(bufptr, offset, shape, stride, |x| $func(x));
                Ok(())
            }
        }
    };
}

macro_rules! impl_cpu_scalar {
    ($name:ident, $func:ident $( where $($extra:tt)+ )?) => {
        paste::paste! {
            fn [<scalar_apply_ $name _1d_strided>]<T: TensorValue>(
                &self, 
                buf: &mut Self::Buf<T>, 
                value: T,
                offset: usize,
                stride: isize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            {
                let bufptr = buf.as_mut();
                elemwise_1d_strided_loop!(bufptr, offset, stride, len, |x| $func(x, value));
                Ok(())
            }

            fn [<scalar_apply_ $name _contiguous>]<T: TensorValue>(
                &self, 
                buf: &mut Self::Buf<T>, 
                value: T,
                start: usize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            {
                let bufptr = buf.as_mut();
                elemwise_contiguous_loop!(bufptr, start, len, |x| $func(x, value));
                Ok(())
            }

            fn [<scalar_apply_ $name _nd>]<T: TensorValue>(
                &self,
                buf: &mut Self::Buf<T>,
                value: T,
                offset: usize,
                shape: &[usize],
                stride: &[isize],
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            {
                let bufptr = buf.as_mut();
                elemwise_nd_loop!(bufptr, offset, shape, stride, |x| $func(x, value));
                Ok(())
            }
        }
    };
}


impl Backend for Cpu {
    type Buf<T: TensorValue> = Box<[T]>;


    fn device_type() -> crate::core::primitives::DeviceType {
        crate::core::primitives::DeviceType::Cpu
    }
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Box<[T]>, TensorError> {
        Ok(vec![T::default(); len].into())
    }

    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), TensorError> {
        if dst.len() != src.len() {
            return Err(TensorError::SizeMismatch(format!(
                "Buffer size mismatch in copy_from_slice: dst size {}, src size {}",
                dst.len(),
                src.len()
            )));
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    fn copy_range_within<T: TensorValue>(
        &self, 
        dst: &mut Self::Buf<T>, 
        src: &Self::Buf<T>, 
        dst_offset: usize, 
        src_offset: usize, 
        len: usize
    ) -> Result<(), TensorError> {
        if dst_offset + len > dst.len() || src_offset + len > src.len() {
            return Err(TensorError::IdxOutOfBounds(format!(
                "Index out of bounds in copy_range_within: dst size {}, src size {}, dst_offset {}, src_offset {}, len {}",
                dst.len(),
                src.len(),
                dst_offset,
                src_offset,
                len
            )));
        }
        dst[dst_offset..dst_offset + len].copy_from_slice(&src[src_offset..src_offset + len]);
        Ok(())
    }

    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, TensorError> {
        Ok(*buf.get(offset).ok_or(
            TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len()
            )),
        )?)
    }

    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), TensorError> {
        match buf.get_mut(offset) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(TensorError::IdxOutOfBounds(format!(
                "Index {} out of bounds for buffer of length {}",
                offset,
                buf.len()
            ))),
        }
    }
    
    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, TensorError> {
        Ok(src)
    }
    
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize {
        buf.len()
    }
    
    fn new() -> Self {
        Self
    }

    fn convert<T: TensorValue, N: TensorValue>(&self, src: &Self::Buf<T>, dst: &mut Self::Buf<N>) -> Result<(), TensorError> {
        if src.len() != dst.len() {
            return Err(TensorError::SizeMismatch(format!(
                "Buffer size mismatch in convert: src size {}, dst size {}",
                src.len(),
                dst.len()
            )));
        }
        for i in 0..src.len() {
            dst[i] = T::convert::<N>(&src[i]);
        }
        Ok(())
    }

    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError> {
        let mut dst = self.alloc(src.len())?;
        dst.copy_from_slice(src);
        Ok(dst)
    }
    
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError> {
        Ok(src.clone())
    }


    fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
    ) -> Result<(), TensorError> {
        // this is a stupid algorithm which is O(rank*size)
        // it can be optimized to O(size) later
        // a cleaner O(rank*size) algorithm just uses the coordinate iterator
        // and converts the, to full offsets
        let (left_buf, left_meta) = left;
        let (right_buf, right_meta) = right;
        let (dst_buf, dst_meta) = dst;

        let rank = dst_meta.rank();

        let sl = left_meta.strides();
        let sr = right_meta.strides();
        let sd = dst_meta.strides();

        
        let mut ol = left_meta.offset() as isize;
        let mut or = right_meta.offset() as isize;
        let mut od = dst_meta.offset() as isize;

        // println!("Strides: left: {:?}, right: {:?}, dst: {:?}", sl, sr, sd);
        // println!("Offsets: left: {}, right: {}, dst: {}", ol, or, od);

        let mut coords = vec![0; rank];

        let mut first = true;

        for new_coord in dst_meta.iter_coords() {
            // println!("Coords: {:?}", new_coord);
            if first {
                first = false;
            } else{
                for d in (0..rank).rev() {
                    if new_coord[d] != coords[d] {
                        let delta = new_coord[d] as isize - coords[d] as isize;
                        ol += delta * sl[d];
                        or += delta * sr[d];
                        od += delta * sd[d];
                    }
                }
            }
            coords = new_coord;
            debug_assert!(od >= 0);
            debug_assert!(ol >= 0);
            debug_assert!(or >= 0);
            // dst_buf[od as usize] = op.apply(left_buf[ol as usize], right_buf[or as usize]);
            unsafe {
                let lval = (*left_buf)[ol as usize];
                let rval = (*right_buf)[or as usize];
                (*dst_buf)[od as usize] = op.apply(lval, rval);
            }
        }

        Ok(())
    }
    impl_cpu_unary!{ neg, _negate where T: std::ops::Neg<Output = T> }
    impl_cpu_unary!{ relu, _relu }
    impl_cpu_unary!{ sigmoid, _sigmoid where T: InvExp}
    impl_cpu_unary!{ silu, _silu where T: InvExp}
    impl_cpu_unary!{ tanh, _tanh where T: Exp + InvExp }
    impl_cpu_unary!{ abs, _abs }
    impl_cpu_unary!{ sqrt, _sqrt where T: SquareRoot }
    impl_cpu_unary!{ sinh, _sinh where T: Exp + InvExp }
    impl_cpu_unary!{ cosh, _cosh where T: Exp + InvExp }
    impl_cpu_unary!{ asinh, _asinh where T: WeightValue }
    impl_cpu_unary!{ acosh, _acosh where T: WeightValue }
    impl_cpu_unary!{ atanh, _atanh where T: WeightValue }
    impl_cpu_unary!{ rsqrt, _rsqrt where T: SquareRoot }
    impl_cpu_unary!{ reciprocal, _reciprocal }
    impl_cpu_unary!{ square, _square }
    impl_cpu_unary!{ cube, _cube }
    impl_cpu_unary!{ exp, _exp where T: Exp }
    impl_cpu_unary!{ sign, _sign where T: std::ops::Neg<Output = T> }
    impl_cpu_unary!{ sin, _sin where T: WeightValue }
    impl_cpu_unary!{ cos, _cos where T: WeightValue }
    impl_cpu_unary!{ tan, _tan where T: WeightValue }
    impl_cpu_unary!{ asin, _asin where T: WeightValue }
    impl_cpu_unary!{ acos, _acos where T: WeightValue }
    impl_cpu_unary!{ atan, _atan where T: WeightValue }

    impl_cpu_unary!{ ln, _ln where T: WeightValue }
    impl_cpu_unary!{ expm1, _expm1 where T: Exp }
    impl_cpu_unary!{ ln1p, _ln1p where T: WeightValue }
    impl_cpu_unary!{ floor, _floor where T: WeightValue }
    impl_cpu_unary!{ ceil, _ceil where T: WeightValue }
    impl_cpu_unary!{ round, _round where T: WeightValue }
    impl_cpu_unary!{ trunc, _trunc where T: WeightValue }
    
    // Scalar binary operations
    impl_cpu_scalar!{ add, _scalar_add }
    impl_cpu_scalar!{ sub, _scalar_sub }
    impl_cpu_scalar!{ mul, _scalar_mul }
    impl_cpu_scalar!{ div, _scalar_div }
    impl_cpu_scalar!{ log, _scalar_log where T: WeightValue }
    impl_cpu_scalar!{ log1p, _scalar_log1p where T: WeightValue }
    impl_cpu_scalar!{ leaky_relu, _scalar_leaky_relu }
    impl_cpu_scalar!{ elu, _scalar_elu where T: WeightValue }

    /// go through entire buffer, take everything
    fn apply_reduce_contiguous_flat<T: WeightValue>(
        &self, 
        src: &Self::Buf<T>, 
        dst: &mut Self::Buf<T>, 
        start: usize, 
        len: usize, 
        op: crate::ops::reduction::ReductionOpTypes
    ) -> Result<(), TensorError> {
        let mut accumulator = op.get_accumulator::<T>();
        for i in src[start..start+len].iter() {
            accumulator.accumulate(*i);
        }
        dst[0] = accumulator.finalize(len);
        Ok(())
    }
        
    fn apply_reduce_contiguous_nd<T: WeightValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<T>, &MetaTensor), 
        dim: crate::core::Dim,
        op: crate::ops::reduction::ReductionOpTypes
    ) -> Result<(), TensorError> {
        let (in_d, in_meta) = src;
        let (out_d, _out_meta) = dst;

        let inner = in_meta.inner_dimensions(dim);
        let outer = in_meta.outer_dimensions(dim);
        let r = in_meta.shape()[dim];

        let r_stride = in_meta.strides()[dim];
            // let r_stride = inner;
        let left_stride = r * inner;

        for i in 0..inner {
            for o in 0..outer {
                // let base = 0; // the bottom of the  TODO row
                let base = o * left_stride + i;
                // let out_idx = 0; // TODO compute output index
                let out_idx = o * inner + i;
                let mut accum = op.get_accumulator::<T>();
                for k in 0..r  {
                    // iterate this rows elements
                    let idx = base + k * r_stride as usize;
                    accum.accumulate(in_d[idx]);
                } 
                // out_d[out_idx] = result;
                out_d[out_idx] = accum.finalize(r);
            }
        }

        Ok(())
    }


    fn apply_argmax_contiguous_flat<T: WeightValue>(
            &self, 
            _src: &Self::Buf<T>, 
            _dst: &mut Self::Buf<u64>, 
            _start: usize, 
            _len: usize, 
            _op: crate::ops::reduction::ReductionOpTypes
        ) -> Result<(), TensorError> {
        todo!()
    }    

    fn apply_argmax_contiguous_nd<T: WeightValue>(
            &self, 
            _src: (&Self::Buf<T>, &MetaTensor), 
            _dst: (&mut Self::Buf<u64>, &MetaTensor), 
            _dim: crate::core::Dim,
            _op: crate::ops::reduction::ReductionOpTypes
        ) -> Result<(), TensorError> {
        todo!()
    }
    
    fn apply_conv_2d<T: WeightValue>(
        &self, 
        _input: (&Self::Buf<T>, &MetaTensor), 
        _kernel: (&Self::Buf<T>, &MetaTensor),
        _output: &mut Self::Buf<T>,
        _config: &crate::ops::linalg::ConvConfig2D,
    ) -> Result<(), TensorError> {
        todo!()
    }

}

#[inline]
fn _sinh<T: TensorValue + Exp + InvExp>(x: &mut T) -> T {
    let a = x.apply_exp();
    let b = x.apply_invexp();
    (a - b) / (T::ONE + T::ONE)
}

#[inline]
fn _cosh<T: TensorValue + Exp + InvExp>(x: &mut T) -> T {
    let a = x.apply_exp();
    let b = x.apply_invexp();
    (a + b) / (T::ONE + T::ONE)
}

#[inline]
fn _asinh<T: WeightValue>(x: &mut T) -> T {
    // asinh(x) = ln(x + sqrt(x^2 + 1))
    let x_sq = *x * *x;
    let one = T::ONE;
    ((*x) + (x_sq + one).apply_sqrt()).nat_log()
}

#[inline]
fn _acosh<T: WeightValue>(x: &mut T) -> T {
    // acosh(x) = ln(x + sqrt(x^2 - 1))
    let x_sq = *x * *x;
    let one = T::ONE;
    ((*x) + (x_sq - one).apply_sqrt()).nat_log()
}

#[inline]
fn _atanh<T: WeightValue>(x: &mut T) -> T {
    // atanh(x) = 0.5 * ln((1 + x) / (1 - x))
    let one = T::ONE;
    let two = one + one;
    ((one + *x) / (one - *x)).nat_log() / two
}

#[inline]
fn _rsqrt<T: TensorValue + SquareRoot>(x: &mut T) -> T {
    T::ONE / x.apply_sqrt()
}

#[inline]
fn _reciprocal<T: TensorValue>(x: &mut T) -> T {
    T::ONE / *x
}

#[inline]
fn _square<T: TensorValue>(x: &mut T) -> T {
    *x * *x
}

#[inline]
fn _cube<T: TensorValue>(x: &mut T) -> T {
    *x * *x * *x
}

#[inline]
fn _exp<T: Exp>(x: &mut T) -> T {
    x.apply_exp()
}

#[inline]
fn _sign<T: TensorValue + std::ops::Neg<Output = T>>(x: &mut T) -> T {
    if *x < T::ZERO {
        -T::ONE
    } else if *x > T::ZERO {
        T::ONE
    } else {
        T::ZERO
    }
}

#[inline]
fn _sin<T: WeightValue>(x: &mut T) -> T {
    x.vsin()
}

#[inline]
fn _cos<T: WeightValue>(x: &mut T) -> T {
    x.vcos()
}

#[inline]
fn _tan<T: WeightValue>(x: &mut T) -> T {
    x.vtan()
}

#[inline]
fn _asin<T: WeightValue>(x: &mut T) -> T {
    x.vasin()
}

#[inline]
fn _acos<T: WeightValue>(x: &mut T) -> T {
    x.vacos()
}

#[inline]
fn _atan<T: WeightValue>(x: &mut T) -> T {
    x.vatan()
}

#[inline]
fn _ln<T: WeightValue>(x: &mut T) -> T {
    x.nat_log()
}

#[inline]
fn _ln1p<T: WeightValue>(x: &mut T) -> T {
    x.nat_log1p()
}

#[inline]
fn _floor<T: WeightValue>(x: &mut T) -> T {
    x.vfloor()
}

#[inline]
fn _ceil<T: WeightValue>(x: &mut T) -> T {
    x.vceil()
}

#[inline]
fn _round<T: WeightValue>(x: &mut T) -> T {
    x.vround()
}

#[inline]
fn _trunc<T: WeightValue>(x: &mut T) -> T {
    x.vtrunc()
}

#[inline]
fn _expm1<T: Exp>(x: &mut T) -> T {
    x.apply_expm1()
}

#[inline]
fn _tanh<T: TensorValue + InvExp + Exp>(x: &mut T) -> T {
    let a = x.apply_exp();
    let b = x.apply_invexp();
    (a - b) / (a + b)
}

#[inline]
fn _abs<T: TensorValue>(x: &mut T) -> T {
    x.absolute()
}



#[inline]
fn _sqrt<T: TensorValue + SquareRoot>(x: &mut T) -> T {
    x.apply_sqrt()
}


#[inline]
fn _negate<T: TensorValue + std::ops::Neg<Output = T>>(x: &mut T) -> T {
    -*x
}


#[inline]
fn _relu<T: TensorValue>(x: &mut T) -> T {
    if *x > T::ZERO {
        *x
    } else {
        T::ZERO
    }
}

/// The sigmoid function, implemented based
/// on a quick google search.
#[inline]
fn _sigmoid<T: TensorValue>(x: &mut T) -> T
where 
    T: InvExp
{
    T::ONE / (T::ONE + x.apply_invexp())
}

#[inline]
fn _silu<T: TensorValue>(x: &mut T) -> T
where 
    T: InvExp
{
    *x * _sigmoid(x)
}

// Scalar binary operation helper functions
#[inline]
fn _scalar_add<T: TensorValue>(x: &mut T, value: T) -> T {
    *x + value
}

#[inline]
fn _scalar_sub<T: TensorValue>(x: &mut T, value: T) -> T {
    *x - value
}

#[inline]
fn _scalar_mul<T: TensorValue>(x: &mut T, value: T) -> T {
    *x * value
}

#[inline]
fn _scalar_div<T: TensorValue>(x: &mut T, value: T) -> T {
    *x / value
}

#[inline]
fn _scalar_log<T: WeightValue>(x: &mut T, value: T) -> T {
    x.vlog(value)
}

#[inline]
fn _scalar_log1p<T: WeightValue>(x: &mut T, value: T) -> T {
    x.vlog1p(value)
}

#[inline]
fn _scalar_leaky_relu<T: TensorValue>(x: &mut T, slope: T) -> T {
    if *x > T::ZERO {
        *x
    } else {
        *x * slope
    }
}

#[inline]
fn _scalar_elu<T: WeightValue>(x: &mut T, alpha: T) -> T {
    if *x >= T::ZERO {
        *x
    } else {
        alpha * (x.apply_expm1())
    }
}

macro_rules! blas_impl {
    ($t:ty, $gemm_fn:ident) => {
        impl BackendMatMul<$t> for Cpu {
            fn matmul(
                &self,
                (lhs_buf, lhs_meta, lhs_contiguity): (&Self::Buf<$t>, &MetaTensor, ContiguityTypes),
                (rhs_buf, rhs_meta, rhs_contiguity): (&Self::Buf<$t>, &MetaTensor, ContiguityTypes),
                dst: &mut Self::Buf<$t>,
                b: usize,
                m: usize,
                k: usize,
                n: usize
            ) -> Result<(), TensorError> {

                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides()[lhs_meta.rank() - 3] as usize
                } else {
                    assert!(b == 1);
                    m * k
                };
                
                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides()[rhs_meta.rank() - 3] as usize
                } else {
                    assert!(b == 1);
                    k * n
                };

                let (
                    transpose_lhs, 
                    transpose_rhs, 
                    m, 
                    n, 
                    lhs, 
                    rhs, 
                    bstride_lhs, 
                    bstride_rhs,
                    offset_lhs,
                    offset_rhs,
                    lda,
                    ldb
                ) = match (lhs_contiguity, rhs_contiguity) {
                    (ContiguityTypes::ColumnMajor, ContiguityTypes::ColumnMajor) => (
                        CBLAS_TRANSPOSE::CblasTrans, 
                        CBLAS_TRANSPOSE::CblasTrans,
                        n, m,
                        rhs_buf.as_ptr(),
                        lhs_buf.as_ptr(),
                        bstride_rhs,
                        bstride_lhs,
                        rhs_meta.offset,
                        lhs_meta.offset,
                        rhs_meta.strides()[lhs_meta.rank() - 1] as blasint,
                        lhs_meta.strides()[rhs_meta.rank() - 1] as blasint
                    ), // because the output should be row major, and we are doing col major gemm, thus we transpose both
                    (ContiguityTypes::RowMajor, ContiguityTypes::ColumnMajor) => (
                        CBLAS_TRANSPOSE::CblasTrans, // tranpose of row major matrix is og matrix in col major
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        n, m,
                        rhs_buf.as_ptr(),
                        lhs_buf.as_ptr(),
                        bstride_rhs,
                        bstride_lhs,
                        rhs_meta.offset,
                        lhs_meta.offset,
                        rhs_meta.strides()[lhs_meta.rank() - 1] as blasint,
                        lhs_meta.strides()[rhs_meta.rank() - 2] as blasint,
                    ),
                    (ContiguityTypes::ColumnMajor, ContiguityTypes::RowMajor) => (
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasTrans,
                        n, m,
                        rhs_buf.as_ptr(),
                        lhs_buf.as_ptr(),
                        bstride_rhs,
                        bstride_lhs,
                        rhs_meta.offset,
                        lhs_meta.offset,
                        rhs_meta.strides()[rhs_meta.rank() - 2] as blasint,
                        lhs_meta.strides()[lhs_meta.rank() - 1] as blasint
                    ),
                    (ContiguityTypes::RowMajor, ContiguityTypes::RowMajor) => (
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        n, m,
                        rhs_buf.as_ptr(),
                        lhs_buf.as_ptr(),
                        bstride_rhs,
                        bstride_lhs,
                        rhs_meta.offset,
                        lhs_meta.offset,
                        rhs_meta.strides()[rhs_meta.rank() - 2] as blasint,
                        lhs_meta.strides()[lhs_meta.rank() - 2] as blasint
                    ), // tranpose of row major matrix is og matrix in col major,
                    _ => panic!("Invalid contiguity for matmul")
                };

                let ldc = m as blasint;
                for batch in 0..b {
                    // base pointers
                    let lhs_batch = offset_lhs + batch * bstride_lhs;
                    let rhs_batch = offset_rhs + batch * bstride_rhs;

                    let out_batch = batch * m * n; // contiguous 0 offset

                    unsafe {
                        $gemm_fn(
                            CBLAS_ORDER::CblasColMajor,
                            transpose_lhs,
                            transpose_rhs,
                            m as blasint,
                            n as blasint,
                            k as blasint,
                            1.0,
                            lhs.add(lhs_batch) as *const $t,
                            lda,
                            rhs.add(rhs_batch) as *const $t,
                            ldb,
                            0.0,
                            dst.as_mut_ptr().add(out_batch) as *mut $t,
                            ldc,
                        );
                    }
                }

                Ok(())
            }
        }
    };
}

macro_rules! generic_backend_matmul {
    ($t:ty) => {
        impl BackendMatMul<$t> for Cpu {
            fn matmul(
                &self,
                lhs: (&Self::Buf<$t>, &MetaTensor, ContiguityTypes),
                rhs: (&Self::Buf<$t>, &MetaTensor, ContiguityTypes),
                dst: &mut Self::Buf<$t>,
                b: usize,
                m: usize,
                k: usize,
                n: usize,
            ) -> Result<(), TensorError> {
                // let mut out_buf = self.alloc(b * m * n)?;
                let (lhs_buf, lhs_meta, lhs_contiguity): (&Self::Buf<$t>, &MetaTensor, ContiguityTypes) = lhs;
                let (rhs_buf, rhs_meta, rhs_contiguity): (&Self::Buf<$t>, &MetaTensor, ContiguityTypes) = rhs;
                if lhs_contiguity != rhs_contiguity {
                    todo!("Kernel needs a refactor to handle different contiguity types between LHS and RHS in generic dtype case.");
                }
                let lda = match lhs_contiguity {
                    ContiguityTypes::ColumnMajor => lhs_meta.strides[lhs_meta.rank() - 1] as usize,
                    ContiguityTypes::RowMajor => lhs_meta.strides[lhs_meta.rank() - 2] as usize,
                    _ => panic!("Invalid contiguity for generic matmul")
                };
                let ldb = match rhs_contiguity {
                    ContiguityTypes::ColumnMajor => rhs_meta.strides[rhs_meta.rank() - 1] as usize,
                    ContiguityTypes::RowMajor => rhs_meta.strides[rhs_meta.rank() - 2] as usize,
                    _ => panic!("Invalid contiguity for generic matmul")
                };

                let bstride_lhs = if lhs_meta.rank() > 2 {
                    lhs_meta.strides[lhs_meta.rank() - 3] as usize
                } else {
                    0 // only 1 batch, we won't stride
                };

                let bstride_rhs = if rhs_meta.rank() > 2 {
                    rhs_meta.strides[rhs_meta.rank() - 3] as usize
                } else {
                    0 // only 1 batch, we won't stride
                };

                for batch in 0..b {
                    let lhs_batch = lhs_meta.offset + batch * bstride_lhs;
                    let rhs_batch = rhs_meta.offset + batch * bstride_rhs;
                    let out_batch = batch * m * n;
                    // this is repeated code, yes, but we want to reduce indirection in the inner loop
                    // as this is a hot path. furthermore, branching in the inner loop will reduce chances of vectorization
                    if lhs_contiguity == ContiguityTypes::RowMajor {
                        for row in 0..m {
                            for col in 0..n {
                                let mut sum: $t = <$t>::ZERO;
                                for inner in 0..k {
                                    let lhs_idx = lhs_batch + row * lda + inner;
                                    let rhs_idx = rhs_batch + inner * ldb + col;
                                    sum += lhs_buf[lhs_idx] * rhs_buf[rhs_idx];
                                }
                                dst[out_batch + row * n + col] = sum;
                            }
                        }
                    }else {
                        for row in 0..m {
                            for col in 0..n {
                                let mut sum: $t = <$t>::ZERO;
                                for inner in 0..k {
                                    let lhs_idx = lhs_batch + row + inner * lda;
                                    let rhs_idx = rhs_batch + inner + col * ldb;
                                    sum += lhs_buf[lhs_idx] * rhs_buf[rhs_idx];
                                }
                                dst[out_batch + row * n + col] = sum;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
        
    };
}

// instead of specialization
blas_impl!(f32, cblas_sgemm);
blas_impl!(f64, cblas_dgemm);
generic_backend_matmul!(i8);
generic_backend_matmul!(i16);
generic_backend_matmul!(i32);
generic_backend_matmul!(i64);
generic_backend_matmul!(i128);
generic_backend_matmul!(u8);
generic_backend_matmul!(u16);
generic_backend_matmul!(u32);
generic_backend_matmul!(u64);
generic_backend_matmul!(u128);
generic_backend_matmul!(types::boolean);

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{idx::Idx, tensor::TensorAccess, Tensor}, openblas::*, ops::reduction::{NormType, ReductionOp, TotalReductionOp}};
    use std::{error::Error, ffi::CStr};

    #[test]
    fn test_openblas_info() {
        unsafe {
            // Get OpenBLAS information
            let config = openblas_get_config();
            let config_str = CStr::from_ptr(config).to_string_lossy();
            println!("OpenBLAS Config: {}", config_str);
            
            let corename = openblas_get_corename();
            let corename_str = CStr::from_ptr(corename).to_string_lossy();
            println!("OpenBLAS Core: {}", corename_str);
            
            let num_procs = openblas_get_num_procs();
            println!("Number of processors: {}", num_procs);
            assert!(num_procs > 0);
            
            let num_threads = openblas_get_num_threads();
            println!("Number of threads: {}", num_threads);
            assert!(num_threads > 0);
        }
    }

    #[test]
    fn test_openblas_set_threads() {
        unsafe {
            let original_threads = openblas_get_num_threads();
            
            // Set to 4 threads
            openblas_set_num_threads(4);
            assert_eq!(openblas_get_num_threads(), 4);
            
            // Restore original
            openblas_set_num_threads(original_threads);
            assert_eq!(openblas_get_num_threads(), original_threads);
        }
    }

    #[test]
    fn test_cblas_dot_product() {
        unsafe {
            // Test single precision dot product
            let x = vec![1.0f32, 2.0, 3.0, 4.0];
            let y = vec![5.0f32, 6.0, 7.0, 8.0];
            
            let result = cblas_sdot(
                x.len() as blasint,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1
            );
            
            // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            let expected = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>();
            assert_eq!(result, expected);
            assert_eq!(result, 70.0);
        }
    }

    #[test]
    fn test_cblas_dot_product_double() {
        unsafe {
            // Test double precision dot product
            let x = vec![1.0f64, 2.0, 3.0, 4.0];
            let y = vec![5.0f64, 6.0, 7.0, 8.0];
            
            let result = cblas_ddot(
                x.len() as blasint,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1
            );
            
            let expected = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
            assert_eq!(result, expected);
            assert_eq!(result, 70.0);
        }
    }

    #[test]
    fn test_cblas_gemv() {
        unsafe {
            // Matrix-vector multiplication: y = A * x
            // A is 2x3, x is 3x1, result should be 2x1
            #[rustfmt::skip]
            let a = vec![
                1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ];
            let x = vec![1.0f32, 2.0, 3.0];
            let mut y = vec![0.0f32, 0.0];
            
            cblas_sgemv(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m: number of rows in A
                3,  // n: number of columns in A
                1.0,  // alpha
                a.as_ptr(),
                3,  // lda: leading dimension of A
                x.as_ptr(),
                1,  // incx
                0.0,  // beta
                y.as_mut_ptr(),
                1,  // incy
            );
            
            // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
            assert_eq!(y[0], 14.0);
            assert_eq!(y[1], 32.0);
        }
    }

    #[test]
    fn test_cblas_gemm() {
        unsafe {
            // Matrix-matrix multiplication: C = A * B
            // A is 2x3, B is 3x2, C should be 2x2
            #[rustfmt::skip]
            let a = vec![
                1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ];
            #[rustfmt::skip]
            let b = vec![
                7.0f32, 8.0,
                9.0, 10.0,
                11.0, 12.0,
            ];
            let mut c = vec![0.0f32; 4];
            
            cblas_sgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m: rows in A and C
                2,  // n: columns in B and C
                3,  // k: columns in A, rows in B
                1.0,  // alpha
                a.as_ptr(),
                3,  // lda
                b.as_ptr(),
                2,  // ldb
                0.0,  // beta
                c.as_mut_ptr(),
                2,  // ldc
            );
            
            // Expected:
            // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
            // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
            // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
            // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
            assert_eq!(c[0], 58.0);
            assert_eq!(c[1], 64.0);
            assert_eq!(c[2], 139.0);
            assert_eq!(c[3], 154.0);
        }
    }

    #[test]
    fn test_cblas_gemm_double() {
        unsafe {
            // Test double precision matrix multiplication
            #[rustfmt::skip]
            let a = vec![
                1.0f64, 2.0,
                3.0, 4.0,
            ];
            #[rustfmt::skip]
            let b = vec![
                5.0f64, 6.0,
                7.0, 8.0,
            ];
            let mut c = vec![0.0f64; 4];
            
            cblas_dgemm(
                CBLAS_ORDER::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                2,  // m
                2,  // n
                2,  // k
                1.0,  // alpha
                a.as_ptr(),
                2,  // lda
                b.as_ptr(),
                2,  // ldb
                0.0,  // beta
                c.as_mut_ptr(),
                2,  // ldc
            );
            
            // Expected:
            // C[0,0] = 1*5 + 2*7 = 19
            // C[0,1] = 1*6 + 2*8 = 22
            // C[1,0] = 3*5 + 4*7 = 43
            // C[1,1] = 3*6 + 4*8 = 50
            assert_eq!(c[0], 19.0);
            assert_eq!(c[1], 22.0);
            assert_eq!(c[2], 43.0);
            assert_eq!(c[3], 50.0);
        }
    }



    #[test]
    pub fn test_reduce_total_sum_case1() {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.sum().unwrap().item().unwrap(), 1.0999999999999999);
    }

    #[test]
    pub fn test_reduce_total_max_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.max_at(&Idx::Item).unwrap().item().unwrap(), 0.3);
    }

     #[test]
    pub fn test_reduce_total_min_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, -0.9, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();
        assert_eq!(cuda.min_at(&Idx::Item).unwrap().item().unwrap(), -0.9);
    }


    #[test]
    pub fn test_reduce_total_prod_case1() {
         let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![1., 2., 3., 4., 5., 6., 7., 8.], (4, 2))
                .unwrap();
        assert_eq!(cuda.prod_at(&Idx::Item).unwrap().item().unwrap(), 40320.);
    }
    
    #[test]
    pub fn test_reduce_sum_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![
                1., 0., 
                1., 0., 
                1., 1., 
                1., 0.
            ], (4, 2))
                .unwrap();
        assert_eq!(cuda.sum_at(&Idx::At(0))?, Tensor::from_buf(vec![4., 1.], (1, 2))?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_max_case1() -> Result<(), Box<dyn Error>> {
        let cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![
                3., 5., 
                6., 8., 
                1., 2., 
                -1., 4.
            ], (4, 2)).unwrap();
        assert_eq!(cuda.max_at(&Idx::At(0))?, Tensor::from_buf(vec![6., 8.], (1, 2))?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_min_case1() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![
                3., 5., 
                6., 8., 
                1., 2., 
                -1., 4.
            ], (4, 2))
                .unwrap();
        assert_eq!(cuda.min_at(&Idx::At(0))?, Tensor::from_buf(vec![-1., 2.], (1, 2))?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_prod_case1() -> Result<(), Box<dyn Error>> {
        let cuda  = Tensor::<f64>::from_buf(vec![
            3., 5., 
            6., 8., 
            1., 2.,
            -1., 4.
        ], (4, 2)).unwrap();
        assert_eq!(cuda.prod_at(&Idx::At(0))?, Tensor::from_buf(vec![-18., 320.], (1, 2))?);
        Ok(())
    }

    #[test]
    pub fn test_reduce_mean_case1() -> Result<(), Box<dyn Error>> {
        let cuda=
            Tensor::<f64>::from_buf(vec![
                1.,  2., 
                3., 4., 
                5., 6., 
                7., 8.
            ], (4, 2))
                .unwrap();
        assert_eq!(cuda.mean_at(&Idx::At(0))?, Tensor::from_buf(vec![4.0, 5.0], (1, 2))?);
        Ok(())
    }


    #[test]
    pub fn test_reduce_mean_case2() -> Result<(), Box<dyn Error>> {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![
                1.,  2., 
                3., 4., 
                5., 6., 
                7., 8.
            ], (4, 2))
                .unwrap();
        assert_eq!(cuda.mean_at(&Idx::At(1))?, Tensor::from_buf(vec![1.5, 3.5, 5.5, 7.5], (4, 1))?);
        Ok(())
    }

 

    // #[test]
    // pub fn test_reduce_variance_case1() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.var(&Idx::At(0))?, Tensor::from_buf(vec![1.6666666666666667f64, 1.6666666666666667], (1, 2))?);
    //     Ok(())
    // // }

    // #[test]
    // pub fn test_reduce_pop_var_case1() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.pop_var(&Idx::At(0))?, Tensor::from_buf(vec![1.25, 1.25], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_stdev_unbiased() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.std(&Idx::At(0), true)?, Tensor::from_buf(vec![1.2909944487358056, 1.2909944487358056], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_stdev_biased() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.std(&Idx::At(0), false)?, Tensor::from_buf(vec![1.118033988749895, 1.118033988749895], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_logsumexp() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.logsumexp(&Idx::At(0))?, Tensor::from_buf(vec![4.440189698561196, 8.440189698561195], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_norm_l1() -> Result<(), Box<dyn Error>> {
    //     let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.norm(&Idx::At(0), NormType::L1)?, Tensor::from_buf(vec![10.0, 26.0], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_norm_l2() -> Result<(), Box<dyn Error>> {
    //     let cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
    //         Tensor::<f64>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.norm(&Idx::At(0), NormType::L2)?, Tensor::from_buf(vec![5.477225575051661, 13.19090595827292], (1, 2))?);
    //     Ok(())
    // }

    // #[test]
    // pub fn test_reduce_norm_l2_f32() -> Result<(), Box<dyn Error>> {
    //     let cuda: crate::core::primitives::TensorBase<f32, crate::backend::cpu::Cpu> =
    //         Tensor::<f32>::from_buf(vec![1.,  2., 3., 4., 5., 6., 7., 8.], (4, 2))
    //             .unwrap();
    //     assert_eq!(cuda.norm(&Idx::At(0), NormType::L2)?, Tensor::from_buf(vec![5.477225575051661, 13.19090595827292], (1, 2))?);
    //     Ok(())
    // }


    #[test]
    pub fn test_reductio_multi() {
        let mut cuda: crate::core::primitives::TensorBase<f64, crate::backend::cpu::Cpu> =
            Tensor::<f64>::from_buf(vec![0.2, 0.3, 0.1, 0.3, 0.3, -0.1, -0.3, 0.3], (4, 2))
                .unwrap();


        println!("Original: {:?}", cuda);

        let result = cuda.sum_at(&Idx::At(1));
        println!("Result: {:?}", result.unwrap());
        
        // let mut out: crate::core::primitives::TensorBase<f64, Cuda> = CudaTensor::from_buf(vec![0.0f64, 0.0f64], (2,))
        //     .unwrap();


        // let in_tensor = (&mut cuda.buf, cuda.meta.clone());
        // let out_tensor = (&mut out.buf, out.meta.clone());

        // _apply_sum_contiguous(&cuda.backend, in_tensor,  out_tensor, 1)
        //     .unwrap();
        

        // println!("Output: {:?}", out.cpu());

        // println!("CUDA: {:?}", cuda.owned().cpu().unwrap());
        // // cuda.tanh_inplace();

        // let start = cuda.offset();
        // let size = cuda.size();

        // let sus = cuda.backend;

        // let mut out = CudaTensor::<f64>::from_buf(vec![0.0], (1,)).unwrap();

        // Cuda::_test_apply_sum_flat_contiguous(&sus, &mut cuda.buf, &mut out.buf, start, size);

        // println!("OUT: {:?}", out.cpu());
    }
}
