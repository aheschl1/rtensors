
use std::fmt::Debug;

use crate::{core::{meta::ContiguityTypes, primops::{Exp, InvExp, SquareRoot}, tensor::TensorError, value::{TensorValue, WeightValue}, Dim, MetaTensor, MetaTensorView}, ops::{base::BinaryOpType, linalg::ConvConfig2D, reduction::ReductionOpTypes}};

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_tests;
#[cfg(feature = "remote")]
pub mod remote;

macro_rules! elementwise_binary_dispatch {
    (
        $self:expr,
        $buf:expr,
        $op:expr,
        $meta:expr,
        contiguous = $contig:ident,
        strided1d  = $strided1d:ident,
        nd         = $nd:ident
    ) => {{
        if $meta.is_contiguous() {
            return $self.$contig(
                $buf,
                $op,
                $meta.offset,
                $meta.size(),
            );
        }

        let non_singleton_dims = $meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return $self.$strided1d(
                $buf,
                $op,
                $meta.offset,
                dim_stride,
                dim_size,
            );
        }

        $self.$nd(
            $buf,
            $op,
            $meta.offset,
            $meta.shape.as_slice(),
            $meta.strides.as_ref(),
        )
    }};
}

macro_rules! elementwise_unary_dispatch {
    (
        $self:expr,
        $buf:expr,
        $meta:expr,
        contiguous = $contig:ident,
        strided1d  = $strided1d:ident,
        nd         = $nd:ident
    ) => {{
        if $meta.is_contiguous() {
            return $self.$contig(
                $buf,
                $meta.offset,
                $meta.size(),
            );
        }

        let non_singleton_dims = $meta.non_singleton_dims();
        if non_singleton_dims.len() == 1 {
            let (_, dim_size, dim_stride) = non_singleton_dims[0];
            return $self.$strided1d(
                $buf,
                $meta.offset,
                dim_stride,
                dim_size,
            );
        }

        $self.$nd(
            $buf,
            $meta.offset,
            $meta.shape.as_slice(),
            $meta.strides.as_ref(),
        )
    }};
}

use paste::paste;


macro_rules! specify_trait_unary_cabal {
    ($name:ident $( where $($extra:tt)+ )?) => {
        paste! {
            fn [<apply_ $name>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, meta: &MetaTensor) -> Result<(), TensorError>
            $( where $($extra)+ )?
            {
                elementwise_unary_dispatch!(
                    self,
                    buf,
                    meta,
                    contiguous = [<apply_ $name _contiguous>],
                    strided1d  = [<apply_ $name _1d_strided>],
                    nd         = [<apply_ $name _nd>]
                )
            }
        
        

            fn [<apply_ $name _nd>]<T: TensorValue>(
                &self,
                buf: &mut Self::Buf<T>,
                offset: usize,
                shape: &[usize],
                stride: &[isize],
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            ;

            fn [<apply_ $name _1d_strided>]<T: TensorValue>(
            &self, buf: &mut Self::Buf<T>, 
                offset: usize,
                stride: isize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            ;

            fn [<apply_ $name _contiguous>]<T: TensorValue>(
            &self, buf: &mut Self::Buf<T>, 
                start: usize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?;
        }
    };
}

macro_rules! specify_trait_scalar_cabal {
    ($name:ident $( where $($extra:tt)+ )?) => {
        paste! {
            fn [<scalar_apply_ $name>]<T: TensorValue>(&self, buf: &mut Self::Buf<T>, value: T, meta: &MetaTensor) -> Result<(), TensorError>
            $( where $($extra)+ )?
            {
                if meta.is_contiguous() {
                    return self.[<scalar_apply_ $name _contiguous>](
                        buf,
                        value,
                        meta.offset,
                        meta.size(),
                    );
                }

                let non_singleton_dims = meta.non_singleton_dims();
                if non_singleton_dims.len() == 1 {
                    let (_, dim_size, dim_stride) = non_singleton_dims[0];
                    return self.[<scalar_apply_ $name _1d_strided>](
                        buf,
                        value,
                        meta.offset,
                        dim_stride,
                        dim_size,
                    );
                }

                self.[<scalar_apply_ $name _nd>](
                    buf,
                    value,
                    meta.offset,
                    meta.shape.as_slice(),
                    meta.strides.as_ref(),
                )
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
            ;

            fn [<scalar_apply_ $name _1d_strided>]<T: TensorValue>(
                &self, 
                buf: &mut Self::Buf<T>, 
                value: T,
                offset: usize,
                stride: isize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?
            ;

            fn [<scalar_apply_ $name _contiguous>]<T: TensorValue>(
                &self, 
                buf: &mut Self::Buf<T>, 
                value: T,
                start: usize,
                len: usize
            ) -> Result<(), TensorError>
            $( where $($extra)+ )?;
        }
    };
}


pub trait Backend: Send + Sync + 'static + Clone + Debug {
    type Buf<T: TensorValue>: Send + Sync + Debug + PartialEq;

    fn device_type() -> crate::core::primitives::DeviceType;
    fn alloc_from_slice<T: TensorValue>(&self, src: Box<[T]>) -> Result<Self::Buf<T>, TensorError>;
    fn alloc<T: TensorValue>(&self, len: usize) -> Result<Self::Buf<T>, TensorError>;
    fn copy_from_slice<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &[T]) -> Result<(), TensorError>;
    fn copy_range_within<T: TensorValue>(&self, dst: &mut Self::Buf<T>, src: &Self::Buf<T>, dst_offset: usize, src_offset: usize, len: usize) -> Result<(), TensorError>;
    fn read<T: TensorValue>(&self, buf: &Self::Buf<T>, offset: usize) -> Result<T, TensorError>;
    fn write<T: TensorValue>(&self, buf: &mut Self::Buf<T>, offset: usize, value: T) -> Result<(), TensorError>;
    fn len<T: TensorValue>(&self, buf: &Self::Buf<T>) -> usize;
    fn copy<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Self::Buf<T>, TensorError>;
    fn dump<T: TensorValue>(&self, src: &Self::Buf<T>) -> Result<Box<[T]>, TensorError>;
    fn convert<T: TensorValue, N: TensorValue>(&self, src: &Self::Buf<T>, dst: &mut Self::Buf<N>) -> Result<(), TensorError>;
    fn new() -> Self;
    /// Broadcast two tensors into a destination tensor according to broadcasting rules
    /// 
    /// # Safety
    /// The caller must ensure that the pointers and metatensors are valid and that the destination
    /// tensor has the correct shape to hold the broadcasted result of the two source tensors.
    /// Dst and left may be the same buffer for in-place operations. It is vital that the caller ensures
    /// the stride for the left buffer contains no zeros in this case.
    fn broadcast<T: TensorValue>(
        &self, 
        left: (*const Self::Buf<T>, &MetaTensor), 
        right: (*const Self::Buf<T>, &MetaTensor),
        dst: (*mut Self::Buf<T>, &MetaTensor),
        op: BinaryOpType
    ) -> Result<(), TensorError>;
    
    specify_trait_unary_cabal!{neg where T: std::ops::Neg<Output = T>}
    specify_trait_unary_cabal!{relu}
    specify_trait_unary_cabal!{sigmoid where T: InvExp}
    specify_trait_unary_cabal!{silu where T: InvExp}
    specify_trait_unary_cabal!{tanh where T: Exp + InvExp}
    specify_trait_unary_cabal!{abs}
    specify_trait_unary_cabal!{sqrt where T: SquareRoot}
    specify_trait_unary_cabal!{ln where T: WeightValue}
    specify_trait_unary_cabal!{expm1 where T: Exp}
    specify_trait_unary_cabal!{ln1p where T: WeightValue}
    specify_trait_unary_cabal!{floor where T: WeightValue}
    specify_trait_unary_cabal!{ceil where T: WeightValue}
    specify_trait_unary_cabal!{round where T: WeightValue}
    specify_trait_unary_cabal!{trunc where T: WeightValue}

    // Scalar binary operations
    specify_trait_scalar_cabal!{add}
    specify_trait_scalar_cabal!{sub}
    specify_trait_scalar_cabal!{mul}
    specify_trait_scalar_cabal!{div}
    specify_trait_scalar_cabal!{log where T: WeightValue}
    specify_trait_scalar_cabal!{log1p where T: WeightValue}
    specify_trait_scalar_cabal!{leaky_relu}
    specify_trait_scalar_cabal!{elu where T: WeightValue}

    specify_trait_unary_cabal!{sin where T: WeightValue}
    specify_trait_unary_cabal!{cos where T: WeightValue}
    specify_trait_unary_cabal!{tan where T: WeightValue}
    specify_trait_unary_cabal!{asin where T: WeightValue}
    specify_trait_unary_cabal!{acos where T: WeightValue}
    specify_trait_unary_cabal!{atan where T: WeightValue}
    
    specify_trait_unary_cabal!{sinh where T: WeightValue}
    specify_trait_unary_cabal!{cosh where T: WeightValue}
    specify_trait_unary_cabal!{asinh where T: WeightValue}
    specify_trait_unary_cabal!{acosh where T: WeightValue}
    specify_trait_unary_cabal!{atanh where T: WeightValue}

    specify_trait_unary_cabal!{ rsqrt where T: WeightValue }

    specify_trait_unary_cabal!{ reciprocal where T: WeightValue }
    specify_trait_unary_cabal!{ square where T: WeightValue }
    specify_trait_unary_cabal!{ cube where T: WeightValue }
    specify_trait_unary_cabal!{ exp where T: WeightValue }
    specify_trait_unary_cabal!{ sign where T: WeightValue }

    fn apply_reduce_contiguous_flat<T: WeightValue>(
        &self, 
        src: &Self::Buf<T>, 
        dst: &mut Self::Buf<T>, 
        start: usize, 
        len: usize, 
        op: ReductionOpTypes
    ) -> Result<(), TensorError>;

    fn apply_reduce_contiguous_nd<T: WeightValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<T>, &MetaTensor), 
        dim: Dim,
        op: ReductionOpTypes
    ) -> Result<(), TensorError>;

    /// currently assuming that the tensor is contiguous
    fn apply_reduce<T: WeightValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<T>, &MetaTensor), 
        dim: Dim,
        op: ReductionOpTypes
    ) -> Result<(), TensorError>{

        let (src_buf, src_meta) = src;
        let (dst_buf, dst_meta) = dst;

        if !src_meta.is_contiguous(){
            return Err(TensorError::WrongDims(
                "Reduction over non-contiguous tensors is not implemented yet.".to_string(),
            ));
        }

        if src_meta.rank() == 1 {
            self.apply_reduce_contiguous_flat(
                src_buf,
                dst_buf,
                src_meta.offset,
                src_meta.size(),
                op
            )
        }else{
            self.apply_reduce_contiguous_nd(
                (src_buf, src_meta),
                (dst_buf, dst_meta),
                dim,
                op
            )
        }

    }


    fn apply_argmax_contiguous_flat<T: WeightValue>(
        &self, 
        src: &Self::Buf<T>, 
        dst: &mut Self::Buf<u64>, 
        start: usize, 
        len: usize, 
        op: ReductionOpTypes
    ) -> Result<(), TensorError>;

    fn apply_argmax_contiguous_nd<T: WeightValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<u64>, &MetaTensor), 
        dim: Dim,
        op: ReductionOpTypes
    ) -> Result<(), TensorError>;

    /// currently assuming that the tensor is contiguous
    fn apply_argmax<T: WeightValue>(
        &self, 
        src: (&Self::Buf<T>, &MetaTensor), 
        dst: (&mut Self::Buf<u64>, &MetaTensor), 
        dim: Dim,
        op: ReductionOpTypes
    ) -> Result<(), TensorError>{

        let (src_buf, src_meta) = src;
        let (dst_buf, dst_meta) = dst;

        if !src_meta.is_contiguous(){
            return Err(TensorError::WrongDims(
                "Reduction over non-contiguous tensors is not implemented yet.".to_string(),
            ));
        }

        if src_meta.rank() == 1 {
            self.apply_argmax_contiguous_flat(
                src_buf,
                dst_buf,
                src_meta.offset,
                src_meta.size(),
                op
            )
        }else{
            self.apply_argmax_contiguous_nd(
                (src_buf, src_meta),
                (dst_buf, dst_meta),
                dim,
                op
            )
        }
    }

    fn apply_conv_2d<T: WeightValue>(
        &self, 
        input: (&Self::Buf<T>, &MetaTensor), 
        kernel: (&Self::Buf<T>, &MetaTensor),
        output: &mut Self::Buf<T>,
        config: &ConvConfig2D
    ) -> Result<(), TensorError>;
}

pub trait BackendMatMul<T: TensorValue>: Backend {
    fn matmul(
        &self,
        lhs: (&Self::Buf<T>, &MetaTensor, ContiguityTypes),
        rhs: (&Self::Buf<T>, &MetaTensor, ContiguityTypes),
        dst: &mut Self::Buf<T>,
        b: usize,
        m: usize,
        k: usize,
        n: usize
    ) -> Result<(), TensorError>;
}
