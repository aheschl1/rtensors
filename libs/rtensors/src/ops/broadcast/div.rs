use std::ops::{Div, DivAssign};


use crate::{backend::Backend, core::{MetaTensor, MetaTensorView, Shape, Strides, TensorView, TensorViewMut, primitives::{OpTensor, TensorBase}, tensor::{TensorAccess, TensorAccessMut}, untyped::AsUntypedTensor, value::TensorValue}, grad::{self, GradNode}, ops::broadcast::compute_broadcasted_params};
use crate::ops::base::BinaryOpType;
use crate::core::tensor::AsTensor;

/// Macro to implement DivAssign for mutable tensor types (TensorBase and TensorViewMut)
macro_rules! impl_div_assign {
    // For owned TensorBase with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> DivAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn div_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place division: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Div,
                ).unwrap();
            }
        }
    };
    // For owned TensorBase with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> DivAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn div_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place division: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Div,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with owned RHS
    (TensorViewMut, $rhs_type:ty, owned) => {
        impl<'a, T, B> DivAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn div_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place division: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Div,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with view RHS
    (TensorViewMut, $rhs_type:ty, view) => {
        impl<'a, T, B> DivAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn div_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place division: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Div,
                ).unwrap();
            }
        }
    };
}

/// Macro to implement Div for all tensor type combinations
macro_rules! impl_div {
    // For owned types (TensorBase) consuming self with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> Div<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = Self;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For owned types (TensorBase) consuming self with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> Div<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = Self;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For view types with owned RHS
    ($lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, T, B> Div<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For view types with view RHS
    ($lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, T, B> Div<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with owned RHS
    (&TensorBase, $rhs_type:ty, owned) => {
        impl<'a, T, B> Div<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with view RHS
    (&TensorBase, $rhs_type:ty, view) => {
        impl<'a, T, B> Div<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with owned RHS
    (&$lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, 'b, T, B> Div<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with view RHS
    (&$lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, 'b, T, B> Div<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;
            fn div(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
                attach_broadcast_div_grad(
                    self.contiguous(),
                    rhs.contiguous(),
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let meta_c = MetaTensor::new(result.shape().clone(), result.strides().clone(), result.offset());

                result.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &meta_c),
                    BinaryOpType::Div,
                ).unwrap();
                result
            }
        }
    };
}


// TensorBase /= TensorBase
impl_div_assign!(TensorBase, TensorBase<T, B>, owned);
// TensorBase /= TensorView
impl_div_assign!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase /= TensorViewMut
impl_div_assign!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase /= &TensorBase
impl_div_assign!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase /= &TensorView
impl_div_assign!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase /= &TensorViewMut
impl_div_assign!(TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorViewMut /= TensorBase
impl_div_assign!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut /= TensorView
impl_div_assign!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut /= TensorViewMut
impl_div_assign!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut /= &TensorBase
impl_div_assign!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut /= &TensorView
impl_div_assign!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut /= &TensorViewMut
impl_div_assign!(TensorViewMut, &TensorViewMut<'_, T, B>, view);


// TensorBase / TensorBase
impl_div!(TensorBase, TensorBase<T, B>, owned);
// TensorBase / TensorView
impl_div!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase / TensorViewMut
impl_div!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase / &TensorBase
impl_div!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase / &TensorView
impl_div!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase / &TensorViewMut
impl_div!(TensorBase, &TensorViewMut<'_, T, B>, view);

// &TensorBase / TensorBase
impl_div!(&TensorBase, TensorBase<T, B>, owned);
// &TensorBase / TensorView
impl_div!(&TensorBase, TensorView<'_, T, B>, view);
// &TensorBase / TensorViewMut
impl_div!(&TensorBase, TensorViewMut<'_, T, B>, view);
// &TensorBase / &TensorBase
impl_div!(&TensorBase, &TensorBase<T, B>, owned);
// &TensorBase / &TensorView
impl_div!(&TensorBase, &TensorView<'_, T, B>, view);
// &TensorBase / &TensorViewMut
impl_div!(&TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorView / TensorBase
impl_div!(TensorView, TensorBase<T, B>, owned);
// TensorView / TensorView
impl_div!(TensorView, TensorView<'_, T, B>, view);
// TensorView / TensorViewMut
impl_div!(TensorView, TensorViewMut<'_, T, B>, view);
// TensorView / &TensorBase
impl_div!(TensorView, &TensorBase<T, B>, owned);
// TensorView / &TensorView
impl_div!(TensorView, &TensorView<'_, T, B>, view);
// TensorView / &TensorViewMut
impl_div!(TensorView, &TensorViewMut<'_, T, B>, view);

// &TensorView / TensorBase
impl_div!(&TensorView, TensorBase<T, B>, owned);
// &TensorView / TensorView
impl_div!(&TensorView, TensorView<'_, T, B>, view);
// &TensorView / TensorViewMut
impl_div!(&TensorView, TensorViewMut<'_, T, B>, view);
// &TensorView / &TensorBase
impl_div!(&TensorView, &TensorBase<T, B>, owned);
// &TensorView / &TensorView
impl_div!(&TensorView, &TensorView<'_, T, B>, view);
// &TensorView / &TensorViewMut
impl_div!(&TensorView, &TensorViewMut<'_, T, B>, view);

// TensorViewMut / TensorBase
impl_div!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut / TensorView
impl_div!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut / TensorViewMut
impl_div!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut / &TensorBase
impl_div!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut / &TensorView
impl_div!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut / &TensorViewMut
impl_div!(TensorViewMut, &TensorViewMut<'_, T, B>, view);

// &TensorViewMut / TensorBase
impl_div!(&TensorViewMut, TensorBase<T, B>, owned);
// &TensorViewMut / TensorView
impl_div!(&TensorViewMut, TensorView<'_, T, B>, view);
// &TensorViewMut / TensorViewMut
impl_div!(&TensorViewMut, TensorViewMut<'_, T, B>, view);
// &TensorViewMut / &TensorBase
impl_div!(&TensorViewMut, &TensorBase<T, B>, owned);
// &TensorViewMut / &TensorView
impl_div!(&TensorViewMut, &TensorView<'_, T, B>, view);
// &TensorViewMut / &TensorViewMut
impl_div!(&TensorViewMut, &TensorViewMut<'_, T, B>, view);

#[inline(always)]
#[grad::if_enabled(ctx)]
fn attach_broadcast_div_grad<T: TensorValue, B: Backend>(
    left: TensorBase<T, B>,
    right: TensorBase<T, B>,
    lhs_strides: &Strides,
    rhs_strides: &Strides,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    result: &impl OpTensor,
) -> Option<()>
{
    // TODO make .reciprical valid for all types
    let mut rhs_reciprocal = TensorBase::<T, B>::zeros(rhs_shape);
    for coord in right.iter_coords() {
        let val = right.get(&coord).unwrap();
        rhs_reciprocal.set(&coord, T::ONE / val).unwrap();
    }

    let op = GradNode::BroadcastDiv {
        left: left.op(),
        right: right.op(),
        lhs_input: left.as_untyped(),
        rhs_input_reciprocal: rhs_reciprocal.as_untyped(),
        lhs_strides: lhs_strides.clone(),
        rhs_strides: rhs_strides.clone(),
        lhs_shape: lhs_shape.clone(),
        rhs_shape: rhs_shape.clone(),
    };
    ctx.attach(result, op);
}
