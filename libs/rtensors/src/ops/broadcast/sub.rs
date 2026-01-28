use std::ops::{Sub, SubAssign};

use crate::{backend::Backend, core::{MetaTensor, MetaTensorView, Shape, Strides, TensorView, TensorViewMut, primitives::{OpTensor, TensorBase}, value::{TensorValue, WeightValue}}, grad::{self, GradNode}, ops::broadcast::compute_broadcasted_params};
use crate::ops::base::BinaryOpType;

/// Macro to implement SubAssign for mutable tensor types (TensorBase and TensorViewMut)
macro_rules! impl_sub_assign {
    // For owned TensorBase with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> SubAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn sub_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place subtraction: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Sub,
                ).unwrap();
            }
        }
    };
    // For owned TensorBase with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> SubAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn sub_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place subtraction: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Sub,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with owned RHS
    (TensorViewMut, $rhs_type:ty, owned) => {
        impl<'a, T, B> SubAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn sub_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place subtraction: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Sub,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with view RHS
    (TensorViewMut, $rhs_type:ty, view) => {
        impl<'a, T, B> SubAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn sub_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    self,
                );
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place subtraction: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Sub,
                ).unwrap();
            }
        }
    };
}

/// Macro to implement Sub for all tensor type combinations
macro_rules! impl_sub {
    // For owned types (TensorBase) consuming self with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> Sub<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
 
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    &self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For owned types (TensorBase) consuming self with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> Sub<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
    
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    &self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For view types with owned RHS
    ($lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, T, B> Sub<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    &self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For view types with view RHS
    ($lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, T, B> Sub<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    &self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with owned RHS
    (&TensorBase, $rhs_type:ty, owned) => {
        impl<'a, T, B> Sub<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with view RHS
    (&TensorBase, $rhs_type:ty, view) => {
        impl<'a, T, B> Sub<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with owned RHS
    (&$lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, 'b, T, B> Sub<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with view RHS
    (&$lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, 'b, T, B> Sub<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn sub(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                    
                let mut result = TensorBase::<T, B>::zeros(out_shape.clone());
                    
                attach_broadcast_sub_grad(
                    self,
                    &rhs,
                    &broadcast_stra,
                    &broadcast_strb,
                    &self.meta.shape,
                    &rhs.meta.shape,
                    &result,
                );

                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape, broadcast_strb, rhs.offset());
                
                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Sub,
                ).unwrap();
                result
            }
        }
    };
}


// TensorBase -= TensorBase
impl_sub_assign!(TensorBase, TensorBase<T, B>, owned);
// TensorBase -= TensorView
impl_sub_assign!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase -= TensorViewMut
impl_sub_assign!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase -= &TensorBase
impl_sub_assign!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase -= &TensorView
impl_sub_assign!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase -= &TensorViewMut
impl_sub_assign!(TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorViewMut -= TensorBase
impl_sub_assign!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut -= TensorView
impl_sub_assign!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut -= TensorViewMut
impl_sub_assign!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut -= &TensorBase
impl_sub_assign!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut -= &TensorView
impl_sub_assign!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut -= &TensorViewMut
impl_sub_assign!(TensorViewMut, &TensorViewMut<'_, T, B>, view);


// TensorBase - TensorBase
impl_sub!(TensorBase, TensorBase<T, B>, owned);
// TensorBase - TensorView
impl_sub!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase - TensorViewMut
impl_sub!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase - &TensorBase
impl_sub!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase - &TensorView
impl_sub!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase - &TensorViewMut
impl_sub!(TensorBase, &TensorViewMut<'_, T, B>, view);

// &TensorBase - TensorBase
impl_sub!(&TensorBase, TensorBase<T, B>, owned);
// &TensorBase - TensorView
impl_sub!(&TensorBase, TensorView<'_, T, B>, view);
// &TensorBase - TensorViewMut
impl_sub!(&TensorBase, TensorViewMut<'_, T, B>, view);
// &TensorBase - &TensorBase
impl_sub!(&TensorBase, &TensorBase<T, B>, owned);
// &TensorBase - &TensorView
impl_sub!(&TensorBase, &TensorView<'_, T, B>, view);
// &TensorBase - &TensorViewMut
impl_sub!(&TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorView - TensorBase
impl_sub!(TensorView, TensorBase<T, B>, owned);
// TensorView - TensorView
impl_sub!(TensorView, TensorView<'_, T, B>, view);
// TensorView - TensorViewMut
impl_sub!(TensorView, TensorViewMut<'_, T, B>, view);
// TensorView - &TensorBase
impl_sub!(TensorView, &TensorBase<T, B>, owned);
// TensorView - &TensorView
impl_sub!(TensorView, &TensorView<'_, T, B>, view);
// TensorView - &TensorViewMut
impl_sub!(TensorView, &TensorViewMut<'_, T, B>, view);

// &TensorView - TensorBase
impl_sub!(&TensorView, TensorBase<T, B>, owned);
// &TensorView - TensorView
impl_sub!(&TensorView, TensorView<'_, T, B>, view);
// &TensorView - TensorViewMut
impl_sub!(&TensorView, TensorViewMut<'_, T, B>, view);
// &TensorView - &TensorBase
impl_sub!(&TensorView, &TensorBase<T, B>, owned);
// &TensorView - &TensorView
impl_sub!(&TensorView, &TensorView<'_, T, B>, view);
// &TensorView - &TensorViewMut
impl_sub!(&TensorView, &TensorViewMut<'_, T, B>, view);

// TensorViewMut - TensorBase
impl_sub!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut - TensorView
impl_sub!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut - TensorViewMut
impl_sub!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut - &TensorBase
impl_sub!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut - &TensorView
impl_sub!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut - &TensorViewMut
impl_sub!(TensorViewMut, &TensorViewMut<'_, T, B>, view);

// &TensorViewMut - TensorBase
impl_sub!(&TensorViewMut, TensorBase<T, B>, owned);
// &TensorViewMut - TensorView
impl_sub!(&TensorViewMut, TensorView<'_, T, B>, view);
// &TensorViewMut - TensorViewMut
impl_sub!(&TensorViewMut, TensorViewMut<'_, T, B>, view);
// &TensorViewMut - &TensorBase
impl_sub!(&TensorViewMut, &TensorBase<T, B>, owned);
// &TensorViewMut - &TensorView
impl_sub!(&TensorViewMut, &TensorView<'_, T, B>, view);
// &TensorViewMut - &TensorViewMut
impl_sub!(&TensorViewMut, &TensorViewMut<'_, T, B>, view);

#[inline(always)]
#[grad::if_enabled(ctx)]
fn attach_broadcast_sub_grad(
    left: &impl OpTensor,
    right: &impl OpTensor,
    lhs_strides: &Strides,
    rhs_strides: &Strides,
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    result: &impl OpTensor,
) -> Option<()>
{
    let op = GradNode::BroadcastSub {
        left: left.op().unwrap_or_default(),
        right: right.op().unwrap_or_default(),
        lhs_strides: lhs_strides.clone(),
        rhs_strides: rhs_strides.clone(),
        lhs_shape: lhs_shape.clone(),
        rhs_shape: rhs_shape.clone(),
    };
    ctx.attach(result, op);
}

// impl<T, B> std::ops::Sub<GradTensor<T, B>> for GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn sub(self, rhs: GradTensor<T, B> ) -> Self::Output {
//         let value = &self.borrow().tensor - &rhs.borrow().tensor;
//         let (_, broadcast_stra, broadcast_strb) = 
//             compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
//         let op = GradNode::BroadcastSub {
//             left: self.node, 
//             right: rhs.node, 
//             lhs_strides: broadcast_stra, 
//             rhs_strides: broadcast_strb,
//             lhs_shape: self.borrow().tensor.meta.shape.clone(),
//             rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
//         };
//         GradTensor::from_op(value, op)
//     }
// }

// impl<T, B> std::ops::Sub<&GradTensor<T, B>> for GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn sub(self, rhs: &GradTensor<T, B> ) -> Self::Output {
//         let (_, broadcast_stra, broadcast_strb) = 
//             compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();

//         let value = &self.borrow().tensor - &rhs.borrow().tensor;
//         let op = GradNode::BroadcastSub { 
//             left: self.node, 
//             right: rhs.node, 
//             lhs_strides: broadcast_stra, 
//             rhs_strides: broadcast_strb,
//             lhs_shape: self.borrow().tensor.meta.shape.clone(),
//             rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
//         };
//         GradTensor::from_op(value, op)
//     }
// }

// impl<T, B> std::ops::Sub<&GradTensor<T, B>> for &GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn sub(self, rhs: &GradTensor<T, B> ) -> Self::Output {
//         let value = &self.borrow().tensor - &rhs.borrow().tensor;
//         let (_, broadcast_stra, broadcast_strb) = 
//             compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
//         let op = GradNode::BroadcastSub { 
//             left: self.node, 
//             right: rhs.node, 
//             lhs_strides: broadcast_stra, 
//             rhs_strides: broadcast_strb,
//             lhs_shape: self.borrow().tensor.meta.shape.clone(),
//             rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
//         };
//         GradTensor::from_op(value, op)
//     }
// }

// impl<T, B> std::ops::Sub<GradTensor<T, B>> for &GradTensor<T, B> 
//     where T: WeightValue,
//           B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn sub(self, rhs: GradTensor<T, B> ) -> Self::Output {
//         let value = &self.borrow().tensor - &rhs.borrow().tensor;
//         let (_, broadcast_stra, broadcast_strb) = 
//             compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
//         let op = GradNode::BroadcastSub { 
//             left: self.node, 
//             right: rhs.node, 
//             lhs_strides: broadcast_stra, 
//             rhs_strides: broadcast_strb,
//             lhs_shape: self.borrow().tensor.meta.shape.clone(),
//             rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
//         };
//         GradTensor::from_op(value, op)
//     }
// }