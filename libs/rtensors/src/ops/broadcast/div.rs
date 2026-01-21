use std::ops::{Div, DivAssign};


use crate::{backend::Backend, core::{primitives::TensorBase, value::{TensorValue, WeightValue}, MetaTensor, MetaTensorView, TensorView, TensorViewMut}, grad::{GradNode}, ops::{broadcast::compute_broadcasted_params, unary::UnaryOp}};
use crate::ops::base::BinaryOpType;

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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                let mut result = TensorBase::<T, B>::zeros(out_shape.as_ref());
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


impl<T, B> std::ops::Div<GradTensor<T, B>> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn div(self, rhs: GradTensor<T, B> ) -> Self::Output {
        let value = &self.borrow().tensor / &rhs.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();

        let rhs_input_reciprocal = rhs.borrow().tensor.reciprocal();
        
        let op = GradNode::BroadcastDiv {
            left: self.node, 
            right: rhs.node,
            lhs_input: self.borrow().tensor.clone(),
            rhs_input_reciprocal,
            lhs_strides: broadcast_stra, 
            rhs_strides: broadcast_strb,
            lhs_shape: self.borrow().tensor.meta.shape.clone(),
            rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
        };
        GradTensor::from_op(value, op)
    }
}

impl<T, B> std::ops::Div<&GradTensor<T, B>> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn div(self, rhs: &GradTensor<T, B> ) -> Self::Output {
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();

        let value = &self.borrow().tensor / &rhs.borrow().tensor;
        let rhs_input_reciprocal = rhs.borrow().tensor.reciprocal();

        let op = GradNode::BroadcastDiv { 
            lhs_input: self.borrow().tensor.clone(),
            rhs_input_reciprocal,
            left: self.node, 
            right: rhs.node, 
            lhs_strides: broadcast_stra, 
            rhs_strides: broadcast_strb,
            lhs_shape: self.borrow().tensor.meta.shape.clone(),
            rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
        };
        GradTensor::from_op(value, op)
    }
}

impl<T, B> std::ops::Div<&GradTensor<T, B>> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn div(self, rhs: &GradTensor<T, B> ) -> Self::Output {
        let value = &self.borrow().tensor / &rhs.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
        
        let rhs_input_reciprocal = rhs.borrow().tensor.reciprocal();

        let op = GradNode::BroadcastDiv { 
            left: self.node, 
            right: rhs.node, 
            lhs_input: self.borrow().tensor.clone(),
            rhs_input_reciprocal,
            lhs_strides: broadcast_stra, 
            rhs_strides: broadcast_strb,
            lhs_shape: self.borrow().tensor.meta.shape.clone(),
            rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
        };
        GradTensor::from_op(value, op)
    }
}

impl<T, B> std::ops::Div<GradTensor<T, B>> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn div(self, rhs: GradTensor<T, B> ) -> Self::Output {
        let value = &self.borrow().tensor / &rhs.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
        
        let rhs_input_reciprocal = rhs.borrow().tensor.reciprocal();

        let op = GradNode::BroadcastDiv { 
            left: self.node, 
            right: rhs.node, 
            lhs_input: self.borrow().tensor.clone(),
            rhs_input_reciprocal,
            lhs_strides: broadcast_stra, 
            rhs_strides: broadcast_strb,
            lhs_shape: self.borrow().tensor.meta.shape.clone(),
            rhs_shape: rhs.borrow().tensor.meta.shape.clone(),
        };
        GradTensor::from_op(value, op)
    }
}