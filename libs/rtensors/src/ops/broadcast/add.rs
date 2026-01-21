use std::ops::{Add, AddAssign};

use crate::{backend::Backend, core::{primitives::TensorBase, value::{TensorValue, WeightValue}, MetaTensor, MetaTensorView, TensorView, TensorViewMut}, grad::{GradNode}, ops::broadcast::compute_broadcasted_params};
use crate::ops::base::BinaryOpType;

/// Macro to implement AddAssign for mutable tensor types (TensorBase and TensorViewMut)
macro_rules! impl_add_assign {
    // For owned TensorBase with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> AddAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn add_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place addition: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Add,
                ).unwrap();
            }
        }
    };
    // For owned TensorBase with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> AddAssign<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn add_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place addition: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Add,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with owned RHS
    (TensorViewMut, $rhs_type:ty, owned) => {
        impl<'a, T, B> AddAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn add_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place addition: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Add,
                ).unwrap();
            }
        }
    };
    // For TensorViewMut with view RHS
    (TensorViewMut, $rhs_type:ty, view) => {
        impl<'a, T, B> AddAssign<$rhs_type> for TensorViewMut<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            fn add_assign(&mut self, rhs: $rhs_type) {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta)
                        .expect("Shapes are not broadcastable");
                
                if self.meta.shape != out_shape {
                    panic!(
                        "Incompatible shapes for in-place addition: {:?} does not broadcast to {:?}",
                        rhs.meta.shape.0, self.meta.shape
                    );
                }
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (self.buf as *mut B::Buf<T>, &meta_a),
                    BinaryOpType::Add,
                ).unwrap();
                
            }
        }
    };
}

/// Macro to implement Add for all tensor type combinations
macro_rules! impl_add {
    // For owned types (TensorBase) consuming self with owned RHS
    (TensorBase, $rhs_type:ty, owned) => {
        impl<T, B> Add<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For owned types (TensorBase) consuming self with view RHS
    (TensorBase, $rhs_type:ty, view) => {
        impl<T, B> Add<$rhs_type> for TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();

                result
            }
        }
    };
    // For view types with owned RHS
    ($lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, T, B> Add<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For view types with view RHS
    ($lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, T, B> Add<$rhs_type> for $lhs_type<'a, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with owned RHS
    (&TensorBase, $rhs_type:ty, owned) => {
        impl<'a, T, B> Add<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For references to owned types with view RHS
    (&TensorBase, $rhs_type:ty, view) => {
        impl<'a, T, B> Add<$rhs_type> for &'a TensorBase<T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (&self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with owned RHS
    (&$lhs_type:ident, $rhs_type:ty, owned) => {
        impl<'a, 'b, T, B> Add<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (&rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
    // For references to view types with view RHS
    (&$lhs_type:ident, $rhs_type:ty, view) => {
        impl<'a, 'b, T, B> Add<$rhs_type> for &'a $lhs_type<'b, T, B>
        where
            T: TensorValue,
            B: Backend,
        {
            type Output = TensorBase<T, B>;

            fn add(self, rhs: $rhs_type) -> Self::Output {
                let (out_shape, broadcast_stra, broadcast_strb) =
                    compute_broadcasted_params(&self.meta, &rhs.meta).unwrap();
                
                let meta_a = MetaTensor::new(out_shape.clone(), broadcast_stra, self.offset());
                let meta_b = MetaTensor::new(out_shape.clone(), broadcast_strb, rhs.offset());
                
                let mut result = TensorBase::<T, B>::zeros(out_shape);

                self.backend.broadcast(
                    (self.buf as *const B::Buf<T>, &meta_a),
                    (rhs.buf as *const B::Buf<T>, &meta_b),
                    (&mut result.buf as *mut B::Buf<T>, &result.meta),
                    BinaryOpType::Add,
                ).unwrap();
                result
            }
        }
    };
}


// TensorBase += TensorBase
impl_add_assign!(TensorBase, TensorBase<T, B>, owned);
// TensorBase += TensorView
impl_add_assign!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase += TensorViewMut
impl_add_assign!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase += &TensorBase
impl_add_assign!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase += &TensorView
impl_add_assign!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase += &TensorViewMut
impl_add_assign!(TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorViewMut += TensorBase
impl_add_assign!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut += TensorView
impl_add_assign!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut += TensorViewMut
impl_add_assign!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut += &TensorBase
impl_add_assign!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut += &TensorView
impl_add_assign!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut += &TensorViewMut
impl_add_assign!(TensorViewMut, &TensorViewMut<'_, T, B>, view);


// TensorBase + TensorBase
impl_add!(TensorBase, TensorBase<T, B>, owned);
// TensorBase + TensorView
impl_add!(TensorBase, TensorView<'_, T, B>, view);
// TensorBase + TensorViewMut
impl_add!(TensorBase, TensorViewMut<'_, T, B>, view);
// TensorBase + &TensorBase
impl_add!(TensorBase, &TensorBase<T, B>, owned);
// TensorBase + &TensorView
impl_add!(TensorBase, &TensorView<'_, T, B>, view);
// TensorBase + &TensorViewMut
impl_add!(TensorBase, &TensorViewMut<'_, T, B>, view);

// &TensorBase + TensorBase
impl_add!(&TensorBase, TensorBase<T, B>, owned);
// &TensorBase + TensorView
impl_add!(&TensorBase, TensorView<'_, T, B>, view);
// &TensorBase + TensorViewMut
impl_add!(&TensorBase, TensorViewMut<'_, T, B>, view);
// &TensorBase + &TensorBase
impl_add!(&TensorBase, &TensorBase<T, B>, owned);
// &TensorBase + &TensorView
impl_add!(&TensorBase, &TensorView<'_, T, B>, view);
// &TensorBase + &TensorViewMut
impl_add!(&TensorBase, &TensorViewMut<'_, T, B>, view);

// TensorView + TensorBase
impl_add!(TensorView, TensorBase<T, B>, owned);
// TensorView + TensorView
impl_add!(TensorView, TensorView<'_, T, B>, view);
// TensorView + TensorViewMut
impl_add!(TensorView, TensorViewMut<'_, T, B>, view);
// TensorView + &TensorBase
impl_add!(TensorView, &TensorBase<T, B>, owned);
// TensorView + &TensorView
impl_add!(TensorView, &TensorView<'_, T, B>, view);
// TensorView + &TensorViewMut
impl_add!(TensorView, &TensorViewMut<'_, T, B>, view);

// &TensorView + TensorBase
impl_add!(&TensorView, TensorBase<T, B>, owned);
// &TensorView + TensorView
impl_add!(&TensorView, TensorView<'_, T, B>, view);
// &TensorView + TensorViewMut
impl_add!(&TensorView, TensorViewMut<'_, T, B>, view);
// &TensorView + &TensorBase
impl_add!(&TensorView, &TensorBase<T, B>, owned);
// &TensorView + &TensorView
impl_add!(&TensorView, &TensorView<'_, T, B>, view);
// &TensorView + &TensorViewMut
impl_add!(&TensorView, &TensorViewMut<'_, T, B>, view);

// TensorViewMut + TensorBase
impl_add!(TensorViewMut, TensorBase<T, B>, owned);
// TensorViewMut + TensorView
impl_add!(TensorViewMut, TensorView<'_, T, B>, view);
// TensorViewMut + TensorViewMut
impl_add!(TensorViewMut, TensorViewMut<'_, T, B>, view);
// TensorViewMut + &TensorBase
impl_add!(TensorViewMut, &TensorBase<T, B>, owned);
// TensorViewMut + &TensorView
impl_add!(TensorViewMut, &TensorView<'_, T, B>, view);
// TensorViewMut + &TensorViewMut
impl_add!(TensorViewMut, &TensorViewMut<'_, T, B>, view);

// &TensorViewMut + TensorBase
impl_add!(&TensorViewMut, TensorBase<T, B>, owned);
// &TensorViewMut + TensorView
impl_add!(&TensorViewMut, TensorView<'_, T, B>, view);
// &TensorViewMut + TensorViewMut
impl_add!(&TensorViewMut, TensorViewMut<'_, T, B>, view);
// &TensorViewMut + &TensorBase
impl_add!(&TensorViewMut, &TensorBase<T, B>, owned);
// &TensorViewMut + &TensorView
impl_add!(&TensorViewMut, &TensorView<'_, T, B>, view);
// &TensorViewMut + &TensorViewMut
impl_add!(&TensorViewMut, &TensorViewMut<'_, T, B>, view);

impl<T, B> std::ops::Add<GradTensor<T, B>> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn add(self, rhs: GradTensor<T, B> ) -> Self::Output {
        let value = &rhs.borrow().tensor + &self.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
        let op = GradNode::BroadcastAdd { 
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

impl<T, B> std::ops::Add<&GradTensor<T, B>> for GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn add(self, rhs: &GradTensor<T, B> ) -> Self::Output {
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();

        let value = &rhs.borrow().tensor + &self.borrow().tensor;
        let op = GradNode::BroadcastAdd { 
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

impl<T, B> std::ops::Add<&GradTensor<T, B>> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn add(self, rhs: &GradTensor<T, B> ) -> Self::Output {
        let value = &rhs.borrow().tensor + &self.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
        let op = GradNode::BroadcastAdd { 
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

impl<T, B> std::ops::Add<GradTensor<T, B>> for &GradTensor<T, B> 
    where T: WeightValue,
          B: Backend,
{
    type Output = GradTensor<T, B>;

    fn add(self, rhs: GradTensor<T, B> ) -> Self::Output {
        let value = &rhs.borrow().tensor + &self.borrow().tensor;
        let (_, broadcast_stra, broadcast_strb) = 
            compute_broadcasted_params(&self.borrow().tensor.meta, &rhs.borrow().tensor.meta).unwrap();
        let op = GradNode::BroadcastAdd { 
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