
use crate::{backend::Backend, core::{primitives::TensorBase, tensor::{TensorAccess, TensorAccessMut}, value::WeightValue, MetaTensorView}, grad::{GradNode}, ops::{reduction::{self, TotalReductionOp}, unary::UnaryOp}};



pub enum ReductionType {
    Mean,
    Sum,
    None,
}

trait L1<T: WeightValue, B: Backend> {
    fn l1(&self, target: &Self, reduction: ReductionType) -> Self;
}

impl<T: WeightValue, B: Backend> L1<T, B> for TensorBase<T, B> {
    fn l1(&self, target: &Self, reduction: ReductionType) -> Self {
        let diff = (self - target).abs();
        match reduction {
            ReductionType::Mean => {
                diff.mean().expect("Failed to reduce")
            },
            ReductionType::Sum => {
                diff.sum().expect("Failed to reduce")
            },
            ReductionType::None => {
                diff
            },
        }
    }
}

// TODO concider braodcasting impact. 
impl<T: WeightValue, B: Backend> L1<T, B> for GradTensor<T, B> {
    fn l1(&self, target: &Self, reduction: ReductionType) -> Self {
        let self_inner = self.borrow();
        let target_inner = target.borrow();
        
        let self_tensor = &self_inner.tensor;
        let target_tensor = &target_inner.tensor;
        assert!(self_tensor.shape() == target_tensor.shape(), "Shapes must be the same for L1 loss. Got {:?} and {:?}", self_tensor.shape(), target_tensor.shape());
        
        let diff = self_tensor - target_tensor;
        let mut grad_map = diff.sign();

        if let ReductionType::Mean = reduction {
            grad_map /= T::from_usize(self_tensor.size());
        }

        let loss_tensor = self_tensor.l1(target_tensor, reduction);
        GradTensor::from_op_self_referential(loss_tensor, |inner| 
            GradNode::L1 {
                input: self.node,
                target: target.node,
                loss: inner,
                grad_map,
            }
        )
    }
}

pub fn mean_l1_loss<T: WeightValue, B: Backend>(
    input: &GradTensor<T, B>,
    target: &GradTensor<T, B>
) -> GradTensor<T, B> {
    input.l1(target, ReductionType::Mean)
}

pub fn sum_l1_loss<T: WeightValue, B: Backend>(
    input: &GradTensor<T, B>,
    target: &GradTensor<T, B>
) -> GradTensor<T, B> {
    input.l1(target, ReductionType::Sum)
}

pub fn l1_loss<T: WeightValue, B: Backend>(
    input: &GradTensor<T, B>,
    target: &GradTensor<T, B>,
    reduction: ReductionType,
) -> GradTensor<T, B> {
    input.l1(target, reduction)
}