
use crate::{backend::Backend, core::{MetaTensorView, primitives::{OpTensor, TensorBase}, tensor::{AsView, TensorAccess, TensorAccessMut}, untyped::AsUntypedTensor, value::WeightValue}, grad::{self, GradNode}, ops::{reduction::{self, TotalReductionOp}, unary::UnaryOp}};



pub enum ReductionType {
    Mean,
    Sum,
    None,
}

trait L1<T: WeightValue, B: Backend> {
    fn l1(&self, target: &impl AsView<T, B>, reduction: ReductionType) -> TensorBase<T, B>;
}

#[grad::if_enabled(ctx)]
fn attach_l1_grad<T: WeightValue, B: Backend>(
    input: &impl OpTensor,
    target: &impl OpTensor,
    loss: &TensorBase<T, B>,
    diff: &Option<TensorBase<T, B>>,
    reduction: ReductionType, 
) -> Option<()> {
    let diff = diff.clone().expect("Gradient map must be provided when attaching L1 grad.");

    let input_node = input.op();
    let target_node = target.op();
    let grad_map = diff.sign();
    let grad_map = match reduction {
        ReductionType::Mean => {
            grad_map / T::from_usize(diff.size())
        },
        _ => grad_map,
    };
    let op = GradNode::L1 { 
        input: input_node, 
        target: target_node, 
        grad_map: grad_map.as_untyped(),
        loss: loss.clone().as_untyped()
    };
    ctx.attach(loss, op);
}

impl<T: WeightValue, B: Backend, V: AsView<T, B>> L1<T, B> for V {
    fn l1(&self, target: &impl AsView<T, B>, reduction: ReductionType) -> TensorBase<T, B> {
        let lhs = self.view();
        let target = target.view();
        let diff = (&lhs - &target).abs();
        
        let gmap = grad::when_enabled(|_| {
            diff.sign()
        });

        let result = match reduction {
            ReductionType::Mean => {
                diff.mean().expect("Failed to reduce")
            },
            ReductionType::Sum => {
                diff.sum().expect("Failed to reduce")
            },
            ReductionType::None => {
                diff
            },
        };

        attach_l1_grad(&lhs, &target, &result, &gmap, reduction);

        result
    }
}

pub fn mean_l1_loss<T: WeightValue, B: Backend>(
    input: &impl AsView<T, B>,
    target: &impl AsView<T, B>
) -> TensorBase<T, B> {
    input.l1(target, ReductionType::Mean)
}

pub fn sum_l1_loss<T: WeightValue, B: Backend>(
    input: &impl AsView<T, B>,
    target: &impl AsView<T, B>
) -> TensorBase<T, B> {
    input.l1(target, ReductionType::Sum)
}

pub fn l1_loss<T: WeightValue, B: Backend>(
    input: &impl AsView<T, B>,
    target: &impl AsView<T, B>,
    reduction: ReductionType,
) -> TensorBase<T, B> {
    input.l1(target, reduction)
}

// // TODO concider braodcasting impact. 
// impl<T: WeightValue, B: Backend> L1<T, B> for GradTensor<T, B> {
//     fn l1(&self, target: &Self, reduction: ReductionType) -> Self {
//         let self_inner = self.borrow();
//         let target_inner = target.borrow();
        
//         let self_tensor = &self_inner.tensor;
//         let target_tensor = &target_inner.tensor;
//         assert!(self_tensor.shape() == target_tensor.shape(), "Shapes must be the same for L1 loss. Got {:?} and {:?}", self_tensor.shape(), target_tensor.shape());
        
//         let diff = self_tensor - target_tensor;
//         let mut grad_map = diff.sign();

//         if let ReductionType::Mean = reduction {
//             grad_map /= T::from_usize(self_tensor.size());
//         }

//         let loss_tensor = self_tensor.l1(target_tensor, reduction);
//         GradTensor::from_op_self_referential(loss_tensor, |inner| 
//             GradNode::L1 {
//                 input: self.node,
//                 target: target.node,
//                 loss: inner,
//                 grad_map,
//             }
//         )
//     }
// }

// pub fn mean_l1_loss<T: WeightValue, B: Backend>(
//     input: &GradTensor<T, B>,
//     target: &GradTensor<T, B>
// ) -> GradTensor<T, B> {
//     input.l1(target, ReductionType::Mean)
// }

// pub fn sum_l1_loss<T: WeightValue, B: Backend>(
//     input: &GradTensor<T, B>,
//     target: &GradTensor<T, B>
// ) -> GradTensor<T, B> {
//     input.l1(target, ReductionType::Sum)
// }

// pub fn l1_loss<T: WeightValue, B: Backend>(
//     input: &GradTensor<T, B>,
//     target: &GradTensor<T, B>,
//     reduction: ReductionType,
// ) -> GradTensor<T, B> {
//     input.l1(target, reduction)
// }