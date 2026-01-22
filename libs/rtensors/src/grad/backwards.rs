use crate::{backend::{Backend, BackendMatMul}, core::{Shape, Strides, idx::Idx, primitives::TensorBase, tensor::{TensorAccess, TensorError}, untyped::AsUntypedTensor, value::WeightValue}, grad::GradNode, ops::{reduction::ReductionOp, unary::UnaryOp}};
use crate::ops::linalg::MatMul;

const MIXED_TYPE_ERROR: &str = "Mixed datatyped training is not supported.";

#[inline]
pub(crate) fn accumulate_grad<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Leaf( grad_ref ) = node else {
        return Err(TensorError::UnsupportedOperation(MIXED_TYPE_ERROR.into()));
    };

    let mut grad_tensor = grad_ref.write();
    if let Some(existing_grad) = grad_tensor.as_mut() {
        // Accumulate gradient
        *existing_grad.typed_mut().expect("Mixed datatyped training is not supported.") += upstream;
    } else {
        // First gradient assignment
        *grad_tensor = Some(upstream.clone().as_untyped());
    }

    Ok(vec![upstream.clone()])
}

#[inline]
pub(crate) fn backwards_add<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::BroadcastAdd { lhs_strides, rhs_strides, lhs_shape, rhs_shape, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Add backwards.".into()));
    };

    let mut lhs_grad = upstream.clone();
    let mut rhs_grad = upstream.clone();
    inverse_broadcast_gradient(&mut lhs_grad, lhs_strides, lhs_shape)?;
    inverse_broadcast_gradient(&mut rhs_grad, rhs_strides, rhs_shape)?;

    Ok(vec![lhs_grad, rhs_grad])
}

#[inline]
pub(crate) fn backwards_sub<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::BroadcastSub { lhs_strides, rhs_strides, lhs_shape, rhs_shape, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sub backwards.".into()));
    };

    let mut lhs_grad = upstream.clone();
    let mut rhs_grad = -upstream.clone();
    inverse_broadcast_gradient(&mut lhs_grad, lhs_strides, lhs_shape)?;
    inverse_broadcast_gradient(&mut rhs_grad, rhs_strides, rhs_shape)?;

    Ok(vec![lhs_grad, rhs_grad])
}

#[inline]
pub(crate) fn backwards_mul<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::BroadcastMul { 
        lhs_input,
        rhs_input,
        lhs_strides, 
        rhs_strides, 
        lhs_shape, 
        rhs_shape, .. 
    } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Mul backwards.".into()));
    };

    let mut lhs_grad = upstream * rhs_input;
    let mut rhs_grad = upstream * lhs_input;
    inverse_broadcast_gradient(&mut lhs_grad, lhs_strides, lhs_shape)?;
    inverse_broadcast_gradient(&mut rhs_grad, rhs_strides, rhs_shape)?;

    Ok(vec![lhs_grad, rhs_grad])
}

#[inline]
pub(crate) fn backwards_div<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::BroadcastDiv { 
        lhs_input,
        rhs_input_reciprocal,
        lhs_strides, 
        rhs_strides, 
        lhs_shape, 
        rhs_shape, .. 
    } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Div backwards.".into()));
    };

    let mut lhs_grad = upstream * rhs_input_reciprocal;
    let mut rhs_grad = upstream * -(lhs_input.typed().expect(MIXED_TYPE_ERROR) * rhs_input_reciprocal.typed().expect(MIXED_TYPE_ERROR).square());
    inverse_broadcast_gradient(&mut lhs_grad, lhs_strides, lhs_shape)?;
    inverse_broadcast_gradient(&mut rhs_grad, rhs_strides, rhs_shape)?;

    Ok(vec![lhs_grad, rhs_grad])
}

#[inline]
pub(crate) fn backwards_add_scalar<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::AddScalar { input: _ } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Add backwards.".into()));
    };

    Ok(vec![upstream.clone()])
}

#[inline]
pub(crate) fn backwards_mul_scalar<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::MulScalar { input: _ , scalar: s} = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Mul backwards.".into()));
    };

    Ok(vec![upstream * s])
}

#[inline]
pub(crate) fn backwards_div_scalar<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::DivScalar { input: _ , scalar: s} = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Div backwards.".into()));
    };

    Ok(vec![upstream / s])
}

#[inline]
pub(crate) fn backwards_l1<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::L1 { grad_map, ..} = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to L1 backwards.".into()));
    };
    let t = grad_map.typed().expect(MIXED_TYPE_ERROR);
    let result = vec![
        t * upstream,
        -t * upstream,
    ];
    Ok(result)
}

/// Backward function for the Permute operation
/// Reverses the permutation applied in the forward pass
#[inline]
pub fn backwards_permute<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Permute { dims, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Permute backwards.".into()));
    };
    let dims_vec = match dims {
        Idx::Coord(dims) => dims.clone(),
        Idx::At(i) => vec![*i],
        Idx::Item => vec![]
    };
    let mut inverse_dims = vec![0; dims_vec.len()];
    for (i, &d) in dims_vec.iter().enumerate() {
        inverse_dims[d] = i;
    }
    let permuted_grad = upstream.permute(inverse_dims)?;
    let mut grad = upstream.clone();
    grad.meta = permuted_grad.meta;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_relu<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::ReLU { grad_map, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to ReLU backwards.".into()));
    };
    let t: &TensorBase<T, B> = grad_map.typed().expect(MIXED_TYPE_ERROR);
    let grad = t * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_negate<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Negate { .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Negate backwards.".into()));
    };
    let grad = -upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_sqrt<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sqrt { output, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sqrt backwards.".into()));
    };
    let two = T::from_f32(2.0);
    let grad = upstream / (output * two);
    Ok(vec![grad])
}

#[inline]
pub fn backwards_abs<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Abs { grad_map, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Abs backwards.".into()));
    };
    let grad = grad_map * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_sigmoid<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sigmoid { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sigmoid backwards.".into()));
    };
    let one = T::from_f32(1.0);
    // TODO allow scalars on left hand side so it is fewer kernel launches
    let rt: &TensorBase<T, B> = result.typed().expect(MIXED_TYPE_ERROR);
    let grad = rt * (-rt + one) * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_ln<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Ln { x_reciprocal, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Ln backwards.".into()));
    };
    let grad = x_reciprocal * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_sin<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sin { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sin backwards.".into()));
    };
    // d/dx(sin(x)) = cos(x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = it.cos() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_cos<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Cos { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Cos backwards.".into()));
    };
    // d/dx(cos(x)) = -sin(x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = -it.sin() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_tan<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Tan { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Tan backwards.".into()));
    };
    // d/dx(tan(x)) = sec^2(x) = 1/cos^2(x)
    let it : &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let cos_x = it.cos();
    let sec_squared = cos_x.square().reciprocal();
    let grad = sec_squared * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_tanh<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Tanh { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Tanh backwards.".into()));
    };
    // d/dx(tanh(x)) = 1 - tanh²(x) = sech²(x)
    let one = T::from_f32(1.0);
    let rt: &TensorBase<T, B> = result.typed().expect(MIXED_TYPE_ERROR);
    let grad = (-rt.square() + one) * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_exp<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Exp { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Exp backwards.".into()));
    };
    // d/dx(exp(x)) = exp(x)
    let grad = result * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_square<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Square { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Square backwards.".into()));
    };
    // d/dx(x²) = 2x
    let two = T::from_f32(2.0);
    let grad = input_tensor * two * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_cube<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Cube { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Cube backwards.".into()));
    };
    // d/dx(x³) = 3x²
    let three = T::from_f32(3.0);
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = it.square() * three * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_reciprocal<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Reciprocal { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Reciprocal backwards.".into()));
    };
    // d/dx(1/x) = -1/x² = -(1/x)²
    let rt: &TensorBase<T, B> = result.typed().expect(MIXED_TYPE_ERROR);
    let grad = -rt.square() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_rsqrt<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Rsqrt { result, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Rsqrt backwards.".into()));
    };
    // d/dx(1/√x) = -1/(2x^(3/2)) = -rsqrt(x)³/2
    let neg_half = T::from_f32(-0.5);
    let rt: &TensorBase<T, B> = result.typed().expect(MIXED_TYPE_ERROR);
    let grad = rt.cube() * neg_half * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_sinh<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Sinh { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Sinh backwards.".into()));
    };
    // d/dx(sinh(x)) = cosh(x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = it.cosh() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_cosh<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Cosh { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Cosh backwards.".into()));
    };
    // d/dx(cosh(x)) = sinh(x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = it.sinh() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_expm1<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::ExpM1 { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to ExpM1 backwards.".into()));
    };
    // d/dx(exp(x) - 1) = exp(x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let grad = it.exp() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_ln1p<T: WeightValue, B: Backend>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::Ln1p { input_tensor, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to Ln1p backwards.".into()));
    };
    // d/dx(ln(1 + x)) = 1/(1 + x)
    let it: &TensorBase<T, B> = input_tensor.typed().expect(MIXED_TYPE_ERROR);
    let p1: TensorBase<T, B> = it + T::ONE;
    let grad = p1.reciprocal() * upstream;
    Ok(vec![grad])
}

#[inline]
pub fn backwards_matmul<T: WeightValue, B: Backend + BackendMatMul<T>>(
    node: &GradNode, 
    upstream: &TensorBase<T, B>,
) -> Result<Vec<TensorBase<T, B>>, TensorError>{
    let GradNode::MatMul { left_input, right_input, .. } = node else {
        return Err(TensorError::UnsupportedOperation("Invalid node type passed to MatMul backwards.".into()));
    };
    // only tranpose the last two dims, leave batch dims intact
    let right_input: &TensorBase<T, B> = right_input.typed().expect(MIXED_TYPE_ERROR);
    let left_input: &TensorBase<T, B> = left_input.typed().expect(MIXED_TYPE_ERROR);
    
    let permute_dims_rhs: Idx = Idx::Coord(vec![right_input.meta.shape.len() - 1, right_input.meta.shape.len() - 2]);
    let permute_dims_lhs: Idx = Idx::Coord(vec![left_input.meta.shape.len() - 1, left_input.meta.shape.len() - 2]);
    let grad_lhs = upstream.matmul(&right_input.permute(permute_dims_rhs)?)?;
    let grad_rhs = left_input.permute(permute_dims_lhs)?.matmul(&upstream)?;
    // BE CAREFUL if broadcasting is added to matmul, then we need to reduce gradients here
    Ok(vec![grad_lhs, grad_rhs])
}

#[inline]
/// collapse upstream according to broadcast strides
/// the strides are zero along a dimension that was broadcasted
/// we will do .sum(dim) along those dimensions. then, we need to squeeze those dimensions out
fn inverse_broadcast_gradient<T: WeightValue, B: Backend>(
    grad: &mut TensorBase<T, B>,
    strides: &Strides,
    shape: &Shape
) -> Result<(), TensorError> {
    // we need to use the shape differential
    let shape_differential = grad.meta.shape.len() - shape.len();
    
    // reductions
    for (dim, &stride) in strides.iter().enumerate() {
        if stride == 0 {
            grad.meta = grad.sum_at(dim)?.meta;
        }
    }
    // squeezing, mutate only the meta to avoid reallocations
    // reverse order to not mess up the dimensions
    for (dim, &stride) in strides.iter().enumerate().rev() {
        if stride == 0 && (dim < shape_differential || shape[dim - shape_differential] != 1) {
            grad.meta = grad.squeeze_at(dim)?.meta;
        }
    }
    Ok(())
}