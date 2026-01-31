
use crate::{
    backend::Backend,
    core::{
        primitives::TensorBase, primops::{Exp, InvExp, SquareRoot}, tensor::{AsTensor, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::{TensorValue, WeightValue}, MetaTensorView, TensorView, TensorViewMut
    }, grad::GradNode,
};
use crate::core::tensor::seal;
use crate::grad;
use crate::core::primitives::OpTensor;
use crate::core::untyped::AsUntypedTensor;

macro_rules! specify_unary_op_template {
    (
        $(
            ($name:ident) $op:ident $(where T: $first:path $(, $extra:path)*)?; |$requires_input:literal, $input:ident, $result:ident, $ctx:ident, $grad_node:ident| $grad_fn:block
        ),+ $(,)?
    ) => {

        paste::paste! {

            pub trait InplaceUnaryOp<T: TensorValue, B: Backend> {
                $(
                    fn [<apply_ $op>](&mut self)
                    where 
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+

            }

            
            pub trait UnaryOp<T: TensorValue, B: Backend> {
                $(
                    fn $op(&self) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+
            }


            $(

                pub trait $name<T: TensorValue, B: Backend> {
                    fn [<$op _inplace>](&mut self)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    ;
                }

                impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> $name<T, B> for V {
                    fn [<$op _inplace>](&mut self)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let $result = self.view_mut();
                        //  ========== GRAPH BUILDING ============
                        let requires_input: bool = $requires_input;
                        let input = grad::when_enabled(|_| {
                            if requires_input { 
                                Some($result.owned()) // TODO extra memory copy here, optimize later
                            } else {
                                None
                            }
                        });
                        // ========== APPLICATION OF OPERATION ============
                        if let Err(e) = $result.backend.[<apply_ $op>]($result.buf, &$result.meta) {
                            panic!("Failed to apply abs: {}", e);
                        }
                        // ========== GRAD NODE CREATION ============
                        grad::when_enabled(|$ctx| {
                            grad::no_grad(|| {
                                // unwrap when_enabled error
                                let $input = input.expect("Input tensor required for gradient computation but not captured.");
                                let $grad_node = $result.op();
                                let _ = (&$grad_node, &$input); // to remove warning if not used
    
                                let node: Result<GradNode, TensorError> = $grad_fn;
                                $ctx.attach(&$result, node.expect("Failed to create gradient node."))
                            });
                        });
                    }
                }

              
            )+



            impl<T: TensorValue, B: Backend, V: AsViewMut<T, B> + seal::Sealed> InplaceUnaryOp<T, B> for V
            {
                $(
                    fn [<apply_ $op>](&mut self)
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        self.[<$op _inplace>]();
                    }
                )+
            }

            
            impl<T: TensorValue, B: Backend, V: AsTensor<T, B> + seal::Sealed> UnaryOp<T, B> for V {
                $(
                    fn $op(&self) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let mut result = self.owned();
                        result.[<apply_ $op>]();
                        result
                    }
                )+
            }

        }
        
    };
}

specify_unary_op_template! {
    (Sin) sin where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Sin {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Cos) cos where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Cos {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Tan) tan where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Tan {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Asin) asin where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Acos) acos where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Atan) atan where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Sinh) sinh where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Sinh {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Cosh) cosh where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Cosh {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Asinh) asinh where T: WeightValue; |false, _input, _result, _ctx, _grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Acosh) acosh where T: WeightValue; |false, _input, _result, _ctx, _grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Atanh) atanh where T: WeightValue; |false, _input, _result, _ctx, _grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for tanh not yet implemented.".into()))
    },
    (Rsqrt) rsqrt where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Ok(GradNode::Rsqrt {
            input: grad_node,
            result: result.owned().as_untyped(),
        })
    },
    (Reciprocal) reciprocal where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Ok(GradNode::Reciprocal {
            input: grad_node,
            result: result.owned().as_untyped(),
        })
    },
    (Square) square where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Square {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Cube) cube where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Cube {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (ExpV) exp where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Exp {
            input: grad_node,
            result: result.owned().as_untyped(),
        })
    },
    (Abs) abs where T: WeightValue; |true, input, result, _ctx, grad_node| {
        // TODO: Make a kernel for this
        let input = input.unwrap();
        let mut grad_map = TensorBase::<T, B>::zeros(input.shape());
        for coord in input.iter_coords() {
            let val = input.get(&coord).unwrap();
            if val > T::ZERO {
                grad_map.set(&coord, T::ONE).unwrap();
            } else {
                grad_map.set(&coord, -T::ONE).unwrap();
            }
        }

        // let grad_map = input.unwrap().sign();

        let node = GradNode::Abs {
            input: grad_node,
            grad_map: grad_map.as_untyped(),
        };
        Ok(node)   
    },
    (Relu) relu; |true, input, result, _ctx, grad_node| {
        let input = input.unwrap();
        let mut grad_map = TensorBase::<T, B>::zeros(input.shape());
        // TODO: Make a kernel for this
        for coord in input.iter_coords() {
            let val = input.get(&coord).unwrap();
            if val > T::ZERO {
                grad_map.set(&coord, T::ONE).unwrap();
            } else {
                grad_map.set(&coord, T::ZERO).unwrap();
            }
        }

        let node = GradNode::ReLU {
            input: grad_node,
            grad_map: grad_map.as_untyped(),
        };
        Ok(node)    
    },
    (Sigmoid) sigmoid where T: InvExp; |false, _input, result, _ctx, grad_node| {
        Ok(GradNode::Sigmoid {
            input: grad_node,
            result: result.owned().as_untyped(),
        })
    },
    (Silu) silu where T: InvExp; |true, input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Gradient for silu not yet implemented.".into()))
    },
    (Tanh) tanh where T: Exp, InvExp; |false, _input, result, _ctx, grad_node| {
        Ok(GradNode::Tanh {
            input: grad_node,
            result: result.owned().as_untyped(),
        })
    },
    (Sqrt) sqrt where T: SquareRoot; |false, _input, result, _ctx, grad_node| {
        Ok(GradNode::Sqrt {
            input: grad_node,
            output: result.owned().as_untyped(),
        })
    },
    (Negate) neg where T: std::ops::Neg<Output = T>; |false, _input, result, ctx, grad_node| {
        Ok(GradNode::Negate {
            input: grad_node,
        })
    },
    (NatLog) ln where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Ln {
            input: grad_node,
            x_reciprocal: input.unwrap().reciprocal().as_untyped(),
        })
    },
    (ExpM1) expm1 where T: Exp; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::ExpM1 {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Ln1p) ln1p where T: WeightValue; |true, input, result, _ctx, grad_node| {
        Ok(GradNode::Ln1p {
            input: grad_node,
            input_tensor: input.unwrap().as_untyped(),
        })
    },
    (Floor) floor where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Floor is not differentiable.".into()))
    },
    (Ceil) ceil where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Ceiling is not differentiable.".into()))
    },
    (Round) round where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Rounding is not differentiable.".into()))
    },
    (Trunc) trunc where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Truncating is not differentiable.".into()))
    },
    (Sign) sign where T: WeightValue; |false, _input, result, _ctx, grad_node| {
        Err(TensorError::UnsupportedOperation("Sign is not differentiable.".into()))
    },
}

impl<T, B> std::ops::Neg for TensorBase<T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorView<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for TensorViewMut<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<T, B> std::ops::Neg for &TensorBase<T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for &TensorView<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

impl<'a, T, B> std::ops::Neg for &TensorViewMut<'a, T, B>
where
    T: TensorValue + std::ops::Neg<Output = T>,
    B: Backend,
{
    type Output = TensorBase<T, B>;

    fn neg(self) -> Self::Output {
        let mut result = self.owned();
        result.neg_inplace();
        result
    }
}

// impl<T, B> std::ops::Neg for GradTensor<T, B>
// where
//     T: WeightValue,
//     B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn neg(self) -> Self::Output {
//         UnaryGradOp::neg(&self)
//     }
// }

// impl<T, B> std::ops::Neg for &GradTensor<T, B>
// where
//     T: WeightValue,
//     B: Backend,
// {
//     type Output = GradTensor<T, B>;

//     fn neg(self) -> Self::Output {
//         UnaryGradOp::neg(self)
//     }
// }


#[cfg(test)]
mod tests {
    use crate::{
        backend::cpu::Cpu,
        ops::unary::*,
        testing::{unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided},
    };

    #[test]
    fn test_unary_negate_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_relu_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [-1.0, 1.0, -1.0],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_relu_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [
                -1.0, 1.0, 0.0, 2.0, 1.0, 2.3, -0.3, 0.4, 0.0, -0.3, 0.4, 0.5, -0.2, 0.1, 0.2, -0.5,
            ],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>(
            [1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [1.0; 16],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>(
            [1.0, 1.0],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [1.0; 16],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>(
            [1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>(
            [1.0; 16],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_ln_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>(
            [1.0, 1.0, 1.0],
            |f| f.ln(),
            |f| f.ln_inplace(),
        );
    }

    #[test]
    fn test_unary_ln_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0; 2], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.5; 16], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.5, 1.5, 1.5],
            |f| f.ln(),
            |f| f.ln_inplace(),
        );
    }

    #[test]
    fn test_unary_ln_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.5; 2], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_expm1_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.5; 16], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_expm1_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.5, 1.5, 1.5],
            |f| f.exp_m1(),
            |f| f.expm1_inplace(),
        );
    }

    #[test]
    fn test_unary_expm1_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.5; 2], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_ln1p_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([0.5; 16], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_ln1p_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [0.5, 0.5, 0.5],
            |f| f.ln_1p(),
            |f| f.ln1p_inplace(),
        );
    }

    #[test]
    fn test_unary_ln1p_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([0.5; 2], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_floor_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.7; 16], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_floor_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.7, 2.3, 3.9],
            |f| f.floor(),
            |f| f.floor_inplace(),
        );
    }

    #[test]
    fn test_unary_floor_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.7, 2.3], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_ceil_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.3; 16], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_ceil_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.3, 2.7, 3.1],
            |f| f.ceil(),
            |f| f.ceil_inplace(),
        );
    }

    #[test]
    fn test_unary_ceil_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.3, 2.7], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_round_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.4; 16], |f| f.round(), |f| f.round_inplace());
    }

    #[test]
    fn test_unary_round_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.4, 2.6, 3.5],
            |f| f.round(),
            |f| f.round_inplace(),
        );
    }

    #[test]
    fn test_unary_round_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.4, 2.6], |f| f.round(), |f| f.round_inplace());
    }

    #[test]
    fn test_unary_trunc_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([-1.4; 16], |f| f.trunc(), |f| f.trunc_inplace());
    }

    #[test]
    fn test_unary_trunc_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [-1.4, 2.6, 3.5],
            |f| f.trunc(),
            |f| f.trunc_inplace(),
        );
    }

    #[test]
    fn test_unary_trunc_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([-1.4, 2.6], |f| f.trunc(), |f| f.trunc_inplace());
    }

    // Trigonometric function tests
    #[test]
    fn test_unary_sin_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::sin, |f| f.sin_inplace());
    }

    #[test]
    fn test_unary_sin_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::sin, |f| f.sin_inplace());
    }

    #[test]
    fn test_unary_sin_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::sin, |f| f.sin_inplace());
    }

    #[test]
    fn test_unary_cos_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::cos, |f| f.cos_inplace());
    }

    #[test]
    fn test_unary_cos_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::cos, |f| f.cos_inplace());
    }

    #[test]
    fn test_unary_cos_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::cos, |f| f.cos_inplace());
    }

    #[test]
    fn test_unary_tan_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::tan, |f| f.tan_inplace());
    }

    #[test]
    fn test_unary_tan_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::tan, |f| f.tan_inplace());
    }

    #[test]
    fn test_unary_tan_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::tan, |f| f.tan_inplace());
    }

    // Inverse trigonometric functions
    #[test]
    fn test_unary_asin_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.5], f64::asin, |f| f.asin_inplace());
    }

    #[test]
    fn test_unary_asin_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.5, 0.5, 0.5], f64::asin, |f| f.asin_inplace());
    }

    #[test]
    fn test_unary_asin_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::asin, |f| f.asin_inplace());
    }

    #[test]
    fn test_unary_acos_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.5], f64::acos, |f| f.acos_inplace());
    }

    #[test]
    fn test_unary_acos_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.5, 0.5, 0.5], f64::acos, |f| f.acos_inplace());
    }

    #[test]
    fn test_unary_acos_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::acos, |f| f.acos_inplace());
    }

    #[test]
    fn test_unary_atan_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::atan, |f| f.atan_inplace());
    }

    #[test]
    fn test_unary_atan_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::atan, |f| f.atan_inplace());
    }

    #[test]
    fn test_unary_atan_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::atan, |f| f.atan_inplace());
    }

    // Hyperbolic functions
    #[test]
    fn test_unary_sinh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::sinh, |f| f.sinh_inplace());
    }

    #[test]
    fn test_unary_sinh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::sinh, |f| f.sinh_inplace());
    }

    #[test]
    fn test_unary_sinh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::sinh, |f| f.sinh_inplace());
    }

    #[test]
    fn test_unary_cosh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::cosh, |f| f.cosh_inplace());
    }

    #[test]
    fn test_unary_cosh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::cosh, |f| f.cosh_inplace());
    }

    #[test]
    fn test_unary_cosh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::cosh, |f| f.cosh_inplace());
    }

    // Inverse hyperbolic functions
    #[test]
    fn test_unary_asinh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::asinh, |f| f.asinh_inplace());
    }

    #[test]
    fn test_unary_asinh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::asinh, |f| f.asinh_inplace());
    }

    #[test]
    fn test_unary_asinh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::asinh, |f| f.asinh_inplace());
    }

    #[test]
    fn test_unary_acosh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([1.0, 1.0], f64::acosh, |f| f.acosh_inplace());
    }

    #[test]
    fn test_unary_acosh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([1.0, 1.0, 1.0], f64::acosh, |f| f.acosh_inplace());
    }

    #[test]
    fn test_unary_acosh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([1.0; 16], f64::acosh, |f| f.acosh_inplace());
    }

    #[test]
    fn test_unary_atanh_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], f64::atanh, |f| f.atanh_inplace());
    }

    #[test]
    fn test_unary_atanh_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], f64::atanh, |f| f.atanh_inplace());
    }

    #[test]
    fn test_unary_atanh_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::atanh, |f| f.atanh_inplace());
    }

    // Other unary operations
    #[test]
    fn test_unary_rsqrt_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], |f| (1. / f.sqrt()), |f| f.rsqrt_inplace());
    }

    #[test]
    fn test_unary_rsqrt_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], |f| (1. / f.sqrt()), |f| f.rsqrt_inplace());
    }

    #[test]
    fn test_unary_rsqrt_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], |f| (1. / f.sqrt()), |f| f.rsqrt_inplace());
    }

    #[test]
    fn test_unary_reciprocal_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], |f| 1. / f, |f| f.reciprocal_inplace());
    }

    #[test]
    fn test_unary_reciprocal_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], |f| 1. / f, |f| f.reciprocal_inplace());
    }

    #[test]
    fn test_unary_reciprocal_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], |f| 1. / f, |f| f.reciprocal_inplace());
    }

    #[test]
    fn test_unary_square_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], |f| f * f, |f| f.square_inplace());
    }

    #[test]
    fn test_unary_square_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], |f| f * f, |f| f.square_inplace());
    }

    #[test]
    fn test_unary_square_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], |f| f * f, |f| f.square_inplace());
    }

    #[test]
    fn test_unary_cube_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], |f| f * f * f, |f| f.cube_inplace());
    }

    #[test]
    fn test_unary_cube_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], |f| f * f * f, |f| f.cube_inplace());
    }

    #[test]
    fn test_unary_cube_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], |f| f * f * f, |f| f.cube_inplace());
    }

    #[test]
    fn test_unary_exp_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, 0.25], f64::exp, |f| f.exp_inplace());
    }

    #[test]
    fn test_unary_exp_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([0.125, 0.5, 0.25], f64::exp, |f| f.exp_inplace());
    }

    #[test]
    fn test_unary_exp_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::exp, |f| f.exp_inplace());
    }

    #[test]
    fn test_unary_sign_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cpu>([0.5, -0.25], f64::signum, |f| f.sign_inplace());
    }

    #[test]
    fn test_unary_sign_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cpu>([-0.125, 0.5, -0.25], f64::signum, |f| f.sign_inplace());
    }

    #[test]
    fn test_unary_sign_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cpu>([0.5; 16], f64::signum, |f| f.sign_inplace());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::{
        backend::cuda::Cuda,
        core::{
            primitives::{CudaTensor, TensorBase},
            tensor::{AsTensor, TensorAccess, TensorAccessMut},
            Tensor,
        },
        ops::unary::*,
        testing::{
            test_with_contiguous_2_elem_tensor, unary_assert_1d_strided, unary_assert_contiguous,
            unary_assert_nd_strided,
        },
    };

    #[test]
    fn test_unary_negate_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_negate_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], std::ops::Neg::neg, |f| {
            f.neg_inplace()
        });
    }

    #[test]
    fn test_unary_cos_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_cos_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_cos_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::cos, |f| {
            f.cos_inplace()
        });
    }

    #[test]
    fn test_unary_sin_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_sin_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_sin_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::sin, |f| {
            f.sin_inplace()
        });
    }

    #[test]
    fn test_unary_tan_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::tan, |f| {
            f.tan_inplace()
        });
    }

    #[test]
    fn test_unary_tan_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::tan, |f| {
            f.tan_inplace()
        });
    }

    #[test]
    fn test_unary_tan_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::tan, |f| {
            f.tan_inplace()
        });
    }


    /*
    ARCTRIG
     */

    #[test]
    fn test_unary_acos_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_acos_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_acos_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::acos, |f| {
            f.acos_inplace()
        });
    }

    #[test]
    fn test_unary_asin_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_asin_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_asin_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::asin, |f| {
            f.asin_inplace()
        });
    }

    #[test]
    fn test_unary_atan_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::atan, |f| {
            f.atan_inplace()
        });
    }

    #[test]
    fn test_unary_atan_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::atan, |f| {
            f.atan_inplace()
        });
    }

    #[test]
    fn test_unary_atan_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::atan, |f| {
            f.atan_inplace()
        });
    }


    /**
     * 
     * HYPERBOLIC ARC TRIG
     */

     #[test]
    fn test_unary_cosh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_cosh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_cosh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::cosh, |f| {
            f.cosh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

    #[test]
    fn test_unary_sinh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::sinh, |f| {
            f.sinh_inplace()
        });
    }

   


    /*
    ARCTRIG
     */

    #[test]
    fn test_unary_acosh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_acosh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_acosh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], f64::acosh, |f| {
            f.acosh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0, 1.0], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([1.0, 1.0, 1.0], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_asinh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::asinh, |f| {
            f.asinh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }

    #[test]
    fn test_unary_atanh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::atanh, |f| {
            f.atanh_inplace()
        });
    }


    /*
    RSQRT */

    #[test]
    fn test_unary_rsqrt_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_rsqrt_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_rsqrt_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| (1. / f.sqrt()), |f| {
            f.rsqrt_inplace()
        });
    }

    #[test]
    fn test_unary_sign_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_sign_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_sign_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::signum, |f| {
            f.sign_inplace()
        });
    }

    #[test]
    fn test_unary_exp_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_exp_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_exp_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], f64::exp, |f| {
            f.exp_inplace()
        });
    }

    #[test]
    fn test_unary_square_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| f * f, |f| {
            f.square_inplace()
        });
    }

    #[test]
    fn test_unary_square_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| f * f, |f| {
            f.square_inplace()
        });
    }

    #[test]
    fn test_unary_square_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| f * f, |f| {
            f.square_inplace()
        });
    }

     #[test]
    fn test_unary_cube_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

    #[test]
    fn test_unary_cube_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

    #[test]
    fn test_unary_cube_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| f * f * f, |f| {
            f.cube_inplace()
        });
    }

     #[test]
    fn test_unary_reciprocal_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5, 0.25], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }

    #[test]
    fn test_unary_reciprocal_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>([0.125, 0.5, 0.25], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }

    #[test]
    fn test_unary_reciprocal_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| 1. / f, |f| {
            f.reciprocal_inplace()
        });
    }


    #[test]
    fn test_unary_abs_continous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.3], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_abs_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, -1.0, 3.0],
            |f| f.abs(),
            |f| f.abs_inplace(),
        );
    }

    #[test]
    fn test_unary_abs_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([-1.0; 16], |f| f.abs(), |f| f.abs_inplace());
    }

    #[test]
    fn test_unary_relu_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.0, 1.0], |f| f.max(0.0), Relu::relu_inplace);
    }

    #[test]
    fn test_unary_relu_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [-1.0, 1.0, -1.0],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_relu_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [
                -1.0, 1.0, 0.0, 2.0, 1.0, 2.3, -0.3, 0.4, 0.0, -0.3, 0.4, 0.5, -0.2, 0.1, 0.2, -0.5,
            ],
            |f| f.max(0.0),
            |f| f.relu_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_sigmoid_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [1.0; 16],
            |f| 1. / (1. + (-f).exp()),
            |f| f.sigmoid_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [1.0, 1.0],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_silu_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [1.0; 16],
            |f| f / (1. + (-f).exp()),
            |f| f.silu_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_contiguous_cuda() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_1d_strided_cuda() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_tanh_nd_strided_cuda() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [1.0; 16],
            |f| (f.exp() - (-f).exp()) / (f.exp() + (-f).exp()),
            |f| f.tanh_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_nd_strided() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.0; 16], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_sqrt_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f.sqrt(),
            |f| f.sqrt_inplace(),
        );
    }

    #[test]
    fn test_unary_sqrt_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.0; 2], |f| f.sqrt(), |f| f.sqrt_inplace());
    }

    #[test]
    fn test_unary_ln_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.0; 16], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.0, 1.0, 1.0],
            |f| f.ln(),
            |f| f.ln_inplace(),
        );
    }

    #[test]
    fn test_unary_ln_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.0; 2], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.5; 16], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_ln_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.5, 1.1, 1.5],
            |f| f.ln(),
            |f| f.ln_inplace(),
        );
    }

    #[test]
    fn test_unary_ln_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.5; 2], |f| f.ln(), |f| f.ln_inplace());
    }

    #[test]
    fn test_unary_expm1_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.5; 16], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_expm1_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.5, 1.1, 1.5],
            |f| f.exp_m1(),
            |f| f.expm1_inplace(),
        );
    }

    #[test]
    fn test_unary_expm1_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.5; 2], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_expm1_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.5; 16], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_expm1_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.5, 1.1, 1.5],
            |f| f.exp_m1(),
            |f| f.expm1_inplace(),
        );
    }

    #[test]
    fn test_unary_expm1_contiguous_f32() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.5; 2], |f| f.exp_m1(), |f| f.expm1_inplace());
    }

    #[test]
    fn test_unary_ln1p_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([0.5; 16], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_ln1p_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [0.5, 0.1, 0.5],
            |f| f.ln_1p(),
            |f| f.ln1p_inplace(),
        );
    }

    #[test]
    fn test_unary_ln1p_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([0.5; 2], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_ln1p_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([0.5; 16], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_ln1p_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [0.5, 0.1, 0.5],
            |f| f.ln_1p(),
            |f| f.ln1p_inplace(),
        );
    }

    #[test]
    fn test_unary_ln1p_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([0.5; 2], |f| f.ln_1p(), |f| f.ln1p_inplace());
    }

    #[test]
    fn test_unary_floor_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.7; 16], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_floor_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.7, 2.3, 3.9],
            |f| f.floor(),
            |f| f.floor_inplace(),
        );
    }

    #[test]
    fn test_unary_floor_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.7, 2.3], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_floor_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.7; 16], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_floor_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.7, 2.3, 3.9],
            |f| f.floor(),
            |f| f.floor_inplace(),
        );
    }

    #[test]
    fn test_unary_floor_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.7, 2.3], |f| f.floor(), |f| f.floor_inplace());
    }

    #[test]
    fn test_unary_ceil_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.3; 16], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_ceil_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.3, 2.7, 3.1],
            |f| f.ceil(),
            |f| f.ceil_inplace(),
        );
    }

    #[test]
    fn test_unary_ceil_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.3, 2.7], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_ceil_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.3; 16], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_ceil_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.3, 2.7, 3.1],
            |f| f.ceil(),
            |f| f.ceil_inplace(),
        );
    }

    #[test]
    fn test_unary_ceil_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.3, 2.7], |f| f.ceil(), |f| f.ceil_inplace());
    }

    #[test]
    fn test_unary_round_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.3; 16], |f| f.round(), |f| {
            f.round_inplace()
        });
    }

    #[test]
    fn test_unary_round_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.3, 2.7, 3.1],
            |f| f.round(),
            |f| f.round_inplace(),
        );
    }

    #[test]
    fn test_unary_round_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.3, 2.7], |f| f.round(), |f| {
            f.round_inplace()
        });
    }

    #[test]
    fn test_unary_round_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.3; 16], |f| f.round(), |f| {
            f.round_inplace()
        });
    }

    #[test]
    fn test_unary_round_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.3, 2.7, 3.1],
            |f| f.round(),
            |f| f.round_inplace(),
        );
    }

    #[test]
    fn test_unary_round_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([-1.3, 2.7], |f| f.round(), |f| {
            f.round_inplace()
        });
    }

    #[test]
    fn test_unary_trunc_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([-1.3; 16], |f| f.trunc(), |f| {
            f.trunc_inplace()
        });
    }

    #[test]
    fn test_unary_trunc_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [-1.3, 2.7, 3.1],
            |f| f.trunc(),
            |f| f.trunc_inplace(),
        );
    }

    #[test]
    fn test_unary_trunc_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([-1.3, 2.7], |f| f.trunc(), |f| {
            f.trunc_inplace()
        });
    }

    #[test]
    fn test_unary_trunc_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([-1.3; 16], |f| f.trunc(), |f| {
            f.trunc_inplace()
        });
    }

    #[test]
    fn test_unary_trunc_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [-1.3, 2.7, 3.1],
            |f| f.trunc(),
            |f| f.trunc_inplace(),
        );
    }

    #[test]
    fn test_unary_trunc_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([-1.3, 2.7], |f| f.trunc(), |f| {
            f.trunc_inplace()
        });
    }
}

#[cfg(all(test, feature = "remote"))]
mod remote_tests {
    use std::{sync::OnceLock, thread};

    use crate::{
        backend::{
            remote::{client::RemoteBackend, get_backend_default, server::RemoteServer},
            Backend,
        },
        core::{
            primitives::{RemoteTensor, TensorBase},
            tensor::TensorError,
            value::TensorValue,
            MetaTensor, Shape,
        },
    };

    // Lazy static backend shared across all tests
    static BACKEND: OnceLock<RemoteBackend> = OnceLock::new();

    fn get_backend() -> RemoteBackend {
        BACKEND
            .get_or_init(|| {
                // Start the server
                let mut server = RemoteServer::new("127.0.0.1".parse().unwrap(), 7878);
                thread::spawn(move || {
                    let _ = server.serve();
                });
                thread::sleep(std::time::Duration::from_millis(10));

                // Create and connect the backend
                let backend = get_backend_default().unwrap();

                backend
            })
            .clone()
    }

    fn make_remote_tensor<T: TensorValue>(
        buf: Vec<T>,
        shape: impl Into<Shape>,
    ) -> Result<RemoteTensor<T>, TensorError> {
        let shape: Shape = shape.into();
        let buf_len = buf.len();
        let expected_len: usize = shape.iter().product();

        if buf_len != expected_len {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                expected_len, buf_len
            )));
        }

        let backend = get_backend();
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);

        // Clone the backend for this tensor
        let tensor_backend = backend.clone();
        drop(backend); // Release the lock

        Ok(TensorBase::from_parts(
            tensor_backend,
            buffer,
            MetaTensor::new(shape, stride, 0),
        ))
    }

    #[test]
    fn test_remote_negate() {
        let tensor: TensorBase<f32, RemoteBackend> =
            make_remote_tensor(vec![1.0f32, -2.0, 3.0], (3,)).unwrap();
        let negated = -tensor;

        let expected = make_remote_tensor(vec![-1.0f32, 2.0, -3.0], (3,)).unwrap();
        assert_eq!(negated.cpu().unwrap(), expected.cpu().unwrap());
    }
}
