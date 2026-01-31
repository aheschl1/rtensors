use crate::core::tensor::TensorAccess;
use crate::core::tensor::TensorAccessMut;
use crate::core::untyped::AsUntypedTensor;
use crate::core::value::WeightValue;
use crate::core::value::TensorValue;
use crate::backend::Backend;
use crate::core::primitives::TensorBase;
use crate::core::tensor::{AsViewMut, AsTensor, TensorError};
use crate::core::MetaTensorView;
use crate::grad::GradNode;
use crate::grad;
use crate::core::primitives::OpTensor;

pub mod add;
pub mod sub;
pub mod mul;
pub mod div;

macro_rules! specify_binary_scalar_op_template {
    (
        $(
            ($name:ident) $op:ident $(where T: $first:path $(, $extra:path)*)?; |$requires_input:literal, $input:ident, $result:ident, $ctx:ident, $grad_node:ident, $scalar:ident| $grad_fn:block
        ),+ $(,)?
    ) => {

        paste::paste! {

            pub trait InplaceBinaryOp<T: TensorValue, B: Backend> {
                $(
                    fn [<apply_scalar $op>](&mut self, value: T)
                    where 
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+

            }

            
            pub trait ScalarOp<T: TensorValue, B: Backend> {
                $(
                    fn $op(&self, value: T) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?;
                )+
            }


            $(

                pub trait $name<T: TensorValue, B: Backend> {
                    fn [<$op _inplace>](&mut self, value: T)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    ;
                }

                impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> $name<T, B> for V {
                    fn [<$op _inplace>](&mut self, value: T)
                    where
                     $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let view = self.view_mut();
                        //  ========== GRAPH BUILDING ============
                        let requires_input: bool = $requires_input;
                        let input = grad::when_enabled(|$ctx| {
                            if requires_input { 
                                Some(view.owned()) // TODO extra memory copy here, optimize later
                            } else {
                                None
                            }
                        });
                        // ========== APPLICATION OF OPERATION ============
                        if let Err(e) = view.backend.[<scalar_apply_$op>](view.buf, value, &view.meta) {
                            panic!("Failed to apply op: {}", e);
                        }

                        // ========== GRADIENT NODE CREATION ============
                        grad::when_enabled(|$ctx| {
                            // unwrap when_enabled error
                            let input = input.expect("Input tensor required for gradient computation but not captured.");
                            let $scalar = value;
                            let $input = input;
                            let $grad_node = view.op();
                            let $result = view;
                            let node: Result<GradNode, TensorError> = $grad_fn;
                            $ctx.attach(&$result, node.expect("Failed to create gradient node."))
                        });
                    }
                }

              
            )+



            impl<T: TensorValue, B: Backend, V: AsViewMut<T, B>> InplaceBinaryOp<T, B> for V
            {
                $(
                    fn [<apply_scalar $op>](&mut self, value: T)
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        self.[<$op _inplace>](value);
                    }
                )+
            }

            
            impl<T: TensorValue, B: Backend, V: AsTensor<T, B>> ScalarOp<T, B> for V {
                $(
                    fn $op(&self, value: T) -> TensorBase<T, B>
                    where
                    $(
                        T: $first $(+ $extra)*
                    )?
                    {
                        let mut result = self.owned();
                        result.[<apply_scalar $op>](value);
                        result
                    }
                )+
            }
           
        }
        
    };
}

specify_binary_scalar_op_template!(
    (LogOp) log where T: WeightValue; |false, _input, _result, _ctx, _grad_node, _scalar| {
        Err(TensorError::UnsupportedOperation("Gradient for log not yet implemented.".into()))
    },
    (Log1POp) log1p where T: WeightValue; |false, _input, _result, _ctx, _grad_node, _scalar| {
        Err(TensorError::UnsupportedOperation("Gradient for log1p not yet implemented.".into()))
    },
    (LeakyReluOp) leaky_relu; |true, input, _result, _ctx, grad_node, scalar| {
        let input = input.unwrap();
        let mut grad_map = TensorBase::<T, B>::zeros(input.shape());
        for coord in input.iter_coords() {
            let val = input.get(&coord).unwrap();
            if val > T::ZERO {
                grad_map.set(&coord, T::ONE);
            } else {
                grad_map.set(&coord, scalar);
            }
        }

        let node = GradNode::ReLU {
            input: grad_node,
            grad_map: grad_map.as_untyped(),
        };
        Ok(node)    
    },
    (EluOp) elu where T: WeightValue; |false, _input, _result, _ctx, _grad_node, _scalar| {
        Err(TensorError::UnsupportedOperation("Gradient for elu not yet implemented.".into()))
    },
);

#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, ops::scalar::{EluOp, LeakyReluOp, Log1POp, LogOp}, testing::{unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided}};

    #[test]
    fn test_scalar_log_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.4, 2.6, 3.5],
            |f| f.log(10.0),
            |f| f.log_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.4, 2.6], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }


    #[test]
    fn test_scalar_log_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.4; 16], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }

    #[test]
    fn test_scalar_log1p_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.4, 2.6, 3.5],
            |f| f.ln_1p() / 10.0_f32.ln(),
            |f| f.log1p_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log1p_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>([1.4, 2.6], |f| f.ln_1p() / 10.0_f32.ln(), |f| f.log1p_inplace(10.0));
    }


    #[test]
    fn test_scalar_log1p_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>([1.4; 16], |f| f.ln_1p() / 10.0_f32.ln(), |f| f.log1p_inplace(10.0));
    }

    #[test]
    fn test_scalar_leaky_relu_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.0, -2.0, 3.0],
            |f| if f > 0.0 { f } else { f * 0.1 },
            |f| f.leaky_relu_inplace(0.1),
        );
    }

    #[test]
    fn test_scalar_leaky_relu_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>(
            [2.0, -1.5],
            |f| if f > 0.0 { f } else { f * 0.1 },
            |f| f.leaky_relu_inplace(0.1)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
            |f| if f > 0.0 { f } else { f * 0.2 },
            |f| f.leaky_relu_inplace(0.2)
        );
    }

    #[test]
    fn test_scalar_elu_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cpu>(
            [1.0, -2.0, 3.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1),
        );
    }

    #[test]
    fn test_scalar_elu_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cpu>(
            [2.0, -1.5],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1)
        );
    }

    #[test]
    fn test_scalar_elu_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cpu>(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.2 },
            |f| f.elu_inplace(0.2)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_1d_strided_i32() {
        unary_assert_1d_strided::<i32, _, _, Cpu>(
            [1, -2, 3],
            |f| if f > 0 { f } else { f * 1 },
            |f| f.leaky_relu_inplace(1),
        );
    }

    #[test]
    fn test_scalar_leaky_relu_contiguous_i32() {
        unary_assert_contiguous::<i32, _, _, Cpu>(
            [2, -1],
            |f| if f > 0 { f } else { f * 1 },
            |f| f.leaky_relu_inplace(1)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_nd_strided_i32() {
        unary_assert_nd_strided::<i32, _, _, Cpu>(
            [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16],
            |f| if f > 0 { f } else { f * 2 },
            |f| f.leaky_relu_inplace(2)
        );
    }

}
#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests{
    use crate::{backend::cuda::Cuda, ops::scalar::{EluOp, LeakyReluOp, Log1POp, LogOp}, testing::{unary_assert_1d_strided, unary_assert_contiguous, unary_assert_nd_strided}};

    #[test]
    fn test_scalar_log_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.4, 2.6, 3.5],
            |f| f.log(10.0),
            |f| f.log_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.4, 2.6], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }


    #[test]
    fn test_scalar_log_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.4; 16], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }

    #[test]
    fn test_scalar_log_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.4, 2.6, 3.5],
            |f| f.log(10.0),
            |f| f.log_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.4, 2.6], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }


    #[test]
    fn test_scalar_log_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.4; 16], |f| f.log(10.0), |f| f.log_inplace(10.0));
    }

    #[test]
    fn test_scalar_log1p_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.4, 2.6, 3.5],
            |f| f.ln_1p() / 10.0_f32.ln(),
            |f| f.log1p_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log1p_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>([1.4, 2.6], |f| f.ln_1p() / 10.0_f32.ln(), |f| f.log1p_inplace(10.0));
    }


    #[test]
    fn test_scalar_log1p_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>([1.4; 16], |f| f.ln_1p() / 10.0_f32.ln(), |f| f.log1p_inplace(10.0));
    }

    #[test]
    fn test_scalar_log1p_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.4, 2.6, 3.5],
            |f| f.ln_1p() / 10.0_f64.ln(),
            |f| f.log1p_inplace(10.0),
        );
    }

    #[test]
    fn test_scalar_log1p_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>([1.4, 2.6], |f| f.ln_1p() / 10.0_f64.ln(), |f| f.log1p_inplace(10.0));
    }


    #[test]
    fn test_scalar_log1p_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>([1.4; 16], |f| f.ln_1p() / 10.0_f64.ln(), |f| f.log1p_inplace(10.0));
    }

    #[test]
    fn test_scalar_leaky_relu_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.0, -2.0, 3.0],
            |f| if f > 0.0 { f } else { f * 0.1 },
            |f| f.leaky_relu_inplace(0.1),
        );
    }

    #[test]
    fn test_scalar_leaky_relu_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>(
            [2.0, -1.5],
            |f| if f > 0.0 { f } else { f * 0.1 },
            |f| f.leaky_relu_inplace(0.1)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
            |f| if f > 0.0 { f } else { f * 0.2 },
            |f| f.leaky_relu_inplace(0.2)
        );
    }
    
    #[test]
    fn test_scalar_leaky_relu_1d_strided_i32() {
        unary_assert_1d_strided::<i32, _, _, Cuda>(
            [1, -2, 3],
            |f| if f > 0 { f } else { f * 1 },
            |f| f.leaky_relu_inplace(1),
        );
    }

    #[test]
    fn test_scalar_elu_1d_strided_f32() {
        unary_assert_1d_strided::<f32, _, _, Cuda>(
            [1.0, -2.0, 3.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1),
        );
    }

    #[test]
    fn test_scalar_elu_contiguous_f32() {
        unary_assert_contiguous::<f32, _, _, Cuda>(
            [2.0, -1.5],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1)
        );
    }

    #[test]
    fn test_scalar_elu_nd_strided_f32() {
        unary_assert_nd_strided::<f32, _, _, Cuda>(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.2 },
            |f| f.elu_inplace(0.2)
        );
    }

    #[test]
    fn test_scalar_elu_1d_strided_f64() {
        unary_assert_1d_strided::<f64, _, _, Cuda>(
            [1.0, -2.0, 3.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1),
        );
    }

    #[test]
    fn test_scalar_elu_contiguous_f64() {
        unary_assert_contiguous::<f64, _, _, Cuda>(
            [2.0, -1.5],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.1 },
            |f| f.elu_inplace(0.1)
        );
    }

    #[test]
    fn test_scalar_elu_nd_strided_f64() {
        unary_assert_nd_strided::<f64, _, _, Cuda>(
            [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0, 12.0, -13.0, 14.0, -15.0, 16.0],
            |f| if f >= 0.0 { f } else { f.exp_m1() * 0.2 },
            |f| f.elu_inplace(0.2)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_contiguous_i32() {
        unary_assert_contiguous::<i32, _, _, Cuda>(
            [2, -1],
            |f| if f > 0 { f } else { f * 1 },
            |f| f.leaky_relu_inplace(1)
        );
    }

    #[test]
    fn test_scalar_leaky_relu_nd_strided_i32() {
        unary_assert_nd_strided::<i32, _, _, Cuda>(
            [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16],
            |f| if f > 0 { f } else { f * 1 },
            |f| f.leaky_relu_inplace(1)
        );
    }
}