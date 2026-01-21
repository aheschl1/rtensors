use std::{cell::RefCell, sync::Arc};

use crate::{backend::Backend, core::{idx::Idx, primitives::TensorBase, tensor::{AsTensor, TensorAccess, TensorError}, value::{TensorValue, WeightValue}, MetaTensorView}, grad::{self, GradNode, NodeKey}};

#[derive(Debug)]
pub struct GradTensor<T: WeightValue, B: Backend> {
    pub(crate) inner: GradTensorRef<T, B>,
    pub(crate) node: NodeKey,
}

// deep clone, not Rc clone
impl<T: WeightValue, B: Backend> Clone for GradTensor<T, B> {
    #[grad::when_enabled(ctx)]
    fn clone(&self) -> Self {
        let tensor = self.borrow().tensor.contiguous();
        let grad = self.borrow().grad.as_ref().map(|g| g.contiguous());
        let inner = GradTensorInner {
            tensor,
            grad,
        };
        let inner_ref = Arc::new(RefCell::new(inner));
        let nodes = ctx.nodes.borrow();
        let curr_node = nodes.get(self.node).expect("Node not found in grad context");
        let node = if let GradNode::Leaf(_) = curr_node {
            GradNode::Leaf( inner_ref.clone() )
        } else {
            curr_node.clone()
        };
        drop(nodes);
        ctx.attach(inner_ref.clone(), node)
    }
}

impl<T: WeightValue, B: Backend> GradTensor<T, B> {

    #[grad::when_enabled(ctx)]
    pub(crate) fn leaf(
        tensor: TensorBase<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.make_leaf(inner)

    }

    pub(crate) fn input( // a tensor that requires grad but is not a parameter, for example, input to the model
        value: TensorBase<T, B>
    ) -> Self {
        Self::from_op(value, GradNode::None)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op(
        tensor: TensorBase<T, B>,
        op: GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        ctx.attach(inner, op)
    }

    #[inline]
    #[grad::when_enabled(ctx)]
    pub(crate) fn from_op_self_referential(
        tensor: TensorBase<T, B>,
        op_builder: impl FnOnce(GradTensorRef<T, B>) -> GradNode<T, B>,
    ) -> Self {
        let inner = GradTensorInner {
            tensor,
            grad: None,
        };
        let inner = Arc::new(RefCell::new(inner));
        let node = op_builder(inner.clone());
        ctx.attach(inner, node)
    } 

    #[grad::when_enabled(ctx)]
    pub(crate) fn is_leaf(&self) -> bool {
        let nodes = ctx.nodes.borrow();
        let node = nodes.get(self.node).expect("Node not found in grad context.");
        node.is_leaf()
    }

    pub(crate) fn copy_tensor(&self) -> TensorBase<T, B> {
        self.borrow().tensor.contiguous()
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, GradTensorInner<T, B>> {
        self.inner.borrow()
    }

    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, GradTensorInner<T, B>> {
        self.inner.borrow_mut()
    }

    pub fn get_ref(&self) -> GradTensorRef<T, B> {
        self.inner.clone()
    }

    #[grad::when_enabled(ctx)]
    pub fn permute(self, dims: impl Into<Idx>) -> Result<Self, TensorError> {
        let idx = dims.into();
        let mut inner = self.borrow_mut();
        let new_view = inner.tensor.permute(idx.clone())?;
        inner.tensor.meta = new_view.meta.clone();
        drop(inner);
        // record node
        let new_node = GradNode::Permute {
            input: self.node,
            dims: idx,
        };

        Ok(ctx.attach(self.inner, new_node))
    }

    pub fn transpose(self) -> Self {
        let rank = self.borrow().tensor.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        unsafe { self.permute(dims).unwrap_unchecked() }
    }
    
}


pub struct GradTensorInner<T: TensorValue, B: Backend> {
    pub(crate) tensor: TensorBase<T, B>,
    pub(crate) grad: Option<TensorBase<T, B>>,
}

impl<T: TensorValue, B: Backend> GradTensorInner<T, B> {
    pub fn item(&self) -> Result<T, TensorError> {
        self.tensor.item()
    }
}

impl<T: TensorValue, B: Backend> MetaTensorView for GradTensorInner<T, B> {
    fn meta(&self) -> &crate::core::MetaTensor {
        &self.tensor.meta
    }
}

// pub type GradTensorRef<T, B> = Arc<RefCell<GradTensorInner<T, B>>>;

impl<T: TensorValue, B: Backend> std::fmt::Debug for GradTensorInner<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradTensorInner")
            .field("grad", &self.grad)
            .field("value", &self.tensor)
            .finish()
    }
}

impl<T: WeightValue, B: Backend> Eq for GradTensor<T, B> {}
impl<T: WeightValue, B: Backend> PartialEq for GradTensor<T, B> { fn eq(&self, other: &Self) -> bool { true }}


#[cfg(test)]
mod tests {
    use crate::{backend::{cpu::Cpu, Backend}, core::{tensor::{RandomTensor, TensorAccess, WithGrad}, value::WeightValue, Tensor}, grad::{self, optim::{Optim, SGD}, primitives::GradTensor}, ops::{broadcast::l1::mean_l1_loss, linalg::MatMul, scalar::ScalarGradOp, unary::UnaryGradOp}};

    #[test]
    fn playground() {

        fn model(wa: &GradTensor<f32, Cpu>, wb: &GradTensor<f32, Cpu>, target: &GradTensor<f32, Cpu>) -> GradTensor<f32, Cpu> {
            let c = wa + wb;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv2(
            wa: &GradTensor<f32, Cpu>, 
            wb: &GradTensor<f32, Cpu>, 
            wc: &GradTensor<f32, Cpu>, 
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let inter = wb + wc;
            // println!("Intermediate: {:?}", inter);
            let c = wa + &inter;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv3(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            wa: &GradTensor<f32, Cpu>, // [2, 3]
            wb: &GradTensor<f32, Cpu>,  // [3, 2]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().abs();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv4(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            wa: &GradTensor<f32, Cpu>, // [2, 3]
            wb: &GradTensor<f32, Cpu>,  // [3, 2]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().relu();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv5(
            input: &GradTensor<f32, Cpu>,  // [2, 3]
            target: &GradTensor<f32, Cpu> // [3, 2]
        ) -> GradTensor<f32, Cpu> {
            let loss = mean_l1_loss(&-input.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv6(
            wa: &GradTensor<f32, Cpu>,  
            input: &GradTensor<f32, Cpu>,  
            target: &GradTensor<f32, Cpu> 
        ) -> GradTensor<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.sigmoid().leaky_relu(1.); // grad should be identical even without leaky relu
            let loss = mean_l1_loss(&-temp2.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv7(
            wa: &GradTensor<f32, Cpu>,  
            input: &GradTensor<f32, Cpu>,  
            target: &GradTensor<f32, Cpu> 
        ) -> GradTensor<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.leaky_relu(0.1).ln();
            let loss = mean_l1_loss(&-temp2, &target.clone().transpose());
            loss
        }

        grad::with::<f32, Cpu>(|ctx| {
            
            let a = Tensor::<f32>::scalar(1.).grad();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let target = Tensor::<f32>::zeros((2, 2)).grad();

            let mut optim = SGD::<f32, Cpu>::new(1.);
            // optim.register_parameter(&a).unwrap();
            optim.register_parameters(&[&b]).unwrap();
            
            let initial_loss = model(&a, &b, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = model(&a, &b, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model(&a, &b, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", a);

            let a = Tensor::<f32>::ones((2, 2)).param();
            let b = Tensor::<f32>::ones((2, 2)).param();
            let c = Tensor::<f32>::ones((2, 2)).param();

            optim.register_parameter(&a).unwrap();
            optim.register_parameter(&b).unwrap();
            optim.register_parameter(&c).unwrap();

            let initial_loss = modelv2(&a, &b, &c, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = modelv2(&a, &b, &c, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv2(&a, &b, &c, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", a);

            let input = Tensor::<f32>::ones((2, 3)).grad();
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((3, 2)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            let initial_loss = modelv3(&input, &wa, &wb, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = modelv3(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv3(&input, &wa, &wb, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);
            println!("{:?}", wa);

            let input = Tensor::<f32>::ones((2, 3)).grad();
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((3, 2)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            let initial_loss = modelv4(&input, &wa, &wb, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = modelv4(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv4(&input, &wa, &wb, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);
            println!("{:?}", wa);

            let input = Tensor::<f32>::ones((2, 3)).param();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&input).unwrap();
            let initial_loss = modelv5(&input, &target).borrow().tensor.item().unwrap();
            for _ in 0..12 {
                let loss = modelv5(&input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv5(&input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);

            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            let initial_loss = modelv6(&wa, &input, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = modelv6(&wa, &input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv6(&wa, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.001, 
                "Loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((3, 2)).grad();
            optim.register_parameter(&wa).unwrap();
            let initial_loss = modelv7(&wa, &input, &target).borrow().tensor.item().unwrap();
            for _ in 0..10 {
                let loss = modelv7(&wa, &input, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv7(&wa, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.2, 
                "Loss should reduce by at least 0.2, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted add
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((1, 2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            let initial_loss = {
                let inter = &input + &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input + &wa; // broadcasted add
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input + &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted sub
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::ones((1, 2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            let initial_loss = {
                let inter = &input - &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input - &wa; // broadcasted sub
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input - &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted mul
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((1, 2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            let initial_loss = {
                let inter = &input * &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input * &wa; // broadcasted mul
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input * &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted div
            let wa = Tensor::<f32>::ones((1, 3)).param();
            let input = Tensor::<f32>::ones((1, 2, 1)).param();
            let target = Tensor::<f32>::zeros((1, 2, 3)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&input).unwrap();
            let initial_loss = {
                let inter = &input / &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input / &wa; // broadcasted div
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input / &wa;
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // mamtmul
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((3, 4)).param();
            let target = Tensor::<f32>::zeros((2, 4)).grad();
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            let initial_loss = {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..1 {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                mean_l1_loss(&inter, &target).borrow().tensor.item().unwrap()
            };
            // Only 1 iteration, so smaller threshold
            assert!(initial_loss - final_loss > 0.01, 
                "Loss should reduce by at least 0.01, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);
            println!("{:?}", wb);

        })
    }

    #[test]
    fn playground_long_model() {
        // we will do a 10 layer dense model with relu activations
        struct Layer<T: WeightValue, B: Backend> {
            pub weight: GradTensor<T, B>,
            pub bias: GradTensor<T, B>,
        }

        struct DenseModel<T: WeightValue, B: Backend> {
            pub layers: Vec<Layer<T, B>>,
        }

        impl DenseModel<f32, Cpu> {
            fn new(input_size: usize, hidden_size: usize, output_size: usize, num_layers: usize) -> Self {
                let mut layers = Vec::new();
                for i in 0..num_layers {
                    let in_size = if i == 0 { input_size } else { hidden_size };
                    let out_size = if i == num_layers - 1 { output_size } else { hidden_size };
                    let weight = Tensor::<f32>::uniform((in_size, out_size))
                        .expect("Failed to create uniform tensor").param();
                    let bias = Tensor::<f32>::zeros((1, out_size)).param();
                    layers.push(Layer { weight, bias });
                }
                Self { layers }
            }

            fn forward(&self, mut x: GradTensor<f32, Cpu>) -> GradTensor<f32, Cpu> {
                for (i, layer) in self.layers.iter().enumerate() {
                    x = x.matmul(&layer.weight).unwrap() + &layer.bias;
                    if i != self.layers.len() - 1 {
                        x = x.relu();
                    }
                }
                x.sigmoid()
            }

            fn register(&self, optim: &mut SGD<f32, Cpu>) {
                for layer in &self.layers {
                    optim.register_parameter(&layer.weight).unwrap();
                    optim.register_parameter(&layer.bias).unwrap();
                }
            }
        }

        grad::with::<f32, Cpu>(|ctx| {
            let model = DenseModel::new(5, 10, 2, 10);

            let input = Tensor::<f32>::ones((1, 5)).grad();
            let target = Tensor::<f32>::uniform((1, 2)).unwrap().grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            model.register(&mut optim);
            
            let initial_loss = {
                let output = model.forward(input.clone());
                mean_l1_loss(&output, &target).borrow().tensor.item().unwrap()
            };
            for _ in 0..100 {
                let output = model.forward(input.clone());
                let loss = mean_l1_loss(&output, &target);
                println!("Loss: {:?}", loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let output = model.forward(input.clone());
                mean_l1_loss(&output, &target).borrow().tensor.item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.01, 
                "Dense model loss should reduce by at least 0.01, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_trig_and_hyperbolic() {
        // Test sin, cos, tan, sinh, cosh, tanh
        fn model_with_trig(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.sin() + x.cos().tanh();
            mean_l1_loss(&y, target)
        }

        fn model_with_hyperbolic(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.sinh() + x.cosh();
            mean_l1_loss(&y, target)
        }

        grad::with::<f32, Cpu>(|ctx| {
            println!("\n=== Testing Trig Functions ===");
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&wa).unwrap();
            
            let initial_loss = model_with_trig(&wa, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_trig(&wa, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_trig(&wa, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.01, 
                "Trig loss should reduce by at least 0.01, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Hyperbolic Functions ===");
            let wb = Tensor::<f32>::ones((2, 3)).param();
            optim.register_parameter(&wb).unwrap();
            
            let initial_loss = model_with_hyperbolic(&wb, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_hyperbolic(&wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_hyperbolic(&wb, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Hyperbolic loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_exp_and_powers() {
        // Test exp, square, cube
        fn model_with_exp(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.exp();
            mean_l1_loss(&y, target)
        }

        fn model_with_powers(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.square() + x.cube();
            mean_l1_loss(&y, target)
        }

        grad::with::<f32, Cpu>(|ctx| {
            println!("\n=== Testing Exp ===");
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::ones((2, 3)).grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            optim.register_parameter(&wa).unwrap();
            
            let initial_loss = model_with_exp(&wa, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_exp(&wa, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_exp(&wa, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Exp loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Powers ===");
            let wb = Tensor::<f32>::ones((2, 3)).param();
            optim.register_parameter(&wb).unwrap();
            
            let initial_loss = model_with_powers(&wb, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_powers(&wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_powers(&wb, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Powers loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_reciprocal_and_rsqrt() {
        // Test reciprocal and rsqrt
        fn model_with_reciprocal(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.reciprocal();
            mean_l1_loss(&y, target)
        }

        fn model_with_rsqrt(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.rsqrt();
            mean_l1_loss(&y, target)
        }

        grad::with::<f32, Cpu>(|ctx| {
            println!("\n=== Testing Reciprocal ===");
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::ones((2, 3)).grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&wa).unwrap();
            
            let initial_loss = model_with_reciprocal(&wa, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_reciprocal(&wa, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_reciprocal(&wa, &input, &target).borrow().tensor.item().unwrap();
            // Reciprocal has small gradients for x > 1, so expect smaller reduction
            assert!(initial_loss - final_loss > 0.001, 
                "Reciprocal loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Rsqrt ===");
            let wb = Tensor::<f32>::ones((2, 3)).param();
            optim.register_parameter(&wb).unwrap();
            
            let initial_loss = model_with_rsqrt(&wb, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_with_rsqrt(&wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_rsqrt(&wb, &input, &target).borrow().tensor.item().unwrap();
            // Rsqrt also has small gradients for x > 1
            assert!(initial_loss - final_loss > 0.001, 
                "Rsqrt loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_combined_operations() {
        // Test combining multiple operations
        fn model_complex(
            wa: &GradTensor<f32, Cpu>,
            wb: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x1 = input + wa;
            let x2 = x1.square().tanh();  // square then tanh
            let x3 = x2 + wb;
            let x4 = x3.sinh().reciprocal();  // sinh then reciprocal
            mean_l1_loss(&x4, target)
        }

        fn model_chain(
            wa: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + wa;
            let y = x.sin().exp().rsqrt();  // chain: sin -> exp -> rsqrt
            mean_l1_loss(&y, target)
        }

        grad::with::<f32, Cpu>(|ctx| {
            println!("\n=== Testing Complex Model ===");
            let wa = Tensor::<f32>::ones((2, 3)).param();
            let wb = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&wa).unwrap();
            optim.register_parameter(&wb).unwrap();
            
            let initial_loss = model_complex(&wa, &wb, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_complex(&wa, &wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_complex(&wa, &wb, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.001, 
                "Complex model loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Chained Operations ===");
            let wc = Tensor::<f32>::ones((2, 3)).param();
            optim.register_parameter(&wc).unwrap();
            
            let initial_loss = model_chain(&wc, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..10 {
                let loss = model_chain(&wc, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_chain(&wc, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 0.0001, 
                "Chained operations loss should reduce by at least 0.0001, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_all_new_ops() {
        // Test all 8 new operations in one model
        fn model_all(
            w: &GradTensor<f32, Cpu>,
            input: &GradTensor<f32, Cpu>,
            target: &GradTensor<f32, Cpu>
        ) -> GradTensor<f32, Cpu> {
            let x = input + w;
            let y1 = x.tanh();
            let y2 = x.exp();
            let y3 = x.square();
            let y4 = x.cube();
            let y5 = x.reciprocal();
            let y6 = x.rsqrt();
            let y7 = x.sinh();
            let y8 = x.cosh();
            
            let combined = &y1 + &y2 + &y3 + &y4 + &y5 + &y6 + &y7 + &y8;
            mean_l1_loss(&combined, target)
        }

        grad::with::<f32, Cpu>(|ctx| {
            println!("\n=== Testing All 8 New Operations ===");
            let w = Tensor::<f32>::ones((2, 3)).param();
            let input = Tensor::<f32>::ones((2, 3)).grad();
            let target = Tensor::<f32>::zeros((2, 3)).grad();
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            optim.register_parameter(&w).unwrap();
            
            let initial_loss = model_all(&w, &input, &target).borrow().tensor.item().unwrap();
            for i in 0..15 {
                let loss = model_all(&w, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.borrow().tensor.item());
                ctx.backwards(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_all(&w, &input, &target).borrow().tensor.item().unwrap();
            assert!(initial_loss - final_loss > 1.0, 
                "All ops loss should reduce by at least 1.0, initial: {}, final: {}", initial_loss, final_loss);
            
            println!("Final w gradient: {:?}", w.borrow().grad.as_ref().map(|g| g.item()));
        });
    }
}