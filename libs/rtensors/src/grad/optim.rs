use crate::{backend::Backend, core::{primitives::{Grad, OpTensor, TensorBase}, tensor::{AsViewMut, TensorError}, value::WeightValue}, grad};

pub trait Optim<T: WeightValue, B: Backend> {
    fn step(&mut self) -> Result<(), TensorError>;
    fn register_parameter(&mut self, param: &mut TensorBase<T, B>) -> Result<(), TensorError>;
    fn register_parameters(&mut self, param: Vec<&mut TensorBase<T, B>>) -> Result<(), TensorError> {
        for p in param.into_iter() {
            self.register_parameter(p)?;
        }
        Ok(())
    }
}

pub struct SGD<T: WeightValue, B: Backend> {
    parameters: Vec<*mut TensorBase<T, B>>,
    learning_rate: T,
}

impl<T: WeightValue, B: Backend> SGD<T, B> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            parameters: Vec::new(),
            learning_rate,
        }
    }
}

impl<T: WeightValue, B: Backend> Optim<T, B> for SGD<T, B> {
    fn step(&mut self) -> Result<(), TensorError> {
        for param_ref in self.parameters {
            let mut param = unsafe { param_ref.as_mut().expect("Null pointer where it shouldn't be") };
            let mut grad = param.grad.write();
            if let Some(grad_inner) =  grad.as_ref(){
                // Update parameter: param = param - learning_rate * grad
                let update = grad_inner * self.learning_rate;
                let mut v = param.view_mut();
                v -= update;
                // Clear gradient after update
                *grad = None;
            }
        }
        Ok(())  
    }

    #[grad::when_enabled(ctx, message = "Cannot register a parameter without a grad context.")]
    fn register_parameter(&mut self, param: &mut TensorBase<T, B>) -> Result<(), TensorError> {
        // check is leaf node
        let nodes = ctx.nodes.borrow();
        let node = param.op().ok_or_else(|| TensorError::GradError("Parameter has no associated node.".into()))?;
        let node = nodes.get(node).ok_or_else(|| TensorError::GradError("Parameter not found in grad context.".into()))?;
        if !node.is_leaf() {
            return Err(TensorError::GradError("Only leaf tensors can be registered as parameters.".into()));
        }
        self.parameters.push(param as *mut TensorBase<T, B>);
        Ok(())
    }
}