use crate::{backend::Backend, core::{primitives::{OpTensor, TensorBase}, tensor::{AsViewMut, TensorError}, value::WeightValue}, grad};

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
    #[grad::no_grad] // do not track the optimization step itself
    fn step(&mut self) -> Result<(), TensorError> {
        for param_ref in &self.parameters {
            let param = unsafe { param_ref.as_mut().expect("Invalid parameter registered with optimizer") };
            let mut grad = param.grad.write();
            let u = if let Some(grad_inner) =  grad.as_ref(){
                // Update parameter: param = param - learning_rate * grad
                let update: TensorBase<T, B> = grad_inner.typed().expect("Mixed precision training not supported.") * self.learning_rate;
                // Clear gradient after update. CONCIDER an explicit zero_grad method instead
                *grad = None;
                Some(update)
            } else {
                None
            };
            drop(grad);
            if let Some(update) = u {
                let mut v = param.view_mut();
                v -= update;
            }
        }
        Ok(())  
    }

    #[grad::when_enabled(ctx, message = "Cannot register a parameter without a grad context.")]
    fn register_parameter(&mut self, param: &mut TensorBase<T, B>) -> Result<(), TensorError> {
        // check is leaf node
        param.param();
        let nodes = ctx.nodes.borrow();
        let node_key = ctx.resolve_maybe_key(param.op());
        let node = nodes.get(node_key).ok_or_else(|| TensorError::GradError("Parameter not found in grad context.".into()))?;
        if !node.is_leaf() {
            return Err(TensorError::GradError("Only leaf tensors can be registered as parameters.".into()));
        }
        self.parameters.push(param as *mut TensorBase<T, B>);
        Ok(())
    }
}