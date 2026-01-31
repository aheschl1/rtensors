use slotmap::{new_key_type, SecondaryMap};

use crate::{backend::{Backend, BackendMatMul, cpu::Cpu}, core::{Shape, Strides, idx::Idx, primitives::{Grad, OpTensor, TensorBase}, tensor::TensorError, untyped::UntypedTensor, value::{TensorValue, Value, WeightValue}}, grad};
use std::{cell::RefCell};
use std::collections::HashMap;

mod backwards;
pub mod optim;
// pub mod primitives;

pub use proc::when_enabled;
pub use proc::if_enabled;
pub use proc::no_grad;

// struct NodeKey;

new_key_type! {
    pub struct NodeKey;
}

pub type MaybeNodeKey = Option<NodeKey>;

/// Each variant of a node holds parents and any tensors that need to be saved for backward.
#[derive(Debug)]
pub(crate) enum GradNode {
    // LEAF NODES
    Leaf( Grad ),
    None,
    // OPS
    BroadcastAdd { 
        left: MaybeNodeKey, 
        right: MaybeNodeKey, 
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastSub { 
        left: MaybeNodeKey, 
        right: MaybeNodeKey, 
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastMul { 
        left: MaybeNodeKey, 
        right: MaybeNodeKey, 
        lhs_input: Box<dyn UntypedTensor>,
        rhs_input: Box<dyn UntypedTensor>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastDiv { 
        left: MaybeNodeKey, 
        right: MaybeNodeKey, 
        lhs_input: Box<dyn UntypedTensor>,
        rhs_input_reciprocal: Box<dyn UntypedTensor>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    AddScalar { input: MaybeNodeKey },
    MulScalar { input: MaybeNodeKey, scalar: Value },
    DivScalar { input: MaybeNodeKey, scalar: Value },
    Abs { input: MaybeNodeKey, grad_map: Box<dyn UntypedTensor> },
    ReLU { input: MaybeNodeKey, grad_map: Box<dyn UntypedTensor> },
    Sigmoid { input: MaybeNodeKey, result: Box<dyn UntypedTensor> },
    Negate { input: MaybeNodeKey },
    Sqrt { input: MaybeNodeKey, output: Box<dyn UntypedTensor> },
    Ln { input: MaybeNodeKey, x_reciprocal: Box<dyn UntypedTensor> }, // store 1/x for backward
    Sin { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cos { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Tan { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Tanh { input: MaybeNodeKey, result: Box<dyn UntypedTensor> },
    Exp { input: MaybeNodeKey, result: Box<dyn UntypedTensor> },
    Square { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cube { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Reciprocal { input: MaybeNodeKey, result: Box<dyn UntypedTensor> },
    Rsqrt { input: MaybeNodeKey, result: Box<dyn UntypedTensor> },
    Sinh { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cosh { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    ExpM1 { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    Ln1p { input: MaybeNodeKey, input_tensor: Box<dyn UntypedTensor> },
    MatMul {
        left: MaybeNodeKey,
        right: MaybeNodeKey,
        left_input: Box<dyn UntypedTensor>,
        right_input: Box<dyn UntypedTensor>,
    },
    // VIEW OPS
    Permute {
        input: MaybeNodeKey,
        dims: Idx
    },
    // LOSSES
    L1 { 
        input: MaybeNodeKey, 
        // it is likely that this is leaf; however, it is not always the case
        // consider siamese networks
        target: MaybeNodeKey,
        grad_map: Box<dyn UntypedTensor>, // where is the diff greater than zero
        loss: Box<dyn UntypedTensor>,
    },
}

impl Default for GradNode {
    fn default() -> Self {
        GradNode::None
    }
}

impl GradNode {
    pub fn is_leaf(&self) -> bool {
        matches!(self, GradNode::Leaf(..))
    }

    pub fn leaf(inner: Grad) -> Self {
        GradNode::Leaf(inner)
    }

    #[inline]
    pub fn parents(&self) -> Vec<MaybeNodeKey> {
        match self {
            GradNode::BroadcastAdd { left, right, .. } => vec![*left, *right],
            GradNode::BroadcastSub { left, right, .. } => vec![*left, *right],
            GradNode::BroadcastMul { left, right, .. } => vec![*left, *right],
            GradNode::BroadcastDiv { left, right, .. } => vec![*left, *right],
            GradNode::AddScalar { input } => vec![*input],
            GradNode::MulScalar { input , ..} => vec![*input],
            GradNode::DivScalar { input , ..} => vec![*input],
            GradNode::Abs { input, .. } => vec![*input],
            GradNode::L1 { input, target, ..} => vec![*input, *target],
            GradNode::Leaf(_) | GradNode::None => vec![],
            GradNode::Permute { input, .. } => vec![*input],
            GradNode::ReLU { input, .. } => vec![*input],
            GradNode::Negate { input } => vec![*input],
            GradNode::Sigmoid { input, .. } => vec![*input],
            GradNode::Sqrt { input, .. } => vec![*input],
            GradNode::Ln { input, .. } => vec![*input],
            GradNode::Sin { input, .. } => vec![*input],
            GradNode::Cos { input, .. } => vec![*input],
            GradNode::Tan { input, .. } => vec![*input],
            GradNode::Tanh { input, .. } => vec![*input],
            GradNode::Exp { input, .. } => vec![*input],
            GradNode::Square { input, .. } => vec![*input],
            GradNode::Cube { input, .. } => vec![*input],
            GradNode::Reciprocal { input, .. } => vec![*input],
            GradNode::Rsqrt { input, .. } => vec![*input],
            GradNode::Sinh { input, .. } => vec![*input],
            GradNode::Cosh { input, .. } => vec![*input],
            GradNode::ExpM1 { input, .. } => vec![*input],
            GradNode::Ln1p { input, .. } => vec![*input],
            GradNode::MatMul { left, right, .. } => vec![*left, *right],
        }
    }

    fn backwards<T, B>(&self, upstream: &TensorBase<T, B>, _ctx: &GradContext) -> Result<Vec<TensorBase<T, B>>, TensorError> 
    where
        T: WeightValue,
        B: BackendMatMul<T>
    {
        match self {
            GradNode::L1 { .. } => backwards::backwards_l1::<T, B>(self, upstream),
            GradNode::Leaf( .. ) => backwards::accumulate_grad::<T, B>(self, upstream),
            GradNode::BroadcastAdd { .. } => backwards::backwards_add::<T, B>(self, upstream),
            GradNode::BroadcastSub { .. } => backwards::backwards_sub::<T, B>(self, upstream),
            GradNode::BroadcastMul { .. } => backwards::backwards_mul::<T, B>(self, upstream),
            GradNode::BroadcastDiv { .. } => backwards::backwards_div::<T, B>(self, upstream),
            GradNode::AddScalar { .. } => backwards::backwards_add_scalar::<T, B>(self, upstream),
            GradNode::MulScalar { .. } => backwards::backwards_mul_scalar::<T, B>(self, upstream),
            GradNode::DivScalar { .. } => backwards::backwards_div_scalar::<T, B>(self, upstream),
            GradNode::Permute { .. } => backwards::backwards_permute::<T, B>(self, upstream),

            GradNode::Negate { .. } => backwards::backwards_negate::<T, B>(self, upstream),
            GradNode::Sigmoid { .. } => backwards::backwards_sigmoid::<T, B>(self, upstream),
            GradNode::ReLU { .. } => backwards::backwards_relu::<T, B>(self, upstream),
            GradNode::Abs { .. } => backwards::backwards_abs::<T, B>(self, upstream),
            GradNode::Sqrt { .. } => backwards::backwards_sqrt::<T, B>(self, upstream),
            GradNode::Ln { .. } => backwards::backwards_ln::<T, B>(self, upstream),
            GradNode::Sin { .. } => backwards::backwards_sin::<T, B>(self, upstream),
            GradNode::Cos { .. } => backwards::backwards_cos::<T, B>(self, upstream),
            GradNode::Tan { .. } => backwards::backwards_tan::<T, B>(self, upstream),
            GradNode::Tanh { .. } => backwards::backwards_tanh::<T, B>(self, upstream),
            GradNode::Exp { .. } => backwards::backwards_exp::<T, B>(self, upstream),
            GradNode::Square { .. } => backwards::backwards_square::<T, B>(self, upstream),
            GradNode::Cube { .. } => backwards::backwards_cube::<T, B>(self, upstream),
            GradNode::Reciprocal { .. } => backwards::backwards_reciprocal::<T, B>(self, upstream),
            GradNode::Rsqrt { .. } => backwards::backwards_rsqrt::<T, B>(self, upstream),
            GradNode::Sinh { .. } => backwards::backwards_sinh::<T, B>(self, upstream),
            GradNode::Cosh { .. } => backwards::backwards_cosh::<T, B>(self, upstream),
            GradNode::ExpM1 { .. } => backwards::backwards_expm1::<T, B>(self, upstream),
            GradNode::Ln1p { .. } => backwards::backwards_ln1p::<T, B>(self, upstream),
            GradNode::MatMul { .. } => backwards::backwards_matmul::<T, B>(self, upstream),
            GradNode::None => Ok(vec![]),
            // _ => Err(TensorError::UnsupportedOperation("Backward not implemented for this node type.".into())),
        }
    }
    fn loss(&self) -> Option<&Box<dyn UntypedTensor>> {
        match self {
            GradNode::L1 { loss, .. } => Some(loss),
            _ => None,
        }
    }
}

pub struct GradContext {
    // tape: Vec<NodeKey>, // holds references to all inner tensors that require gradients
    pub(crate) nodes: RefCell<slotmap::SlotMap<NodeKey, GradNode>>,
    pub(crate) none_node: NodeKey,
}

impl Default for GradContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GradContext {
    pub fn new() -> Self {
        let mut sm = slotmap::SlotMap::with_key();
        let none_node = sm.insert(GradNode::None);
        Self { 
            nodes: RefCell::new(sm),
            none_node,
        }
    }

    /// Clears the tape, removing all recorded tensors.
    // pub fn clear(&mut self) {
    //     self.tape.clear();
    // }

    #[inline]
    pub(crate) fn make_leaf<T: TensorValue, B: Backend>(
        &self,
        inner: &TensorBase<T, B>,
    ) {
        let node = GradNode::leaf(inner.grad.clone());
        self.attach(inner, node);
    }

    #[inline]
    pub(crate) fn attach(
        &self,
        inner: &impl OpTensor,
        op: GradNode,
    ) {
        let node_id = self.nodes.borrow_mut().insert(op);
        inner.set_op(node_id);
    }

    #[inline]
    fn resolve_maybe_key(&self, maybe_key: MaybeNodeKey) -> NodeKey {
        match maybe_key {
            Some(key) => key,
            None => self.none_node,
        }
    }

    #[grad::no_grad]
    pub fn backwards<T, B>(&self, root: &impl OpTensor) -> Result<(), TensorError> 
    where
        T: WeightValue,
        B: BackendMatMul<T>
    {
        // holds nodes to visit along with their upstream gradients
        // topo sort, because concider a graph like A->C<-B<-D where BFS should visit C too early
        let mut stack = Vec::new();
        let mut marks = SecondaryMap::new();
        let mut node_order = Vec::new();
        stack.push(StackState::Enter(self.resolve_maybe_key(root.op())));

        enum StackState {
            Enter (NodeKey),
            Exit (NodeKey),
        }

        while let Some(state) = stack.pop() {
            match state {
                StackState::Enter (nkey) => {
                    match marks.get(nkey) {
                        Some(true) => continue,
                        Some(false) => return Err(TensorError::GradError("Graph contains a cycle.".into())),
                        None => {
                            marks.insert(nkey, false);
                            stack.push(StackState::Exit(nkey));
                            if let Some(node) = self.nodes.borrow().get(nkey) {
                                let node: &GradNode = node;
                                let ps: Vec<MaybeNodeKey> = node.parents();
                                for parent in ps {
                                    stack.push(StackState::Enter(self.resolve_maybe_key(parent)));
                                }
                            } else {
                                return Err(TensorError::GradError("Node not found during backward pass.".into()));
                            }
                        }
                    }
                },
                StackState::Exit (nkey) => {
                    marks.insert(nkey, true);
                    node_order.push(nkey);
                }
            }
        }
        // could in theory move this into the above loop but this is clearer
        let root_node_key = self.resolve_maybe_key(root.op());
        let nodes = self.nodes.borrow();
        let loss = nodes.get(root_node_key)
            .and_then(|n: &GradNode| n.loss().as_ref().cloned())
            .ok_or_else(|| TensorError::GradError("Root node does not contain a loss.".into()))?;
        let mut accumulations = HashMap::new();
        accumulations.insert(root_node_key, vec![loss.typed::<T, B>().expect("Loss is the wrong datatype.").clone()]);

        // println!("{:?}", node_order
        //     .iter()
        //     .map(|k| self.nodes.borrow().get(*k).unwrap().clone())
        //     .collect::<Vec<_>>()
        // );

        drop(nodes); // free borrow
        for node_key in node_order.into_iter().rev() {
            // accumulate grad. because of topo sort, we can assume to just sum the upstreams present to us
            // and then propagate downstream
            let dldy = accumulations.remove(&node_key)
                .unwrap() // t=since this node is in the visited list, it must have upstream grads
                .into_iter()
                .fold(None, |acc: Option<TensorBase<T, B>>, grad| {
                    if let Some(accum) = acc {
                        Some(accum + grad)
                    } else {
                        Some(grad) // clone here is lame
                    }
                })
                .unwrap(); // must have at least one upstream grad
            
            let mut nodes = self.nodes.borrow_mut();
            let node = nodes.get(node_key).unwrap(); // we would never have discovered this node if it was not present
            let upstreams: Vec<TensorBase<T, B>> = node.backwards(&dldy, self)?;

            let parents = node.parents();
            for (parent, grad) in parents.into_iter().zip(upstreams.into_iter()) {
                accumulations.entry(self.resolve_maybe_key(parent)).or_insert_with(Vec::new).push(grad);
            }

            match node {
                GradNode::None | GradNode::Leaf(_) => {},
                _ => {
                    // free memory by removing upstream grads after use
                    accumulations.remove(&node_key);
                    nodes.remove(node_key);
                }
            }
        }
        Ok(())
        // gradient is now accumulated in leaf nodes
    }
}

impl std::fmt::Debug for GradContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradContext {{ nodes_len: {} }}", self.nodes.borrow().len())
    }
}

thread_local! {
    static GRAD_CONTEXT: std::cell::RefCell<Option<GradContext>> = std::cell::RefCell::new(None);
    static GRAD_DISABLED: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

pub fn no_grad<R>(
    f: impl FnOnce() -> R
) -> R {
    GRAD_DISABLED.with(|d| {
        let previous = *d.borrow();
        d.replace(true);
        let result = f();
        d.replace(previous);
        result
    })
}

pub fn with(
    f: impl FnOnce(&GradContext)
){
    if GRAD_DISABLED.with(|d| *d.borrow()) {
        return;
    }
    GRAD_CONTEXT.with(|ctx_cell| {
        let mut ctx_map = ctx_cell.borrow_mut();
        ctx_map.get_or_insert_with(|| GradContext::new());
        drop(ctx_map);
        let ctx = ctx_cell.borrow();
        f(ctx.as_ref().unwrap());
    });
    return;
}

#[inline]
pub fn is_enabled() -> bool {
    if GRAD_DISABLED.with(|d| *d.borrow()) {
        return false;
    }
    GRAD_CONTEXT.with(|ctx_cell| {
        let ctx_map = ctx_cell.borrow();
        ctx_map.is_some()
    })
}

/// Runs the provided closure if the gradient context is enabled for the given backend.
pub fn when_enabled<R>(
    f: impl FnOnce(&GradContext) -> R
) -> Option<R>{
    if GRAD_DISABLED.with(|d| *d.borrow()) {
        return None;
    }
    let mut result = None;
    GRAD_CONTEXT.with(|ctx_cell| {
        let ctx_map = ctx_cell.borrow();
        if let Some(ctx) = ctx_map.as_ref() {
            result = Some(f(ctx));
        }
    });
    result
}

pub fn when_disabled(
    f: impl FnOnce()
) {
    if GRAD_DISABLED.with(|d| *d.borrow()) {
        f();
        return;
    }
    GRAD_CONTEXT.with(|ctx_cell| {
        let ctx_map = ctx_cell.borrow();
        if ctx_map.is_none() {
            f();
        }
    });
}

/// same as anabled but warns if not enabled
pub fn enabled_or_warn(
    f: impl FnOnce(&GradContext),
    msg: &str,
) {
    when_disabled(|| {
        tracing::warn!("{}", msg);
    });
    with(f);
}

#[cfg(test)]
mod tests {
    use crate::{backend::{Backend, cpu::Cpu}, core::{
        Tensor, primitives::{OpTensor, TensorBase}, tensor::{RandomTensor, TensorAccess}, value::WeightValue}, 
        grad::{self, optim::{Optim, SGD}}, ops::{broadcast::l1::mean_l1_loss, linalg::MatMul, scalar::ScalarOp, unary::UnaryOp}};

    #[test]
    fn playground() {

        fn model(wa: &TensorBase<f32, Cpu>, wb: &TensorBase<f32, Cpu>, target: &TensorBase<f32, Cpu>) -> TensorBase<f32, Cpu> {
            let c = wa + wb;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv2(
            wa: &TensorBase<f32, Cpu>, 
            wb: &TensorBase<f32, Cpu>, 
            wc: &TensorBase<f32, Cpu>, 
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let inter = wb + wc;
            // println!("Intermediate: {:?}", inter);
            let c = wa + &inter;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv3(
            input: &TensorBase<f32, Cpu>,  // [2, 3]
            wa: &TensorBase<f32, Cpu>, // [2, 3]
            wb: &TensorBase<f32, Cpu>,  // [3, 2]
            target: &TensorBase<f32, Cpu> // [3, 2]
        ) -> TensorBase<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().abs();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv4(
            input: &TensorBase<f32, Cpu>,  // [2, 3]
            wa: &TensorBase<f32, Cpu>, // [2, 3]
            wb: &TensorBase<f32, Cpu>,  // [3, 2]
            target: &TensorBase<f32, Cpu> // [3, 2]
        ) -> TensorBase<f32, Cpu> {
            let inter = input + wa; // [2, 3]
            let inter2 = inter.permute((1, 0)).unwrap().relu();
            // println!("Intermediate: {:?}", inter);
            let c = wb + &inter2;
            let loss = mean_l1_loss(&c, target);
            loss
        }

        fn modelv5(
            input: &TensorBase<f32, Cpu>,  // [2, 3]
            target: &TensorBase<f32, Cpu> // [3, 2]
        ) -> TensorBase<f32, Cpu> {
            let loss = mean_l1_loss(&-input.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv6(
            wa: &TensorBase<f32, Cpu>,  
            input: &TensorBase<f32, Cpu>,  
            target: &TensorBase<f32, Cpu> 
        ) -> TensorBase<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.sigmoid().leaky_relu(1.); // grad should be identical even without leaky relu
            let loss = mean_l1_loss(&-temp2.sqrt(), &target.clone().transpose());
            loss
        }

        fn modelv7(
            wa: &TensorBase<f32, Cpu>,  
            input: &TensorBase<f32, Cpu>,  
            target: &TensorBase<f32, Cpu> 
        ) -> TensorBase<f32, Cpu> {
            let temp = input + wa;
            let temp2 = temp.leaky_relu(0.1).ln();
            let loss = mean_l1_loss(&-temp2, &target.clone().transpose());
            loss
        }

        grad::with(|ctx| {
            
            let a = Tensor::<f32>::scalar(1.);
            let mut b = Tensor::<f32>::ones((2, 2));
            let target = Tensor::<f32>::zeros((2, 2));

            let mut optim = SGD::<f32, Cpu>::new(1.);
            // optim.register_parameter(&a).unwrap();
            optim.register_parameters(vec![&mut b]).unwrap();
            
            let initial_loss = model(&a, &b, &target).item().unwrap();
            for _ in 0..10 {
                let loss = model(&a, &b, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model(&a, &b, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", a);

            let mut a = Tensor::<f32>::ones((2, 2));
            let mut b = Tensor::<f32>::ones((2, 2));
            let mut c = Tensor::<f32>::ones((2, 2));

            optim.register_parameter(&mut a).unwrap();
            optim.register_parameter(&mut b).unwrap();
            optim.register_parameter(&mut c).unwrap();

            let initial_loss = modelv2(&a, &b, &c, &target).item().unwrap();
            for _ in 0..10 {
                let loss = modelv2(&a, &b, &c, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv2(&a, &b, &c, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", a);

            let input = Tensor::<f32>::ones((2, 3));
            let mut wa = Tensor::<f32>::ones((2, 3));
            let mut wb = Tensor::<f32>::ones((3, 2));
            let target = Tensor::<f32>::zeros((3, 2));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut wb).unwrap();
            let initial_loss = modelv3(&input, &wa, &wb, &target).item().unwrap();
            for _ in 0..10 {
                let loss = modelv3(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv3(&input, &wa, &wb, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);
            println!("{:?}", wa);

            let input = Tensor::<f32>::ones((2, 3));
            let mut wa = Tensor::<f32>::ones((2, 3));
            let mut wb = Tensor::<f32>::ones((3, 2));
            let target = Tensor::<f32>::zeros((3, 2));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut wb).unwrap();
            let initial_loss = modelv4(&input, &wa, &wb, &target).item().unwrap();
            for _ in 0..10 {
                let loss = modelv4(&input, &wa, &wb, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv4(&input, &wa, &wb, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);
            println!("{:?}", wa);

            let mut input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((3, 2));
            optim.register_parameter(&mut input).unwrap();
            let initial_loss = modelv5(&input, &target).item().unwrap();
            for _ in 0..12 {
                let loss = modelv5(&input, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv5(&input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.5, 
                "Loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);

            println!("{:?}", input);

            let mut wa = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((3, 2));
            optim.register_parameter(&mut wa).unwrap();
            let initial_loss = modelv6(&wa, &input, &target).item().unwrap();
            for _ in 0..10 {
                let loss = modelv6(&wa, &input, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv6(&wa, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.001, 
                "Loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            let mut wa = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((3, 2));
            optim.register_parameter(&mut wa).unwrap();
            let initial_loss = modelv7(&wa, &input, &target).item().unwrap();
            for _ in 0..10 {
                let loss = modelv7(&wa, &input, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = modelv7(&wa, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.2, 
                "Loss should reduce by at least 0.2, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted add
            let mut wa = Tensor::<f32>::ones((1, 3));
            let mut input = Tensor::<f32>::ones((1, 2, 1));
            let target = Tensor::<f32>::zeros((1, 2, 3));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut input).unwrap();
            let initial_loss = {
                let inter = &input + &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input + &wa; // broadcasted add
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input + &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted sub
            let mut wa = Tensor::<f32>::ones((1, 3));
            let mut input = Tensor::<f32>::ones((1, 2, 1));
            let target = Tensor::<f32>::ones((1, 2, 3));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut input).unwrap();
            let initial_loss = {
                let inter = &input - &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input - &wa; // broadcasted sub
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input - &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted mul
            let mut wa = Tensor::<f32>::ones((1, 3));
            let mut input = Tensor::<f32>::ones((1, 2, 1));
            let target = Tensor::<f32>::zeros((1, 2, 3));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut input).unwrap();
            let initial_loss = {
                let inter = &input * &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input * &wa; // broadcasted mul
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input * &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // use model, but do a broadcasted div
            let mut wa = Tensor::<f32>::ones((1, 3));
            let mut input = Tensor::<f32>::ones((1, 2, 1));
            let target = Tensor::<f32>::zeros((1, 2, 3));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut input).unwrap();
            let initial_loss = {
                let inter = &input / &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            for _ in 0..10 {
                let inter = &input / &wa; // broadcasted div
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = &input / &wa;
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.1, 
                "Loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
            println!("{:?}", wa);

            // mamtmul
            let mut wa = Tensor::<f32>::ones((2, 3));
            let mut wb = Tensor::<f32>::ones((3, 4));
            let target = Tensor::<f32>::zeros((2, 4));
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut wb).unwrap();
            let initial_loss = {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                mean_l1_loss(&inter, &target).item().unwrap()
            };
            for _ in 0..1 {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                let loss = mean_l1_loss(&inter, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let inter = wa.matmul(&wb).expect("MatMul failed");
                mean_l1_loss(&inter, &target).item().unwrap()
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
            pub weight: TensorBase<T, B>,
            pub bias: TensorBase<T, B>,
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
                        .expect("Failed to create uniform tensor");
                    let bias = Tensor::<f32>::zeros((1, out_size));
                    layers.push(Layer { weight, bias });
                }
                Self { layers }
            }

            fn forward(&self, mut x: TensorBase<f32, Cpu>) -> TensorBase<f32, Cpu> {
                for (i, layer) in self.layers.iter().enumerate() {
                    x = x.matmul(&layer.weight).unwrap() + &layer.bias;
                    if i != self.layers.len() - 1 {
                        x = x.relu();
                    }
                }
                x.sigmoid()
            }

            fn register(&mut self, optim: &mut SGD<f32, Cpu>) {
                for layer in &mut self.layers {
                    optim.register_parameter(&mut layer.weight).unwrap();
                    optim.register_parameter(&mut layer.bias).unwrap();
                }
            }
        }

        grad::with(|ctx| {
            let mut model = DenseModel::new(5, 10, 2, 10);

            let input = Tensor::<f32>::ones((1, 5));
            let target = Tensor::<f32>::uniform((1, 2)).unwrap();
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            model.register(&mut optim);
            
            let initial_loss = {
                let output = model.forward(input.clone());
                mean_l1_loss(&output, &target).item().unwrap()
            };
            for _ in 0..100 {
                let output = model.forward(input.clone());
                let loss = mean_l1_loss(&output, &target);
                println!("Loss: {:?}", loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = {
                let output = model.forward(input.clone());
                mean_l1_loss(&output, &target).item().unwrap()
            };
            assert!(initial_loss - final_loss > 0.01, 
                "Dense model loss should reduce by at least 0.01, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_trig_and_hyperbolic() {
        // Test sin, cos, tan, sinh, cosh, tanh
        #[grad::when_enabled(ctx)]
        fn model_with_trig(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(input.op())));
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(wa.op())));
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(x.op())));
            let t = x.sin();
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(t.op())));
            let t2 = x.cos();
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(t2.op())));
            let t3 = t2.tanh();
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(t3.op())));
            let y = t + t3;
            // println!("{:?}", ctx.nodes.borrow().get(ctx.resolve_maybe_key(y.op())));
            mean_l1_loss(&y, target)
        }

        fn model_with_hyperbolic(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.sinh() + x.cosh();
            mean_l1_loss(&y, target)
        }

        grad::with(|ctx| {
            println!("\n=== Testing Trig Functions ===");
            let mut wa = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((2, 3));
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&mut wa).unwrap();
            
            let initial_loss = model_with_trig(&wa, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_with_trig(&wa, &input, &target);
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                println!("Iter {}: Loss = {:?}", i, loss.item());
                optim.step().unwrap();
            }
            let final_loss = model_with_trig(&wa, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.01, 
                "Trig loss should reduce by at least 0.01, initial: {}, final: {}", initial_loss, final_loss);

            // println!("\n=== Testing Hyperbolic Functions ===");
            // let mut wb = Tensor::<f32>::ones((2, 3));
            // optim.register_parameter(&mut wb).unwrap();
            
            // let initial_loss = model_with_hyperbolic(&wb, &input, &target).item().unwrap();
            // for i in 0..10 {
            //     let loss = model_with_hyperbolic(&wb, &input, &target);
            //     println!("Iter {}: Loss = {:?}", i, loss.item());
            //     ctx.backwards::<f32, Cpu>(&loss).unwrap();
            //     optim.step().unwrap();
            // }
            // let final_loss = model_with_hyperbolic(&wb, &input, &target).item().unwrap();
            // assert!(initial_loss - final_loss > 0.5, 
            //     "Hyperbolic loss should reduce by at least 0.5, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_exp_and_powers() {
        // Test exp, square, cube
        fn model_with_exp(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.exp();
            mean_l1_loss(&y, target)
        }

        fn model_with_powers(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.square() + x.cube();
            mean_l1_loss(&y, target)
        }

        grad::with(|ctx| {
            println!("\n=== Testing Exp ===");
            let mut wa = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::ones((2, 3));
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            optim.register_parameter(&mut wa).unwrap();
            
            let initial_loss = model_with_exp(&wa, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_with_exp(&wa, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_exp(&wa, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Exp loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Powers ===");
            let mut wb = Tensor::<f32>::ones((2, 3));
            optim.register_parameter(&mut wb).unwrap();
            
            let initial_loss = model_with_powers(&wb, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_with_powers(&wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_powers(&wb, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.1, 
                "Powers loss should reduce by at least 0.1, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_reciprocal_and_rsqrt() {
        // Test reciprocal and rsqrt
        fn model_with_reciprocal(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.reciprocal();
            mean_l1_loss(&y, target)
        }

        fn model_with_rsqrt(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.rsqrt();
            mean_l1_loss(&y, target)
        }

        grad::with(|ctx| {
            println!("\n=== Testing Reciprocal ===");
            let mut wa = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::ones((2, 3));
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&mut wa).unwrap();
            
            let initial_loss = model_with_reciprocal(&wa, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_with_reciprocal(&wa, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_reciprocal(&wa, &input, &target).item().unwrap();
            // Reciprocal has small gradients for x > 1, so expect smaller reduction
            assert!(initial_loss - final_loss > 0.001, 
                "Reciprocal loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Rsqrt ===");
            let mut wb = Tensor::<f32>::ones((2, 3));
            optim.register_parameter(&mut wb).unwrap();
            
            let initial_loss = model_with_rsqrt(&wb, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_with_rsqrt(&wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_with_rsqrt(&wb, &input, &target).item().unwrap();
            // Rsqrt also has small gradients for x > 1
            assert!(initial_loss - final_loss > 0.001, 
                "Rsqrt loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_combined_operations() {
        // Test combining multiple operations
        fn model_complex(
            wa: &TensorBase<f32, Cpu>,
            wb: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x1 = input + wa;
            let x2 = x1.square().tanh();  // square then tanh
            let x3 = x2 + wb;
            let x4 = x3.sinh().reciprocal();  // sinh then reciprocal
            mean_l1_loss(&x4, target)
        }

        fn model_chain(
            wa: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
            let x = input + wa;
            let y = x.sin().exp().rsqrt();  // chain: sin -> exp -> rsqrt
            mean_l1_loss(&y, target)
        }

        grad::with(|ctx| {
            println!("\n=== Testing Complex Model ===");
            let mut wa = Tensor::<f32>::ones((2, 3));
            let mut wb = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((2, 3));
            
            let mut optim = SGD::<f32, Cpu>::new(0.1);
            optim.register_parameter(&mut wa).unwrap();
            optim.register_parameter(&mut wb).unwrap();
            
            let initial_loss = model_complex(&wa, &wb, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_complex(&wa, &wb, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_complex(&wa, &wb, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.001, 
                "Complex model loss should reduce by at least 0.001, initial: {}, final: {}", initial_loss, final_loss);

            println!("\n=== Testing Chained Operations ===");
            let mut wc = Tensor::<f32>::ones((2, 3));
            optim.register_parameter(&mut wc).unwrap();
            
            let initial_loss = model_chain(&wc, &input, &target).item().unwrap();
            for i in 0..10 {
                let loss = model_chain(&wc, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_chain(&wc, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 0.0001, 
                "Chained operations loss should reduce by at least 0.0001, initial: {}, final: {}", initial_loss, final_loss);
        });
    }

    #[test]
    fn test_all_new_ops() {
        // Test all 8 new operations in one model
        fn model_all(
            w: &TensorBase<f32, Cpu>,
            input: &TensorBase<f32, Cpu>,
            target: &TensorBase<f32, Cpu>
        ) -> TensorBase<f32, Cpu> {
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

        grad::with(|ctx| {
            println!("\n=== Testing All 8 New Operations ===");
            let mut w = Tensor::<f32>::ones((2, 3));
            let input = Tensor::<f32>::ones((2, 3));
            let target = Tensor::<f32>::zeros((2, 3));
            
            let mut optim = SGD::<f32, Cpu>::new(0.01);
            optim.register_parameter(&mut w).unwrap();
            
            let initial_loss = model_all(&w, &input, &target).item().unwrap();
            for i in 0..15 {
                let loss = model_all(&w, &input, &target);
                println!("Iter {}: Loss = {:?}", i, loss.item());
                ctx.backwards::<f32, Cpu>(&loss).unwrap();
                optim.step().unwrap();
            }
            let final_loss = model_all(&w, &input, &target).item().unwrap();
            assert!(initial_loss - final_loss > 1.0, 
                "All ops loss should reduce by at least 1.0, initial: {}, final: {}", initial_loss, final_loss);
            
            println!("Final w: {:?}", w);
        });
    }
}