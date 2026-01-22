use slotmap::{new_key_type, SecondaryMap};

use crate::{backend::{Backend, BackendMatMul, cpu::Cpu}, core::{Shape, Strides, idx::Idx, primitives::{Grad, OpTensor, TensorBase}, tensor::TensorError, untyped::UntypedTensor, value::{TensorValue, Value, WeightValue}}};
#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;
use std::{any::{Any, TypeId}, cell::RefCell};
use std::collections::HashMap;

mod backwards;
pub mod optim;
// pub mod primitives;

pub use proc::when_enabled;

// struct NodeKey;

new_key_type! {
    pub(crate) struct NodeKey;
}

/// Each variant of a node holds parents and any tensors that need to be saved for backward.
#[derive(Debug)]
pub(crate) enum GradNode {
    // LEAF NODES
    Leaf( Grad ),
    None,
    // OPS
    BroadcastAdd { 
        left: NodeKey, 
        right: NodeKey, 
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastSub { 
        left: NodeKey, 
        right: NodeKey, 
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastMul { 
        left: NodeKey, 
        right: NodeKey, 
        lhs_input: Box<dyn UntypedTensor>,
        rhs_input: Box<dyn UntypedTensor>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastDiv { 
        left: NodeKey, 
        right: NodeKey, 
        lhs_input: Box<dyn UntypedTensor>,
        rhs_input_reciprocal: Box<dyn UntypedTensor>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    AddScalar { input: NodeKey },
    MulScalar { input: NodeKey, scalar: Value },
    DivScalar { input: NodeKey, scalar: Value },
    Abs { input: NodeKey, grad_map: Box<dyn UntypedTensor> },
    ReLU { input: NodeKey, grad_map: Box<dyn UntypedTensor> },
    Sigmoid { input: NodeKey, result: Box<dyn UntypedTensor> },
    Negate { input: NodeKey },
    Sqrt { input: NodeKey, output: Box<dyn UntypedTensor> },
    Ln { input: NodeKey, x_reciprocal: Box<dyn UntypedTensor> }, // store 1/x for backward
    Sin { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cos { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Tan { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Tanh { input: NodeKey, result: Box<dyn UntypedTensor> },
    Exp { input: NodeKey, result: Box<dyn UntypedTensor> },
    Square { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cube { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Reciprocal { input: NodeKey, result: Box<dyn UntypedTensor> },
    Rsqrt { input: NodeKey, result: Box<dyn UntypedTensor> },
    Sinh { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Cosh { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    ExpM1 { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    Ln1p { input: NodeKey, input_tensor: Box<dyn UntypedTensor> },
    MatMul {
        left: NodeKey,
        right: NodeKey,
        left_input: Box<dyn UntypedTensor>,
        right_input: Box<dyn UntypedTensor>,
    },
    // VIEW OPS
    Permute {
        input: NodeKey,
        dims: Idx
    },
    // LOSSES
    L1 { 
        input: NodeKey, 
        // it is likely that this is leaf; however, it is not always the case
        // consider siamese networks
        target: NodeKey,
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
    pub fn parents(&self) -> Vec<NodeKey> {
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
}

impl Default for GradContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GradContext {
    pub fn new() -> Self {
        Self { 
            nodes: RefCell::new(slotmap::SlotMap::with_key()),
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
        stack.push(StackState::Enter(root.op().expect("Root tensor has no associated grad node.")));

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
                                for parent in node.parents() {
                                    stack.push(StackState::Enter(parent));
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
        let root_node_key = root.op().expect("Root tensor has no associated grad node.");
        let nodes = self.nodes.borrow();
        let loss = nodes.get(root_node_key)
            .and_then(|n| n.loss().as_ref().cloned())
            .ok_or_else(|| TensorError::GradError("Root node does not contain a loss.".into()))?;
        let mut accumulations = HashMap::new();
        accumulations.insert(root_node_key, vec![loss.typed::<T, B>().expect("Loss is the wrong datatype.").clone()]);

        // println!("{:?}", node_order
        //     .iter()
        //     .map(|k| self.nodes.borrow().get(*k).unwrap().clone())
        //     .collect::<Vec<_>>()
        // );

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
            let upstreams = node.backwards(&dldy, self)?;

            // println!("upstreams: {:?}", upstreams);
            let parents = node.parents();
            for (parent, grad) in parents.into_iter().zip(upstreams.into_iter()) {
                accumulations.entry(parent).or_insert_with(Vec::new).push(grad);
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
}


pub fn with(
    f: impl FnOnce(&GradContext)
){    
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
    GRAD_CONTEXT.with(|ctx_cell| {
        let ctx_map = ctx_cell.borrow();
        ctx_map.is_some()
    })
}

/// Runs the provided closure if the gradient context is enabled for the given backend.
pub fn when_enabled<R>(
    f: impl FnOnce(&GradContext) -> R
) -> Option<R>{
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