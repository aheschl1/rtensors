use slotmap::{new_key_type, SecondaryMap};

use crate::{backend::{cpu::Cpu, Backend, BackendMatMul}, core::{idx::Idx, primitives::TensorBase, tensor::TensorError, value::{TensorValue, WeightValue}, Shape, Strides}, grad::primitives::{GradTensor, GradTensorRef}};
#[cfg(feature = "cuda")]
use crate::backend::cuda::Cuda;
use std::{any::{Any, TypeId}, cell::RefCell};
use std::collections::HashMap;

mod backwards;
pub mod optim;
pub mod primitives;

pub use proc::when_enabled;

// struct NodeKey;

new_key_type! {
    pub(crate) struct NodeKey;
}

/// Each variant of a node holds parents and any tensors that need to be saved for backward.
#[derive(Debug, Clone)]
pub(crate) enum GradNode<T: TensorValue, B: Backend> {
    // LEAF NODES
    Leaf( GradTensorRef<T, B> ),
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
        lhs_input: TensorBase<T, B>,
        rhs_input: TensorBase<T, B>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    BroadcastDiv { 
        left: NodeKey, 
        right: NodeKey, 
        lhs_input: TensorBase<T, B>,
        rhs_input_reciprocal: TensorBase<T, B>,
        lhs_strides: Strides, // strides so we know when to reduce
        rhs_strides: Strides, 
        lhs_shape: Shape,  // shapes so that we know when to squeeze
        rhs_shape: Shape 
    },
    AddScalar { input: NodeKey },
    MulScalar { input: NodeKey, scalar: T },
    DivScalar { input: NodeKey, scalar: T },
    Abs { input: NodeKey, grad_map: TensorBase<T, B> },
    ReLU { input: NodeKey, grad_map: TensorBase<T, B> },
    Sigmoid { input: NodeKey, result: TensorBase<T, B> },
    Negate { input: NodeKey },
    Sqrt { input: NodeKey, output: TensorBase<T, B> },
    Ln { input: NodeKey, x_reciprocal: TensorBase<T, B> }, // store 1/x for backward
    Sin { input: NodeKey, input_tensor: TensorBase<T, B> },
    Cos { input: NodeKey, input_tensor: TensorBase<T, B> },
    Tan { input: NodeKey, input_tensor: TensorBase<T, B> },
    Tanh { input: NodeKey, result: TensorBase<T, B> },
    Exp { input: NodeKey, result: TensorBase<T, B> },
    Square { input: NodeKey, input_tensor: TensorBase<T, B> },
    Cube { input: NodeKey, input_tensor: TensorBase<T, B> },
    Reciprocal { input: NodeKey, result: TensorBase<T, B> },
    Rsqrt { input: NodeKey, result: TensorBase<T, B> },
    Sinh { input: NodeKey, input_tensor: TensorBase<T, B> },
    Cosh { input: NodeKey, input_tensor: TensorBase<T, B> },
    ExpM1 { input: NodeKey, input_tensor: TensorBase<T, B> },
    Ln1p { input: NodeKey, input_tensor: TensorBase<T, B> },
    MatMul {
        left: NodeKey,
        right: NodeKey,
        left_input: TensorBase<T, B>,
        right_input: TensorBase<T, B>,
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
        grad_map: TensorBase<T, B>, // where is the diff greater than zero
        loss: GradTensorRef<T, B>,
    },
}

impl<T: WeightValue, B: Backend> GradNode<T, B> {
    pub fn is_leaf(&self) -> bool {
        matches!(self, GradNode::Leaf(..))
    }

    pub fn leaf(inner: GradTensorRef<T, B>) -> Self {
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

    fn backwards(&self, upstream: &TensorBase<T, B>, _ctx: &GradContext<T, B>) -> Result<Vec<TensorBase<T, B>>, TensorError> 
    where
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
}

pub struct GradContext<T: TensorValue, B: Backend> {
    // tape: Vec<NodeKey>, // holds references to all inner tensors that require gradients
    pub(crate) nodes: RefCell<slotmap::SlotMap<NodeKey, GradNode<T, B>>>,
}

impl<T: WeightValue, B: Backend> Default for GradContext<T, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: WeightValue, B: Backend> GradContext<T, B> {
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
    pub(crate) fn make_leaf(
        &self,
        inner: GradTensorRef<T, B>,
    ) -> GradTensor<T, B> {
        let node = GradNode::leaf(inner.clone());
        self.attach(inner, node)
    }

    #[inline]
    pub(crate) fn attach(
        &self,
        inner: GradTensorRef<T, B>,
        op: GradNode<T, B>,
    ) -> GradTensor<T, B> {
        let node_id = self.nodes.borrow_mut().insert(op);
        GradTensor {
            inner,
            node: node_id,
        }
    }

    pub fn backwards(&self, root: &GradTensor<T, B>) -> Result<(), TensorError> 
    where 
        B: BackendMatMul<T>
    {
        // holds nodes to visit along with their upstream gradients
        // topo sort, because concider a graph like A->C<-B<-D where BFS should visit C too early
        let mut stack = Vec::new();
        let mut marks = SecondaryMap::new();
        let mut node_order = Vec::new();
        stack.push(StackState::Enter(root.node));

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
        let mut accumulations = HashMap::new();
        accumulations.insert(root.node, vec![root.borrow().tensor.clone()]);

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
                        Some(grad)
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

impl<T: TensorValue, B: Backend> std::fmt::Debug for GradContext<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GradContext {{ nodes_len: {} }}", self.nodes.borrow().len())
    }
}

thread_local! {
    static GRAD_CONTEXT_CPU: std::cell::RefCell<HashMap<TypeId, Box<dyn Any>>> = std::cell::RefCell::new(HashMap::new());
    #[cfg(feature = "cuda")]
    static GRAD_CONTEXT_CUDA: std::cell::RefCell<HashMap<TypeId, Box<dyn Any>>> = std::cell::RefCell::new(HashMap::new());
}


pub fn with<T: WeightValue, B: Backend>(
    f: impl FnOnce(&GradContext<T, B>)
){
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let mut ctx_map = ctx_cell.borrow_mut();
            let _ = ctx_map
                .entry(type_id)
                .or_insert_with(|| Box::new(GradContext::<T, Cpu>::new()));
            drop(ctx_map);
            let ctx = ctx_cell.borrow();
            let ctx = ctx.get(&type_id).unwrap();
            
            // SAFETY: We know the TypeId matches T, and we've verified B is Cpu
            let ctx = ctx.downcast_ref::<GradContext<T, Cpu>>().unwrap();
            let ctx = unsafe { &*(ctx as *const GradContext<T, Cpu> as *const GradContext<T, B>) };
            f(ctx);
        });
        return;
    }
    
    #[cfg(feature = "cuda")]
    {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
            GRAD_CONTEXT_CUDA.with(|ctx_cell| {
                let mut ctx_map = ctx_cell.borrow_mut();
                let _ = ctx_map
                    .entry(type_id)
                    .or_insert_with(|| Box::new(GradContext::<T, Cuda>::new()));
                drop(ctx_map);
                let ctx = ctx_cell.borrow();
                let ctx = ctx.get(&type_id).unwrap();
                // SAFETY: We know the TypeId matches T, and we've verified B is Cuda
                let ctx = ctx.downcast_ref::<GradContext<T, Cuda>>().unwrap();
                let ctx = unsafe { &*(ctx as *const GradContext<T, Cuda> as *const GradContext<T, B>) };
                f(ctx);
            });
            return;
        }
    }
    
    panic!("Unsupported backend for GradContext");
}

#[inline]
pub fn is_enabled<T: TensorValue, B: Backend>() -> bool {
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        return GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            ctx_map.contains_key(&type_id)
        });
    }
    
    #[cfg(feature = "cuda")]
    {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
            return GRAD_CONTEXT_CUDA.with(|ctx_cell| {
                let ctx_map = ctx_cell.borrow();
                ctx_map.contains_key(&type_id)
            });
        }
    }
    
    panic!("Unsupported backend for GradContext");
}

/// Runs the provided closure if the gradient context is enabled for the given backend.
pub fn when_enabled<T: TensorValue, B: Backend, R>(
    f: impl FnOnce(&GradContext<T, B>) -> R
) -> Option<R>{
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        return GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            if let Some(ctx_box) = ctx_map.get(&type_id) {
                // SAFETY: We know the TypeId matches T, and we've verified B is Cpu
                if let Some(ctx) = ctx_box.downcast_ref::<GradContext<T, Cpu>>() {
                    let ctx = unsafe { &*(ctx as *const GradContext<T, Cpu> as *const GradContext<T, B>) };
                    Some(f(ctx))
                }else{
                    None
                }
            } else {
                None
            }
        });
    }
    
    #[cfg(feature = "cuda")]
    {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
            return GRAD_CONTEXT_CUDA.with(|ctx_cell| {
                let ctx_map = ctx_cell.borrow();
                if let Some(ctx_box) = ctx_map.get(&type_id) {
                    // SAFETY: We know the TypeId matches T, and we've verified B is Cuda
                    if let Some(ctx) = ctx_box.downcast_ref::<GradContext<T, Cuda>>() {
                        let ctx = unsafe { &*(ctx as *const GradContext<T, Cuda> as *const GradContext<T, B>) };
                        Some(f(ctx))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
        }
    }
    
    panic!("Unsupported backend for GradContext");
}

pub fn when_disabled<T: TensorValue, B: Backend>(
    f: impl FnOnce()
) {
    let type_id = TypeId::of::<T>();
    
    if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cpu>() {
        GRAD_CONTEXT_CPU.with(|ctx_cell| {
            let ctx_map = ctx_cell.borrow();
            if !ctx_map.contains_key(&type_id) {
                f();
            }
        });
        return;
    }
    
    #[cfg(feature = "cuda")]
    {
        if std::any::TypeId::of::<B>() == std::any::TypeId::of::<Cuda>() {
            GRAD_CONTEXT_CUDA.with(|ctx_cell| {
                let ctx_map = ctx_cell.borrow();
                if !ctx_map.contains_key(&type_id) {
                    f();
                }
            });
            return;
        }
    }
    
    panic!("Unsupported backend for GradContext");
}

/// same as anabled but warns if not enabled
pub fn enabled_or_warn<T: WeightValue, B: Backend>(
    f: impl FnOnce(&GradContext<T, B>),
    msg: &str,
) {
    when_disabled::<T, B>(|| {
        tracing::warn!("{}", msg);
    });
    with::<T, B>(f);
}