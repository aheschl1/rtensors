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


