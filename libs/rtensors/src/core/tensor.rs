use crate::{backend::Backend, core::{Dim, MetaTensor, MetaTensorView, Shape, Strides, TensorView, TensorViewMut, idx::Idx, meta::is_contiguous_relaxed, primitives::{DeviceType, OpTensor, TensorBase}, value::{TensorValue, WeightValue}}, grad::{self, GradNode, NodeKey}, ops::linalg::PaddingType};
use super::slice::{Slice, compute_sliced_parameters};
use thiserror::Error;

#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

/// Errors that can occur during tensor operations.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum TensorError {
    #[error("index out of bounds {0}")]
    IdxOutOfBounds(String),

    #[error("wrong number of dimensions {0}")]
    WrongDims(String),

    #[error("invalid tensor shape {0}")]
    InvalidShape(String),

    #[error("invalid dimension {0}")]
    InvalidDim(String),

    #[error("size mismatch between tensors {0}")]
    SizeMismatch(String),

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("broadcast error: {0}")]
    BroadcastError(String),

    #[error("tensor is not contiguous {0}")]
    ContiguityError(String),

    #[error("operation not supported: {0}")]
    UnsupportedOperation(String),

    #[error("conversion error: {0}")]
    ConversionError(String),

    #[cfg(feature = "cuda")]
    #[error("cuda error: {0}")]
    CudaError(String),

    
    #[error("gradient error: {0}")]
    GradError(String),

    #[cfg(feature = "remote")]
    #[error("connection not open: {0}")]
    RemoteError(String),
}

pub(crate) mod seal {
    use crate::{backend::Backend, core::{primitives::{TensorBase}, value::TensorValue, TensorView, TensorViewMut}};

    pub(crate) trait Sealed {}
    impl<T: TensorValue, B: Backend> Sealed for TensorBase<T, B> {}
    impl<T: TensorValue, B: Backend> Sealed for &TensorBase<T, B> {}
    impl<T: TensorValue, B: Backend> Sealed for TensorView<'_, T, B> {}
    impl<T: TensorValue, B: Backend> Sealed for TensorViewMut<'_, T, B> {}
}

/// Provides immutable view access to tensor data.
pub trait AsView<T: TensorValue, B: Backend> : OpTensor{
    /// Returns the device type where this tensor resides.
    fn device(&self) -> DeviceType {
        B::device_type()
    }
    
    /// Returns an immutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view(&self) -> TensorView<'_, T, B>;
    
    /// Returns a view with a different shape, collapsing dimensions as needed.
    /// The tensor must be contiguous.
    fn view_as(&self, shape: impl Into<Shape>) -> Result<TensorView<'_, T, B>, TensorError>;
}

/// Provides mutable view access to tensor data.
pub trait AsViewMut<T: TensorValue, B: Backend> : AsView<T, B> {
    /// Returns a mutable view over the tensor data, sharing the same
    /// underlying buffer and metadata (shape/stride/offset) without copying.
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B>;
    
    /// Returns a mutable view with a different shape, collapsing dimensions as needed.
    /// The tensor must be contiguous.
    fn view_as_mut(&'_ mut self, shape: impl Into<Shape>) -> Result<TensorViewMut<'_, T, B>, TensorError>;
}

/// Converts tensor views or references to owned tensors.
pub trait AsTensor<T: TensorValue, B: Backend> {
    /// Converts to an owned tensor, copying data if necessary.
    fn owned(&self) -> TensorBase<T, B>;
    
    /// Ensures the tensor has a contiguous memory layout, copying if needed.
    fn contiguous(&self) -> TensorBase<T, B>;

    /// Adds padding around tensor
    fn pad(&self, padding: impl Into<Shape>, padding_type: &PaddingType) -> Result<TensorBase<T, B>, TensorError>;

    /// Reshapes a tensor and makes contiguous
    fn reshape(&self, shape: impl Into<Shape>) -> Result<TensorBase<T, B>, TensorError>;
}

impl<T: TensorValue, B: Backend> AsView<T, B> for TensorBase<T, B> {
    fn view(&self) -> TensorView<'_, T, B> {
        let mut v = TensorView::<T, B>::from_parts(
            &self.buf, 
            &self.backend, 
            self.meta.clone(),
            None
        );
        v.op = self.op.clone(); // keeps the same operation node
        v
    }
    
    /// Logical reinterpretation of a contiguous memory layout.
    fn view_as(&self, shape: impl Into<Shape>) -> Result<TensorView<'_, T, B>, TensorError> {
        view_as_inner(self, shape.into())
    }
} 

impl<T: TensorValue, B: Backend> AsView<T, B> for &TensorBase<T, B> {
    fn view(&self) -> TensorView<'_, T, B> {
        let mut v = TensorView::<T, B>::from_parts(
            &self.buf, 
            &self.backend, 
            self.meta.clone(),
            None
        );
        v.op = self.op.clone();
        v
    }

    fn view_as(&self, shape: impl Into<Shape>) -> Result<TensorView<'_, T, B>, TensorError> {
        view_as_inner(self, shape.into())
    }
} 

impl<T: TensorValue, B: Backend> AsViewMut<T, B> for TensorBase<T, B> {
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> {
        let mut v = TensorViewMut::<T, B>::from_parts(
            &mut self.buf, 
            &self.backend, 
            self.meta.clone(),
            None
        );
        v.op = self.op.clone();
        v
    }

    fn view_as_mut(&'_ mut self, shape: impl Into<Shape>) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        view_as_mut_inner(self, shape.into())
    }
}

impl<T: TensorValue, B: Backend> AsView<T, B> for TensorView<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        let mut v = TensorView::from_parts(
            self.buf, 
            self.backend,
            self.meta.clone(),
            None
        );
        v.op = self.op.clone();
        v
    }

    fn view_as(&self, shape: impl Into<Shape>) -> Result<TensorView<'_, T, B>, TensorError> {
        view_as_inner(self, shape.into())
    }

}

impl<T: TensorValue, B: Backend> AsView<T, B> for TensorViewMut<'_, T, B> 
{
    fn view(&self) -> TensorView<'_, T, B> {
        let mut v = TensorView::from_parts(
            self.buf,
            self.backend,
            self.meta.clone(),
            None
        );
        v.op = self.op.clone();
        v
    }

    fn view_as(&self, shape: impl Into<Shape>) -> Result<TensorView<'_, T, B>, TensorError> {
        view_as_inner(self, shape.into())
    }
}

impl<T: TensorValue, B: Backend> AsViewMut<T, B> for TensorViewMut<'_, T, B> 
{
    fn view_mut(&'_ mut self) -> TensorViewMut<'_, T, B> {
        let mut v = TensorViewMut::from_parts(
            self.buf,
            self.backend,
            self.meta.clone(),
            None
        );
        v.op = self.op.clone();
        v
    }

    fn view_as_mut(&'_ mut self, shape: impl Into<Shape>) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        view_as_mut_inner(self, shape.into())
    }
}


impl <T: TensorValue, B: Backend> AsTensor<T, B> for TensorBase<T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        self.clone()
    }
    
    fn contiguous(&self) -> TensorBase<T, B> {
        if self.meta.is_contiguous() {
            // fast path: already contiguous
            self.clone()
        } else {
            view_to_contiguous(&self.meta, &self.buf, &self.backend, self.op()).unwrap()
        }
    }

    #[grad::incomplete]
    fn pad(&self, padding: impl Into<Shape>, padding_type: &PaddingType) -> Result<TensorBase<T, B>, TensorError> {
        pad_inner(
            self,
            padding,
            padding_type
        )
    }

    fn reshape(&self, shape: impl Into<Shape>) -> Result<TensorBase<T, B>, TensorError> {
        let mut contiguous = self.contiguous();
        let contig_view = contiguous.view_as(shape)?;
        contiguous.meta = contig_view.meta.clone();
        Ok(contiguous)
    }
}

impl<'a, T: TensorValue, B: Backend> AsTensor<T, B> for TensorView<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_contiguous(&self.meta, self.buf, self.backend, self.op()).unwrap()
    }

    fn contiguous(&self) -> TensorBase<T, B> {
        self.owned()
    }

    #[grad::incomplete]
    fn pad(&self, padding: impl Into<Shape>, padding_type: &PaddingType) -> Result<TensorBase<T, B>, TensorError> {
        pad_inner(
            self,
            padding,
            padding_type
        )
    }

    fn reshape(&self, shape: impl Into<Shape>) -> Result<TensorBase<T, B>, TensorError> {
        let mut contiguous = self.contiguous();
        let contig_view = contiguous.view_as(shape)?;
        contiguous.meta = contig_view.meta.clone();
        Ok(contiguous)
    }
}

impl<'a, T: TensorValue, B: Backend> AsTensor<T, B> for TensorViewMut<'a, T, B> {
    fn owned(&self) -> TensorBase<T, B> {
        view_to_contiguous(&self.meta, self.buf, self.backend, self.op()).unwrap()
    }

    fn contiguous(&self) -> TensorBase<T, B> {
        self.owned()
    }

    #[grad::incomplete]
    fn pad(&self, padding: impl Into<Shape>, padding_type: &PaddingType) -> Result<TensorBase<T, B>, TensorError> {
        pad_inner(
            self,
            padding,
            padding_type
        )
    }

    fn reshape(&self, shape: impl Into<Shape>) -> Result<TensorBase<T, B>, TensorError> {
        let mut contiguous = self.contiguous();
        let contig_view = contiguous.view_as(shape)?;
        contiguous.meta = contig_view.meta.clone();
        Ok(contiguous)
    }
}

#[inline]
fn view_to_contiguous<T: TensorValue, B: Backend>(meta: &MetaTensor, raw: &B::Buf<T>, backend: &B, op: Option<NodeKey>) -> Result<TensorBase<T, B>, TensorError> {
    let size = meta.size();
    let new_backend = backend.clone();
    let mut new_buf = new_backend.alloc(size)?;
    
    // Copy element by element from the view to the new contiguous buffer
    // The view might be non-contiguous (e.g., a column slice), so we iterate
    // through all logical positions and copy to sequential positions in the new buffer
    // for (new_idx, old_offset) in meta.iter_offsets().enumerate() {
    //     new_backend.copy_range_within(&mut new_buf, raw, new_idx, old_offset, 1)?
    // }
    
    let mut new_idx = 0;
    for range in meta.iter_forward_contiguous_ranges() {
        let len = range.end - range.start;
        new_backend.copy_range_within(&mut new_buf, raw, new_idx, range.start, len)?;
        new_idx += len;
    }


    // Create a new tensor with contiguous layout (standard row-major stride)
    let new_shape = meta.shape().clone();
    let new_stride = super::shape_to_stride(&new_shape);
    let new_meta = MetaTensor::new(new_shape, new_stride, 0);
    
    Ok(TensorBase::from_parts(new_backend, new_buf, new_meta, op))
}

#[inline]
fn view_as_inner<T: TensorValue, B: Backend>(
    tensor: &impl AsView<T, B>,
    shape: Shape
) -> Result<TensorView<'_, T, B>, TensorError> {
    let mut tensor: TensorView<'_, T, B> = tensor.view();

    if !is_contiguous_relaxed(&tensor.meta.shape, &tensor.meta.strides){
        return Err(TensorError::ContiguityError("Cannot view_as non contiguous tensor".to_string()));
    }
    if shape.size() != tensor.meta.size() {
        return Err(TensorError::InvalidShape(format!("Invalid size {} for initial shape {}", shape.size(), tensor.meta.size())));
    }

    grad::when_enabled(|ctx| {
        let node = GradNode::Reshape {
            input: tensor.op(),
            original_shape: tensor.shape().clone(),
        };
        ctx.attach(&tensor, node);
    });

    // correct element count, one subspace.
    // so, we can just create new meta
    let new_stride = super::shape_to_stride(&shape);
    let new_meta = MetaTensor::new(shape, new_stride, tensor.meta.offset());
    tensor.meta = new_meta;
    Ok(tensor)
}

#[inline]
fn view_as_mut_inner<T: TensorValue, B: Backend>(
    tensor: &mut impl AsViewMut<T, B>,
    shape: Shape
) -> Result<TensorViewMut<'_, T, B>, TensorError> {
    let mut tensor: TensorViewMut<'_, T, B> = tensor.view_mut();

    if !is_contiguous_relaxed(&tensor.meta.shape, &tensor.meta.strides){
        return Err(TensorError::ContiguityError("Cannot view_as non contiguous tensor".to_string()));
    }
    if shape.size() != tensor.meta.size() {
        return Err(TensorError::InvalidShape(format!("Invalid size {} for initial shape {}", shape.size(), tensor.meta.size())));
    }
    grad::when_enabled(|ctx| {
        let node = GradNode::Reshape {
            input: tensor.op(),
            original_shape: tensor.shape().clone(),
        };
        ctx.attach(&tensor, node);
    });
    // correct element count, one subspace.
    // so, we can just create new meta
    let new_stride = super::shape_to_stride(&shape);
    let new_meta = MetaTensor::new(shape, new_stride, tensor.meta.offset());
    tensor.meta = new_meta;
    Ok(tensor)
}

#[inline]
// TODO optimize padding operation
fn pad_inner<T: TensorValue, B: Backend>(
    tensor: &impl AsView<T, B>,
    padding: impl Into<Shape>,
    padding_type: &PaddingType
) -> Result<TensorBase<T, B>, TensorError> {
    let padding: Shape = padding.into();
    let tensor: TensorView<'_, T, B> = tensor.view();

    if padding.len() != tensor.rank() {
        return Err(TensorError::WrongDims("Padding rank must match tensor rank".to_string()));
    }
    let output_shape: Vec<usize> = tensor.shape().iter().zip(padding.iter()).map(|(dim_size, pad)| dim_size + 2 * pad).collect();
    let output_shape = Shape::from(output_shape);
    let mut output_tensor = match padding_type {
        PaddingType::Zeros => {
            TensorBase::<T, B>::zeros(output_shape)
        }
    };

    // temporary go through each input coordinate and copy to output
    for input_coord in tensor.meta.iter_coords() {
        let output_coord: Vec<Dim> = input_coord.iter().zip(padding.iter()).map(|(c, p)| c + p).collect();
        let input_offset = tensor.meta.idx_to_offset(&input_coord);
        let output_offset = output_tensor.meta.idx_to_offset(&output_coord);
        tensor.backend.copy_range_within(&mut output_tensor.buf, tensor.buf, output_offset, input_offset, 1)?;
    }

    Ok(output_tensor)   
}

/// Provides read access to tensor elements and slicing operations.
pub trait TensorAccess<T: TensorValue, B: Backend>: Sized {
    /// Get element at given index.
    /// 
    /// # Examples
    /// ```ignore
    /// let value = tensor.get((0, 1)).unwrap();
    /// let value = tensor.get(coord![2, 3]).unwrap();
    /// ```
    fn get<I: Into<Idx>>(&self, idx: I) -> Result<T, TensorError>;

    /// Get the single element from a scalar tensor (rank 0).
    fn item(&self) -> Result<T, TensorError> {
        self.get(Idx::Item)
    }
    
    /// Create a slice/view of the tensor along a specific dimension.
    /// 
    /// # Examples
    /// ```ignore
    /// let slice = tensor.slice(0, 2..5).unwrap();  // rows 2-4
    /// let slice = tensor.slice(1, 3).unwrap();     // column 3
    /// ```
    fn slice<S: Into<Slice>>(&self, dim: Dim, idx: S) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized;
    
    /// Take a slice at a specific index along a dimension.
    fn slice_at(&self, dim: Dim, at: usize) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized{
        self.slice(dim, at)
    }

    /// Permute the dimensions of the tensor.
    /// 
    /// # Examples
    /// ```ignore
    /// let permuted = tensor.permute(vec![2, 0, 1]).unwrap();
    /// ```
    fn permute(&self, dims: impl Into<Idx>) -> Result<TensorView<'_, T, B>, TensorError>;
    
    /// Transpose all dimensions (reverse dimension order).
    fn transpose(&self) -> TensorView<'_, T, B>;
    
    /// Add a dimension of size 1 at the specified position.
    fn unsqueeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError>;
    
    /// Add a dimension of size 1 at the beginning.
    fn unsqueeze(&self) -> TensorView<'_, T, B> {
        unsafe{self.unsqueeze_at(0).unwrap_unchecked()}
    }
    
    /// Remove a dimension of size 1 at the specified position.
    fn squeeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError>;
    
    /// Remove all dimensions of size 1.
    fn squeeze(&self) -> TensorView<'_, T, B>;

    // fn squeeze_in_place(&self);
}

/// Provides mutable access to tensor elements and slicing operations.
pub trait TensorAccessMut<T: TensorValue, B: Backend>: TensorAccess<T, B> {
    /// Create a mutable slice/view of the tensor along a specific dimension.
    fn slice_mut<S: Into<Slice>>(&mut self, dim: Dim, idx: S) -> Result<TensorViewMut<'_, T, B>, TensorError> where Self: Sized;
    
    /// Sets a value at given index.
    /// 
    /// # Examples
    /// ```ignore
    /// tensor.set((0, 1), 42.0).unwrap();
    /// tensor.set(coord![2, 3], 1.5).unwrap();
    /// ```
    fn set<I: Into<Idx>>(&mut self, idx: I, value: T) -> Result<(), TensorError>;
    
    /// Take a mutable slice at given index.
    fn slice_at_mut(&mut self, dim: Dim, idx: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> where Self: Sized{
        self.slice_mut(dim, idx)
    }

    /// Permute the dimensions of the tensor (mutable).
    fn permute_mut(&mut self, dims: impl Into<Idx>) -> Result<TensorViewMut<'_, T, B>, TensorError> ;
    
    /// Transpose all dimensions (mutable).
    fn transpose_mut(&mut self) -> TensorViewMut<'_, T, B>;
    
    /// Add a dimension of size 1 at the specified position (mutable).
    fn unsqueeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError>;
    
    /// Add a dimension of size 1 at the beginning (mutable).
    fn unsqueeze_mut(&mut self) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        self.unsqueeze_at_mut(0)
    }
    
    /// Remove a dimension of size 1 at the specified position (mutable).
    fn squeeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError>;
    
    /// Remove all dimensions of size 1 (mutable).
    fn squeeze_mut(&mut self) -> TensorViewMut<'_, T, B>;


}

impl<T: TensorValue, B: Backend, V> TensorAccess<T, B> for V
where B: Backend, V: AsView<T, B> + seal::Sealed
{
    /// Returns a reference to the element at a logical index, converting
    /// coordinates into a buffer position via stride and offset.
    ///
    /// Errors
    /// - `WrongDims` if the index rank doesn't match the tensor rank.
    /// - `IdxOutOfBounds` if the computed buffer index is outside the backing slice.
    fn get<I: Into<Idx>>(&self, idx: I) -> Result<T, TensorError> {
        let view = self.view();
        let idx = logical_to_buffer_idx(&idx.into(), view.meta.strides(), view.meta.offset())?;
        view.backend.read(view.buf, idx)
    }

    /// Creates a new immutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    #[grad::incomplete]
    fn slice<S: Into<Slice>>(&self, dim: Dim, idx: S) -> Result<TensorView<'_, T, B>, TensorError> where Self: Sized {
        let view = self.view();
        let (new_shape, new_stride, offset) = compute_sliced_parameters(
            view.meta.shape(), 
            view.meta.strides(), 
            view.meta.offset(),
            dim, 
            idx
        )?;
        
        let v = TensorView::from_parts(
            view.buf, 
            view.backend, 
            MetaTensor::new(new_shape, new_stride, offset),
            view.op() // TODO this should be a special slice op
        );
        Ok(v)
    }

    fn permute(&self, dims: impl Into<Idx>) -> Result<TensorView<'_, T, B>, TensorError> {
        let dims = dims.into();
        let input_op = self.view().op();
        let mut view = self.view();
        let (new_shape, new_stride) = compute_permuted_parameters(
            view.meta.shape(),
            view.meta.strides(),
            &dims
        )?;

        view.meta.shape = new_shape;
        view.meta.strides = new_stride;
        
        attach_permute_grad::<T, B>(&view, input_op, &dims);
        Ok(view)
    }
    
    /// permute all dims
    fn transpose(&self) -> TensorView<'_, T, B> {
        let rank = self.view().meta.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        unsafe { self.permute(dims).unwrap_unchecked() }
    }

    fn unsqueeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError> {
        let view = self.view();
        let (new_shape, new_strides) = compute_unsqueezed_parameters(
            view.meta.shape(),
            view.meta.strides(),
            dim
        )?;

        let res = TensorView::from_parts(
            view.buf, 
            view.backend, 
            MetaTensor::new(new_shape, new_strides, view.meta.offset()),
            view.op()
        );
        attach_unsqueeze_grad(&res, dim);
        Ok(res)
    }
    
    /// removes dimension at given dim, if its size is 1
    fn squeeze_at(&self, dim: Dim) -> Result<TensorView<'_, T, B>, TensorError> {
        let mut view = self.view();
        let original_shape = view.shape().clone();
        let (new_shape, new_stride) = compute_squeezed_parameters(view.shape(), view.strides(), Some(dim))?;
        view.meta.shape = new_shape;
        view.meta.strides = new_stride;
        attach_squeeze_grad(&view, original_shape);
        Ok(view)
    }

    fn squeeze(&self) -> TensorView<'_, T, B> {
        let mut res = self.view();
        let original_shape = res.shape().clone();
        let (new_shape, new_strides) = unsafe { compute_squeezed_parameters(res.shape(), res.strides(), None).unwrap_unchecked() };
        res.meta.shape = new_shape;
        res.meta.strides = new_strides;
        attach_squeeze_grad(&res, original_shape);
        res
    }

   
}

impl<T: TensorValue, B: Backend, V> TensorAccessMut<T, B> for V
where V: AsViewMut<T, B> + seal::Sealed
{
    /// Creates a new mutable view by fixing `dim` to `idx`, effectively
    /// removing that dimension and adjusting shape/stride/offset accordingly.
    ///
    /// Errors
    /// - `InvalidDim` if `dim` is out of range.
    /// - `IdxOutOfBounds` if `idx` exceeds the size of `dim`.
    #[grad::incomplete]
    fn slice_mut<S: Into<Slice>>(&mut self, dim: Dim, idx: S) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let view = self.view_mut();
        let (new_shape, new_stride, offset) =
            compute_sliced_parameters(view.meta.shape(), view.meta.strides(), view.meta.offset(), dim, idx)?;
    
        Ok(TensorViewMut::from_parts(
            view.buf, 
            view.backend, 
            MetaTensor::new(new_shape, new_stride, offset),
            view.op() 
        ))// TODO this should be a special slice op
    }
    
    fn set<I: Into<Idx>>(&mut self, idx: I, value: T) -> Result<(), TensorError> {
        let view = self.view_mut();
        grad::when_enabled(|ctx| {
            if let Some(k) = view.op() {
                let nodes = ctx.nodes.borrow();
                let op = nodes.get(k);
                if let Some(op) = op {
                    if let GradNode::Leaf(_) = op {
                        panic!("Cannot set value of a leaf tensor that requires grad");
                    }
                }
            }
        });
        let idx = idx.into();
        let buf_idx = logical_to_buffer_idx(&idx, view.meta.strides(), view.meta.offset())?;
        view.backend.write(view.buf, buf_idx, value)
    }

    fn permute_mut(&mut self, dims: impl Into<Idx>) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let dims = dims.into();
        let input_op = self.view().op();
        let mut view = self.view_mut();
        let (new_shape, new_stride) = compute_permuted_parameters(
            view.meta.shape(),
            view.meta.strides(),
            &dims
        )?;

        view.meta.shape = new_shape;
        view.meta.strides = new_stride;
        
        attach_permute_grad::<T, B>(&view, input_op, &dims);

        Ok(view)
    }

    fn transpose_mut(&mut self) -> TensorViewMut<'_, T, B> {
        let rank = self.view().meta.rank();
        let dims: Idx = Idx::Coord((0..rank).rev().collect());
        unsafe { self.permute_mut(dims).unwrap_unchecked() }
    }

    fn unsqueeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let mut view = self.view_mut();
        let (new_shape, new_strides) = compute_unsqueezed_parameters(
            view.meta.shape(),
            view.meta.strides(),
            dim
        )?;

        view.meta.shape = new_shape;
        view.meta.strides = new_strides;
        attach_unsqueeze_grad(&view, dim);
        Ok(view)
    }

    fn squeeze_at_mut(&mut self, dim: Dim) -> Result<TensorViewMut<'_, T, B>, TensorError> {
        let mut view = self.view_mut();
        let original_shape = view.shape().clone();
        let (new_shape, new_stride) = compute_squeezed_parameters(view.shape(), view.strides(), Some(dim))?;
        view.meta.shape = new_shape;
        view.meta.strides = new_stride;
        attach_squeeze_grad(&view, original_shape);
        Ok(view)
    }

    fn squeeze_mut(&mut self) -> TensorViewMut<'_, T, B> {
        let mut res = self.view_mut();
        let original_shape = res.shape().clone();
        let (new_shape, new_strides) = unsafe { compute_squeezed_parameters(res.shape(), res.strides(), None).unwrap_unchecked() };
        res.meta.shape = new_shape;
        res.meta.strides = new_strides;
        attach_squeeze_grad(&res, original_shape);
        res
    }

}

fn scaled_uniform<T: WeightValue>(
    size: usize,
    scale: f32,
) -> Vec<T> {
    let mut rng = rand::rng();
    let mut v = Vec::with_capacity(size);

    for _ in 0..size {
        let x: f32 = rand::Rng::random_range(&mut rng, -scale..scale);
        v.push(T::from_f32(x));
    }

    v
}

fn fan_in_out(shape: &Shape) -> (usize, usize) {
    let slice = shape.as_slice();
    match slice {
        [fan_out, fan_in] => (*fan_in, *fan_out),
        _ => {
            // fallback: treat last dim as fan_in
            let fan_in = *slice.last().unwrap();
            let fan_out = shape.size() / fan_in;
            (fan_in, fan_out)
        }
    }
}


pub trait RandomTensor<T: TensorValue + rand::distr::uniform::SampleUniform, B: Backend> {
    fn uniform(shape: impl Into<Shape>) -> TensorBase<T, B>;
    fn xavier_uniform(shape: impl Into<Shape>) -> TensorBase<T, B>;
    fn kaiming_uniform(shape: impl Into<Shape>) -> TensorBase<T, B>;
    fn lecun_uniform(shape: impl Into<Shape>) -> TensorBase<T, B>;
}

impl<T: WeightValue, B: Backend> RandomTensor<T, B> for TensorBase<T, B> {
    fn uniform(shape: impl Into<Shape>) -> TensorBase<T, B> {
        let shape = shape.into();
        // random vector of size shape.size(), fill with uniform random values
        let size = shape.size();
        let mut raw = vec![T::default(); size];

        // fill with random values
        let mut rng = rand::rng();
        for v in raw.iter_mut() {
            *v = rand::Rng::random_range(&mut rng, T::from_f32(-1.0)..T::from_f32(1.0));
        }

        let backend = B::new();
        let buf = backend.alloc_from_slice(raw.into_boxed_slice()).expect("Allocation failed");
        let stride = super::shape_to_stride(&shape);
        TensorBase::from_parts(backend, buf, MetaTensor::new(
            shape,
            stride,
            0
        ), None)
    }

    fn xavier_uniform(shape: impl Into<Shape>) -> TensorBase<T, B> {
        let shape = shape.into();
        let (fan_in, fan_out) = fan_in_out(&shape);
        let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
        let v = scaled_uniform(shape.size(), limit);
        let backend = B::new();
        let buf = backend.alloc_from_slice(v.into_boxed_slice()).expect("Allocation failed");
        let stride = super::shape_to_stride(&shape);
        TensorBase::from_parts(backend, buf, MetaTensor::new(
            shape.clone(),
            stride,
            0
        ), None)
    }

    fn kaiming_uniform(shape: impl Into<Shape>) -> TensorBase<T, B> {
        let shape = shape.into();
        let (fan_in, _) = fan_in_out(&shape);
        let limit = (6.0f32 / fan_in as f32).sqrt();

        let v = scaled_uniform(shape.size(), limit);
        
        let backend = B::new();
        let buf = backend.alloc_from_slice(v.into_boxed_slice()).expect("Allocation failed");
        let stride = super::shape_to_stride(&shape);
        TensorBase::from_parts(backend, buf, MetaTensor::new(
            shape.clone(),
            stride,
            0
        ), None)
    }

    fn lecun_uniform(shape: impl Into<Shape>) -> TensorBase<T, B> {
        let shape = shape.into();

        let (fan_in, _) = fan_in_out(&shape);
        let limit = (3.0f32 / fan_in as f32).sqrt();
        let v = scaled_uniform(shape.size(), limit);
        let backend = B::new();
        let buf = backend.alloc_from_slice(v.into_boxed_slice()).expect("Allocation failed");
        let stride = super::shape_to_stride(&shape);
        TensorBase::from_parts(backend, buf, MetaTensor::new(
            shape.clone(),
            stride,
            0
        ), None)
    }

}

/// Converts a logical index (coordinate, single position, or scalar) into a
/// linear buffer index using the provided stride and offset.
///
/// Behavior
/// - `Coord(&[d0, d1, ...])` computes `offset + sum(di*stride[i])`.
/// - `At(i)` is treated as `Coord(&[i])`.
/// - `Item` is only valid for scalars (rank 0).
///
/// Errors
/// - `WrongDims` if index rank differs from stride length, or `Item` is used on non-scalars.
/// - `IdxOutOfBounds` is not checked here (caller validates against buffer length).
#[inline]
fn logical_to_buffer_idx(idx: &Idx, stride: &Strides, offset: usize) -> Result<usize, TensorError> {
    match idx {
        Idx::Coord(idx) => {
            if idx.len() != stride.len() {
                Err(TensorError::WrongDims(format!(
                    "Index rank {} does not match tensor rank {}",
                    idx.len(),
                    stride.len()
                )))
            }else{
                let bidx = idx
                    .iter()
                    .zip(stride.iter())
                    .fold(offset as isize, |acc, (a, b)| acc + (*a as isize) * *b);
                if bidx < 0 {
                    return Err(TensorError::IdxOutOfBounds("Buffer index is negative".to_string()));
                }
                Ok(bidx as usize)
            }
        },
        Idx::Item => {
            if stride.is_empty() {
                Ok(offset)
            }else{
                Err(TensorError::WrongDims(format!(
                    "Item index used on non-scalar tensor with rank {}",
                    stride.len()
                )))
            }
        },
        Idx::At(i) => {
            // Single-dimensional index; only valid when there is exactly one dimension
            logical_to_buffer_idx(&Idx::Coord(vec![*i]), stride, offset)
        }
    }
}

#[inline]
fn compute_permuted_parameters(shape: &Shape, stride: &Strides, dims: &Idx) -> Result<(Shape, Strides), TensorError> {
    let rank = shape.len();
    let dims_vec = match dims {
        Idx::Coord(v) => v.clone(),
        Idx::At(i) => vec![*i],
        Idx::Item => vec![],
    };

    if dims_vec.len() != rank {
        return Err(TensorError::WrongDims(format!(
            "Permutation dims length {} does not match tensor rank {}",
            dims_vec.len(),
            rank
        )));
    }

    let mut new_shape = Vec::with_capacity(rank);
    let mut new_stride = Vec::with_capacity(rank);

    for &d in &dims_vec {
        if d >= rank {
            return Err(TensorError::InvalidDim(format!(
                "Permutation dim {} is out of bounds for tensor rank {}",
                d,
                rank
            )));
        }
        new_shape.push(shape[d]);
        new_stride.push(stride[d]);
    }

    Ok((new_shape.into(), new_stride.into()))
}


#[inline]
#[grad::if_enabled(ctx)]
fn attach_unsqueeze_grad(
    result: &impl OpTensor,
    dim: Dim,
) -> Option<()>{
    let input_op = result.op();
    let op = GradNode::Unsqueeze {
        input: input_op,
        dim,
    };
    ctx.attach(result, op);
}

#[inline]
#[grad::if_enabled(ctx)]
fn attach_squeeze_grad(
    result: &impl OpTensor,
    original_shape: Shape,
) -> Option<()>{
    let input_op = result.op();
    let op = GradNode::Squeeze {
        input: input_op,
        original_shape,
    };
    ctx.attach(result, op);
}

#[inline]
pub(crate) fn compute_unsqueezed_parameters(shape: &Shape, stride: &Strides, dim: Dim) -> Result<(Shape, Strides), TensorError> {
    if dim > shape.len() {
        return Err(TensorError::InvalidDim(format!(
            "Unsqueeze dim {} is out of bounds for tensor rank {}",
            dim,
            shape.len()
        )));
    }
    let mut new_strides = stride.clone();
    let mut new_shape = shape.clone();

    let lstr = *new_strides.0.get(dim).unwrap_or(&1);
    let lsh = *new_shape.0.get(dim).unwrap_or(&1) as isize;
    new_strides.0.insert(dim, lstr * lsh);
    new_shape.0.insert(dim, 1);

    Ok((new_shape, new_strides))
}

#[inline]
pub(crate) fn compute_squeezed_parameters(shape: &Shape, stride: &Strides, dim: Option<Dim>) -> Result<(Shape, Strides), TensorError> {
    let mut result_shape = shape.clone();
    let mut result_stride = stride.clone();

    // Validate the dimension if specified
    if let Some(target_dim) = dim {
        if target_dim >= shape.len() {
            return Err(TensorError::InvalidDim(format!(
                "Dimension {} is out of bounds for tensor with rank {}",
                target_dim,
                shape.len()
            )));
        }
    }

    for d in (0..shape.len()).rev() {
        let should_squeeze = match dim {
            None => shape[d] == 1,
            Some(target_dim) => target_dim == d,
        };
        
        if should_squeeze {
            if shape[d] != 1 {
                return Err(TensorError::InvalidDim(format!(
                    "Cannot squeeze dimension {} with size {}",
                    d,
                    shape[d]
                )));
            }
            result_shape.0.remove(d);
            result_stride.0.remove(d);
        }
    }
    Ok((result_shape, result_stride))
}

#[inline(always)]
#[grad::if_enabled(ctx)]
fn attach_permute_grad<T: TensorValue, B: Backend>(
    result: &impl OpTensor,
    input_op: Option<NodeKey>,
    dims: &Idx,
) -> Option<()>
{
    let op = GradNode::Permute {
        input: input_op,
        dims: dims.clone(),
    };
    ctx.attach(result, op);
}
