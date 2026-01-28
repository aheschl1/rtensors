use std::cell::RefCell;
use std::fmt::Debug;
#[cfg(feature = "remote")]
use std::net::IpAddr;
use std::sync::{Arc, RwLock};


use crate::backend::Backend;
use crate::backend::cpu::Cpu;
use crate::core::untyped::UntypedTensor;
use crate::core::value::TensorValue;
use crate::core::{shape_to_stride, MetaTensor, MetaTensorView, Shape};
use crate::core::tensor::{compute_squeezed_parameters, compute_unsqueezed_parameters, TensorError};

#[derive(Debug, Clone)]
pub struct Grad(Arc<RwLock<Option<Box<dyn UntypedTensor>>>>);

impl Grad {
    pub fn default() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }

    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, Option<Box<dyn UntypedTensor>>> {
        self.0.read().unwrap()
    } 

    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, Option<Box<dyn UntypedTensor>>> {
        self.0.write().unwrap()
    }
}

pub type NodeOp = Arc<RwLock<Option<NodeKey>>>;

/// A generic tensor with backend-specific storage.
/// 
/// This is the base type for all tensors, parameterized by element type `T` and backend `B`.
/// Most users will use type aliases like `Tensor<T>` (CPU) or `CudaTensor<T>` (GPU).
#[derive(Debug)]
pub struct TensorBase<T: TensorValue, B: Backend> {
    pub(crate) backend: B,
    pub(crate) buf: B::Buf<T>,
    pub(crate) meta: MetaTensor,
    pub(crate) grad: Grad,    // relevant for params with auto-grad
    pub(crate) op: NodeOp, // relevant when there is a ctx for tracking ops
}

impl<T: TensorValue, B: Backend> PartialEq for TensorBase<T, B> {
    fn eq(&self, other: &Self) -> bool {
        if self.meta != other.meta { return false; }
        if self.buf != other.buf { return false; }
        if self.grad != other.grad { return false; }
        true
    }
}

impl PartialEq for Grad {
    fn eq(&self, other: &Self) -> bool {
        let self_guard = self.0.read().unwrap();
        let other_guard = other.0.read().unwrap();
        match (&*self_guard, &*other_guard) {
            (Some(t1), Some(t2)) => t1.typed_unknown() == t2.typed_unknown(),
            (None, None) => true,
            _ => false,
        }
    }
}

impl<B: Backend, T: TensorValue> Clone for TensorBase<T, B> {
    fn clone(&self) -> Self {
        let new_backend = self.backend.clone();
        let new_buffer = new_backend.copy(&self.buf).unwrap();
        Self {
            backend: new_backend,
            buf: new_buffer,
            meta: self.meta.clone(),
            grad: Grad::default(),
            op: RwLock::new(None).into(),
        }

    }
}

/// An owned CPU tensor stored in row-major order.
/// 
/// # Examples
/// ```ignore
/// let tensor = Tensor::<f32>::zeros((3, 4));
/// let tensor = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], (2, 2)).unwrap();
/// ```
pub type Tensor<T> = TensorBase<T, Cpu>;

#[cfg(feature = "remote")]
use crate::backend::remote::client::RemoteBackend;
use crate::grad::NodeKey;

#[cfg(feature = "remote")]
pub type RemoteTensor<T> = TensorBase<T, RemoteBackend>;

#[cfg(feature = "cuda")]
/// An owned GPU tensor stored on CUDA device.
pub type CudaTensor<T> = TensorBase<T, crate::backend::cuda::Cuda>;

#[cfg(feature = "cuda")]
impl<T: TensorValue> CudaTensor<T> {
    /// Transfers this tensor from the CUDA device to CPU memory.
    pub fn cpu(&self) -> Result<Tensor<T>, TensorError> {
        let cpu_backend = Cpu;
        let cpu_buffer = self.backend.dump(&self.buf)?;
        let cpu = Tensor::from_parts(cpu_backend, cpu_buffer, self.meta.clone());
        Ok(cpu)
    }
}

#[cfg(feature = "cuda")]
impl<T: TensorValue> Tensor<T> {
    /// Transfers this tensor from CPU to the CUDA device.
    pub fn cuda(&self) -> Result<CudaTensor<T>, TensorError> {
        let cuda_backend = crate::backend::cuda::Cuda::construct(0)?;
        let cuda_buffer = cuda_backend.alloc_from_slice(self.backend.dump(&self.buf)?)?;
        let cuda = CudaTensor::from_parts(cuda_backend, cuda_buffer, self.meta.clone());
        Ok(cuda)
    }
}

#[cfg(feature = "remote")]
impl<T: TensorValue> RemoteTensor<T> {
    /// Transfers this tensor from remote backend to CPU memory.
    pub fn cpu(&self) -> Result<Tensor<T>, TensorError> {
        let cpu_backend = Cpu;
        let cpu_buffer = self.backend.dump(&self.buf)?;
        let cpu = Tensor::from_parts(cpu_backend, cpu_buffer, self.meta.clone());
        Ok(cpu)
    }

    pub fn with_remote(ip: IpAddr, port: u16) -> Result<Self, TensorError> {
        let remote_backend = RemoteBackend::new_with_address(ip, port)
            .map_err(|e| TensorError::RemoteError(format!("Failed to create remote backend: {}", e)))?;
        let buf = remote_backend.alloc::<T>(0)?;
        Ok(Self {
            backend: remote_backend,
            buf,
            meta: MetaTensor::new(vec![], vec![], 0),
            _t: PhantomData,
        })
    }
}

/// A non-owning immutable view over tensor data.
/// 
/// Views share the underlying buffer with the source tensor and have their own
/// metadata (shape, stride, offset) to represent different interpretations of the data.
pub struct TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    pub(crate) buf: &'a B::Buf<T>,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
    pub(crate) op: NodeOp
}

/// A non-owning mutable view over tensor data.
/// 
/// Like `TensorView` but allows mutation of the underlying data.
pub struct TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    pub(crate) buf: &'a mut B::Buf<T>,
    pub(crate) backend: &'a B,
    pub(crate) meta: MetaTensor,
    pub(crate) op: NodeOp
}

impl<'a, T, B> TensorView<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        buf: &'a B::Buf<T>,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            buf,
            backend,
            meta,
            op: RwLock::new(None).into(),
        }
    }

    pub fn unsqueeze_at_inplace(&mut self, dim: usize) -> Result<(), TensorError> {
        let (new_shape, new_strides) = unsafe { compute_unsqueezed_parameters(self.shape(), self.strides(), dim).unwrap_unchecked() };
        self.meta.shape = new_shape;
        self.meta.strides = new_strides;
        Ok(())
    }

    pub fn unsqueeze_inplace(&mut self) {
        self.unsqueeze_at_inplace(0).unwrap();
    }
}

impl<'a, T, B> TensorViewMut<'a, T, B>
where
    T: TensorValue,
    B: Backend + 'a,
{
    /// Builds a tensor view from raw storage and metadata. No copying occurs;
    /// caller guarantees that `meta` correctly describes the layout within `raw`.
    pub(crate) fn from_parts(
        raw: &'a mut B::Buf<T>,
        backend: &'a B,
        meta: MetaTensor
    ) -> Self {
        Self {
            buf: raw,
            backend,
            meta,
            op: RwLock::new(None).into(),
        }
    }

    pub fn unsqueeze_at_inplace(&mut self, dim: usize) -> Result<(), TensorError> {
        let (new_shape, new_strides) = unsafe { compute_unsqueezed_parameters(self.shape(), self.strides(), dim).unwrap_unchecked() };
        self.meta.shape = new_shape;
        self.meta.strides = new_strides;
        Ok(())
    }

    pub fn unsqueeze_inplace(&mut self) {
        self.unsqueeze_at_inplace(0).unwrap();
    }
}

pub(crate) trait OpTensor {
    fn op(&self) -> Option<NodeKey>;
    fn set_op(&self, op: NodeKey);
}

impl<T: TensorValue, B: Backend> OpTensor for TensorBase<T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}

impl<T: TensorValue, B: Backend> OpTensor for &TensorBase<T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}

impl<'a, T: TensorValue, B: Backend> OpTensor for TensorView<'a, T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}

impl<'a, T: TensorValue, B: Backend> OpTensor for &TensorView<'a, T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}


impl<'a, T: TensorValue, B: Backend> OpTensor for TensorViewMut<'a, T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}

impl<'a, T: TensorValue, B: Backend> OpTensor for &TensorViewMut<'a, T, B> {
    fn op(&self) -> Option<NodeKey> {
        self.op.read().unwrap().clone()
    }

    fn set_op(&self, op: NodeKey) {
        self.op.write().unwrap().replace(op);
    }
}

pub type CpuTensorView<'a, T> = TensorView<'a, T, Cpu>;
pub type CpuTensorViewMut<'a, T> = TensorViewMut<'a, T, Cpu>;
#[cfg(feature = "cuda")]
pub type CudaTensorView<'a, T> = TensorView<'a, T, crate::backend::cuda::Cuda>;
#[cfg(feature = "cuda")]
pub type CudaTensorViewMut<'a, T> = TensorViewMut<'a, T, crate::backend::cuda::Cuda>;

#[cfg(feature = "remote")]
pub type RemoteTensorView<'a, T> = TensorView<'a, T, RemoteBackend>;
#[cfg(feature = "remote")]
pub type RemoteTensorViewMut<'a, T> = TensorViewMut<'a, T, RemoteBackend>;

impl<B, T: TensorValue> TensorBase<T, B> 
where 
    B: Backend,
{
    /// Internal constructor from raw parts. Used for creating tensors from
    /// existing backend buffers without copying.
    pub(crate) fn from_parts(backend: B, raw: B::Buf<T>, meta: MetaTensor) -> Self {
        Self {
            backend,
            buf: raw,
            meta,
            grad: Grad::default(),
            op: RwLock::new(None).into(),
        }
    }

    /// Constructs a tensor from a buffer and shape.
    /// 
    /// The buffer must be contiguous and in row-major order.
    /// 
    /// # Errors
    /// - `InvalidShape` if the buffer size doesn't match the shape.
    /// - `InvalidShape` if the shape has more than 128 dimensions.
    /// 
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).unwrap();
    /// ```
    pub fn from_buf(raw: impl Into<Box<[T]>>, shape: impl Into<Shape>) -> Result<Self, TensorError> {
        let shape: Shape = shape.into();
        if shape.len() > 128 {
            // artificial cap due to broadcast cuda kernel...
            return Err(TensorError::InvalidShape(format!(
                "Tensors with more than 128 dimensions are not supported, got {} dimensions",
                shape.len()
            )));
        }
        let backend = B::new();
        let buffer = backend.alloc_from_slice(raw.into())?;
        if shape.iter().product::<usize>() != backend.len(&buffer) {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                shape.iter().product::<usize>(),
                backend.len(&buffer)
            )));
        }
        let stride = shape_to_stride(&shape);
        Ok(Self {
            backend,
            buf: buffer,
            meta: MetaTensor::new(shape, stride, 0),
            grad: Grad::default(),
            op: RwLock::new(None).into(),
        })
    }

    pub fn to_box(&self) -> Result<Box<[T]>, TensorError> {
        self.backend.dump(&self.buf)
    }

    pub fn zero_grad(&mut self) {
        let mut grad_ref = self.grad.write();
        *grad_ref = None;
    }

    /// Creates a rank-0 (scalar) tensor.
    /// 
    /// # Examples
    /// ```ignore
    /// let scalar = Tensor::<f32>::scalar(42.0);
    /// ```
    pub fn scalar(value: T) -> Self {
        Self::from_buf(vec![value], vec![]).unwrap()
    }

    /// Creates a 1-D column tensor from values.
    /// 
    /// # Examples
    /// ```ignore
    /// let col = Tensor::<i32>::column(vec![1, 2, 3]);
    /// ```
    pub fn column(column: impl Into<Box<[T]>>) -> Self {
        let column = column.into();
        let shape = vec![column.len()];
        Self::from_buf(column, shape).unwrap()
    }

    /// Creates a 1xN row tensor from values.
    /// 
    /// # Examples
    /// ```ignore
    /// let row = Tensor::<f32>::row(vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn row(row: impl Into<Box<[T]>>) -> Self {
        let row = row.into();
        let shape = vec![1, row.len()];
        Self::from_buf(row, shape).unwrap()
    }

    /// Creates a tensor filled with zeros.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// 
    /// # Examples
    /// ```ignore
    /// let zeros = Tensor::<f32>::zeros((3, 4));
    /// ```
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let zero_buf = vec![T::ZERO; element_count];
        Self::from_buf(zero_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with ones.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    /// 
    /// # Examples
    /// ```ignore
    /// let ones = Tensor::<f32>::ones((2, 2));
    /// ```
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let one_buf = vec![T::ONE; element_count];
        Self::from_buf(one_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the maximum value for type `T`.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    pub fn max(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let max_buf = vec![T::MAX; element_count];
        Self::from_buf(max_buf, shape).expect("Failed to allocate memory")
    }

    /// Creates a tensor filled with the minimum value for type `T`.
    /// 
    /// # Panics
    /// Panics if memory allocation fails.
    pub fn min(shape: impl Into<Shape>) -> Self {
        let shape: Shape = shape.into();
        let element_count = shape.iter().product::<usize>();
        let min_buf = vec![T::MIN; element_count];
        Self::from_buf(min_buf, shape).expect("Failed to allocate memory")
    }

    /// Squeezes the tensor in place, preventing a new allocation from being made.
    pub fn squeeze_inplace(&mut self) {
        let (new_shape, new_strides) = unsafe { compute_squeezed_parameters(self.shape(), self.strides(), None).unwrap_unchecked() };
        self.meta.shape = new_shape;
        self.meta.strides = new_strides;
    }

    pub fn unsqueeze_at_inplace(&mut self, dim: usize) -> Result<(), TensorError> {
        let (new_shape, new_strides) = unsafe { compute_unsqueezed_parameters(self.shape(), self.strides(), dim).unwrap_unchecked() };
        self.meta.shape = new_shape;
        self.meta.strides = new_strides;
        Ok(())
    }

    pub fn unsqueeze_inplace(&mut self) {
        self.unsqueeze_at_inplace(0).unwrap();
    }

    pub fn into_dtype<N: TensorValue>(&self) -> Result<TensorBase<N, B>, TensorError> {

        let mut new_buf = self.backend.alloc::<N>(self.size())?;
        self.backend.convert::<T, N>(&self.buf, &mut new_buf)?;

        Ok(TensorBase::<N, B>::from_parts(
            self.backend.clone(),
            new_buf,
            self.meta.clone()
        ))
    }

}

use image::imageops::FilterType::Triangle;
#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

/// Indicates where a tensor's data resides.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum DeviceType {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "remote")]
    Remote {
        ip: IpAddr,
        port: u16,
        remote_type: Box<DeviceType>
    }
}