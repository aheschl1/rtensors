use std::fmt::Debug;

use crate::{backend::{Backend, cpu::Cpu}, core::{MetaTensor, TensorView, TensorViewMut, primitives::{DeviceType, TensorBase}, tensor::{AsView, AsViewMut}, value::{DType, TensorValue, types}}};

/// Trait for erased tensors, allowing dynamic dispatch on tensor types.
/// Implemented for all `TensorBase<T, B>` where `T: TensorValue` and `B: Backend<T>`.
pub trait UntypedTensor: Send + Sync + Debug {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn device(&self) -> DeviceType;
    fn dtype(&self) -> DType;
    fn meta(&self) -> &MetaTensor;
}

impl<T, B> UntypedTensor for TensorBase<T, B>
where
    T: crate::core::value::TensorValue + 'static,
    B: crate::backend::Backend + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn device(&self) -> DeviceType {
        B::device_type()
    }

    fn dtype(&self) -> DType {
        T::DTYPE
    }

    fn meta(&self) -> &MetaTensor {
        &self.meta
    }
}

/// Downcasting methods for `ErasedTensor`.
/// Allows retrieving the concrete tensor type from the erased trait object.
/// Requires knowing the original `T` and `B` types.
impl dyn UntypedTensor {
    pub fn typed_unknown(&self) -> UnknownTensor<'_> {
        match (self.dtype(), self.device()) {
            (DType::U8, DeviceType::Cpu) => UnknownTensor::U8Cpu(self.typed::<u8, Cpu>().unwrap()),
            (DType::U16, DeviceType::Cpu) => UnknownTensor::U16Cpu(self.typed::<u16, Cpu>().unwrap()),
            (DType::U32, DeviceType::Cpu) => UnknownTensor::U32Cpu(self.typed::<u32, Cpu>().unwrap()),
            (DType::U64, DeviceType::Cpu) => UnknownTensor::U64Cpu(self.typed::<u64, Cpu>().unwrap()),
            (DType::U128, DeviceType::Cpu) => UnknownTensor::U128Cpu(self.typed::<u128, Cpu>().unwrap()),
            (DType::I8, DeviceType::Cpu) => UnknownTensor::I8Cpu(self.typed::<i8, Cpu>().unwrap()),
            (DType::I16, DeviceType::Cpu) => UnknownTensor::I16Cpu(self.typed::<i16, Cpu>().unwrap()),
            (DType::I32, DeviceType::Cpu) => UnknownTensor::I32Cpu(self.typed::<i32, Cpu>().unwrap()),
            (DType::I64, DeviceType::Cpu) => UnknownTensor::I64Cpu(self.typed::<i64, Cpu>().unwrap()),
            (DType::I128, DeviceType::Cpu) => UnknownTensor::I128Cpu(self.typed::<i128, Cpu>().unwrap()),
            (DType::F32, DeviceType::Cpu) => UnknownTensor::F32Cpu(self.typed::<f32, Cpu>().unwrap()),
            (DType::F64, DeviceType::Cpu) => UnknownTensor::F64Cpu(self.typed::<f64, Cpu>().unwrap()),
            (DType::BOOL, DeviceType::Cpu) => UnknownTensor::BOOLCpu(self.typed::<types::boolean, Cpu>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U8, DeviceType::Cuda(_)) => UnknownTensor::U8Cuda(self.typed::<u8, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U16, DeviceType::Cuda(_)) => UnknownTensor::U16Cuda(self.typed::<u16, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U32, DeviceType::Cuda(_)) => UnknownTensor::U32Cuda(self.typed::<u32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]            
            (DType::U64, DeviceType::Cuda(_)) => UnknownTensor::U64Cuda(self.typed::<u64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U128, DeviceType::Cuda(_)) => UnknownTensor::U128Cuda(self.typed::<u128, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I8, DeviceType::Cuda(_)) => UnknownTensor::I8Cuda(self.typed::<i8, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I16, DeviceType::Cuda(_)) => UnknownTensor::I16Cuda(self.typed::<i16, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I32, DeviceType::Cuda(_)) => UnknownTensor::I32Cuda(self.typed::<i32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I64, DeviceType::Cuda(_)) => UnknownTensor::I64Cuda(self.typed::<i64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I128, DeviceType::Cuda(_)) => UnknownTensor::I128Cuda(self.typed::<i128, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::F32, DeviceType::Cuda(_)) => UnknownTensor::F32Cuda(self.typed::<f32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::F64, DeviceType::Cuda(_)) => UnknownTensor::F64Cuda(self.typed::<f64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::BOOL, DeviceType::Cuda(_)) => UnknownTensor::BOOLCuda(self.typed::<types::boolean, crate::backend::cuda::Cuda>().unwrap()),
        }
    }

    pub fn typed_mut_unknown(&mut self) -> UnknownTensorMut<'_> {
        match (self.dtype(), self.device()) {
            (DType::U8, DeviceType::Cpu) => UnknownTensorMut::U8Cpu(self.typed_mut::<u8, Cpu>().unwrap()),
            (DType::U16, DeviceType::Cpu) => UnknownTensorMut::U16Cpu(self.typed_mut::<u16, Cpu>().unwrap()),
            (DType::U32, DeviceType::Cpu) => UnknownTensorMut::U32Cpu(self.typed_mut::<u32, Cpu>().unwrap()),
            (DType::U64, DeviceType::Cpu) => UnknownTensorMut::U64Cpu(self.typed_mut::<u64, Cpu>().unwrap()),
            (DType::U128, DeviceType::Cpu) => UnknownTensorMut::U128Cpu(self.typed_mut::<u128, Cpu>().unwrap()),
            (DType::I8, DeviceType::Cpu) => UnknownTensorMut::I8Cpu(self.typed_mut::<i8, Cpu>().unwrap()),
            (DType::I16, DeviceType::Cpu) => UnknownTensorMut::I16Cpu(self.typed_mut::<i16, Cpu>().unwrap()),
            (DType::I32, DeviceType::Cpu) => UnknownTensorMut::I32Cpu(self.typed_mut::<i32, Cpu>().unwrap()),
            (DType::I64, DeviceType::Cpu) => UnknownTensorMut::I64Cpu(self.typed_mut::<i64, Cpu>().unwrap()),
            (DType::I128, DeviceType::Cpu) => UnknownTensorMut::I128Cpu(self.typed_mut::<i128, Cpu>().unwrap()),
            (DType::F32, DeviceType::Cpu) => UnknownTensorMut::F32Cpu(self.typed_mut::<f32, Cpu>().unwrap()),
            (DType::F64, DeviceType::Cpu) => UnknownTensorMut::F64Cpu(self.typed_mut::<f64, Cpu>().unwrap()),
            (DType::BOOL, DeviceType::Cpu) => UnknownTensorMut::BOOLCpu(self.typed_mut::<types::boolean, Cpu>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U8, DeviceType::Cuda(_)) => UnknownTensorMut::U8Cuda(self.typed_mut::<u8, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U16, DeviceType::Cuda(_)) => UnknownTensorMut::U16Cuda(self.typed_mut::<u16, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U32, DeviceType::Cuda(_)) => UnknownTensorMut::U32Cuda(self.typed_mut::<u32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]            
            (DType::U64, DeviceType::Cuda(_)) => UnknownTensorMut::U64Cuda(self.typed_mut::<u64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::U128, DeviceType::Cuda(_)) => UnknownTensorMut::U128Cuda(self.typed_mut::<u128, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I8, DeviceType::Cuda(_)) => UnknownTensorMut::I8Cuda(self.typed_mut::<i8, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I16, DeviceType::Cuda(_)) => UnknownTensorMut::I16Cuda(self.typed_mut::<i16, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I32, DeviceType::Cuda(_)) => UnknownTensorMut::I32Cuda(self.typed_mut::<i32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I64, DeviceType::Cuda(_)) => UnknownTensorMut::I64Cuda(self.typed_mut::<i64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::I128, DeviceType::Cuda(_)) => UnknownTensorMut::I128Cuda(self.typed_mut::<i128, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::F32, DeviceType::Cuda(_)) => UnknownTensorMut::F32Cuda(self.typed_mut::<f32, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::F64, DeviceType::Cuda(_)) => UnknownTensorMut::F64Cuda(self.typed_mut::<f64, crate::backend::cuda::Cuda>().unwrap()),
            #[cfg(feature = "cuda")]
            (DType::BOOL, DeviceType::Cuda(_)) => UnknownTensorMut::BOOLCuda(self.typed_mut::<types::boolean, crate::backend::cuda::Cuda>().unwrap()),
        }
    }

    pub fn typed<T, B>(&self) -> Option<&TensorBase<T, B>>
    where
        T: TensorValue + 'static,
        B: Backend
    {
        self.as_any().downcast_ref::<TensorBase<T, B>>()
    }
    
    pub fn typed_mut<T, B>(&mut self) -> Option<&mut TensorBase<T, B>>
    where
        T: TensorValue + 'static,
        B: Backend
    {
        self.as_any_mut().downcast_mut::<TensorBase<T, B>>()
    }

    pub fn view_typed<T, B>(&self) -> Option<TensorView<'_, T, B>>
    where
        T: TensorValue + 'static,
        B: Backend
    {
        self.typed::<T, B>().map(|t| t.view())
    }

    pub fn view_typed_mut<T, B>(&mut self) -> Option<TensorViewMut<'_, T, B>>
    where
        T: TensorValue + 'static,
        B: Backend
    {
        self.typed_mut::<T, B>().map(|t| t.view_mut())
    }
}

pub trait AsUntypedTensor {
    fn as_untyped(self) -> Box<dyn UntypedTensor>;
}

impl<T, B> AsUntypedTensor for TensorBase<T, B>
where
    T: TensorValue + 'static,
    B: Backend + 'static,
{
    fn as_untyped(self) -> Box<dyn UntypedTensor> {
        Box::new(self)
    }
}


// TODO: Do not duplicate this logic and find a better way 

#[derive(Debug, PartialEq)]
pub enum UnknownTensor<'a> {
    U8Cpu(&'a TensorBase<u8, Cpu>),
    U16Cpu(&'a TensorBase<u16, Cpu>),
    U32Cpu(&'a TensorBase<u32, Cpu>),
    U64Cpu(&'a TensorBase<u64, Cpu>),
    U128Cpu(&'a TensorBase<u128, Cpu>),
    I8Cpu(&'a TensorBase<i8, Cpu>),
    I16Cpu(&'a TensorBase<i16, Cpu>),
    I32Cpu(&'a TensorBase<i32, Cpu>),
    I64Cpu(&'a TensorBase<i64, Cpu>),
    I128Cpu(&'a TensorBase<i128, Cpu>),
    F32Cpu(&'a TensorBase<f32, Cpu>),
    F64Cpu(&'a TensorBase<f64, Cpu>),
    BOOLCpu(&'a TensorBase<types::boolean, Cpu>),
    #[cfg(feature = "cuda")]
    U8Cuda(&'a TensorBase<u8, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U16Cuda(&'a TensorBase<u16, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U32Cuda(&'a TensorBase<u32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U64Cuda(&'a TensorBase<u64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U128Cuda(&'a TensorBase<u128, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I8Cuda(&'a TensorBase<i8, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I16Cuda(&'a TensorBase<i16, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I32Cuda(&'a TensorBase<i32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I64Cuda(&'a TensorBase<i64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I128Cuda(&'a TensorBase<i128, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    F32Cuda(&'a TensorBase<f32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    F64Cuda(&'a TensorBase<f64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    BOOLCuda(&'a TensorBase<types::boolean, crate::backend::cuda::Cuda>),
}

pub enum UnknownTensorMut<'a> {
    U8Cpu(&'a mut TensorBase<u8, Cpu>),
    U16Cpu(&'a mut TensorBase<u16, Cpu>),
    U32Cpu(&'a mut TensorBase<u32, Cpu>),
    U64Cpu(&'a mut TensorBase<u64, Cpu>),
    U128Cpu(&'a mut TensorBase<u128, Cpu>),
    I8Cpu(&'a mut TensorBase<i8, Cpu>),
    I16Cpu(&'a mut TensorBase<i16, Cpu>),
    I32Cpu(&'a mut TensorBase<i32, Cpu>),
    I64Cpu(&'a mut TensorBase<i64, Cpu>),
    I128Cpu(&'a mut TensorBase<i128, Cpu>),
    F32Cpu(&'a mut TensorBase<f32, Cpu>),
    F64Cpu(&'a mut TensorBase<f64, Cpu>),
    BOOLCpu(&'a mut TensorBase<types::boolean, Cpu>),
    #[cfg(feature = "cuda")]
    U8Cuda(&'a mut TensorBase<u8, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U16Cuda(&'a mut TensorBase<u16, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U32Cuda(&'a mut TensorBase<u32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U64Cuda(&'a mut TensorBase<u64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    U128Cuda(&'a mut TensorBase<u128, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I8Cuda(&'a mut TensorBase<i8, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I16Cuda(&'a mut TensorBase<i16, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I32Cuda(&'a mut TensorBase<i32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I64Cuda(&'a mut TensorBase<i64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    I128Cuda(&'a mut TensorBase<i128, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    F32Cuda(&'a mut TensorBase<f32, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    F64Cuda(&'a mut TensorBase<f64, crate::backend::cuda::Cuda>),
    #[cfg(feature = "cuda")]
    BOOLCuda(&'a mut TensorBase<types::boolean, crate::backend::cuda::Cuda>),
}



#[cfg(test)]
mod tests {
    use crate::{backend::cpu::Cpu, core::{untyped::UntypedTensor, primitives::DeviceType, tensor::TensorError, value::TensorValue, Shape, Tensor}};

    #[test]
    fn test_erased_tensor_downcast() -> Result<(), TensorError> {
        // cpu, f32 tensor
        let tensor = Tensor::<f32>::zeros((2, 3));
        let erased: Box<dyn UntypedTensor> = Box::new(tensor);
        assert_eq!(erased.device(), DeviceType::Cpu);
        assert_eq!(erased.dtype(), f32::DTYPE);
        let downcasted = erased.typed::<f32, Cpu>().unwrap();
        assert_eq!(downcasted.meta().shape, Shape::from((2, 3)));
        Ok(())
    }
}