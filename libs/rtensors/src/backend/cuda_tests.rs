#![allow(non_snake_case)]

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::core::tensor::TensorError;

// ignore styling for generated code
#[allow(non_camel_case_types)]
mod bindings{
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct CudaBackend {
    context: Arc<CudaContext>,
}

impl CudaBackend {
    /// Create a new CUDA backend instance
    /// 
    /// # Errors
    /// Returns an error if CUDA initialization fails or no CUDA device is found
    pub fn new() -> Result<Self, TensorError> {
        let context = CudaContext::new(0)
            .map_err(|e| TensorError::CudaError(e.to_string()))?;
        Ok(Self { context })
    }
    
    pub fn context(&self) -> &CudaContext {
        &self.context
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize CUDA backend")
    }
}


#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::core::{idx::Idx, primitives::CudaTensor, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::TensorValue, MetaTensorView, Shape, Slice};
    
    fn make_cuda_tensor<T: TensorValue + cudarc::driver::DeviceRepr>(buf: Vec<T>, shape: impl Into<Shape>) -> CudaTensor<T> {
        CudaTensor::from_buf(buf, shape).unwrap()
    }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug + cudarc::driver::DeviceRepr>(
        index: Idx, 
        tensor: &'a impl TensorAccess<T, crate::backend::cuda::Cuda>
    ) -> Result<T, TensorError> {
        let r: Result<T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Some(*v),
            Err(_) => None,
        };
        let b = match &index {
            Idx::Item => tensor.item().ok(),
            _ => tensor.get(&index).ok(),
        };
        assert_eq!(a, b);
        r
    }
    
    #[test]
    fn test_cuda_backend_init() {
        let backend = CudaBackend::new();
        assert!(backend.is_ok(), "Failed to initialize CUDA backend");
    }

    #[test]
    fn test_cuda_scalar() {
        let tensor = CudaTensor::scalar(42);
        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.meta.is_scalar());
    }

    #[test]
    fn test_cuda_column() {
        let tensor = CudaTensor::column(vec![1, 2, 3]);
        assert_eq!(*tensor.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_cuda_row() {
        let tensor = CudaTensor::row(vec![1, 2, 3]);
        assert_eq!(*tensor.meta.shape(), vec![1, 3]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_cuda_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_cuda_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);

        tensor.set(&Idx::At(1), 10).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 10);
    }

    #[test]
    fn test_cuda_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 6);

        tensor.view_mut().set(&Idx::Coord(vec![1, 2]), 100).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 100);
    }

    #[test]
    fn test_cuda_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &tensor.view()).unwrap(), 9);

        tensor.set(&Idx::Coord(vec![1, 0, 0]), 67).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 67);
    }

    #[test]
    fn test_cuda_slice_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(*slice.meta.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 3);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap();
        assert_eq!(*slice2.meta.shape(), vec![2]);
        assert_eq!(*slice2.meta.strides(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![1]), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
    }

    #[test]
    fn test_cuda_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.meta.shape(), vec![2, 2]);
        assert_eq!(*slice.meta.strides(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 5);

        let view = tensor.view();
        let slice_second_depth = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice_second_depth.meta.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.meta.strides(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice_second_depth).unwrap(), 9);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap();
        assert_eq!(*slice2.meta.shape(), vec![2, 2]);
        assert_eq!(*slice2.meta.strides(), vec![4, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice2).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice2).unwrap(), 7);

        let view = tensor.view();
        let slice3 = view.slice(2, 0..0).unwrap();
        assert_eq!(*slice3.meta.shape(), vec![2, 2]);
        assert_eq!(*slice3.meta.strides(), vec![4, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice3).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice3).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice3).unwrap(), 8);
    }

    #[test]
    fn test_cuda_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_cuda_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);

        let slice_of_slice = slice.slice(0, 2..2).unwrap();
        assert_eq!(*slice_of_slice.meta.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Coord(vec![]), &slice_of_slice).unwrap(), 6);
    }

    #[test]
    fn test_cuda_slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_cuda_tensor(buf, shape);

        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice.meta.shape(), vec![2, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 9);

        let slice_of_slice = slice.slice(1, 0..0).unwrap();
        assert_eq!(*slice_of_slice.meta.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &slice_of_slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice_of_slice).unwrap(), 8);

        let slice_of_slice_of_slice = slice_of_slice.slice(0, 1..1).unwrap();
        assert_eq!(*slice_of_slice_of_slice.meta.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &slice_of_slice_of_slice).unwrap(), 8);
    }

    #[test]
    fn test_cuda_mut_slices() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap();
        
        assert_eq!(*slice.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        
        slice.set(&Idx::At(1), 50).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 50);
    }

    #[test]
    fn test_cuda_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            CudaTensor::from_buf(buf, shape),
            Err(TensorError::InvalidShape(_))
        ));
    }

    #[test]
    fn test_cuda_get_errors() {
        let tensor = make_cuda_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.get(vec![0, 0, 0]),
            Err(TensorError::WrongDims(_))
        ));
        assert!(matches!(
            tensor.view().get(vec![2, 0]),
            Err(TensorError::IdxOutOfBounds(_))
        ));
    }

    #[test]
    fn test_cuda_slice_errors() {
        let tensor = make_cuda_tensor(
            vec![1, 2, 3, 4],
            vec![2, 2]
        );
        assert!(matches!(
            tensor.view().slice(3, 0..0),
            Err(TensorError::InvalidDim(_))
        ));
        assert!(matches!(
            tensor.view().slice(0, 5..5),
            Err(TensorError::IdxOutOfBounds(_))
        ));
    }

    #[test]
    fn test_cuda_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_cuda_tensor(buf, shape);

        assert_eq!(tensor.view().get(&Idx::Coord(vec![0, 1])).unwrap(), 2);
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 6);

        tensor.view_mut().set(vec![1, 1], 55).unwrap();
        assert_eq!(tensor.view().get(&Idx::Coord(vec![1, 1])).unwrap(), 55);
        assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 55);

        let view = tensor.view();
        let view = view.slice(0, 1..1).unwrap();
        assert_eq!(view.get(vec![0]).unwrap(), 4);
        assert_eq!(view.get(vec![1]).unwrap(), 55);
        assert_eq!(view.get(vec![2]).unwrap(), 6);

        let mut mut_view = tensor.view_mut();
        let mut mut_slice = mut_view.slice_mut(0, 0..0).unwrap();
        mut_slice.set(vec![2], 33).unwrap();
        assert_eq!(mut_slice.get(&Idx::Coord(vec![2])).unwrap(), 33);
        assert_eq!(mut_slice.get(vec![2]).unwrap(), 33);

        assert_eq!(tensor.view().get(vec![0, 2]).unwrap(), 33);
    }

    #[test]
    #[should_panic]
    fn test_cuda_index_out_of_bounds_panic() {
        let tensor = make_cuda_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_cuda_index_wrong_dims_panic() {
        let tensor = make_cuda_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![0, 0]).unwrap();
    }

    #[test]
    fn rediculously_large_cuda_tensor() {
        let n = 10_000_000_usize;
        let buf: Vec<u8> = vec![1; n];
        let shape = vec![n];
        let tensor = make_cuda_tensor(buf, shape);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(n - 1), &tensor.view()).unwrap(), 1);
    }

    #[test]
    fn test_cuda_custom_positive_step() {
        // Test CUDA slicing with custom positive step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_cuda_tensor(buf, shape);
        
        // Step by 2: take every other element
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 14);
        
        // Step by 3: from index 1 to 10
        let slice2 = view.slice(0, Slice::from(1..10).step(3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 7);
    }

    #[test]
    fn test_cuda_custom_negative_step() {
        // Test CUDA slicing with custom negative step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_cuda_tensor(buf, shape);
        let view = tensor.view();
        
        // Step by -2: every other element, reversed
        let slice = view.slice(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 15);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 13);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 1);
        
        // Step by -3: from index 12 to 3
        let slice2 = view.slice(0, Slice::from(12..3).step(-3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 12);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 9);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 6);
    }

    #[test]
    fn test_cuda_custom_positive_step_mut() {
        // Test mutable CUDA slicing with custom positive step
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        // Step by 2: modify every other element
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        
        slice.set(&Idx::At(0), 100).unwrap(); // index 0
        slice.set(&Idx::At(1), 102).unwrap(); // index 2
        slice.set(&Idx::At(7), 114).unwrap(); // index 14
        
        // Verify changes in original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 100);
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 102);
        assert_eq!(index_tensor(Idx::At(14), &view).unwrap(), 114);
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 15); // Unchanged
    }

    #[test]
    fn test_cuda_custom_positive_step_mut_with_range() {
        // Test mutable CUDA slicing with step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        // Step by 3: from index 1 to 10
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(1..10).step(3)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 1, 4, 7
        
        slice.set(&Idx::At(0), 101).unwrap(); // index 1
        slice.set(&Idx::At(1), 104).unwrap(); // index 4
        slice.set(&Idx::At(2), 107).unwrap(); // index 7
        
        // Verify
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 0);  // Unchanged
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 101);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 104);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 107);
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 8);  // Unchanged
    }

    #[test]
    fn test_cuda_custom_negative_step_mut() {
        // Test mutable CUDA slicing with custom negative step
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        // Step by -2: every other element, reversed
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        
        slice.set(&Idx::At(0), 115).unwrap(); // index 15
        slice.set(&Idx::At(1), 113).unwrap(); // index 13
        slice.set(&Idx::At(7), 101).unwrap(); // index 1
        
        // Verify changes
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 101);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 2);   // Unchanged
        assert_eq!(index_tensor(Idx::At(13), &view).unwrap(), 113);
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 115);
    }

    #[test]
    fn test_cuda_custom_negative_step_mut_with_range() {
        // Test mutable CUDA slicing with negative step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_cuda_tensor(buf, shape);
        
        // Step by -3: from index 12 to 3
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(12..3).step(-3)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 12, 9, 6
        
        slice.set(&Idx::At(0), 212).unwrap(); // index 12
        slice.set(&Idx::At(1), 209).unwrap(); // index 9
        slice.set(&Idx::At(2), 206).unwrap(); // index 6
        
        // Verify
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 206);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 7);   // Unchanged
        assert_eq!(index_tensor(Idx::At(9), &view).unwrap(), 209);
        assert_eq!(index_tensor(Idx::At(12), &view).unwrap(), 212);
    }

    #[test]
    fn test_cuda_fill_contiguous() {
        let buf: Vec<i32> = (0..12).collect();
        let mut tensor = make_cuda_tensor(buf, vec![3, 4]);
        tensor.view_mut().fill(42).unwrap();
        
        // Verify all elements are 42
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(index_tensor(Idx::Coord(vec![i, j]), &tensor.view()).unwrap(), 42);
            }
        }
    }

    #[test]
    fn test_cuda_fill_contiguous_slice() {
        let buf: Vec<i32> = (0..12).collect();
        let mut tensor = make_cuda_tensor(buf, vec![3, 4]);
        // Fill row 1
        tensor.view_mut().slice_mut(0, 1).unwrap().fill(99).unwrap();
        
        // Verify row 0 unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 0);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 3]), &tensor.view()).unwrap(), 3);
        
        // Verify row 1 filled with 99
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &tensor.view()).unwrap(), 99);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 99);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 99);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 3]), &tensor.view()).unwrap(), 99);
        
        // Verify row 2 unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &tensor.view()).unwrap(), 9);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2]), &tensor.view()).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 3]), &tensor.view()).unwrap(), 11);
    }

    #[test]
    fn test_cuda_fill_strided_column() {
        let buf: Vec<i32> = (0..12).collect();
        let mut tensor = make_cuda_tensor(buf, vec![3, 4]);
        // Fill column 2 (strided access)
        tensor.view_mut().slice_mut(1, 2).unwrap().fill(7).unwrap();
        
        // Check the filled column
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2]), &tensor.view()).unwrap(), 7);
        
        // Check other columns are unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 0);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 3]), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_cuda_fill_2d_slice() {
        let buf: Vec<i32> = (0..24).collect();
        let mut tensor = make_cuda_tensor(buf, vec![4, 6]);
        // Fill a 2D slice [1:3, 2:5]
        {
            let mut view = tensor.view_mut();
            let mut slice = view.slice_mut(0, 1..3).unwrap();
            let mut slice2d = slice.slice_mut(1, 2..5).unwrap();
            slice2d.fill(5).unwrap();
        }
        
        // Verify the filled region
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 3]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 4]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 3]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 4]), &tensor.view()).unwrap(), 5);
        
        // Verify unchanged regions
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 5]), &tensor.view()).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 2]), &tensor.view()).unwrap(), 20);
    }

    #[test]
    fn test_cuda_fill_f32() {
        let buf: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let mut tensor = make_cuda_tensor(buf, vec![2, 3]);
        tensor.view_mut().fill(3.14).unwrap();
        
        for i in 0..2 {
            for j in 0..3 {
                let val = index_tensor(Idx::Coord(vec![i, j]), &tensor.view()).unwrap();
                assert!((val - 3.14).abs() < 1e-6);
            }
        }
    }
}
