pub mod base;
pub mod scalar;
pub mod unary;
pub mod broadcast;
pub mod linalg;
pub mod reduction;

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use crate::{backend::cpu::Cpu, core::{meta::MetaTensorView, primitives::TensorBase, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}, Tensor, Slice}};

    
    #[test]
    fn test_add() {
        let mut tensor: TensorBase<i32, Cpu> = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += &5;

        let expected = TensorBase::<i32, Cpu>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_add_ref() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view += &value;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_add_negative() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // same for sub
    #[test]
    fn test_sub() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= 5;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_sub_ref() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view -= &value;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![0, 10, 20], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_sub_negative() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= -5;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // same for mul

    #[test]
    fn test_mul() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_div() {
        let mut tensor = TensorBase::<f32, Cpu>::from_buf(vec![1., 2., 3.], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view /= 5.;
        let expected = TensorBase::<f32, Cpu>::from_buf(vec![0.2, 0.4, 0.6], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_mul_ref() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view *= &value;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    #[test]
    fn test_mul_negative() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // Tests with reshaping/slicing

    // #[test]
    // fn test_add_after_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![3, 2]).unwrap();
    //     reshaped += 10;
        
    //     // Original tensor should be modified
    //     let expected = TensorBase::<i32, Cpu>::from_buf(vec![11, 12, 13, 14, 15, 16], vec![2, 3]).unwrap();
    //     assert_eq!(tensor.raw.clone(), expected.raw);
    // }

    #[test]
    fn test_add_after_slice() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0]).unwrap(), 4, "Slice[0] should be 4");
        assert_eq!(slice.get(vec![1]).unwrap(), 5, "Slice[1] should be 5");
        assert_eq!(slice.get(vec![2]).unwrap(), 6, "Slice[2] should be 6");
        
        slice += 100;
        
        // Only the sliced part should be modified
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 104, 105, 106], vec![2, 3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // #[test]
    // fn test_add_after_slice_and_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    //     let mut view = tensor.view_mut();
    //     let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
    //     let mut reshaped = slice.view_as(vec![1, 3]).unwrap();
    //     reshaped += 50;
        
    //     // Only the sliced part should be modified
    //     let expected = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 54, 55, 56], vec![2, 3]).unwrap();
    //     assert_eq!(tensor.raw.clone(), expected.raw);
    // }

    // #[test]
    // fn test_sub_after_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4]).unwrap();
    //     reshaped -= 5;
        
    //     let expected = TensorBase::<i32, Cpu>::from_buf(vec![5, 15, 25, 35], vec![2, 2]).unwrap();
    //     assert_eq!(tensor.raw.clone(), expected.raw);
    // }

    #[test]
    fn test_mul_after_slice() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 0..0).unwrap(); // First depth slice: [1, 2, 3, 4]
        
        // Verify slice values before mutation
        use crate::core::tensor::TensorAccess;
        assert_eq!(slice.get(vec![0, 0]).unwrap(), 1, "Slice[0,0] should be 1");
        assert_eq!(slice.get(vec![0, 1]).unwrap(), 2, "Slice[0,1] should be 2");
        assert_eq!(slice.get(vec![1, 0]).unwrap(), 3, "Slice[1,0] should be 3");
        assert_eq!(slice.get(vec![1, 1]).unwrap(), 4, "Slice[1,1] should be 4");
        
        slice *= 10;
        
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30, 40, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // #[test]
    // fn test_add_scalar_reshaped_to_matrix() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![42], vec![1]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![1, 1]).unwrap();
    //     reshaped += 8;
        
    //     let expected = TensorBase::<i32, Cpu>::from_buf(vec![50], vec![1]).unwrap();
    //     assert_eq!(tensor.raw.clone(), expected.raw);
    // }

    #[test]
    fn test_mul_after_column_slice() {
        // Create a matrix and slice a column (non-contiguous)
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // Verify slice values (non-contiguous access)
        use crate::core::tensor::TensorAccess;
        assert_eq!(col_slice.get(vec![0]).unwrap(), 2, "Column slice[0] should be 2");
        assert_eq!(col_slice.get(vec![1]).unwrap(), 5, "Column slice[1] should be 5");
        
        col_slice *= 3;
        
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![1, 6, 3, 4, 15, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // #[test]
    // fn test_sub_ref_after_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![100, 200, 300, 400], vec![2, 2]).unwrap();
    //     let value = 50;
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4]).unwrap();
    //     reshaped -= &value;
        
    //     let expected = TensorBase::<i32, Cpu>::from_buf(vec![50, 150, 250, 350], vec![2, 2]).unwrap();
    //     assert_eq!(tensor.raw.clone(), expected.raw);
    // }

    #[test]
    fn test_add_ref_after_slice_chain() {
        // Create a 3D tensor and chain multiple slices
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let value = 1000;
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth: [6, 7, 8, 9] -> wait, buffer is [5,6,7,8]
        
        // Verify depth slice values
        use crate::core::tensor::TensorAccess;
        assert_eq!(depth_slice.get(vec![0, 0]).unwrap(), 5, "Depth slice[0,0] should be 5");
        assert_eq!(depth_slice.get(vec![0, 1]).unwrap(), 6, "Depth slice[0,1] should be 6");
        assert_eq!(depth_slice.get(vec![1, 0]).unwrap(), 7, "Depth slice[1,0] should be 7");
        assert_eq!(depth_slice.get(vec![1, 1]).unwrap(), 8, "Depth slice[1,1] should be 8");
        
        let mut row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row of that: [5, 6]
        assert_eq!(row_slice.get(vec![0]).unwrap(), 5, "Row slice[0] should be 5");
        assert_eq!(row_slice.get(vec![1]).unwrap(), 6, "Row slice[1] should be 6");
        
        row_slice += value;
        
        let expected = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 1005, 1006, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.buf.clone(), expected.buf);
    }

    // Tests for non-inplace operations (consume view, return new tensor)

    #[test]
    fn test_add_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + 5;
        
        // Result should be a new tensor with added values
        assert_eq!(result.buf, vec![6, 7, 8].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original tensor should be unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_ref_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view + value;
        
        assert_eq!(result.buf, vec![11, 12, 13].into_boxed_slice());
        assert_eq!(tensor.buf, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view - 5;
        
        assert_eq!(result.buf, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.buf, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_sub_ref_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view - value;
        
        assert_eq!(result.buf, vec![0, 10, 20].into_boxed_slice());
        assert_eq!(tensor.buf, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * 5;
        
        assert_eq!(result.buf, vec![5, 10, 15].into_boxed_slice());
        assert_eq!(tensor.buf, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_mul_ref_not_inplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view * &value;
        
        assert_eq!(result.buf, vec![10, 20, 30].into_boxed_slice());
        assert_eq!(tensor.buf, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_with_slice() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let result = slice + 100;
        
        // Result should be a new 1D tensor with shape [3]
        assert_eq!(result.buf, vec![104, 105, 106].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original tensor should be unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice() {
        // Test with non-contiguous column slice
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous(), "Column slice should be non-contiguous");
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        assert_eq!(result.buf, vec![6, 15].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    // #[test]
    // fn test_sub_not_inplace_with_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    //     let view = tensor.view_mut();
    //     let reshaped = view.view_as(vec![4]).unwrap();
    //     let result = reshaped - 5;
        
    //     // Result should have the reshaped dimensions
    //     assert_eq!(result.raw, vec![5, 15, 25, 35].into_boxed_slice());
    //     assert_eq!(*result.shape(), vec![4]);
        
    //     // Original unchanged
    //     assert_eq!(tensor.raw, vec![10, 20, 30, 40].into_boxed_slice());
    // }

    #[test]
    fn test_add_not_inplace_negative_values() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + (-5);
        
        assert_eq!(result.buf, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(tensor.buf, vec![10, 20, 30].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_negative_values() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * (-5);
        
        assert_eq!(result.buf, vec![-5, -10, -15].into_boxed_slice());
        assert_eq!(tensor.buf, vec![1, 2, 3].into_boxed_slice());
    }

    #[test]
    fn test_add_not_inplace_chained_slices() {
        // Test with chained slices to ensure view_to_owned handles complex cases
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth: [5, 6, 7, 8]
        let row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row: [5, 6]
        
        let result = row_slice + 1000;
        
        assert_eq!(result.buf, vec![1005, 1006].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_not_inplace_matrix() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let view = tensor.view_mut();
        let result = view * 10;
        
        assert_eq!(result.buf, vec![10, 20, 30, 40].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_sub_not_inplace_scalar() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![100], vec![]).unwrap();
        let view = tensor.view_mut();
        let result = view - 50;
        
        assert_eq!(result.buf, vec![50].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![100].into_boxed_slice());
    }

    // Tests for non-inplace operations on non-mutable views (TensorView)
    
    #[test]
    fn test_add_immutable_view_inline() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        // Call operation directly on view() without storing in variable
        let result = tensor.view() + 10;
        
        assert_eq!(result.buf, vec![11, 12, 13, 14].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4].into_boxed_slice());
    }

    #[test]
    fn test_add_immutable_view_ref() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        let value = 100;
        let result = tensor.view() + value;
        
        assert_eq!(result.buf, vec![105, 110, 115].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![5, 10, 15].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_inline() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![100, 200, 300], vec![3]).unwrap();
        let result = tensor.view() - 50;
        
        assert_eq!(result.buf, vec![50, 150, 250].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![100, 200, 300].into_boxed_slice());
    }

    #[test]
    fn test_sub_immutable_view_with_slice() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        // Slice to get first row, then subtract
        let result = tensor.view().slice(0, 0..0).unwrap() - 5;
        
        assert_eq!(result.buf, vec![5, 15, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![3]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_inline() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![2, 4, 6, 8], vec![4]).unwrap();
        let result = tensor.view() * 3;
        
        assert_eq!(result.buf, vec![6, 12, 18, 24].into_boxed_slice());
        assert_eq!(*result.shape(), vec![4]);
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![2, 4, 6, 8].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_matrix() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.view() * 10;
        
        assert_eq!(result.buf, vec![10, 20, 30, 40, 50, 60].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2, 3]);
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    // #[test]
    // fn test_add_immutable_view_after_reshape() {
    //     let tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
    //     // Reshape then add, all inline
    //     let result = tensor.view().view_as(vec![2, 2]).unwrap() + 100;
        
    //     assert_eq!(result.raw, vec![101, 102, 103, 104].into_boxed_slice());
    //     assert_eq!(*result.shape(), vec![2, 2]);
        
    //     // Original unchanged
    //     assert_eq!(tensor.raw, vec![1, 2, 3, 4].into_boxed_slice());
    // }

    #[test]
    fn test_sub_immutable_view_scalar() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![999], vec![]).unwrap();
        let result = tensor.view() - 99;
        
        assert_eq!(result.buf, vec![900].into_boxed_slice());
        assert_eq!(*result.shape(), vec![]);
        assert!(result.is_scalar());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![999].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_noncontiguous_slice() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        // Get column (non-contiguous) then multiply
        let result = tensor.view().slice(1, 1..1).unwrap() * 5;
        
        assert_eq!(result.buf, vec![10, 25].into_boxed_slice());
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    // #[test]
    // fn test_add_immutable_view_chained_operations() {
    //     let tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4]).unwrap();
    //     // Chain slice and reshape, then add
    //     let result = tensor.view()
    //         .slice(0, 0..0).unwrap()  // Get first row: [1, 2, 3, 4]
    //         .view_as(vec![2, 2]).unwrap()  // Reshape to 2x2
    //         + 1000;
        
    //     assert_eq!(result.raw, vec![1001, 1002, 1003, 1004].into_boxed_slice());
    //     assert_eq!(*result.shape(), vec![2, 2]);
        
    //     // Original unchanged
    //     assert_eq!(tensor.raw, vec![1, 2, 3, 4, 5, 6, 7, 8].into_boxed_slice());
    // }

    #[test]
    fn test_sub_immutable_view_negative_values() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![-10, -20, -30], vec![3]).unwrap();
        let result = tensor.view() - 5;
        
        assert_eq!(result.buf, vec![-15, -25, -35].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![-10, -20, -30].into_boxed_slice());
    }

    #[test]
    fn test_mul_immutable_view_with_ref() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![7, 14, 21], vec![3]).unwrap();
        let multiplier = 2;
        let result = tensor.view() * &multiplier;
        
        assert_eq!(result.buf, vec![14, 28, 42].into_boxed_slice());
        
        // Original unchanged
        assert_eq!(tensor.buf, vec![7, 14, 21].into_boxed_slice());
    }

    // ============================================================================
    // EDGE CASE TESTS - Priority 1: Critical Coverage
    // ============================================================================

    // --- Scalar Tensor Tests ---

    #[test]
    fn test_scalar_add_operation() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![42], vec![]).unwrap();
        assert!(tensor.is_scalar());
        let mut view = tensor.view_mut();
        view += 10;
        assert_eq!(tensor.buf, vec![52].into_boxed_slice());
    }

    #[test]
    fn test_scalar_mul_operation() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![7], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= 6;
        assert_eq!(tensor.buf, vec![42].into_boxed_slice());
    }

    #[test]
    fn test_scalar_sub_operation() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![100], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view -= 58;
        assert_eq!(tensor.buf, vec![42].into_boxed_slice());
    }

    #[test]
    fn test_scalar_negative_value() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![-10], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= -3;
        view += 5;
        assert_eq!(tensor.buf, vec![35].into_boxed_slice());
    }

    #[test]
    fn test_scalar_zero_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![0], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view += 42;
        view *= 0;
        view += 10;
        assert_eq!(tensor.buf, vec![10].into_boxed_slice());
    }

    #[test]
    fn test_scalar_non_inplace() {
        let tensor = TensorBase::<i32, Cpu>::from_buf(vec![100], vec![]).unwrap();
        let result = tensor.view() + 50;
        assert_eq!(result.buf, vec![150].into_boxed_slice());
        assert!(result.is_scalar());
        assert_eq!(tensor.buf, vec![100].into_boxed_slice());
    }

    // --- Single Element Tensor Tests ---

    #[test]
    fn test_single_element_1d() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![5], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        assert_eq!(tensor.buf, vec![30].into_boxed_slice());
    }

    #[test]
    fn test_single_element_2d() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![8], vec![1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        assert_eq!(tensor.buf, vec![40].into_boxed_slice());
    }

    #[test]
    fn test_single_element_3d() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![3], vec![1, 1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view += 7;
        view *= 3;
        assert_eq!(tensor.buf, vec![30].into_boxed_slice());
    }

    // --- Non-Square Tensor Tests ---

    #[test]
    fn test_tall_matrix_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![9, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        assert_eq!(tensor.buf, vec![2, 4, 6, 8, 10, 12, 14, 16, 18].into_boxed_slice());
    }

    #[test]
    fn test_wide_matrix_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![1, 9]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        assert_eq!(tensor.buf, vec![11, 12, 13, 14, 15, 16, 17, 18, 19].into_boxed_slice());
    }

    #[test]
    fn test_rectangular_matrix_2x5() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![2, 5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3;
        assert_eq!(
            tensor.buf, 
            vec![3, 6, 9, 12, 15, 18, 21, 24, 27, 30].into_boxed_slice()
        );
    }

    #[test]
    fn test_rectangular_matrix_5x2() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![5, 2]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 1;
        assert_eq!(
            tensor.buf, 
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_boxed_slice()
        );
    }

    #[test]
    fn test_rectangular_3d_tensor() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 3, 4]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected: Vec<i32> = (101..=124).collect();
        assert_eq!(tensor.buf, expected.into_boxed_slice());
    }

    #[test]
    fn test_non_square_slice_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [5, 6, 7, 8]
        slice *= 10;
        
        let expected = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        assert_eq!(tensor.buf, expected.buf);
    }

    // --- Edge Value Tests ---

    #[test]
    fn test_zero_tensor_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        assert_eq!(tensor.buf, vec![20, 20, 20, 20].into_boxed_slice());
    }

    #[test]
    fn test_operations_resulting_in_zero() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        assert_eq!(tensor.buf, vec![0, 0, 0, 0].into_boxed_slice());
    }

    #[test]
    fn test_mixed_positive_negative_values() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![-5, 10, -15, 20, -25, 30], 
            vec![6]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 10;
        
        assert_eq!(
            tensor.buf, 
            vec![0, 30, -20, 50, -40, 70].into_boxed_slice()
        );
    }

    #[test]
    fn test_large_values() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 500_000;
        
        assert_eq!(
            tensor.buf, 
            vec![1_500_000, 2_500_000, 3_500_000].into_boxed_slice()
        );
    }

    // --- Multiple Data Type Tests ---

    #[test]
    fn test_f32_operations() {
        let mut tensor = TensorBase::<f32, Cpu>::from_buf(
            vec![1.5, 2.5, 3.5, 4.5], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 1.0;
        
        let expected = vec![4.0, 6.0, 8.0, 10.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.buf[i] - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_operations() {
        let mut tensor = TensorBase::<f64, Cpu>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 0.5;
        view *= 2.0;
        
        let expected = vec![3.0, 5.0, 7.0, 9.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.buf[i] - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_i64_operations() {
        let mut tensor = TensorBase::<i64, Cpu>::from_buf(
            vec![100, 200, 300, 400], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 50;
        view *= 3;
        
        assert_eq!(tensor.buf, vec![150, 450, 750, 1050].into_boxed_slice());
    }

    #[test]
    fn test_u32_operations() {
        let mut tensor = TensorBase::<u32, Cpu>::from_buf(
            vec![10, 20, 30, 40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        view *= 2;
        
        assert_eq!(tensor.buf, vec![30, 50, 70, 90].into_boxed_slice());
    }

    #[test]
    fn test_i16_operations() {
        let mut tensor = TensorBase::<i16, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 10;
        view += 5;
        
        assert_eq!(tensor.buf, vec![15, 25, 35, 45, 55, 65].into_boxed_slice());
    }

    // --- Complex Non-Contiguous View Tests ---

    #[test]
    fn test_multiple_noncontiguous_operations() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![4, 3]
        ).unwrap();
        
        // Get middle column (non-contiguous)
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Column 1: [2, 5, 8, 11]
        col += 100;
        col *= 2;
        
        let expected = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9, 10, 222, 12], 
            vec![4, 3]
        ).unwrap();
        assert_eq!(tensor.buf, expected.buf);
    }

    // #[test]
    // fn test_noncontiguous_slice_after_reshape() {
    //     let mut tensor = TensorBase::<i32, Cpu>::from_buf(
    //         vec![1, 2, 3, 4, 5, 6, 7, 8], 
    //         vec![8]
    //     ).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4, 2]).unwrap();
    //     let mut col_slice = reshaped.slice_mut(1, 1..1).unwrap(); // Second column
    //     col_slice *= 10;
        
    //     // Second column in 4x2 is indices 1, 3, 5, 7
    //     let expected = TensorBase::<i32, Cpu>::from_buf(
    //         vec![1, 20, 3, 40, 5, 60, 7, 80], 
    //         vec![8]
    //     ).unwrap();
    //     assert_eq!(tensor.raw, expected.raw);
    // }

    #[test]
    fn test_3d_noncontiguous_slice() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 3, 4]).unwrap();
        
        // Slice along middle dimension at index 1 (middle row)
        // Shape is [2, 3, 4] - we're slicing dim 1 at idx 1
        // This removes the middle dimension, leaving shape [2, 4]
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(1, 1..1).unwrap(); // Middle "row" at each depth level
        // make sure that the stride and all is expected
        assert_eq!(*slice.shape(), vec![2, 4]);
        assert!(!slice.is_contiguous(), "Slice should be non-contiguous");
        assert_eq!(*slice.strides(), vec![12, 1]); // Original strides were [12,4,1]
        assert_eq!(slice.offset(), 4); // Starting at index 4 in flat array

        slice += 100;

        
        // Original data is laid out as:
        // Depth 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // Depth 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
        // Slicing at dim=1, idx=1 selects the middle row from each depth:
        // Result: [[5,6,7,8], [17,18,19,20]]
        // These are at indices: 4-7 and 16-19 in the flat array
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in [5, 6, 7, 8, 17, 18, 19, 20].iter() {
            expected_data[i - 1] += 100;
        }
        
        assert_eq!(tensor.buf, expected_data.into_boxed_slice());
    }
    #[test]
    fn test_3d_noncontiguous_slice_view() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 3, 4]).unwrap();
        
        // Slice along middle dimension at index 1 (middle row)
        // Shape is [2, 3, 4] - we're slicing dim 1 at idx 1
        // This removes the middle dimension, leaving shape [2, 4]
        let view = tensor.view_mut();
        let slice = view.slice(1, 1..1).unwrap(); // Middle "row" at each depth level
        
        let expected_data: Vec<i32> = vec![5, 6, 7, 8, 17, 18, 19, 20];
        let mut actual_data: Vec<i32> = Vec::new();
        for i in 0..slice.size() {
            // shape is [2,4] => size 8
            let idx = vec![i / 4, i % 4]; // Map flat index to 2D indices
            actual_data.push(slice.get(idx).unwrap());
        }

        assert_eq!(actual_data, expected_data);
    }

    // --- 4D and 5D Tensor Tests ---

    #[test]
    fn test_4d_tensor_operations() {
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        
        let expected: Vec<i32> = (11..=26).collect();
        assert_eq!(tensor.buf, expected.into_boxed_slice());
    }

    #[test]
    fn test_4d_slice_operation() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 3, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second slice along first dim
        slice *= 10;
        
        // Second half of data (indices 12-23)
        let mut expected: Vec<i32> = (1..=24).collect();
        for i in 12..24 {
            expected[i] *= 10;
        }
        
        assert_eq!(tensor.buf, expected.into_boxed_slice());
    }

    #[test]
    fn test_5d_tensor_operations() {
        let data: Vec<i32> = (1..=32).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 5;
        
        let expected: Vec<i32> = (1..=32).map(|x| x * 2 + 5).collect();
        assert_eq!(tensor.buf, expected.into_boxed_slice());
    }

    // --- Operation Chain Tests ---

    #[test]
    fn test_long_operation_chain() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        view += 1;   // [2, 3, 4, 5]
        view *= 2;   // [4, 6, 8, 10]
        view -= 1;   // [3, 5, 7, 9]
        view *= 3;   // [9, 15, 21, 27]
        view += 10;  // [19, 25, 31, 37]
        view -= 5;   // [14, 20, 26, 32]
        view *= 2;   // [28, 40, 52, 64]
        view += 2;   // [30, 42, 54, 66]
        view -= 10;  // [20, 32, 44, 56]
        view *= 1;   // [20, 32, 44, 56]
        
        assert_eq!(tensor.buf, vec![20, 32, 44, 56].into_boxed_slice());
    }

    #[test]
    fn test_alternating_inplace_noninplace() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        
        // Inplace
        let mut view1 = tensor.view_mut();
        view1 += 5;
        
        // Non-inplace
        let result1 = tensor.view() * 2;
        assert_eq!(result1.buf, vec![20, 30, 40].into_boxed_slice());
        
        // Inplace again
        let mut view2 = tensor.view_mut();
        view2 *= 3;
        
        // Non-inplace again
        let result2 = tensor.view() + 10;
        assert_eq!(result2.buf, vec![40, 55, 70].into_boxed_slice());
        
        // Original tensor should reflect inplace ops
        assert_eq!(tensor.buf, vec![30, 45, 60].into_boxed_slice());
    }

    #[test]
    fn test_multiple_views_different_ops() {
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        
        // First view - operate on first row
        {
            let mut view = tensor.view_mut();
            let mut row1 = view.slice_mut(0, 0..0).unwrap();
            row1 += 10;
        }
        
        // Second view - operate on second row
        {
            let mut view = tensor.view_mut();
            let mut row2 = view.slice_mut(0, 1..1).unwrap();
            row2 *= 5;
        }
        
        let expected = TensorBase::<i32, Cpu>::from_buf(
            vec![11, 12, 13, 20, 25, 30], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.buf, expected.buf);
    }

    // --- Float Edge Cases ---

    #[test]
    fn test_f32_small_values() {
        let mut tensor = TensorBase::<f32, Cpu>::from_buf(
            vec![0.0001, 0.0002, 0.0003], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= &1000.0;
        
        let expected = vec![0.1, 0.2, 0.3];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.buf[i] - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_precision() {
        let mut tensor = TensorBase::<f64, Cpu>::from_buf(
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3.0;
        
        let expected = vec![1.0, 2.0, 3.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((tensor.buf[i] - exp).abs() < 1e-10);
        }
    }

    // --- Stress Test: Very Imbalanced Shapes ---

    #[test]
    fn test_very_imbalanced_shape_1000x2() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![1000, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        
        // Check a few values
        assert_eq!(tensor.buf[0], 6);
        assert_eq!(tensor.buf[1], 7);
        assert_eq!(tensor.buf[1998], 2004);
        assert_eq!(tensor.buf[1999], 2005);
    }

    #[test]
    fn test_very_imbalanced_shape_2x1000() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = TensorBase::<i32, Cpu>::from_buf(data, vec![2, 1000]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        
        // Check a few values
        assert_eq!(tensor.buf[0], 2);
        assert_eq!(tensor.buf[999], 2000);
        assert_eq!(tensor.buf[1000], 2002);
        assert_eq!(tensor.buf[1999], 4000);
    }

        // BROADCASTING TESTS
    #[test]
    fn test_broadcast_flipped() {
        let mut veca = Tensor::<f32>::ones((1, 3));
        let vecb = Tensor::<f32>::ones((3, 1));
        veca.view_mut().set(vec![0, 0], 22.0).unwrap();

        let vecc = &veca.add(vecb.view());

        assert_eq!(vecc.shape().clone(), vec![3, 3]);
        for i in 0..3usize {
            for j in 0..3usize {
                if j == 0 {
                    assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 23.0);
                    continue;
                }
                assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 2.0);
            }
        }
    }

    #[test]
    fn test_broadcast_same() {
        let mut veca = Tensor::<f32>::ones((3,1));
        let vecb = Tensor::<f32>::ones((3, 1));
        veca.view_mut().set(vec![0, 0], 22.0).unwrap();

        let vecc = &veca.add(vecb.view());

        assert_eq!(vecc.shape().clone(), vec![3, 1]);
        for i in 0..3usize {
            for j in 0..1usize {
                if i == 0 {
                    assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 23.0);
                    continue;
                }
                assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 2.0);
            }
        }
    }

    // ============================================================================
    // COMPREHENSIVE BROADCAST TESTS - Higher Rank & Expected Shapes
    // ============================================================================

    // --- 3D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_3d_scalar_to_tensor() {
        let veca = Tensor::<f32>::from_buf(vec![5.0], vec![]).unwrap(); // scalar
        let vecb = Tensor::<f32>::ones((2, 3, 4));
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 6.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_vector_along_last_dim() {
        let veca = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap(); // (3,)
        let vecb = Tensor::<f32>::ones((2, 4, 3)); // (2, 4, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 3]);
        // Check pattern: last dimension should be [2, 3, 4] everywhere
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(vecc.view().get(vec![i, j, 0]).unwrap(), 2.0);
                assert_eq!(vecc.view().get(vec![i, j, 1]).unwrap(), 3.0);
                assert_eq!(vecc.view().get(vec![i, j, 2]).unwrap(), 4.0);
            }
        }
    }

    #[test]
    fn test_broadcast_3d_matrix_to_tensor() {
        let veca = Tensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = Tensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_singleton_dims() {
        let veca = Tensor::<f32>::ones((1, 3, 1)); // (1, 3, 1)
        let vecb = Tensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_prepend_dims() {
        let veca = Tensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = Tensor::<f32>::ones((5, 3, 4)); // (5, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![5, 3, 4]);
        for i in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    // --- 4D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_4d_vector_to_tensor() {
        let veca = Tensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap(); // (2,)
        let vecb = Tensor::<f32>::ones((3, 4, 5, 2)); // (3, 4, 5, 2)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5, 2]);
        // Check pattern in last dimension
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k, 0]).unwrap(), 11.0);
                    assert_eq!(vecc.view().get(vec![i, j, k, 1]).unwrap(), 21.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_4d_complex_singletons() {
        let veca = Tensor::<f32>::ones((1, 1, 5, 1)); // (1, 1, 5, 1)
        let vecb = Tensor::<f32>::ones((2, 3, 5, 4)); // (2, 3, 5, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 5, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_4d_3d_to_4d() {
        let veca = Tensor::<f32>::ones((3, 5, 4)); // (3, 5, 4)
        let vecb = Tensor::<f32>::ones((2, 3, 5, 4)); // (2, 3, 5, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 5, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    // --- 5D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_5d_scalar_to_tensor() {
        let veca = Tensor::<f32>::from_buf(vec![100.0], vec![]).unwrap(); // scalar
        let vecb = Tensor::<f32>::ones((2, 2, 2, 2, 2)); // (2, 2, 2, 2, 2)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 2, 2, 2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            assert_eq!(vecc.view().get(vec![i, j, k, l, m]).unwrap(), 101.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_5d_multiple_singletons() {
        let veca = Tensor::<f32>::ones((1, 2, 1, 3, 1)); // (1, 2, 1, 3, 1)
        let vecb = Tensor::<f32>::ones((4, 2, 5, 3, 6)); // (4, 2, 5, 3, 6)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![4, 2, 5, 3, 6]);
        // Check a few representative values
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![3, 1, 4, 2, 5]).unwrap(), 2.0);
    }

    // --- Mixed Dimension Broadcasts (Valid Cases) ---

    #[test]
    fn test_broadcast_2d_to_4d() {
        let veca = Tensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = Tensor::<f32>::ones((2, 5, 3, 4)); // (2, 5, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 5, 3, 4]);
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..3 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_1d_to_5d() {
        let veca = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap(); // (3,)
        let vecb = Tensor::<f32>::ones((2, 4, 5, 6, 3)); // (2, 4, 5, 6, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 5, 6, 3]);
        // Check pattern in last dimension
        for i in 0..2 {
            for j in 0..4 {
                for k in 0..5 {
                    for l in 0..6 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 0]).unwrap(), 2.0);
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 1]).unwrap(), 3.0);
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 2]).unwrap(), 4.0);
                    }
                }
            }
        }
    }

    // ============================================================================
    // BROADCAST DIVISION TESTS
    // ============================================================================

    #[test]
    fn test_broadcast_div_scalar_to_tensor() {
        let veca = Tensor::<f32>::from_buf(vec![10.0], vec![]).unwrap(); // scalar
        let vecb = Tensor::<f32>::from_buf(vec![2.0; 6], vec![2, 3]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 3]);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(vecc.buf[i * 3 + j], 5.0); // 10.0 / 2.0
            }
        }
    }

    #[test]
    fn test_broadcast_div_3d_vector_along_last_dim() {
        let veca = Tensor::<f32>::ones((2, 3, 4));
        let vecb = Tensor::<f32>::from_buf(vec![1.0, 2.0, 4.0, 8.0], vec![4]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let expected = 1.0 / (1 << k) as f32; // 1.0, 0.5, 0.25, 0.125
                    assert_eq!(vecc.buf[i * 12 + j * 4 + k], expected);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_div_in_place() {
        let mut veca = Tensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let vecb = Tensor::<f32>::from_buf(vec![2.0], vec![]).unwrap(); // scalar
        
        veca /= vecb.view();
        
        assert_eq!(veca.buf[0], 5.0);
        assert_eq!(veca.buf[1], 10.0);
        assert_eq!(veca.buf[2], 15.0);
        assert_eq!(veca.buf[3], 20.0);
    }

    #[test]
    fn test_broadcast_div_2d_matrices() {
        let veca = Tensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let vecb = Tensor::<f32>::from_buf(vec![10.0, 20.0], vec![1, 2]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 2]);
        assert_eq!(vecc.buf[0], 10.0);  // 100 / 10
        assert_eq!(vecc.buf[1], 10.0);  // 200 / 20
        assert_eq!(vecc.buf[2], 30.0);  // 300 / 10
        assert_eq!(vecc.buf[3], 20.0);  // 400 / 20
    }

    // --- FAILING BROADCAST TESTS (Should Not Work) ---

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_dimensions_3d() {
        let veca = Tensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = Tensor::<f32>::ones((2, 5, 6)); // (2, 5, 6) - incompatible!
        
        // This should panic because 4 != 6 and 3 != 5
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_inner_dim() {
        let veca = Tensor::<f32>::ones((5,)); // (5,)
        let vecb = Tensor::<f32>::ones((2, 3, 7)); // (2, 3, 7)
        
        // Should fail because 5 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_middle_dim() {
        let veca = Tensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        let vecb = Tensor::<f32>::ones((2, 5, 4)); // (2, 5, 4)
        
        // Should fail because middle dimension 3 != 5 and neither is 1
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_4d() {
        let veca = Tensor::<f32>::ones((2, 3, 4, 5)); // (2, 3, 4, 5)
        let vecb = Tensor::<f32>::ones((2, 7, 4, 5)); // (2, 7, 4, 5)
        
        // Should fail because 3 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_5d_multiple() {
        let veca = Tensor::<f32>::ones((1, 2, 3, 4, 5)); // (1, 2, 3, 4, 5)
        let vecb = Tensor::<f32>::ones((6, 2, 7, 4, 5)); // (6, 2, 7, 4, 5)
        
        // Should fail because 3 != 7 (dim 2)
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_non_singleton_mismatch() {
        let veca = Tensor::<f32>::ones((4, 5, 6)); // (4, 5, 6)
        let vecb = Tensor::<f32>::ones((4, 3, 6)); // (4, 3, 6)
        
        // Should fail because 5 != 3 and neither is 1
        let _vecc = veca.add(vecb.view());
    }

    // --- Edge Case Broadcasts ---

    #[test]
    fn test_broadcast_all_singletons() {
        let veca = Tensor::<f32>::ones((1, 1, 1)); // (1, 1, 1)
        let vecb = Tensor::<f32>::ones((5, 6, 7)); // (5, 6, 7)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![5, 6, 7]);
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![4, 5, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_identity() {
        let veca = Tensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        let vecb = Tensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_alternating_singletons() {
        let veca = Tensor::<f32>::ones((1, 4, 1, 6)); // (1, 4, 1, 6)
        let vecb = Tensor::<f32>::ones((3, 1, 5, 1)); // (3, 1, 5, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5, 6]);
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![2, 3, 4, 5]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_trailing_singleton() {
        let veca = Tensor::<f32>::ones((3, 4, 1)); // (3, 4, 1)
        let vecb = Tensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_leading_singleton() {
        let veca = Tensor::<f32>::ones((1, 4, 5)); // (1, 4, 5)
        let vecb = Tensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    // --- High Rank Shape Tests (Focus on Expected Output Shapes) ---

    #[test]
    fn test_broadcast_shape_2d_plus_3d() {
        let veca = Tensor::<f32>::ones((4, 5)); // (4, 5)
        let vecb = Tensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (3, 4, 5)
        assert_eq!(*vecc.shape(), vec![3, 4, 5]);
    }

    #[test]
    fn test_broadcast_shape_1d_plus_4d() {
        let veca = Tensor::<f32>::ones((7,)); // (7,)
        let vecb = Tensor::<f32>::ones((2, 3, 4, 7)); // (2, 3, 4, 7)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 7)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 7]);
    }

    #[test]
    fn test_broadcast_shape_singleton_expansion() {
        let veca = Tensor::<f32>::ones((1, 5, 1)); // (1, 5, 1)
        let vecb = Tensor::<f32>::ones((4, 5, 6)); // (4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (4, 5, 6)
        assert_eq!(*vecc.shape(), vec![4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_prepending() {
        let veca = Tensor::<f32>::ones((5, 6)); // (5, 6)
        let vecb = Tensor::<f32>::ones((2, 3, 4, 5, 6)); // (2, 3, 4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_both_expand() {
        let veca = Tensor::<f32>::ones((1, 4, 1, 6)); // (1, 4, 1, 6)
        let vecb = Tensor::<f32>::ones((3, 1, 5, 6)); // (3, 1, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_scalar_expansion() {
        let veca = Tensor::<f32>::from_buf(vec![42.0], vec![]).unwrap(); // scalar
        let vecb = Tensor::<f32>::ones((2, 3, 4, 5, 6)); // (2, 3, 4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_complex_5d() {
        let veca = Tensor::<f32>::ones((1, 1, 3, 1, 5)); // (1, 1, 3, 1, 5)
        let vecb = Tensor::<f32>::ones((2, 4, 3, 6, 5)); // (2, 4, 3, 6, 5)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 4, 3, 6, 5)
        assert_eq!(*vecc.shape(), vec![2, 4, 3, 6, 5]);
    }

    // ============================================================================
    // BOTH TENSORS WITH SINGLETONS - Mutual Broadcasting Tests
    // ============================================================================

    #[test]
    fn test_broadcast_both_singletons_2d_pattern1() {
        // Classic row vs column vector
        let mut veca = Tensor::<f32>::ones((1, 4)); // (1, 4) - row vector
        let vecb = Tensor::<f32>::ones((3, 1)); // (3, 1) - column vector
        veca.view_mut().set(vec![0, 0], 5.0).unwrap();
        veca.view_mut().set(vec![0, 1], 6.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First column should be 5+1=6, second column 6+1=7, rest 1+1=2
        for i in 0..3 {
            assert_eq!(vecc.view().get(vec![i, 0]).unwrap(), 6.0);
            assert_eq!(vecc.view().get(vec![i, 1]).unwrap(), 7.0);
            assert_eq!(vecc.view().get(vec![i, 2]).unwrap(), 2.0);
            assert_eq!(vecc.view().get(vec![i, 3]).unwrap(), 2.0);
        }
    }

    #[test]
    fn test_broadcast_both_singletons_2d_pattern2() {
        // Flipped from classic pattern
        let mut veca = Tensor::<f32>::ones((4, 1)); // (4, 1) - column vector
        let vecb = Tensor::<f32>::ones((1, 5)); // (1, 5) - row vector
        veca.view_mut().set(vec![0, 0], 10.0).unwrap();
        veca.view_mut().set(vec![1, 0], 20.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![4, 5]);
        // First row should all be 10+1=11, second row 20+1=21, rest 1+1=2
        for j in 0..5 {
            assert_eq!(vecc.view().get(vec![0, j]).unwrap(), 11.0);
            assert_eq!(vecc.view().get(vec![1, j]).unwrap(), 21.0);
            assert_eq!(vecc.view().get(vec![2, j]).unwrap(), 2.0);
            assert_eq!(vecc.view().get(vec![3, j]).unwrap(), 2.0);
        }
    }

    #[test]
    fn test_broadcast_both_singletons_3d_complementary() {
        // (1, 3, 1) and (2, 1, 4) -> (2, 3, 4)
        let veca = Tensor::<f32>::ones((1, 3, 1)); // (1, 3, 1)
        let vecb = Tensor::<f32>::ones((2, 1, 4)); // (2, 1, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_both_singletons_3d_alternating() {
        // (1, 4, 5) and (3, 1, 5) -> (3, 4, 5)
        let mut veca = Tensor::<f32>::ones((1, 4, 5)); // (1, 4, 5)
        let mut vecb = Tensor::<f32>::ones((3, 1, 5)); // (3, 1, 5)
        veca.view_mut().set(vec![0, 0, 0], 100.0).unwrap();
        vecb.view_mut().set(vec![0, 0, 0], 200.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 4, 5]);
        // [0,0,0] should be 100+200=300
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 300.0);
        // veca broadcasts along dim 0, vecb broadcasts along dim 1
        // At [1,0,0]: veca[0,0,0]=100 + vecb[1,0,0]=1 = 101
        assert_eq!(vecc.view().get(vec![1, 0, 0]).unwrap(), 101.0);
        assert_eq!(vecc.view().get(vec![2, 0, 0]).unwrap(), 101.0);
        // At [0,1,0]: veca[0,1,0]=1 + vecb[0,0,0]=200 = 201
        assert_eq!(vecc.view().get(vec![0, 1, 0]).unwrap(), 201.0);
        assert_eq!(vecc.view().get(vec![0, 2, 0]).unwrap(), 201.0);
    }

    #[test]
    fn test_broadcast_both_singletons_3d_partial_overlap() {
        // (5, 1, 3) and (1, 4, 3) -> (5, 4, 3)
        let veca = Tensor::<f32>::ones((5, 1, 3)); // (5, 1, 3)
        let vecb = Tensor::<f32>::ones((1, 4, 3)); // (1, 4, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![5, 4, 3]);
        // All values should be 2.0
        for i in 0..5 {
            for j in 0..4 {
                for k in 0..3 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_both_singletons_4d_zigzag() {
        // (1, 3, 1, 5) and (2, 1, 4, 1) -> (2, 3, 4, 5)
        let veca = Tensor::<f32>::ones((1, 3, 1, 5)); // (1, 3, 1, 5)
        let vecb = Tensor::<f32>::ones((2, 1, 4, 1)); // (2, 1, 4, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5]);
        // Spot check some values
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 2, 3, 4]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_4d_complex() {
        // (1, 1, 6, 7) and (5, 4, 1, 1) -> (5, 4, 6, 7)
        let veca = Tensor::<f32>::ones((1, 1, 6, 7)); // (1, 1, 6, 7)
        let vecb = Tensor::<f32>::ones((5, 4, 1, 1)); // (5, 4, 1, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![5, 4, 6, 7]);
        // Check boundaries
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![4, 3, 5, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_5d_checkerboard() {
        // (1, 2, 1, 3, 1) and (4, 1, 5, 1, 6) -> (4, 2, 5, 3, 6)
        let veca = Tensor::<f32>::ones((1, 2, 1, 3, 1)); // (1, 2, 1, 3, 1)
        let vecb = Tensor::<f32>::ones((4, 1, 5, 1, 6)); // (4, 1, 5, 1, 6)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![4, 2, 5, 3, 6]);
        // Spot check
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![3, 1, 4, 2, 5]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_5d_all_different() {
        // (1, 3, 1, 1, 7) and (2, 1, 4, 5, 1) -> (2, 3, 4, 5, 7)
        let veca = Tensor::<f32>::ones((1, 3, 1, 1, 7)); // (1, 3, 1, 1, 7)
        let vecb = Tensor::<f32>::ones((2, 1, 4, 5, 1)); // (2, 1, 4, 5, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 7]);
        // Verify a few positions
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 1, 1, 1, 1]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 2, 3, 4, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_mixed_ranks() {
        // (1, 5) and (3, 1, 1) -> (3, 1, 5) - wait, this should be (3, 1, 5)? Let me recalculate
        // Actually: (1, 5) needs to be prepended -> (1, 1, 5), then broadcast with (3, 1, 1) -> (3, 1, 5)
        let veca = Tensor::<f32>::ones((1, 5)); // (1, 5)
        let vecb = Tensor::<f32>::ones((3, 1, 1)); // (3, 1, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 1, 5]);
        for i in 0..3 {
            for k in 0..5 {
                assert_eq!(vecc.view().get(vec![i, 0, k]).unwrap(), 2.0);
            }
        }
    }

    // --- FAILING TESTS: Both tensors with singletons but incompatible ---

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_2d() {
        // (3, 1) and (1, 5) would work, but (3, 4) and (5, 1) won't
        let veca = Tensor::<f32>::ones((3, 4)); // (3, 4) - no singleton!
        let vecb = Tensor::<f32>::ones((5, 1)); // (5, 1)
        
        // Should fail: 3 != 5 and 4 != 1
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_3d() {
        // (2, 1, 4) and (1, 3, 5) won't work because 4 != 5
        let veca = Tensor::<f32>::ones((2, 1, 4)); // (2, 1, 4)
        let vecb = Tensor::<f32>::ones((1, 3, 5)); // (1, 3, 5)
        
        // Should fail: 4 != 5
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_4d() {
        // (1, 3, 1, 6) and (2, 1, 4, 7) won't work because 6 != 7
        let veca = Tensor::<f32>::ones((1, 3, 1, 6)); // (1, 3, 1, 6)
        let vecb = Tensor::<f32>::ones((2, 1, 4, 7)); // (2, 1, 4, 7)
        
        // Should fail: 6 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_middle() {
        // (1, 4, 1) and (3, 1, 5) would give (3, 4, 5)
        // but (2, 4, 1) and (3, 1, 5) fails because 2 != 3
        let veca = Tensor::<f32>::ones((2, 4, 1)); // (2, 4, 1)
        let vecb = Tensor::<f32>::ones((3, 1, 5)); // (3, 1, 5)
        
        // Should fail: 2 != 3
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    fn test_add_assign_broadcast() {
        let mut tensor = Tensor::<i32>::zeros((3, 3));
        let tensor2 = Tensor::<i32>::ones((3,));
        tensor += tensor2.view();
        let expected = Tensor::<i32>::ones((3, 3));
        assert_eq!(tensor, expected);
    }

    #[test]
    #[should_panic]
    fn test_add_assign_broadcast_incompatible() {
        let mut tensor = Tensor::<i32>::zeros((1, 3));
        let tensor2 = Tensor::<i32>::ones((4, 3));
        tensor += tensor2.view(); // Should panic
    }

    // ============================================================================
    // SUBTRACTION TESTS - Broadcasting & Type Combinations
    // ============================================================================

    #[test]
    fn test_sub_broadcast_basic() {
        let mut veca = Tensor::<f32>::ones((3, 3));
        veca.view_mut().set(vec![0, 0], 5.0).unwrap();
        let vecb = Tensor::<f32>::ones((3, 1));
        
        let vecc = veca - vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![3, 3]);
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 4.0); // 5 - 1
        for i in 1..3 {
            for j in 0..3 {
                assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 0.0); // 1 - 1
            }
        }
    }

    #[test]
    fn test_sub_3d_broadcast() {
        let veca = Tensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let vecb = Tensor::<f32>::ones((2, 4, 3));
        
        let vecc = veca - vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 3]);
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(vecc.view().get(vec![i, j, 0]).unwrap(), 9.0);
                assert_eq!(vecc.view().get(vec![i, j, 1]).unwrap(), 19.0);
                assert_eq!(vecc.view().get(vec![i, j, 2]).unwrap(), 29.0);
            }
        }
    }

    #[test]
    fn test_sub_assign_broadcast() {
        let mut tensor = Tensor::<i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        let tensor2 = Tensor::<i32>::ones((3,));
        tensor -= tensor2.view();
        let expected = Tensor::<i32>::from_buf(vec![4, 9, 14], vec![3]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    #[should_panic]
    fn test_sub_assign_broadcast_incompatible() {
        let mut tensor = Tensor::<i32>::zeros((1, 3));
        let tensor2 = Tensor::<i32>::ones((4, 3));
        tensor -= tensor2.view();
    }

    // ============================================================================
    // MULTIPLICATION TESTS - Broadcasting & Type Combinations
    // ============================================================================

    #[test]
    fn test_mul_broadcast_basic() {
        let mut veca = Tensor::<f32>::ones((3, 3));
        veca.view_mut().set(vec![0, 0], 5.0).unwrap();
        let mut vecb = Tensor::<f32>::ones((3, 1));
        vecb.view_mut().set(vec![1, 0], 2.0).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![3, 3]);
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 5.0); // 5 * 1
        assert_eq!(vecc.view().get(vec![1, 0]).unwrap(), 2.0); // 1 * 2
        assert_eq!(vecc.view().get(vec![2, 2]).unwrap(), 1.0); // 1 * 1
    }

    #[test]
    fn test_mul_3d_broadcast() {
        let veca = Tensor::<f32>::from_buf(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let vecb = Tensor::<f32>::ones((2, 4, 3));
        
        let vecc = veca * vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 3]);
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(vecc.view().get(vec![i, j, 0]).unwrap(), 2.0);
                assert_eq!(vecc.view().get(vec![i, j, 1]).unwrap(), 3.0);
                assert_eq!(vecc.view().get(vec![i, j, 2]).unwrap(), 4.0);
            }
        }
    }

    #[test]
    fn test_mul_scalar_broadcast() {
        let veca = Tensor::<f32>::from_buf(vec![5.0], vec![]).unwrap();
        let vecb = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(vecc.shape().clone(), vec![2, 2]);
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 5.0);
        assert_eq!(vecc.view().get(vec![0, 1]).unwrap(), 10.0);
        assert_eq!(vecc.view().get(vec![1, 0]).unwrap(), 15.0);
        assert_eq!(vecc.view().get(vec![1, 1]).unwrap(), 20.0);
    }

    #[test]
    fn test_mul_assign_broadcast() {
        let mut tensor = Tensor::<i32>::from_buf(vec![2, 4, 6], vec![3]).unwrap();
        let tensor2 = Tensor::<i32>::from_buf(vec![3, 3, 3], vec![3]).unwrap();
        tensor *= tensor2.view();
        let expected = Tensor::<i32>::from_buf(vec![6, 12, 18], vec![3]).unwrap();
        assert_eq!(tensor, expected);
    }

    #[test]
    #[should_panic]
    fn test_mul_assign_broadcast_incompatible() {
        let mut tensor = Tensor::<i32>::zeros((1, 3));
        let tensor2 = Tensor::<i32>::ones((4, 3));
        tensor *= tensor2.view();
    }

    // ============================================================================
    // TYPE COMBINATION TESTS - Add, Sub, Mul with different LHS/RHS types
    // ============================================================================

    // --- TensorBase op TensorBase ---
    #[test]
    fn test_tensorbase_add_tensorbase() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a + b;
        assert_eq!(*c.shape(), vec![2, 3]);
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorbase_sub_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![5.0, 3.0], vec![2]).unwrap();
        let b = Tensor::<f32>::ones((2,));
        let c = a - b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 4.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorbase_mul_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![4.0, 5.0], vec![2]).unwrap();
        let c = a * b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 8.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 15.0);
    }

    // --- TensorBase op TensorView ---
    #[test]
    fn test_tensorbase_add_tensorview() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a + b.view();
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorbase_sub_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![3.0, 7.0], vec![2]).unwrap();
        let c = a - b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 7.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensorbase_mul_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![6.0, 7.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        let c = a * b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 12.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 21.0);
    }

    // --- TensorBase op TensorViewMut ---
    #[test]
    fn test_tensorbase_add_tensorviewmut() {
        let a = Tensor::<f32>::ones((2, 3));
        let mut b = Tensor::<f32>::ones((2, 3));
        let c = a + b.view_mut();
        assert_eq!(c.view().get(vec![1, 2]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorbase_sub_tensorviewmut() {
        let a = Tensor::<f32>::from_buf(vec![8.0, 9.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![3.0, 4.0], vec![2]).unwrap();
        let c = a - b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 5.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 5.0);
    }

    #[test]
    fn test_tensorbase_mul_tensorviewmut() {
        let a = Tensor::<f32>::from_buf(vec![5.0, 6.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![2.0, 2.0], vec![2]).unwrap();
        let c = a * b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 10.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 12.0);
    }

    // --- TensorBase op &TensorBase ---
    #[test]
    fn test_tensorbase_add_ref_tensorbase() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a + &b;
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorbase_sub_ref_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![15.0, 25.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![5.0, 10.0], vec![2]).unwrap();
        let c = a - &b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 10.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 15.0);
    }

    #[test]
    fn test_tensorbase_mul_ref_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![3.0, 4.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![5.0, 6.0], vec![2]).unwrap();
        let c = a * &b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 15.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 24.0);
    }

    // --- &TensorBase op TensorBase ---
    #[test]
    fn test_ref_tensorbase_add_tensorbase() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = &a + b;
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_ref_tensorbase_sub_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![20.0, 30.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![8.0, 12.0], vec![2]).unwrap();
        let c = &a - b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 12.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 18.0);
    }

    #[test]
    fn test_ref_tensorbase_mul_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![7.0, 8.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        let c = &a * b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 14.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 24.0);
    }

    // --- &TensorBase op &TensorBase ---
    #[test]
    fn test_ref_tensorbase_add_ref_tensorbase() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = &a + &b;
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_ref_tensorbase_sub_ref_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![100.0, 200.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![30.0, 70.0], vec![2]).unwrap();
        let c = &a - &b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 70.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 130.0);
    }

    #[test]
    fn test_ref_tensorbase_mul_ref_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![9.0, 10.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 4.0], vec![2]).unwrap();
        let c = &a * &b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 18.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 40.0);
    }

    // --- TensorView op TensorBase ---
    #[test]
    fn test_tensorview_add_tensorbase() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a.view() + b;
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorview_sub_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![50.0, 60.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![20.0, 25.0], vec![2]).unwrap();
        let c = a.view() - b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 30.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 35.0);
    }

    #[test]
    fn test_tensorview_mul_tensorbase() {
        let a = Tensor::<f32>::from_buf(vec![11.0, 12.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![3.0, 4.0], vec![2]).unwrap();
        let c = a.view() * b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 33.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 48.0);
    }

    // --- TensorView op TensorView ---
    #[test]
    fn test_tensorview_add_tensorview() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a.view() + b.view();
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorview_sub_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![70.0, 80.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![30.0, 35.0], vec![2]).unwrap();
        let c = a.view() - b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 40.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 45.0);
    }

    #[test]
    fn test_tensorview_mul_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![13.0, 14.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 5.0], vec![2]).unwrap();
        let c = a.view() * b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 26.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 70.0);
    }

    // --- TensorViewMut op TensorBase ---
    #[test]
    fn test_tensorviewmut_add_tensorbase() {
        let mut a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = a.view_mut() + b;
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorviewmut_sub_tensorbase() {
        let mut a = Tensor::<f32>::from_buf(vec![90.0, 100.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![40.0, 45.0], vec![2]).unwrap();
        let c = a.view_mut() - b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 50.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 55.0);
    }

    #[test]
    fn test_tensorviewmut_mul_tensorbase() {
        let mut a = Tensor::<f32>::from_buf(vec![15.0, 16.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        let c = a.view_mut() * b;
        assert_eq!(c.view().get(vec![0]).unwrap(), 30.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 48.0);
    }

    // --- TensorViewMut op TensorViewMut ---
    #[test]
    fn test_tensorviewmut_add_tensorviewmut() {
        let mut a = Tensor::<f32>::ones((2, 3));
        let mut b = Tensor::<f32>::ones((2, 3));
        let c = a.view_mut() + b.view_mut();
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorviewmut_sub_tensorviewmut() {
        let mut a = Tensor::<f32>::from_buf(vec![110.0, 120.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![50.0, 60.0], vec![2]).unwrap();
        let c = a.view_mut() - b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 60.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 60.0);
    }

    #[test]
    fn test_tensorviewmut_mul_tensorviewmut() {
        let mut a = Tensor::<f32>::from_buf(vec![17.0, 18.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![3.0, 4.0], vec![2]).unwrap();
        let c = a.view_mut() * b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 51.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 72.0);
    }

    // --- &TensorView op &TensorView ---
    #[test]
    fn test_ref_tensorview_add_ref_tensorview() {
        let a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        let c = &a.view() + &b.view();
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_ref_tensorview_sub_ref_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![150.0, 200.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![50.0, 100.0], vec![2]).unwrap();
        let c = &a.view() - &b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 100.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 100.0);
    }

    #[test]
    fn test_ref_tensorview_mul_ref_tensorview() {
        let a = Tensor::<f32>::from_buf(vec![19.0, 20.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 5.0], vec![2]).unwrap();
        let c = &a.view() * &b.view();
        assert_eq!(c.view().get(vec![0]).unwrap(), 38.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 100.0);
    }

    // --- &TensorViewMut op &TensorViewMut ---
    #[test]
    fn test_ref_tensorviewmut_add_ref_tensorviewmut() {
        let mut a = Tensor::<f32>::ones((2, 3));
        let mut b = Tensor::<f32>::ones((2, 3));
        let c = &a.view_mut() + &b.view_mut();
        assert_eq!(c.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_ref_tensorviewmut_sub_ref_tensorviewmut() {
        let mut a = Tensor::<f32>::from_buf(vec![250.0, 300.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![100.0, 150.0], vec![2]).unwrap();
        let c = &a.view_mut() - &b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 150.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 150.0);
    }

    #[test]
    fn test_ref_tensorviewmut_mul_ref_tensorviewmut() {
        let mut a = Tensor::<f32>::from_buf(vec![21.0, 22.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![3.0, 4.0], vec![2]).unwrap();
        let c = &a.view_mut() * &b.view_mut();
        assert_eq!(c.view().get(vec![0]).unwrap(), 63.0);
        assert_eq!(c.view().get(vec![1]).unwrap(), 88.0);
    }

    // --- AddAssign/SubAssign/MulAssign for TensorViewMut ---
    #[test]
    fn test_tensorviewmut_add_assign_tensorbase() {
        let mut a = Tensor::<f32>::ones((2, 3));
        let b = Tensor::<f32>::ones((2, 3));
        {
            let mut view = a.view_mut();
            view += b;
        }
        assert_eq!(a.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorviewmut_sub_assign_tensorview() {
        let mut a = Tensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![3.0, 7.0], vec![2]).unwrap();
        {
            let mut view = a.view_mut();
            view -= b.view();
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 7.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensorviewmut_mul_assign_tensorviewmut() {
        let mut a = Tensor::<f32>::from_buf(vec![5.0, 6.0], vec![2]).unwrap();
        let mut b = Tensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        {
            let mut view_a = a.view_mut();
            let view_b = b.view_mut();
            view_a *= view_b;
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 18.0);
    }

    #[test]
    fn test_broadcast_post_slice() {
        let mut a = Tensor::<f32>::ones((4, 3));
        let mut b = Tensor::<f32>::ones((2, 1));

        let c = a.slice_mut(0, Slice::from(0..3).step(2)).unwrap();
        assert_eq!(*c.shape(), vec![2, 3]);
        
        b.set(vec![0, 0], 2.0).unwrap();

        let d = c + b;
        let expected = vec![
            3.0, 3.0, 3.0, 
            2.0, 2.0, 2.0
        ];
        let expected = Tensor::from_buf(expected, vec![2, 3]).unwrap();
        assert_eq!(expected, d);

    }

    #[test]
    fn test_broadcast_with_negative_step_slice() {
        // Test broadcasting with reversed slice (negative step)
        let mut a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![4, 3]
        ).unwrap();
        
        // Slice with negative step: reverse the rows
        let sliced = a.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*sliced.shape(), vec![4, 3]);
        
        // Broadcast a vector along the reversed dimension
        let b = Tensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let result = sliced + b;
        
        // Expected: rows reversed, then each row gets [10, 20, 30] added
        let expected = vec![
            20.0, 31.0, 42.0,  // row 3 (10, 11, 12) + (10, 20, 30)
            17.0, 28.0, 39.0,  // row 2 (7, 8, 9) + (10, 20, 30)
            14.0, 25.0, 36.0,  // row 1 (4, 5, 6) + (10, 20, 30)
            11.0, 22.0, 33.0,  // row 0 (1, 2, 3) + (10, 20, 30)
        ];
        let expected = Tensor::from_buf(expected, vec![4, 3]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_with_column_slice_negative_step() {
        // Test broadcasting with reversed column slice
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        // Slice columns in reverse
        let sliced = a.slice(1, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3]);
        
        // Broadcast a column vector
        let b = Tensor::<f32>::from_buf(vec![100.0, 200.0], vec![2, 1]).unwrap();
        let result = sliced + b;
        
        // Expected: columns reversed [3,2,1,6,5,4], then broadcast
        let expected = vec![
            103.0, 102.0, 101.0,  // [3, 2, 1] + 100
            206.0, 205.0, 204.0,  // [6, 5, 4] + 200
        ];
        let expected = Tensor::from_buf(expected, vec![2, 3]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_with_step_2_slice() {
        // Test broadcasting with step=2 slice
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8]
        ).unwrap();
        
        // Every other element
        let sliced = a.slice(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        
        // Broadcast a scalar
        let b = Tensor::<f32>::from_buf(vec![10.0], vec![]).unwrap();
        let result = sliced + b;
        
        // Expected: [1, 3, 5, 7] + 10
        let expected = vec![11.0, 13.0, 15.0, 17.0];
        let expected = Tensor::from_buf(expected, vec![4]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_with_negative_step_2_slice() {
        // Test broadcasting with negative step=-2 slice
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8]
        ).unwrap();
        
        // Every other element in reverse
        let sliced = a.slice(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        
        // Broadcast a scalar
        let result = sliced + 100.0;
        
        // Expected: [8, 6, 4, 2] + 100
        let expected = vec![108.0, 106.0, 104.0, 102.0];
        let expected = Tensor::from_buf(expected, vec![4]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_with_3d_negative_step() {
        // Test 3D tensor with negative step slice and broadcasting
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 2, 2]
        ).unwrap();
        
        // Reverse the first dimension
        let sliced = a.slice(0, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*sliced.shape(), vec![3, 2, 2]);
        
        // Broadcast a matrix
        let b = Tensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let result = sliced + b;
        
        // Expected: depth reversed, then add [[10, 20], [30, 40]] to each slice
        let expected = vec![
            19.0, 30.0, 41.0, 52.0,  // [9, 10, 11, 12] + [10, 20, 30, 40]
            15.0, 26.0, 37.0, 48.0,  // [5, 6, 7, 8] + [10, 20, 30, 40]
            11.0, 22.0, 33.0, 44.0,  // [1, 2, 3, 4] + [10, 20, 30, 40]
        ];
        let expected = Tensor::from_buf(expected, vec![3, 2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_inplace_with_negative_step() {
        // Test inplace broadcasting with negative step slice
        let mut a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        // Reverse columns - view is [[3,2,1], [6,5,4]]
        let mut sliced = a.slice_mut(1, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3]);
        
        // Broadcast and modify in place - adds 10 to each element through the reversed view
        let b = Tensor::<f32>::from_buf(vec![10.0], vec![1, 1]).unwrap();
        sliced += b;
        
        // Original tensor modified: each position gets modified through the reversed view
        // Original [[1,2,3], [4,5,6]] with reversed columns means:
        // position [0,0]=1 gets +10 (accessed as view[0,2]), position [0,1]=2 gets +10 (view[0,1]), etc.
        let expected = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let expected = Tensor::from_buf(expected, vec![2, 3]).unwrap();
        assert_eq!(a, expected);
    }

    #[test]
    fn test_broadcast_chained_slices_with_negative_steps() {
        // Test broadcasting after chaining multiple slices with different steps
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();
        
        // First slice: reverse rows
        let sliced1 = a.slice(0, Slice::from(..).step(-1)).unwrap();
        // Second slice: take every other column
        let sliced2 = sliced1.slice(1, Slice::from(..).step(2)).unwrap();
        assert_eq!(*sliced2.shape(), vec![4, 2]);
        
        // Broadcast a row vector
        let b = Tensor::<f32>::from_buf(vec![100.0, 200.0], vec![1, 2]).unwrap();
        let result = sliced2 + b;
        
        // Expected: rows reversed, then cols [0, 2], then add [100, 200]
        let expected = vec![
            113.0, 215.0,  // row 3: [13, 15] + [100, 200]
            109.0, 211.0,  // row 2: [9, 11] + [100, 200]
            105.0, 207.0,  // row 1: [5, 7] + [100, 200]
            101.0, 203.0,  // row 0: [1, 3] + [100, 200]
        ];
        let expected = Tensor::from_buf(expected, vec![4, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_scalar_with_complex_negative_slice() {
        // Test scalar broadcasting to complexly sliced tensor
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![10]
        ).unwrap();
        
        // Slice from index 8 down to 2 with step -2
        let sliced = a.slice(0, Slice::new(Some(8), Some(2), -2)).unwrap();
        assert_eq!(*sliced.shape(), vec![3]);
        
        // Broadcast scalar
        let result = sliced + 100;
        
        // Expected: [9, 7, 5] + 100 (indices 8, 6, 4)
        let expected = vec![109, 107, 105];
        let expected = Tensor::from_buf(expected, vec![3]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_vector_to_negative_step_matrix_slice() {
        // Test vector broadcast to matrix with negative step slice
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![4, 3]
        ).unwrap();
        
        // Take rows 3, 1 (reverse with step -2 starting from index 3)
        let sliced = a.slice(0, Slice::new(Some(3), None, -2)).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3]);
        
        // Broadcast vector along columns
        let b = Tensor::<f32>::from_buf(vec![5.0, 10.0], vec![2, 1]).unwrap();
        let result = sliced + b;
        
        // Expected: rows [3, 1] = [[10,11,12], [4,5,6]] + [[5], [10]]
        let expected = vec![
            15.0, 16.0, 17.0,  // [10, 11, 12] + 5
            14.0, 15.0, 16.0,  // [4, 5, 6] + 10
        ];
        let expected = Tensor::from_buf(expected, vec![2, 3]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_both_tensors_sliced_negative_steps() {
        // Test broadcasting when both tensors have negative step slices
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        let b = Tensor::<f32>::from_buf(
            vec![10.0, 20.0, 30.0],
            vec![1, 3]
        ).unwrap();
        
        // Reverse columns of a, reverse columns of b
        let sliced_a = a.slice(1, Slice::from(..).step(-1)).unwrap();
        let sliced_b = b.slice(1, Slice::from(..).step(-1)).unwrap();
        
        let result = sliced_a + sliced_b;
        
        // a columns reversed: [[3, 2, 1], [6, 5, 4]]
        // b columns reversed: [30, 20, 10]
        let expected = vec![
            33.0, 22.0, 11.0,  // [3, 2, 1] + [30, 20, 10]
            36.0, 25.0, 14.0,  // [6, 5, 4] + [30, 20, 10]
        ];
        let expected = Tensor::from_buf(expected, vec![2, 3]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_broadcast_4d_with_negative_step_multiple_dims() {
        // Test 4D broadcasting with negative steps on multiple dimensions
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
            vec![2, 2, 2, 2]
        ).unwrap();
        
        // Reverse dimension 0 and dimension 2
        let sliced1 = a.slice(0, Slice::from(..).step(-1)).unwrap();
        let sliced2 = sliced1.slice(2, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*sliced2.shape(), vec![2, 2, 2, 2]);
        
        // Broadcast a vector along last dimension
        let b = Tensor::<f32>::from_buf(vec![100.0, 200.0], vec![2]).unwrap();
        let result = sliced2 + b;
        
        // Complex reversal pattern + broadcasting
        assert_eq!(*result.shape(), vec![2, 2, 2, 2]);
        // After reversing dims 0 and 2: position [0,0,0,:] has values [11, 12]
        assert_eq!(result.get(vec![0, 0, 0, 0]).unwrap(), 111.0); // 11 + 100
        assert_eq!(result.get(vec![0, 0, 0, 1]).unwrap(), 212.0); // 12 + 200
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod cuda_tests {
    use std::ops::Add;

    use crate::{core::{primitives::CudaTensor, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut}, Tensor, MetaTensorView}};


    #[test]
    fn test_add_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        let expected = Tensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= 4;
        let expected = Tensor::<i32>::from_buf(vec![4, 8, 12], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_div_cuda() {
        let mut tensor = CudaTensor::<f32>::from_buf(vec![1., 2., 3.], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view /= 4.;
        let expected = Tensor::<f32>::from_buf(vec![0.25, 0.5, 0.75], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= 7;
        let expected = Tensor::<i32>::from_buf(vec![3, 13, 23], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view += value;
        let expected = Tensor::<i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }
    
    #[test]
    fn test_sub_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view -= value;
        let expected = Tensor::<i32>::from_buf(vec![0, 10, 20], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_ref_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let mut view = tensor.view_mut();
        view *= &value;
        let expected = Tensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_add_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = Tensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_sub_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view -= -5;
        let expected = Tensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_negative_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = Tensor::<i32>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // Tests with reshaping/slicing

    // #[test]
    // fn test_add_after_reshape_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![3, 2]).unwrap();
    //     reshaped += 10;
        
    //     let expected = CpuTensor::<i32>::from_buf(vec![11, 12, 13, 14, 15, 16], vec![2, 3]).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_add_after_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        // Verify slice values before mutation
        assert_eq!(slice.get(vec![0]).unwrap(), 4);
        assert_eq!(slice.get(vec![1]).unwrap(), 5);
        assert_eq!(slice.get(vec![2]).unwrap(), 6);
        
        slice += 100;
        
        let expected = Tensor::<i32>::from_buf(vec![1, 2, 3, 104, 105, 106], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // #[test]
    // fn test_add_after_slice_and_reshape_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
    //     let mut view = tensor.view_mut();
    //     let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
    //     let mut reshaped = slice.view_as(vec![1, 3]).unwrap();
    //     reshaped += 50;
        
    //     let expected = CpuTensor::<i32>::from_buf(vec![1, 2, 3, 54, 55, 56], vec![2, 3]).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    // #[test]
    // fn test_sub_after_reshape_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4]).unwrap();
    //     reshaped -= 5;
        
    //     let expected = CpuTensor::<i32>::from_buf(vec![5, 15, 25, 35], vec![2, 2]).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_mul_after_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 0..0).unwrap(); // First depth slice
        
        // Verify slice values before mutation
        assert_eq!(slice.get(vec![0, 0]).unwrap(), 1);
        assert_eq!(slice.get(vec![0, 1]).unwrap(), 2);
        assert_eq!(slice.get(vec![1, 0]).unwrap(), 3);
        assert_eq!(slice.get(vec![1, 1]).unwrap(), 4);
        
        slice *= 10;
        
        let expected = Tensor::<i32>::from_buf(vec![10, 20, 30, 40, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // #[test]
    // fn test_add_scalar_reshaped_to_matrix_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(vec![42], vec![1]).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![1, 1]).unwrap();
    //     reshaped += 8;
        
    //     let expected = CpuTensor::<i32>::from_buf(vec![50], vec![1]).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_mul_after_column_slice_cuda() {
        // Create a matrix and slice a column (non-contiguous)
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // Verify slice values (non-contiguous access)
        assert_eq!(col_slice.get(vec![0]).unwrap(), 2);
        assert_eq!(col_slice.get(vec![1]).unwrap(), 5);
        
        col_slice *= 3;
        
        let expected = Tensor::<i32>::from_buf(vec![1, 6, 3, 4, 15, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // #[test]
    // fn test_sub_ref_after_reshape_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(vec![100, 200, 300, 400], vec![2, 2]).unwrap();
    //     let value = 50;
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4]).unwrap();
    //     reshaped -= &value;
        
    //     let expected = CpuTensor::<i32>::from_buf(vec![50, 150, 250, 350], vec![2, 2]).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_add_ref_after_slice_chain_cuda() {
        // Create a 3D tensor and chain multiple slices
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let value = 1000;
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Second depth
        
        // Verify depth slice values
        assert_eq!(depth_slice.get(vec![0, 0]).unwrap(), 5);
        assert_eq!(depth_slice.get(vec![0, 1]).unwrap(), 6);
        assert_eq!(depth_slice.get(vec![1, 0]).unwrap(), 7);
        assert_eq!(depth_slice.get(vec![1, 1]).unwrap(), 8);
        
        let mut row_slice = depth_slice.slice_mut(0, 0..0).unwrap(); // First row of that
        assert_eq!(row_slice.get(vec![0]).unwrap(), 5);
        assert_eq!(row_slice.get(vec![1]).unwrap(), 6);
        
        row_slice += value;
        
        let expected = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 1005, 1006, 7, 8], vec![2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // Tests for non-inplace operations (consume view, return new tensor)

    #[test]
    fn test_add_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view + 5;
        
        // Result should be a new tensor with added values
        let expected_result = Tensor::<i32>::from_buf(vec![6, 7, 8], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original tensor should be unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_add_ref_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let value = 10;
        let view = tensor.view_mut();
        let result = view + value;
        
        let expected_result = Tensor::<i32>::from_buf(vec![11, 12, 13], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_sub_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view - 5;
        
        let expected_result = Tensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = Tensor::<i32>::from_buf(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_not_inplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let view = tensor.view_mut();
        let result = view * 5;
        
        let expected_result = Tensor::<i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_add_not_inplace_with_slice_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        let result = slice + 100;
        
        // Result should be a new 1D tensor
        let expected_result = Tensor::<i32>::from_buf(vec![104, 105, 106], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert_eq!(*result.shape(), vec![3]);
        
        // Original tensor should be unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice_cuda() {
        // Test with non-contiguous column slice
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous());
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        let expected_result = Tensor::<i32>::from_buf(vec![6, 15], vec![2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert_eq!(*result.shape(), vec![2]);
        assert!(result.is_contiguous());
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // Tests for non-inplace operations on non-mutable views (TensorView)
    
    #[test]
    fn test_add_immutable_view_inline_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let result = tensor.view() + 10;
        
        let expected_result = Tensor::<i32>::from_buf(vec![11, 12, 13, 14], vec![4]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_sub_immutable_view_with_slice_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        let result = tensor.view().slice(0, 0..0).unwrap() - 5;
        
        let expected_result = Tensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_mul_immutable_view_matrix_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let result = tensor.view() * 10;
        
        let expected_result = Tensor::<i32>::from_buf(vec![10, 20, 30, 40, 50, 60], vec![2, 3]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // Large tensor test - avoid full CPU copy, only check select indices
    #[test]
    fn test_large_tensor_cuda() {
        const SIZE: usize = 10_000_000; // 10 million elements
        
        // Create a large tensor filled with sequential values
        let data: Vec<i32> = (0..SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![SIZE]).unwrap();
        
        // Apply operation
        let mut view = tensor.view_mut();
        view += 100;
        
        // Check only select indices to avoid copying 10M elements
        let indices_to_check = vec![
            0,              // First element
            1,              // Second element
            SIZE / 4,       // Quarter way
            SIZE / 2,       // Halfway
            3 * SIZE / 4,   // Three quarters
            SIZE - 2,       // Second to last
            SIZE - 1,       // Last element
        ];
        
        for &idx in &indices_to_check {
            let value = view.get(vec![idx]).unwrap();
            let expected = idx as i32 + 100;
            assert_eq!(
                value, 
                expected,
                "Mismatch at index {}: got {}, expected {}",
                idx, value, expected
            );
        }
    }

    #[test]
    fn test_large_tensor_mul_cuda() {
        const SIZE: usize = 5_000_000; // 5 million elements
        
        let data: Vec<i32> = (1..=SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![SIZE]).unwrap();
        
        let mut view = tensor.view_mut();
        view *= 3;
        
        // Check select indices
        let indices = vec![0, 100, 1000, SIZE / 2, SIZE - 100, SIZE - 1];
        
        for &idx in &indices {
            let value = view.get(vec![idx]).unwrap();
            let expected = (idx as i32 + 1) * 3;
            assert_eq!(value, expected, "Mismatch at index {}", idx);
        }
    }

    #[test]
    fn test_large_tensor_non_contiguous_cuda() {
        // Large matrix - test with non-contiguous slicing
        const ROWS: usize = 10_000;
        const COLS: usize = 1_000;
        const SIZE: usize = ROWS * COLS;
        
        let data: Vec<i32> = (0..SIZE as i32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![ROWS, COLS]).unwrap();
        
        // Slice a single column (non-contiguous) and add to it
        let mut view = tensor.view_mut();
        let mut col_slice = view.slice_mut(1, (COLS / 2)..(COLS / 2)).unwrap(); // Middle column
        col_slice += 999;
        
        // Check a few values in the modified column
        let row_indices = vec![0, ROWS / 4, ROWS / 2, ROWS - 1];
        
        for &row_idx in &row_indices {
            let value = col_slice.get(vec![row_idx]).unwrap();
            let original_value = (row_idx * COLS + COLS / 2) as i32;
            let expected = original_value + 999;
            assert_eq!(value, expected, "Mismatch at row {}", row_idx);
        }
    }

    #[test]
    fn test_large_tensor_f32_cuda() {
        const SIZE: usize = 8_000_000; // 8 million f32 elements
        
        let data: Vec<f32> = (0..SIZE).map(|i| i as f32 * 0.5).collect();
        let mut tensor = CudaTensor::<f32>::from_buf(data, vec![SIZE]).unwrap();
        
        let mut view = tensor.view_mut();
        view *= 2.0;
        
        // Check select indices
        let indices = vec![0, 1000, SIZE / 3, SIZE / 2, SIZE - 1000, SIZE - 1];
        
        for &idx in &indices {
            let value = view.get(vec![idx]).unwrap();
            let expected = (idx as f32 * 0.5) * 2.0;
            assert!((value - expected).abs() < 1e-5, "Mismatch at index {}: got {}, expected {}", idx, value, expected);
        }
    }

    #[test]
    fn test_cuda_performance_vs_cpu() {
        use std::time::Instant;
        use crate::backend::cpu::Cpu;
        
        const SIZE: usize = 20_000_000; // 20 million elements
        
        // Create identical data for CPU and GPU
        let data: Vec<f32> = (0..SIZE).map(|i| (i % 1000) as f32).collect();
        
        // CPU timing
        let mut cpu_tensor = crate::core::primitives::TensorBase::<f32, Cpu>::from_buf(
            data.clone(), 
            vec![SIZE]
        ).unwrap();
        
        let cpu_start = Instant::now();
        {
            let mut view = cpu_tensor.view_mut();
            view *= 3.14159;
            view += 2.71828;
            view -= 1.41421;
        }
        let cpu_duration = cpu_start.elapsed();
        
        // GPU timing - including data transfer
        let mut gpu_tensor = CudaTensor::<f32>::from_buf(data, vec![SIZE]).unwrap();
        
        let gpu_start = Instant::now();
        {
            let mut view = gpu_tensor.view_mut();
            view *= 3.14159;
            view += 2.71828;
            view -= 1.41421;
        }
        let gpu_duration = gpu_start.elapsed();
        
        // Verify results match (check select indices)
        let gpu_result = gpu_tensor.cpu().unwrap();
        
        let check_indices = vec![0, SIZE / 4, SIZE / 2, 3 * SIZE / 4, SIZE - 1];
        for &idx in &check_indices {
            let cpu_val = cpu_tensor.view().get(vec![idx]).unwrap();
            let gpu_val = gpu_result.view().get(vec![idx]).unwrap();
            assert!(
                (cpu_val - gpu_val).abs() < 1e-3,
                "CPU/GPU mismatch at index {}: CPU={}, GPU={}", idx, cpu_val, gpu_val
            );
        }
        
        println!("\nPerformance comparison ({} elements):", SIZE);
        println!("  CPU time: {:?}", cpu_duration);
        println!("  GPU time: {:?}", gpu_duration);
        println!("  Speedup: {:.2}x", cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64());
        
        // Assert GPU is faster - even with memory transfer overhead, GPU should be faster
        // for this size of operation. We use a conservative 1.1x threshold since we're including
        // data transfer time (both to GPU initially and back to CPU for verification)
        let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
        assert!(
            speedup > 1.1,
            "GPU should be faster than CPU for large tensors (even with transfer overhead). Got {:.2}x speedup",
            speedup
        );
    }

    // Additional comprehensive CUDA tests for edge cases and data types
    
    #[test]
    fn test_cuda_f64_operations() {
        let mut tensor = CudaTensor::<f64>::from_buf(vec![1.5, 2.5, 3.5, 4.5], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10.25;
        view *= 2.0;
        view -= 3.5;
        
        let expected = Tensor::<f64>::from_buf(
            vec![20.0, 22.0, 24.0, 26.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-10, "Mismatch at {}: {} vs {}", i, val, exp);
        }
    }

    #[test]
    fn test_cuda_i64_operations() {
        let mut tensor = CudaTensor::<i64>::from_buf(
            vec![100, 200, 300, 400, 500], 
            vec![5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 50;
        view *= 3;
        
        let expected = Tensor::<i64>::from_buf(
            vec![150, 450, 750, 1050, 1350], 
            vec![5]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_u32_operations() {
        let mut tensor = CudaTensor::<u32>::from_buf(
            vec![10, 20, 30, 40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        view *= 2;
        
        let expected = Tensor::<u32>::from_buf(
            vec![30, 50, 70, 90], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_i16_operations() {
        let mut tensor = CudaTensor::<i16>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 10;
        view += 5;
        
        let expected = Tensor::<i16>::from_buf(
            vec![15, 25, 35, 45, 55, 65], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_chained_operations_complex() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 2, 2]
        ).unwrap();
        
        // Complex chain: multiply, add, subtract, multiply again
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 10.0;
        view -= 5.0;
        view *= 0.5;
        
        // Expected: ((x * 2 + 10) - 5) * 0.5 = (x * 2 + 5) * 0.5 = x + 2.5
        let expected = Tensor::<f32>::from_buf(
            vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..8 {
            let idx = vec![i / 4, (i / 2) % 2, i % 2];
            let val = result.view().get(idx.clone()).unwrap();
            let exp = expected.view().get(idx.clone()).unwrap();
            assert!((val - exp).abs() < 1e-5, "Mismatch at {:?}: {} vs {}", idx, val, exp);
        }
    }

    #[test]
    fn test_cuda_zero_operations() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 0;
        view *= 1;
        view -= 0;
        
        let expected = Tensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_multiply_by_zero() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![100, 200, 300, 400], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        
        let expected = Tensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_very_small_tensor() {
        // Single element tensor
        let mut tensor = CudaTensor::<f32>::from_buf(vec![42.0], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 8.0;
        
        let expected = Tensor::<f32>::from_buf(vec![92.0], vec![1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_4d_tensor() {
        // 2x2x2x2 tensor
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected_data: Vec<i32> = (101..=116).collect();
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_mixed_operations_on_slice() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        
        let mut view = tensor.view_mut();
        let mut middle_row = view.slice_mut(0, 1..1).unwrap(); // [4, 5, 6]
        middle_row *= 10.0;
        middle_row += 5.0;
        
        let expected = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 45.0, 55.0, 65.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_alternating_operations() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![10, 20, 30, 40], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        // Alternating pattern
        view += 5;   // [15, 25, 35, 45]
        view *= 2;   // [30, 50, 70, 90]
        view -= 10;  // [20, 40, 60, 80]
        view *= 3;   // [60, 120, 180, 240]
        view += 10;  // [70, 130, 190, 250]
        
        let expected = Tensor::<i32>::from_buf(
            vec![70, 130, 190, 250], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // #[test]
    // fn test_cuda_reshaped_operations() {
    //     let mut tensor = CudaTensor::<f32>::from_buf(
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
    //         vec![6]
    //     ).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![2, 3]).unwrap();
    //     reshaped *= 5.0;
        
    //     let expected = CpuTensor::<f32>::from_buf(
    //         vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0], 
    //         vec![6]
    //     ).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_cuda_non_inplace_multiple_results() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let view = tensor.view_mut();
        
        // Create multiple independent results
        let result1 = view + 10;
        let view2 = tensor.view_mut();
        let result2 = view2 * 5;
        let view3 = tensor.view_mut();
        let result3 = view3 - 1;
        
        let expected1 = Tensor::<i32>::from_buf(vec![11, 12, 13, 14], vec![4]).unwrap();
        let expected2 = Tensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let expected3 = Tensor::<i32>::from_buf(vec![0, 1, 2, 3], vec![4]).unwrap();
        
        assert_eq!(result1.cpu().unwrap(), expected1);
        assert_eq!(result2.cpu().unwrap(), expected2);
        assert_eq!(result3.cpu().unwrap(), expected3);
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    #[test]
    fn test_cuda_scalar_edge_cases() {
        // Test with actual scalar (0-dimensional tensor)
        let mut tensor = CudaTensor::<f32>::from_buf(vec![3.14159], vec![]).unwrap();
        assert!(tensor.is_scalar());
        
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 1.0;
        
        let result = tensor.cpu().unwrap();
        let value = result.view().get(vec![]).unwrap();
        assert!((value - 7.28318).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_negative_to_positive() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![-10, -20, -30, -40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= -1;
        view += 5;
        
        let expected = Tensor::<i32>::from_buf(vec![15, 25, 35, 45], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_large_values_i32() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 1_000_000;
        view -= 500_000;
        
        let expected = Tensor::<i32>::from_buf(
            vec![1_500_000, 2_500_000, 3_500_000], 
            vec![3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_fractional_multiplication() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![10.0, 20.0, 30.0, 40.0], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0.5;
        view *= 0.25;
        
        let expected = Tensor::<f32>::from_buf(
            vec![1.25, 2.5, 3.75, 5.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cuda_column_slice_multiple_ops() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 
            vec![3, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Middle column [2, 5, 8]
        
        col += 100;
        col *= 2;
        
        let expected = Tensor::<i32>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9], 
            vec![3, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_3d_depth_slice_operations() {
        let data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        let mut tensor = CudaTensor::<f32>::from_buf(
            data, 
            vec![3, 3, 3]
        ).unwrap();
        
        let mut view = tensor.view_mut();
        let mut depth_slice = view.slice_mut(0, 1..1).unwrap(); // Middle depth slice
        depth_slice *= 10.0;
        
        let mut expected_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
        // Middle depth is indices 9-17 (0-indexed)
        for i in 9..18 {
            expected_data[i] *= 10.0;
        }
        
        let expected = Tensor::<f32>::from_buf(expected_data, vec![3, 3, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_cuda_ref_value_consistency() {
        // Test that ref and non-ref operations produce same results
        let data = vec![5, 10, 15, 20, 25];
        
        let mut tensor1 = CudaTensor::<i32>::from_buf(data.clone(), vec![5]).unwrap();
        let mut tensor2 = CudaTensor::<i32>::from_buf(data, vec![5]).unwrap();
        
        let value = 7;
        let mut view1 = tensor1.view_mut();
        view1 += value;  // Using owned value
        
        let mut view2 = tensor2.view_mut();
        view2 += value;  // Using owned value
        
        assert_eq!(tensor1.cpu().unwrap(), tensor2.cpu().unwrap());
    }

    // #[test]
    // fn test_cuda_immutable_view_complex_chain() {
    //     let tensor = CudaTensor::<f32>::from_buf(
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
    //         vec![2, 4]
    //     ).unwrap();
        
    //     // Complex immutable chain
    //     let result = tensor.view()
    //         .slice(0, 1..1).unwrap()  // Second row: [5, 6, 7, 8]
    //         .view_as(vec![2, 2]).unwrap()  // Reshape
    //         + 100.0;
        
    //     let expected = CpuTensor::<f32>::from_buf(
    //         vec![105.0, 106.0, 107.0, 108.0], 
    //         vec![2, 2]
    //     ).unwrap();
    //     assert_eq!(result.cpu().unwrap(), expected);
        
    //     // Original unchanged
    //     let original_expected = CpuTensor::<f32>::from_buf(
    //         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
    //         vec![2, 4]
    //     ).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), original_expected);
    // }

    // ============================================================================
    // EDGE CASE TESTS - Priority 1: Critical Coverage (CUDA)
    // ============================================================================

    // --- Scalar Tensor Tests ---

    #[test]
    fn test_scalar_add_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert!(tensor.is_scalar());
        let mut view = tensor.view_mut();
        view += 10;
        let expected = Tensor::<i32>::from_buf(vec![52], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_mul_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![7], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= 6;
        let expected = Tensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_sub_operation_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view -= 58;
        let expected = Tensor::<i32>::from_buf(vec![42], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_negative_value_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![-10], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view *= -3;
        view += 5;
        let expected = Tensor::<i32>::from_buf(vec![35], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_zero_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![0], vec![]).unwrap();
        let mut view = tensor.view_mut();
        view += 42;
        view *= 0;
        view += 10;
        let expected = Tensor::<i32>::from_buf(vec![10], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_scalar_non_inplace_cuda() {
        let tensor = CudaTensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        let result = tensor.view() + 50;
        let expected_result = Tensor::<i32>::from_buf(vec![150], vec![]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert!(result.is_scalar());
        let expected_original = Tensor::<i32>::from_buf(vec![100], vec![]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // --- Single Element Tensor Tests ---

    #[test]
    fn test_single_element_1d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5], vec![1]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        let expected = Tensor::<i32>::from_buf(vec![30], vec![1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_single_element_2d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![8], vec![1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 5;
        let expected = Tensor::<i32>::from_buf(vec![40], vec![1, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_single_element_3d_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![3], vec![1, 1, 1]).unwrap();
        let mut view = tensor.view_mut();
        view += 7;
        view *= 3;
        let expected = Tensor::<i32>::from_buf(vec![30], vec![1, 1, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Non-Square Tensor Tests ---

    #[test]
    fn test_tall_matrix_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![9, 1]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        let expected = Tensor::<i32>::from_buf(vec![2, 4, 6, 8, 10, 12, 14, 16, 18], vec![9, 1]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_wide_matrix_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![1, 9]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        let expected = Tensor::<i32>::from_buf(vec![11, 12, 13, 14, 15, 16, 17, 18, 19], vec![1, 9]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_matrix_2x5_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![2, 5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3;
        let expected = Tensor::<i32>::from_buf(
            vec![3, 6, 9, 12, 15, 18, 21, 24, 27, 30], 
            vec![2, 5]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_matrix_5x2_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
            vec![5, 2]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 1;
        let expected = Tensor::<i32>::from_buf(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
            vec![5, 2]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_rectangular_3d_tensor_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        let mut view = tensor.view_mut();
        view += 100;
        
        let expected_data: Vec<i32> = (101..=124).collect();
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_non_square_slice_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second row: [5, 6, 7, 8]
        slice *= 10;
        
        let expected = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 50, 60, 70, 80, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Edge Value Tests ---

    #[test]
    fn test_zero_tensor_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        view *= 2;
        let expected = Tensor::<i32>::from_buf(vec![20, 20, 20, 20], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_operations_resulting_in_zero_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        let expected = Tensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mixed_positive_negative_values_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![-5, 10, -15, 20, -25, 30], 
            vec![6]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 10;
        
        let expected = Tensor::<i32>::from_buf(
            vec![0, 30, -20, 50, -40, 70], 
            vec![6]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_large_values_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1_000_000, 2_000_000, 3_000_000], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 500_000;
        
        let expected = Tensor::<i32>::from_buf(
            vec![1_500_000, 2_500_000, 3_500_000], 
            vec![3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Complex Non-Contiguous View Tests ---

    #[test]
    fn test_multiple_noncontiguous_operations_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![4, 3]
        ).unwrap();
        
        // Get middle column (non-contiguous)
        let mut view = tensor.view_mut();
        let mut col = view.slice_mut(1, 1..1).unwrap(); // Column 1: [2, 5, 8, 11]
        col += 100;
        col *= 2;
        
        let expected = Tensor::<i32>::from_buf(
            vec![1, 204, 3, 4, 210, 6, 7, 216, 9, 10, 222, 12], 
            vec![4, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // #[test]
    // fn test_noncontiguous_slice_after_reshape_cuda() {
    //     let mut tensor = CudaTensor::<i32>::from_buf(
    //         vec![1, 2, 3, 4, 5, 6, 7, 8], 
    //         vec![8]
    //     ).unwrap();
    //     let view = tensor.view_mut();
    //     let mut reshaped = view.view_as(vec![4, 2]).unwrap();
    //     let mut col_slice = reshaped.slice_mut(1, 1..1).unwrap(); // Second column
    //     col_slice *= 10;
        
    //     // Second column in 4x2 is indices 1, 3, 5, 7
    //     let expected = CpuTensor::<i32>::from_buf(
    //         vec![1, 20, 3, 40, 5, 60, 7, 80], 
    //         vec![8]
    //     ).unwrap();
    //     assert_eq!(tensor.cpu().unwrap(), expected);
    // }

    #[test]
    fn test_3d_noncontiguous_slice_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 4]).unwrap();
        
        // Slice along middle dimension at index 1 (middle row)
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(1, 1..1).unwrap(); // Middle "row" at each depth level
        slice += 100; // in og view, this is skipping first 8 and last 8 elements. impacts 8 elements total
        assert_eq!(*slice.shape(), vec![2, 4]);
        
        // Indices affected: 5-8, 17-20 (middle row of each depth, 1-indexed becomes 4-7, 16-19 in 0-indexed)
        // first row skipped, second row hit, third row skipped fourth row skipped fifth row hit sixth row skipped
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in [5, 6, 7, 8, 17, 18, 19, 20].iter() {
            expected_data[i - 1] += 100;
        }
        
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 3, 4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- 4D and 5D Tensor Tests ---

    #[test]
    fn test_4d_tensor_operations_cuda() {
        let data: Vec<i32> = (1..=16).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 10;
        
        let expected_data: Vec<i32> = (11..=26).collect();
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_4d_slice_operation_cuda() {
        let data: Vec<i32> = (1..=24).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 3, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // Second slice along first dim
        slice *= 10;
        
        // Second half of data (indices 12-23)
        let mut expected_data: Vec<i32> = (1..=24).collect();
        for i in 12..24 {
            expected_data[i] *= 10;
        }
        
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 3, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_5d_tensor_operations_cuda() {
        let data: Vec<i32> = (1..=32).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 2, 2, 2, 2]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        view += 5;
        
        let expected_data: Vec<i32> = (1..=32).map(|x| x * 2 + 5).collect();
        let expected = Tensor::<i32>::from_buf(expected_data, vec![2, 2, 2, 2, 2]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Operation Chain Tests ---

    #[test]
    fn test_long_operation_chain_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        view += 1;   // [2, 3, 4, 5]
        view *= 2;   // [4, 6, 8, 10]
        view -= 1;   // [3, 5, 7, 9]
        view *= 3;   // [9, 15, 21, 27]
        view += 10;  // [19, 25, 31, 37]
        view -= 5;   // [14, 20, 26, 32]
        view *= 2;   // [28, 40, 52, 64]
        view += 2;   // [30, 42, 54, 66]
        view -= 10;  // [20, 32, 44, 56]
        view *= 1;   // [20, 32, 44, 56]
        
        let expected = Tensor::<i32>::from_buf(vec![20, 32, 44, 56], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_alternating_inplace_noninplace_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(vec![5, 10, 15], vec![3]).unwrap();
        
        // Inplace
        let mut view1 = tensor.view_mut();
        view1 += 5;
        
        // Non-inplace
        let result1 = tensor.view() * 2;
        let expected_result1 = Tensor::<i32>::from_buf(vec![20, 30, 40], vec![3]).unwrap();
        assert_eq!(result1.cpu().unwrap(), expected_result1);
        
        // Inplace again
        let mut view2 = tensor.view_mut();
        view2 *= 3;
        
        // Non-inplace again
        let result2 = tensor.view() + 10;
        let expected_result2 = Tensor::<i32>::from_buf(vec![40, 55, 70], vec![3]).unwrap();
        assert_eq!(result2.cpu().unwrap(), expected_result2);
        
        // Original tensor should reflect inplace ops
        let expected_final = Tensor::<i32>::from_buf(vec![30, 45, 60], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_final);
    }

    #[test]
    fn test_multiple_views_different_ops_cuda() {
        let mut tensor = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        
        // First view - operate on first row
        {
            let mut view = tensor.view_mut();
            let mut row1 = view.slice_mut(0, 0..0).unwrap();
            row1 += 10;
        }
        
        // Second view - operate on second row
        {
            let mut view = tensor.view_mut();
            let mut row2 = view.slice_mut(0, 1..1).unwrap();
            row2 *= 5;
        }
        
        let expected = Tensor::<i32>::from_buf(
            vec![11, 12, 13, 20, 25, 30], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // --- Float Edge Cases ---

    #[test]
    fn test_f32_small_values_cuda() {
        let mut tensor = CudaTensor::<f32>::from_buf(
            vec![0.0001, 0.0002, 0.0003], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 1000.0;
        
        let result = tensor.cpu().unwrap();
        let expected = vec![0.1, 0.2, 0.3];
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_f64_precision_cuda() {
        let mut tensor = CudaTensor::<f64>::from_buf(
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0], 
            vec![3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 3.0;
        
        let result = tensor.cpu().unwrap();
        let expected = vec![1.0, 2.0, 3.0];
        for (i, &exp) in expected.iter().enumerate() {
            let val = result.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-10);
        }
    }

    // --- Stress Test: Very Imbalanced Shapes ---

    #[test]
    fn test_very_imbalanced_shape_1000x2_cuda() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![1000, 2]).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        
        let result = tensor.cpu().unwrap();
        // Check a few values
        assert_eq!(result.view().get(vec![0, 0]).unwrap(), 6);
        assert_eq!(result.view().get(vec![0, 1]).unwrap(), 7);
        assert_eq!(result.view().get(vec![999, 0]).unwrap(), 2004);
        assert_eq!(result.view().get(vec![999, 1]).unwrap(), 2005);
    }

    #[test]
    fn test_very_imbalanced_shape_2x1000_cuda() {
        let data: Vec<i32> = (1..=2000).collect();
        let mut tensor = CudaTensor::<i32>::from_buf(data, vec![2, 1000]).unwrap();
        let mut view = tensor.view_mut();
        view *= 2;
        
        let result = tensor.cpu().unwrap();
        // Check a few values
        assert_eq!(result.view().get(vec![0, 0]).unwrap(), 2);
        assert_eq!(result.view().get(vec![0, 999]).unwrap(), 2000);
        assert_eq!(result.view().get(vec![1, 0]).unwrap(), 2002);
        assert_eq!(result.view().get(vec![1, 999]).unwrap(), 4000);
    }

        // BROADCASTING TESTS
    #[test]
    fn test_broadcast_flipped_cuda() {
        let mut veca = CudaTensor::<f32>::ones((1, 3));
        let vecb = CudaTensor::<f32>::ones((3, 1));
        veca.view_mut().set(vec![0, 0], 22.0).unwrap();

        let vecc = &veca.add(vecb.view());

        assert_eq!(vecc.shape().clone(), vec![3, 3]);
        for i in 0..3usize {
            for j in 0..3usize {
                if j == 0 {
                    assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 23.0);
                    continue;
                }
                assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 2.0);
            }
        }
    }

    #[test]
    fn test_broadcast_same_cuda() {
        let mut veca = CudaTensor::<f32>::ones((3,1));
        let vecb = CudaTensor::<f32>::ones((3, 1));
        veca.view_mut().set(vec![0, 0], 22.0).unwrap();

        let vecc = &veca.add(vecb.view());

        assert_eq!(vecc.shape().clone(), vec![3, 1]);
        for i in 0..3usize {
            for j in 0..1usize {
                if i == 0 {
                    assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 23.0);
                    continue;
                }
                assert_eq!(vecc.view().get(vec![i, j]).unwrap(), 2.0);
            }
        }
    }

    // ============================================================================
    // COMPREHENSIVE BROADCAST TESTS - Higher Rank & Expected Shapes
    // ============================================================================

    // --- 3D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_3d_scalar_to_tensor_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![5.0], vec![]).unwrap(); // scalar
        let vecb = CudaTensor::<f32>::ones((2, 3, 4));
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 6.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_vector_along_last_dim_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap(); // (3,)
        let vecb = CudaTensor::<f32>::ones((2, 4, 3)); // (2, 4, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 3]);
        // Check pattern: last dimension should be [2, 3, 4] everywhere
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(vecc.view().get(vec![i, j, 0]).unwrap(), 2.0);
                assert_eq!(vecc.view().get(vec![i, j, 1]).unwrap(), 3.0);
                assert_eq!(vecc.view().get(vec![i, j, 2]).unwrap(), 4.0);
            }
        }
    }

    #[test]
    fn test_broadcast_3d_matrix_to_tensor_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = CudaTensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_singleton_dims_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 3, 1)); // (1, 3, 1)
        let vecb = CudaTensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_3d_prepend_dims_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = CudaTensor::<f32>::ones((5, 3, 4)); // (5, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![5, 3, 4]);
        for i in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    // --- 4D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_4d_vector_to_tensor_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap(); // (2,)
        let vecb = CudaTensor::<f32>::ones((3, 4, 5, 2)); // (3, 4, 5, 2)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5, 2]);
        // Check pattern in last dimension
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k, 0]).unwrap(), 11.0);
                    assert_eq!(vecc.view().get(vec![i, j, k, 1]).unwrap(), 21.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_4d_complex_singletons_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 1, 5, 1)); // (1, 1, 5, 1)
        let vecb = CudaTensor::<f32>::ones((2, 3, 5, 4)); // (2, 3, 5, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 5, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_4d_3d_to_4d_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 5, 4)); // (3, 5, 4)
        let vecb = CudaTensor::<f32>::ones((2, 3, 5, 4)); // (2, 3, 5, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 3, 5, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..5 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    // --- 5D Broadcast Tests (Valid Cases) ---

    #[test]
    fn test_broadcast_5d_scalar_to_tensor_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![100.0], vec![]).unwrap(); // scalar
        let vecb = CudaTensor::<f32>::ones((2, 2, 2, 2, 2)); // (2, 2, 2, 2, 2)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 2, 2, 2, 2]);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            assert_eq!(vecc.view().get(vec![i, j, k, l, m]).unwrap(), 101.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_5d_multiple_singletons_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 2, 1, 3, 1)); // (1, 2, 1, 3, 1)
        let vecb = CudaTensor::<f32>::ones((4, 2, 5, 3, 6)); // (4, 2, 5, 3, 6)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![4, 2, 5, 3, 6]);
        // Check a few representative values
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![3, 1, 4, 2, 5]).unwrap(), 2.0);
    }

    // --- Mixed Dimension Broadcasts (Valid Cases) ---

    #[test]
    fn test_broadcast_2d_to_4d_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = CudaTensor::<f32>::ones((2, 5, 3, 4)); // (2, 5, 3, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 5, 3, 4]);
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..3 {
                    for l in 0..4 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l]).unwrap(), 2.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_broadcast_1d_to_5d_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap(); // (3,)
        let vecb = CudaTensor::<f32>::ones((2, 4, 5, 6, 3)); // (2, 4, 5, 6, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![2, 4, 5, 6, 3]);
        // Check pattern in last dimension
        for i in 0..2 {
            for j in 0..4 {
                for k in 0..5 {
                    for l in 0..6 {
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 0]).unwrap(), 2.0);
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 1]).unwrap(), 3.0);
                        assert_eq!(vecc.view().get(vec![i, j, k, l, 2]).unwrap(), 4.0);
                    }
                }
            }
        }
    }

    // --- FAILING BROADCAST TESTS (Should Not Work) ---

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_dimensions_3d_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4)); // (3, 4)
        let vecb = CudaTensor::<f32>::ones((2, 5, 6)); // (2, 5, 6) - incompatible!
        
        // This should panic because 4 != 6 and 3 != 5
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_inner_dim_cuda() {
        let veca = CudaTensor::<f32>::ones((5,)); // (5,)
        let vecb = CudaTensor::<f32>::ones((2, 3, 7)); // (2, 3, 7)
        
        // Should fail because 5 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_middle_dim_cuda() {
        let veca = CudaTensor::<f32>::ones((2, 3, 4)); // (2, 3, 4)
        let vecb = CudaTensor::<f32>::ones((2, 5, 4)); // (2, 5, 4)
        
        // Should fail because middle dimension 3 != 5 and neither is 1
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_4d_cuda() {
        let veca = CudaTensor::<f32>::ones((2, 3, 4, 5)); // (2, 3, 4, 5)
        let vecb = CudaTensor::<f32>::ones((2, 7, 4, 5)); // (2, 7, 4, 5)
        
        // Should fail because 3 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_5d_multiple_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 2, 3, 4, 5)); // (1, 2, 3, 4, 5)
        let vecb = CudaTensor::<f32>::ones((6, 2, 7, 4, 5)); // (6, 2, 7, 4, 5)
        
        // Should fail because 3 != 7 (dim 2)
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_non_singleton_mismatch_cuda() {
        let veca = CudaTensor::<f32>::ones((4, 5, 6)); // (4, 5, 6)
        let vecb = CudaTensor::<f32>::ones((4, 3, 6)); // (4, 3, 6)
        
        // Should fail because 5 != 3 and neither is 1
        let _vecc = veca.add(vecb.view());
    }

    // --- Edge Case Broadcasts ---

    #[test]
    fn test_broadcast_all_singletons_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 1, 1)); // (1, 1, 1)
        let vecb = CudaTensor::<f32>::ones((5, 6, 7)); // (5, 6, 7)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![5, 6, 7]);
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![4, 5, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_identity_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        let vecb = CudaTensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_alternating_singletons_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 4, 1, 6)); // (1, 4, 1, 6)
        let vecb = CudaTensor::<f32>::ones((3, 1, 5, 1)); // (3, 1, 5, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5, 6]);
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![2, 3, 4, 5]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_trailing_singleton_cuda() {
        let veca = CudaTensor::<f32>::ones((3, 4, 1)); // (3, 4, 1)
        let vecb = CudaTensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_leading_singleton_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 4, 5)); // (1, 4, 5)
        let vecb = CudaTensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(vecc.shape().clone(), vec![3, 4, 5]);
        for i in 0..3 {
            for j in 0..4 {
                for k in 0..5 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    // --- High Rank Shape Tests (Focus on Expected Output Shapes) ---

    #[test]
    fn test_broadcast_shape_2d_plus_3d_cuda() {
        let veca = CudaTensor::<f32>::ones((4, 5)); // (4, 5)
        let vecb = CudaTensor::<f32>::ones((3, 4, 5)); // (3, 4, 5)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (3, 4, 5)
        assert_eq!(*vecc.shape(), vec![3, 4, 5]);
    }

    #[test]
    fn test_broadcast_shape_1d_plus_4d_cuda() {
        let veca = CudaTensor::<f32>::ones((7,)); // (7,)
        let vecb = CudaTensor::<f32>::ones((2, 3, 4, 7)); // (2, 3, 4, 7)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 7)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 7]);
    }

    #[test]
    fn test_broadcast_shape_singleton_expansion_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 5, 1)); // (1, 5, 1)
        let vecb = CudaTensor::<f32>::ones((4, 5, 6)); // (4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (4, 5, 6)
        assert_eq!(*vecc.shape(), vec![4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_prepending_cuda() {
        let veca = CudaTensor::<f32>::ones((5, 6)); // (5, 6)
        let vecb = CudaTensor::<f32>::ones((2, 3, 4, 5, 6)); // (2, 3, 4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_both_expand_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 4, 1, 6)); // (1, 4, 1, 6)
        let vecb = CudaTensor::<f32>::ones((3, 1, 5, 6)); // (3, 1, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_scalar_expansion_cuda() {
        let veca = CudaTensor::<f32>::from_buf(vec![42.0], vec![]).unwrap(); // scalar
        let vecb = CudaTensor::<f32>::ones((2, 3, 4, 5, 6)); // (2, 3, 4, 5, 6)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 3, 4, 5, 6)
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_broadcast_shape_complex_5d_cuda() {
        let veca = CudaTensor::<f32>::ones((1, 1, 3, 1, 5)); // (1, 1, 3, 1, 5)
        let vecb = CudaTensor::<f32>::ones((2, 4, 3, 6, 5)); // (2, 4, 3, 6, 5)
        
        let vecc = veca.add(vecb.view());
        
        // Expected output shape: (2, 4, 3, 6, 5)
        assert_eq!(*vecc.shape(), vec![2, 4, 3, 6, 5]);
    }

    // ============================================================================
    // BOTH TENSORS WITH SINGLETONS - Mutual Broadcasting Tests
    // ============================================================================

    #[test]
    fn test_broadcast_both_singletons_2d_pattern1_cuda() {
        // Classic row vs column vector
        let mut veca = CudaTensor::<f32>::ones((1, 4)); // (1, 4) - row vector
        let vecb = CudaTensor::<f32>::ones((3, 1)); // (3, 1) - column vector
        veca.view_mut().set(vec![0, 0], 5.0).unwrap();
        veca.view_mut().set(vec![0, 1], 6.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First column should be 5+1=6, second column 6+1=7, rest 1+1=2
        for i in 0..3 {
            assert_eq!(vecc.view().get(vec![i, 0]).unwrap(), 6.0);
            assert_eq!(vecc.view().get(vec![i, 1]).unwrap(), 7.0);
            assert_eq!(vecc.view().get(vec![i, 2]).unwrap(), 2.0);
            assert_eq!(vecc.view().get(vec![i, 3]).unwrap(), 2.0);
        }
    }

    #[test]
    fn test_broadcast_both_singletons_2d_pattern2_cuda() {
        // Flipped from classic pattern
        let mut veca = CudaTensor::<f32>::ones((4, 1)); // (4, 1) - column vector
        let vecb = CudaTensor::<f32>::ones((1, 5)); // (1, 5) - row vector
        veca.view_mut().set(vec![0, 0], 10.0).unwrap();
        veca.view_mut().set(vec![1, 0], 20.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![4, 5]);
        // First row should all be 10+1=11, second row 20+1=21, rest 1+1=2
        for j in 0..5 {
            assert_eq!(vecc.view().get(vec![0, j]).unwrap(), 11.0);
            assert_eq!(vecc.view().get(vec![1, j]).unwrap(), 21.0);
            assert_eq!(vecc.view().get(vec![2, j]).unwrap(), 2.0);
            assert_eq!(vecc.view().get(vec![3, j]).unwrap(), 2.0);
        }
    }

    #[test]
    fn test_broadcast_both_singletons_3d_complementary_cuda() {
        // (1, 3, 1) and (2, 1, 4) -> (2, 3, 4)
        let veca = CudaTensor::<f32>::ones((1, 3, 1)); // (1, 3, 1)
        let vecb = CudaTensor::<f32>::ones((2, 1, 4)); // (2, 1, 4)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_both_singletons_3d_alternating_cuda() {
        // (1, 4, 5) and (3, 1, 5) -> (3, 4, 5)
        let mut veca = CudaTensor::<f32>::ones((1, 4, 5)); // (1, 4, 5)
        let mut vecb = CudaTensor::<f32>::ones((3, 1, 5)); // (3, 1, 5)
        veca.view_mut().set(vec![0, 0, 0], 100.0).unwrap();
        vecb.view_mut().set(vec![0, 0, 0], 200.0).unwrap();
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 4, 5]);
        // [0,0,0] should be 100+200=300
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 300.0);
        // veca broadcasts along dim 0, vecb broadcasts along dim 1
        // At [1,0,0]: veca[0,0,0]=100 + vecb[1,0,0]=1 = 101
        assert_eq!(vecc.view().get(vec![1, 0, 0]).unwrap(), 101.0);
        assert_eq!(vecc.view().get(vec![2, 0, 0]).unwrap(), 101.0);
        // At [0,1,0]: veca[0,1,0]=1 + vecb[0,0,0]=200 = 201
        assert_eq!(vecc.view().get(vec![0, 1, 0]).unwrap(), 201.0);
        assert_eq!(vecc.view().get(vec![0, 2, 0]).unwrap(), 201.0);
    }

    #[test]
    fn test_broadcast_both_singletons_3d_partial_overlap_cuda() {
        // (5, 1, 3) and (1, 4, 3) -> (5, 4, 3)
        let veca = CudaTensor::<f32>::ones((5, 1, 3)); // (5, 1, 3)
        let vecb = CudaTensor::<f32>::ones((1, 4, 3)); // (1, 4, 3)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![5, 4, 3]);
        // All values should be 2.0
        for i in 0..5 {
            for j in 0..4 {
                for k in 0..3 {
                    assert_eq!(vecc.view().get(vec![i, j, k]).unwrap(), 2.0);
                }
            }
        }
    }

    #[test]
    fn test_broadcast_both_singletons_4d_zigzag_cuda() {
        // (1, 3, 1, 5) and (2, 1, 4, 1) -> (2, 3, 4, 5)
        let veca = CudaTensor::<f32>::ones((1, 3, 1, 5)); // (1, 3, 1, 5)
        let vecb = CudaTensor::<f32>::ones((2, 1, 4, 1)); // (2, 1, 4, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5]);
        // Spot check some values
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 2, 3, 4]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_4d_complex_cuda() {
        // (1, 1, 6, 7) and (5, 4, 1, 1) -> (5, 4, 6, 7)
        let veca = CudaTensor::<f32>::ones((1, 1, 6, 7)); // (1, 1, 6, 7)
        let vecb = CudaTensor::<f32>::ones((5, 4, 1, 1)); // (5, 4, 1, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![5, 4, 6, 7]);
        // Check boundaries
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![4, 3, 5, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_5d_checkerboard_cuda() {
        // (1, 2, 1, 3, 1) and (4, 1, 5, 1, 6) -> (4, 2, 5, 3, 6)
        let veca = CudaTensor::<f32>::ones((1, 2, 1, 3, 1)); // (1, 2, 1, 3, 1)
        let vecb = CudaTensor::<f32>::ones((4, 1, 5, 1, 6)); // (4, 1, 5, 1, 6)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![4, 2, 5, 3, 6]);
        // Spot check
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![3, 1, 4, 2, 5]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_5d_all_different_cuda() {
        // (1, 3, 1, 1, 7) and (2, 1, 4, 5, 1) -> (2, 3, 4, 5, 7)
        let veca = CudaTensor::<f32>::ones((1, 3, 1, 1, 7)); // (1, 3, 1, 1, 7)
        let vecb = CudaTensor::<f32>::ones((2, 1, 4, 5, 1)); // (2, 1, 4, 5, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4, 5, 7]);
        // Verify a few positions
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 1, 1, 1, 1]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 2, 3, 4, 6]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_both_singletons_mixed_ranks_cuda() {
        // (1, 5) and (3, 1, 1) -> (3, 1, 5) - wait, this should be (3, 1, 5)? Let me recalculate
        // Actually: (1, 5) needs to be prepended -> (1, 1, 5), then broadcast with (3, 1, 1) -> (3, 1, 5)
        let veca = CudaTensor::<f32>::ones((1, 5)); // (1, 5)
        let vecb = CudaTensor::<f32>::ones((3, 1, 1)); // (3, 1, 1)
        
        let vecc = veca.add(vecb.view());
        
        assert_eq!(*vecc.shape(), vec![3, 1, 5]);
        for i in 0..3 {
            for k in 0..5 {
                assert_eq!(vecc.view().get(vec![i, 0, k]).unwrap(), 2.0);
            }
        }
    }

    // --- FAILING TESTS: Both tensors with singletons but incompatible ---

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_2d_cuda() {
        // (3, 1) and (1, 5) would work, but (3, 4) and (5, 1) won't
        let veca = CudaTensor::<f32>::ones((3, 4)); // (3, 4) - no singleton!
        let vecb = CudaTensor::<f32>::ones((5, 1)); // (5, 1)
        
        // Should fail: 3 != 5 and 4 != 1
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_3d_cuda() {
        // (2, 1, 4) and (1, 3, 5) won't work because 4 != 5
        let veca = CudaTensor::<f32>::ones((2, 1, 4)); // (2, 1, 4)
        let vecb = CudaTensor::<f32>::ones((1, 3, 5)); // (1, 3, 5)
        
        // Should fail: 4 != 5
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_4d_cuda() {
        // (1, 3, 1, 6) and (2, 1, 4, 7) won't work because 6 != 7
        let veca = CudaTensor::<f32>::ones((1, 3, 1, 6)); // (1, 3, 1, 6)
        let vecb = CudaTensor::<f32>::ones((2, 1, 4, 7)); // (2, 1, 4, 7)
        
        // Should fail: 6 != 7
        let _vecc = veca.add(vecb.view());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_middle_cuda() {
        // (1, 4, 1) and (3, 1, 5) would give (3, 4, 5)
        // but (2, 4, 1) and (3, 1, 5) fails because 2 != 3
        let veca = CudaTensor::<f32>::ones((2, 4, 1)); // (2, 4, 1)
        let vecb = CudaTensor::<f32>::ones((3, 1, 5)); // (3, 1, 5)
        
        // Should fail: 2 != 3
        let _vecc = veca.add(vecb.view());
    }

    // ==================== SUBTRACTION BROADCASTING TESTS ====================

    #[test]
    fn test_broadcast_sub_scalar_to_vector_cuda() {
        // (3,) - scalar -> (3,)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let scalar = 5.0;
        
        let vecc = veca - scalar;
        
        assert_eq!(*vecc.shape(), vec![3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 15.0, 25.0], vec![3]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_vector_to_matrix_cuda() {
        // (2, 3) - (3,) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        
        let vecc = veca - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_column_to_matrix_cuda() {
        // (2, 3) - (2, 1) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![5.0, 10.0], vec![2, 1]).unwrap();
        
        let vecc = veca - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 15.0, 25.0, 30.0, 40.0, 50.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_3d_singleton_cuda() {
        // (2, 3, 4) - (1, 1, 4) -> (2, 3, 4)
        let data_a: Vec<f32> = (0..24).map(|i| i as f32 * 10.0).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![2, 3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]).unwrap();
        
        let vecc = veca - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4]);
        // Check first row: [0, 10, 20, 30] - [1, 2, 3, 4] = [-1, 8, 17, 26]
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), -1.0);
        assert_eq!(vecc.view().get(vec![0, 0, 1]).unwrap(), 8.0);
        assert_eq!(vecc.view().get(vec![0, 0, 2]).unwrap(), 17.0);
        assert_eq!(vecc.view().get(vec![0, 0, 3]).unwrap(), 26.0);
    }

    #[test]
    fn test_broadcast_sub_different_ranks_cuda() {
        // (3, 4) - (4,) -> (3, 4)
        let data_a: Vec<f32> = (0..12).map(|i| i as f32 * 5.0).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        
        let vecc = veca - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First row: [0, 5, 10, 15] - [1, 2, 3, 4] = [-1, 3, 7, 11]
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), -1.0);
        assert_eq!(vecc.view().get(vec![0, 1]).unwrap(), 3.0);
        assert_eq!(vecc.view().get(vec![0, 2]).unwrap(), 7.0);
        assert_eq!(vecc.view().get(vec![0, 3]).unwrap(), 11.0);
    }

    #[test]
    fn test_broadcast_sub_tensorview_cuda() {
        // Test with TensorView (immutable view)
        let veca = CudaTensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 20.0], vec![2, 1]).unwrap();
        
        let vecc = veca.view() - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![90.0, 190.0, 280.0, 380.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_tensorviewmut_cuda() {
        // Test with TensorViewMut
        let mut veca = CudaTensor::<f32>::from_buf(vec![50.0, 60.0, 70.0, 80.0, 90.0, 100.0], vec![2, 3]).unwrap();
        let mut vecb = CudaTensor::<f32>::from_buf(vec![5.0, 10.0, 15.0], vec![3]).unwrap();
        
        let vecc = veca.view_mut() - vecb.view_mut();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![45.0, 50.0, 55.0, 75.0, 80.0, 85.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_tensorbase_owned_cuda() {
        // Test with owned TensorBase
        let veca = CudaTensor::<f32>::from_buf(vec![20.0, 30.0, 40.0, 50.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![3.0, 7.0], vec![2]).unwrap();
        
        let vecc = veca - vecb;
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![17.0, 23.0, 37.0, 43.0], vec![2, 2]).unwrap());
    }

    // ==================== MULTIPLICATION BROADCASTING TESTS ====================

    #[test]
    fn test_broadcast_mul_scalar_to_vector_cuda() {
        // (3,) * scalar -> (3,)
        let veca = CudaTensor::<f32>::from_buf(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        let scalar = 5.0;
        
        let vecc = veca * scalar;
        
        assert_eq!(*vecc.shape(), vec![3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 15.0, 20.0], vec![3]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_vector_to_matrix_cuda() {
        // (2, 3) * (3,) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_column_to_matrix_cuda() {
        // (2, 3) * (2, 1) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 100.0], vec![2, 1]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 400.0, 500.0, 600.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_3d_singleton_cuda() {
        // (2, 3, 4) * (1, 1, 4) -> (2, 3, 4)
        let data_a: Vec<f32> = (1..=24).map(|i| i as f32).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![2, 3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 3.0, 4.0, 5.0], vec![1, 1, 4]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4]);
        // First row: [1, 2, 3, 4] * [2, 3, 4, 5] = [2, 6, 12, 20]
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![0, 0, 1]).unwrap(), 6.0);
        assert_eq!(vecc.view().get(vec![0, 0, 2]).unwrap(), 12.0);
        assert_eq!(vecc.view().get(vec![0, 0, 3]).unwrap(), 20.0);
    }

    #[test]
    fn test_broadcast_mul_different_ranks_cuda() {
        // (3, 4) * (4,) -> (3, 4)
        let data_a: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First row: [1, 2, 3, 4] * [2, 3, 4, 5] = [2, 6, 12, 20]
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![0, 1]).unwrap(), 6.0);
        assert_eq!(vecc.view().get(vec![0, 2]).unwrap(), 12.0);
        assert_eq!(vecc.view().get(vec![0, 3]).unwrap(), 20.0);
        // Second row: [5, 6, 7, 8] * [2, 3, 4, 5] = [10, 18, 28, 40]
        assert_eq!(vecc.view().get(vec![1, 0]).unwrap(), 10.0);
        assert_eq!(vecc.view().get(vec![1, 1]).unwrap(), 18.0);
        assert_eq!(vecc.view().get(vec![1, 2]).unwrap(), 28.0);
        assert_eq!(vecc.view().get(vec![1, 3]).unwrap(), 40.0);
    }

    #[test]
    fn test_broadcast_mul_tensorview_cuda() {
        // Test with TensorView (immutable view)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 3.0], vec![2, 1]).unwrap();
        
        let vecc = veca.view() * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![20.0, 40.0, 90.0, 120.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_tensorviewmut_cuda() {
        // Test with TensorViewMut
        let mut veca = CudaTensor::<f32>::from_buf(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]).unwrap();
        let mut vecb = CudaTensor::<f32>::from_buf(vec![2.0, 3.0, 4.0], vec![3]).unwrap();
        
        let vecc = veca.view_mut() * vecb.view_mut();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 18.0, 28.0, 16.0, 27.0, 40.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_tensorbase_owned_cuda() {
        // Test with owned TensorBase
        let veca = CudaTensor::<f32>::from_buf(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 100.0], vec![2]).unwrap();
        
        let vecc = veca * vecb;
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![20.0, 300.0, 40.0, 500.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_high_dimensional_cuda() {
        // (2, 1, 3, 1) * (1, 4, 1, 5) -> (2, 4, 3, 5)
        let veca = CudaTensor::<f32>::ones((2, 1, 3, 1));
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0; 20], vec![1, 4, 1, 5]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 4, 3, 5]);
        // All should be 1.0 * 2.0 = 2.0
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![1, 3, 2, 4]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_sub_high_dimensional_cuda() {
        // (2, 1, 3, 1) - (1, 4, 1, 5) -> (2, 4, 3, 5)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0; 6], vec![2, 1, 3, 1]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![3.0; 20], vec![1, 4, 1, 5]).unwrap();
        
        let vecc = veca - vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 4, 3, 5]);
        // All should be 10.0 - 3.0 = 7.0
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 7.0);
        assert_eq!(vecc.view().get(vec![1, 3, 2, 4]).unwrap(), 7.0);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_sub_incompatible_cuda() {
        // (3, 4) and (5,) are incompatible (4 != 5)
        let veca = CudaTensor::<f32>::ones((3, 4));
        let vecb = CudaTensor::<f32>::ones((5,));
        
        let _vecc = veca - vecb.view();
    }

    #[test]
    #[should_panic]
    fn test_broadcast_mul_incompatible_cuda() {
        // (2, 3) and (4,) are incompatible (3 != 4)
        let veca = CudaTensor::<f32>::ones((2, 3));
        let vecb = CudaTensor::<f32>::ones((4,));
        
        let _vecc = veca * vecb.view();
    }

    #[test]
    fn test_tensorviewmut_add_assign_tensorbase_cuda() {
        let mut a = CudaTensor::<f32>::ones((2, 3));
        let b = CudaTensor::<f32>::ones((2, 3));
        {
            let mut view = a.view_mut();
            view += b;
        }
        assert_eq!(a.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorviewmut_sub_assign_tensorview_cuda() {
        let mut a = CudaTensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap();
        let b = CudaTensor::<f32>::from_buf(vec![3.0, 7.0], vec![2]).unwrap();
        {
            let mut view = a.view_mut();
            view -= b.view();
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 7.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensorviewmut_mul_assign_tensorviewmut_cuda() {
        let mut a = CudaTensor::<f32>::from_buf(vec![5.0, 6.0], vec![2]).unwrap();
        let mut b = CudaTensor::<f32>::from_buf(vec![2.0, 3.0], vec![2]).unwrap();
        {
            let mut view_a = a.view_mut();
            let view_b = b.view_mut();
            view_a *= view_b;
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 18.0);
    }

    // ==================== DIVISION BROADCASTING TESTS ====================

    #[test]
    fn test_broadcast_div_scalar_to_vector_cuda() {
        // (3,) / scalar -> (3,)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let scalar = 5.0;
        
        let vecc = veca / scalar;
        
        assert_eq!(*vecc.shape(), vec![3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![2.0, 4.0, 6.0], vec![3]).unwrap());
    }

    #[test]
    fn test_broadcast_div_vector_to_matrix_cuda() {
        // (2, 3) / (3,) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 4.0, 5.0], vec![3]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 5.0, 6.0, 20.0, 12.5, 12.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_div_column_to_matrix_cuda() {
        // (2, 3) / (2, 1) -> (2, 3)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 10.0], vec![2, 1]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 10.0, 15.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_div_3d_singleton_cuda() {
        // (2, 3, 4) / (1, 1, 4) -> (2, 3, 4)
        let data_a: Vec<f32> = (1..=24).map(|i| (i * 2) as f32).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![2, 3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 4.0, 8.0, 16.0], vec![1, 1, 4]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 3, 4]);
        // First row: [2, 4, 6, 8] / [2, 4, 8, 16] = [1, 1, 0.75, 0.5]
        assert_eq!(vecc.view().get(vec![0, 0, 0]).unwrap(), 1.0);
        assert_eq!(vecc.view().get(vec![0, 0, 1]).unwrap(), 1.0);
        assert_eq!(vecc.view().get(vec![0, 0, 2]).unwrap(), 0.75);
        assert_eq!(vecc.view().get(vec![0, 0, 3]).unwrap(), 0.5);
    }

    #[test]
    fn test_broadcast_div_different_ranks_cuda() {
        // (3, 4) / (4,) -> (3, 4)
        let data_a: Vec<f32> = (1..=12).map(|i| (i * 10) as f32).collect();
        let veca = CudaTensor::<f32>::from_buf(data_a, vec![3, 4]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0, 5.0, 10.0, 20.0], vec![4]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First row: [10, 20, 30, 40] / [2, 5, 10, 20] = [5, 4, 3, 2]
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 5.0);
        assert_eq!(vecc.view().get(vec![0, 1]).unwrap(), 4.0);
        assert_eq!(vecc.view().get(vec![0, 2]).unwrap(), 3.0);
        assert_eq!(vecc.view().get(vec![0, 3]).unwrap(), 2.0);
    }

    #[test]
    fn test_broadcast_div_tensorview_cuda() {
        // Test with TensorView (immutable view)
        let veca = CudaTensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 20.0], vec![2, 1]).unwrap();
        
        let vecc = veca.view() / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 20.0, 15.0, 20.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn test_broadcast_div_tensorviewmut_cuda() {
        // Test with TensorViewMut
        let mut veca = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let mut vecb = CudaTensor::<f32>::from_buf(vec![2.0, 4.0, 5.0], vec![3]).unwrap();
        
        let vecc = veca.view_mut() / vecb.view_mut();
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 5.0, 6.0, 20.0, 12.5, 12.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_div_tensorbase_owned_cuda() {
        // Test with owned TensorBase
        let veca = CudaTensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 100.0], vec![2]).unwrap();
        
        let vecc = veca / vecb;
        
        assert_eq!(*vecc.shape(), vec![2, 2]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 2.0, 30.0, 4.0], vec![2, 2]).unwrap());
    }

    #[test]
    fn test_broadcast_div_high_dimensional_cuda() {
        // (2, 1, 3, 1) / (1, 4, 1, 5) -> (2, 4, 3, 5)
        let veca = CudaTensor::<f32>::from_buf(vec![10.0; 6], vec![2, 1, 3, 1]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![2.0; 20], vec![1, 4, 1, 5]).unwrap();
        
        let vecc = veca / vecb.view();
        
        assert_eq!(*vecc.shape(), vec![2, 4, 3, 5]);
        // All should be 10.0 / 2.0 = 5.0
        assert_eq!(vecc.view().get(vec![0, 0, 0, 0]).unwrap(), 5.0);
        assert_eq!(vecc.view().get(vec![1, 3, 2, 4]).unwrap(), 5.0);
    }

    #[test]
    fn test_broadcast_div_in_place_cuda() {
        let mut veca = CudaTensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2]).unwrap();
        let vecb = CudaTensor::<f32>::from_buf(vec![10.0, 20.0], vec![2]).unwrap();
        
        veca /= vecb.view();
        
        assert_eq!(veca.cpu().unwrap(), Tensor::<f32>::from_buf(vec![10.0, 10.0, 30.0, 20.0], vec![2, 2]).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_broadcast_div_incompatible_cuda() {
        // (2, 3) and (4,) are incompatible (3 != 4)
        let veca = CudaTensor::<f32>::ones((2, 3));
        let vecb = CudaTensor::<f32>::ones((4,));
        
        let _vecc = veca / vecb.view();
    }

    #[test]
    fn test_tensorviewmut_div_assign_tensorbase_cuda() {
        let mut a = CudaTensor::<f32>::from_buf(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0], vec![2, 3]).unwrap();
        let b = CudaTensor::<f32>::from_buf(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        {
            let mut view = a.view_mut();
            view /= b;
        }
        assert_eq!(a.view().get(vec![0, 0]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![0, 1]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![0, 2]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![1, 0]).unwrap(), 40.0);
        assert_eq!(a.view().get(vec![1, 1]).unwrap(), 25.0);
        assert_eq!(a.view().get(vec![1, 2]).unwrap(), 20.0);
    }
}


#[cfg(feature = "remote")]
#[cfg(test)]
mod remote_tests {
    use std::{ops::Add, sync::OnceLock, thread};

    use crate::{backend::{remote::{self, client::RemoteBackend, server::RemoteServer}, Backend}, core::{primitives::{RemoteTensor, TensorBase}, tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape, Tensor}};


    // Lazy static backend shared across all tests
    static BACKEND: OnceLock<RemoteBackend> = OnceLock::new();
    
    fn get_backend() -> RemoteBackend {

        BACKEND.get_or_init(|| {
            // Start the server
            let mut server = RemoteServer::new("127.0.0.1".parse().unwrap(), 7878);
            thread::spawn(move || {
                let _ = server.serve();
            });
            thread::sleep(std::time::Duration::from_millis(10));

            // Create and connect the backend
            let backend = remote::get_backend_default().unwrap();
            
            backend
        }).clone()
    }
    
    fn make_remote_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Result<RemoteTensor<T>, TensorError> {
        let shape: Shape = shape.into();
        let buf_len = buf.len();
        let expected_len: usize = shape.iter().product();
        
        if buf_len != expected_len {
            return Err(TensorError::InvalidShape(format!(
                "Element count mismatch: shape implies {} elements, but buffer has {} elements",
                expected_len,
                buf_len
            )));
        }
        
        let backend = get_backend();
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);
        
        // Clone the backend for this tensor
        let tensor_backend = backend.clone();
        drop(backend); // Release the lock
        
        Ok(TensorBase::from_parts(tensor_backend, buffer, MetaTensor::new(shape, stride, 0), None))
    }


    // Edge case: negative values
    #[test]
    fn test_add_negative_remote() {
        let mut tensor = make_remote_tensor(vec![10, 20, 30], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view += -5;
        let expected = Tensor::<i32>::from_buf(vec![5, 15, 25], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_mul_negative_remote() {
        let mut tensor = make_remote_tensor(vec![1, 2, 3], vec![3]).unwrap();
        let mut view = tensor.view_mut();
        view *= -5;
        let expected = Tensor::<i32>::from_buf(vec![-5, -10, -15], vec![3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // Edge case: non-contiguous slices
    #[test]
    fn test_mul_not_inplace_with_noncontiguous_slice_remote() {
        let mut tensor = make_remote_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let mut view = tensor.view_mut();
        let col_slice = view.slice_mut(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        assert!(!col_slice.is_contiguous());
        
        let result = col_slice * 3;
        
        // Result should be a new contiguous tensor
        let expected_result = Tensor::<i32>::from_buf(vec![6, 15], vec![2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected_result);
        assert!(result.is_contiguous());
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // ==================== DATATYPE TESTS ====================
    
    #[test]
    fn test_remote_f64_operations() {
        let mut tensor = make_remote_tensor(vec![1.5, 2.5, 3.5, 4.5], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 10.25;
        view *= 2.0;
        view -= 3.5;
        
        let expected = Tensor::<f64>::from_buf(
            vec![20.0, 22.0, 24.0, 26.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-10, "Mismatch at {}: {} vs {}", i, val, exp);
        }
    }

    #[test]
    fn test_remote_i64_operations() {
        let mut tensor = make_remote_tensor(
            vec![100, 200, 300, 400, 500], 
            vec![5]
        ).unwrap();
        let mut view = tensor.view_mut();
        view -= 50;
        view *= 3;
        
        let expected = Tensor::<i64>::from_buf(
            vec![150, 450, 750, 1050, 1350], 
            vec![5]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_u32_operations() {
        let mut tensor = make_remote_tensor(
            vec![10, 20, 30, 40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view += 5;
        view *= 2;
        
        let expected = Tensor::<u32>::from_buf(
            vec![30, 50, 70, 90], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_i16_operations() {
        let mut tensor = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 10;
        view += 5;
        
        let expected: TensorBase<i16, crate::backend::cpu::Cpu> = Tensor::<i16>::from_buf(
            vec![15, 25, 35, 45, 55, 65], 
            vec![2, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_f32_precision() {
        let mut tensor = make_remote_tensor(
            vec![10.0, 20.0, 30.0, 40.0], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0.5;
        view *= 0.25;
        
        let expected = Tensor::<f32>::from_buf(
            vec![1.25, 2.5, 3.75, 5.0], 
            vec![4]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..4 {
            let val = result.view().get(vec![i]).unwrap();
            let exp = expected.view().get(vec![i]).unwrap();
            assert!((val - exp).abs() < 1e-5);
        }
    }

    // ==================== MIXED OPERATIONS TESTS ====================

    #[test]
    fn test_remote_chained_operations_complex() {
        let mut tensor = make_remote_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 
            vec![2, 2, 2]
        ).unwrap();
        
        // Complex chain: multiply, add, subtract, multiply again
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 10.0;
        view -= 5.0;
        view *= 0.5;
        
        // Expected: ((x * 2 + 10) - 5) * 0.5 = (x * 2 + 5) * 0.5 = x + 2.5
        let expected = Tensor::<f32>::from_buf(
            vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = tensor.cpu().unwrap();
        for i in 0..8 {
            let idx = vec![i / 4, (i / 2) % 2, i % 2];
            let val = result.view().get(idx.clone()).unwrap();
            let exp = expected.view().get(idx.clone()).unwrap();
            assert!((val - exp).abs() < 1e-5, "Mismatch at {:?}: {} vs {}", idx, val, exp);
        }
    }

    #[test]
    fn test_remote_alternating_operations() {
        let mut tensor = make_remote_tensor(vec![10, 20, 30, 40], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        
        // Alternating pattern
        view += 5;   // [15, 25, 35, 45]
        view *= 2;   // [30, 50, 70, 90]
        view -= 10;  // [20, 40, 60, 80]
        view *= 3;   // [60, 120, 180, 240]
        view += 10;  // [70, 130, 190, 250]
        
        let expected = Tensor::<i32>::from_buf(
            vec![70, 130, 190, 250], 
            vec![4]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_mixed_operations_on_slice() {
        let mut tensor = make_remote_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        
        let mut view = tensor.view_mut();
        let mut middle_row = view.slice_mut(0, 1..1).unwrap(); // [4, 5, 6]
        middle_row *= 10.0;
        middle_row += 5.0;
        
        let expected = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 45.0, 55.0, 65.0, 7.0, 8.0, 9.0], 
            vec![3, 3]
        ).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    // ==================== EDGE CASE TESTS ====================

    #[test]
    fn test_remote_zero_operations() {
        let mut tensor = make_remote_tensor(vec![5, 10, 15, 20], vec![4]).unwrap();
        let mut view = tensor.view_mut();
        view += 0;
        view *= 1;
        view -= 0;
        
        let expected = Tensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_multiply_by_zero() {
        let mut tensor = make_remote_tensor(
            vec![100, 200, 300, 400], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= 0;
        
        let expected = Tensor::<i32>::from_buf(vec![0, 0, 0, 0], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_scalar_edge_cases() {
        // Test with actual scalar (0-dimensional tensor)
        let mut tensor = make_remote_tensor(vec![3.14159], vec![]).unwrap();
        assert!(tensor.is_scalar());
        
        let mut view = tensor.view_mut();
        view *= 2.0;
        view += 1.0;
        
        let result = tensor.cpu().unwrap();
        let value = result.view().get(vec![]).unwrap();
        assert!((value - 7.28318) < 1e-5);
    }

    #[test]
    fn test_remote_negative_to_positive() {
        let mut tensor = make_remote_tensor(
            vec![-10, -20, -30, -40], 
            vec![4]
        ).unwrap();
        let mut view = tensor.view_mut();
        view *= -1;
        view += 5;
        
        let expected = Tensor::<i32>::from_buf(vec![15, 25, 35, 45], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_non_inplace_multiple_results() {
        let mut tensor = make_remote_tensor(vec![1, 2, 3, 4], vec![4]).unwrap();
        let view = tensor.view_mut();
        
        // Create multiple independent results
        let result1 = view + 10;
        let view2 = tensor.view_mut();
        let result2 = view2 * 5;
        let view3 = tensor.view_mut();
        let result3 = view3 - 1;
        
        let expected1 = Tensor::<i32>::from_buf(vec![11, 12, 13, 14], vec![4]).unwrap();
        let expected2 = Tensor::<i32>::from_buf(vec![5, 10, 15, 20], vec![4]).unwrap();
        let expected3 = Tensor::<i32>::from_buf(vec![0, 1, 2, 3], vec![4]).unwrap();
        
        assert_eq!(result1.cpu().unwrap(), expected1);
        assert_eq!(result2.cpu().unwrap(), expected2);
        assert_eq!(result3.cpu().unwrap(), expected3);
        
        // Original unchanged
        let expected_original = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![4]).unwrap();
        assert_eq!(tensor.cpu().unwrap(), expected_original);
    }

    // ==================== BROADCASTING TESTS ====================

    #[test]
    fn test_broadcast_vector_to_matrix_remote() {
        // (2, 3) + (3,) -> (2, 3)
        let veca = make_remote_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let vecb = make_remote_tensor(vec![10.0, 20.0, 30.0], vec![3]).unwrap();

        println!("vecablabla id {:?}", veca.buf.id);
        println!("vecbblabla id {:?}", vecb.buf.id);
        println!("penis {:?}", veca.backend);
        println!("penis {:?}", vecb.backend);
        
        let vecc = veca + vecb.view();

        println!("penis {:?}", vecc.backend);
        
        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_sub_column_to_matrix_remote() {
        // (2, 3) - (2, 1) -> (2, 3)
        let veca = make_remote_tensor(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]).unwrap();
        let vecb = make_remote_tensor(vec![5.0, 10.0], vec![2, 1]).unwrap();

        println!("veca id {:?}", veca.buf.id);
        let vecc = veca - vecb.view();
        println!("vecc id {:?}", vecc.buf.id);
        println!("vccb id {:?}", vecb.buf.id);

        assert_eq!(*vecc.shape(), vec![2, 3]);
        assert_eq!(vecc.cpu().unwrap(), Tensor::<f32>::from_buf(vec![5.0, 15.0, 25.0, 30.0, 40.0, 50.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_mul_different_ranks_remote() {
        // (3, 4) * (4,) -> (3, 4)
        let data_a: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let veca = make_remote_tensor(data_a, vec![3, 4]).unwrap();
        let vecb = make_remote_tensor(vec![2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        
        let vecc = veca * vecb.view();
        
        assert_eq!(*vecc.shape(), vec![3, 4]);
        // First row: [1, 2, 3, 4] * [2, 3, 4, 5] = [2, 6, 12, 20]
        assert_eq!(vecc.view().get(vec![0, 0]).unwrap(), 2.0);
        assert_eq!(vecc.view().get(vec![0, 1]).unwrap(), 6.0);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_incompatible_shapes_remote() {
        // (3, 4) and (5,) are incompatible (4 != 5)
        let veca = make_remote_tensor(vec![1.0; 12], vec![3, 4]).unwrap();
        let vecb = make_remote_tensor(vec![1.0; 5], vec![5]).unwrap();
        
        let _vecc = veca + vecb.view();
    }

    #[test]
    #[should_panic]
    fn test_broadcast_both_singletons_incompatible_remote() {
        // (3, 4) and (5, 1) won't work - 3 != 5
        let veca = make_remote_tensor(vec![1.0; 12], vec![3, 4]).unwrap();
        let vecb = make_remote_tensor(vec![1.0; 5], vec![5, 1]).unwrap();
        
        let _vecc = veca.add(vecb.view());
    }

    // ==================== TENSOR VIEW TESTS ====================

    #[test]
    fn test_tensorviewmut_add_assign_tensorbase_remote() {
        let mut a = make_remote_tensor(vec![1.0; 6], vec![2, 3]).unwrap();
        let b = make_remote_tensor(vec![1.0; 6], vec![2, 3]).unwrap();
        {
            let mut view = a.view_mut();
            view += b;
        }
        assert_eq!(a.view().get(vec![0, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensorviewmut_sub_assign_tensorview_remote() {
        let mut a = make_remote_tensor(vec![10.0, 20.0], vec![2]).unwrap();
        let b = make_remote_tensor(vec![3.0, 7.0], vec![2]).unwrap();
        {
            let mut view = a.view_mut();
            view -= b.view();
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 7.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensorviewmut_mul_assign_tensorviewmut_remote() {
        let mut a = make_remote_tensor(vec![5.0, 6.0], vec![2]).unwrap();
        let mut b = make_remote_tensor(vec![2.0, 3.0], vec![2]).unwrap();
        {
            let mut view_a = a.view_mut();
            let view_b = b.view_mut();
            view_a *= view_b;
        }
        assert_eq!(a.view().get(vec![0]).unwrap(), 10.0);
        assert_eq!(a.view().get(vec![1]).unwrap(), 18.0);
    }
}