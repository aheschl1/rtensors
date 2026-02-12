
pub mod primitives;
pub mod meta;
pub mod tensor;
pub mod idx;
pub mod value;
pub mod slice;
pub mod untyped;
pub mod primops;
mod display;

pub use meta::{Dim, Shape, Strides, MetaTensor, MetaTensorView, shape_to_stride};
pub use primitives::{Tensor, TensorView, CpuTensorView, TensorViewMut};
pub use slice::Slice;


#[cfg(test)]
mod tests {
    use crate::{backend::Backend, coord, core::{idx::Idx, tensor::{AsTensor, AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, value::TensorValue, MetaTensor, MetaTensorView, Shape, Slice, Strides, Tensor}, get, ops::linalg::PaddingType};

    fn make_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Tensor<T> {
        Tensor::from_buf(buf, shape.into()).unwrap()
    }

    #[test]
    fn test_slice_matrix() {
        let buf = vec![
            1, 2, 3, 
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*slice.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 3);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap(); // slice along columns, should give a view of shape [2]
        assert_eq!(*slice2.shape(), vec![2]);
        assert_eq!(*slice2.strides(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(coord![1], &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
    }

    #[test]
    fn test_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(*slice.strides(), vec![2, 1]);
        assert_eq!(index_tensor(coord![0, 0], &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 5);

        // second depth
        let view = tensor.view();
        let slice_second_depth = view.slice(0, 1..1).unwrap();
        assert_eq!(*slice_second_depth.shape(), vec![2, 2]);
        assert_eq!(*slice_second_depth.strides(), vec![2, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice_second_depth).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice_second_depth).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice_second_depth).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice_second_depth).unwrap(), 9);
        
        let view = tensor.view();
        let slice2 = view.slice(1, 0..0).unwrap(); // slice along row, should give a view of shape [2, 2]
        assert_eq!(*slice2.shape(), vec![2, 2]);
        assert_eq!(*slice2.strides(), vec![4, 1]);
        assert_eq!(index_tensor(coord![0, 0], &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice2).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice2).unwrap(), 7);

        // column slice
        let view = tensor.view();
        let slice3 = view.slice(2, 0..0).unwrap(); // slice along column
        assert_eq!(*slice3.shape(), vec![2, 2]);
        assert_eq!(*slice3.strides(), vec![4, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice3).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice3).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice3).unwrap(), 8);
    }

    #[test]
    fn test_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // slice along rows, should give a view of shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);

        let slice_of_slice = slice.slice(0, 2..2).unwrap(); // slice along columns, should give a view of shape []
        assert_eq!(*slice_of_slice.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Coord(vec![]), &slice_of_slice).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_matrix_rows() {
        // Test slicing a range of rows (keeps dimension)
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![4, 3]; // 4 rows, 3 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice rows 1..3 (middle 2 rows)
        let view = tensor.view();
        let slice = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![2, 3]); // 2 rows, 3 columns
        assert_eq!(*slice.strides(), vec![3, 1]);
        
        // Should contain rows 1 and 2 (4,5,6 and 7,8,9)
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice).unwrap(), 9);
    }

    #[test]
    fn test_range_slice_matrix_columns() {
        // Test slicing a range of columns (keeps dimension)
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ];
        let shape = vec![3, 4]; // 3 rows, 4 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice columns 1..3 (middle 2 columns)
        let view = tensor.view();
        let slice = view.slice(1, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![3, 2]); // 3 rows, 2 columns
        assert_eq!(*slice.strides(), vec![4, 1]);
        
        // Should contain columns 1 and 2 from each row
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &slice).unwrap(), 11);
    }

    #[test]
    fn test_range_slice_cube_depth() {
        // Test slicing a range along depth dimension
        let buf = vec![
            // Depth 0
            1, 2,
            3, 4,
            // Depth 1
            5, 6,
            7, 8,
            // Depth 2
            9, 10,
            11, 12,
            // Depth 3
            13, 14,
            15, 16
        ];
        let shape = vec![4, 2, 2]; // 4 depth, 2 rows, 2 columns
        let tensor = make_tensor(buf, shape);
        
        // Slice depth 1..3 (middle 2 slices)
        let view = tensor.view();
        let slice = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![2, 2, 2]); // 2 depth, 2 rows, 2 columns
        assert_eq!(*slice.strides(), vec![4, 2, 1]);
        
        // Should contain depth slices 1 and 2
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &slice).unwrap(), 9);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 1]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 0]), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &slice).unwrap(), 12);
    }

    #[test]
    fn test_range_slice_first_n_elements() {
        // Test slicing from start: 0..n
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..5).unwrap(); // First 5 elements
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(*slice.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 5);
    }

    #[test]
    fn test_range_slice_last_n_elements() {
        // Test slicing to end: n..size
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 5..8).unwrap(); // Last 3 elements
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(*slice.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 8);
    }

    #[test]
    fn test_range_slice_of_range_slice() {
        // Test chaining range slices
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // First slice: get elements 2..8
        let view = tensor.view();
        let slice1 = view.slice(0, 2..8).unwrap();
        assert_eq!(*slice1.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice1).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(5), &slice1).unwrap(), 8);
        
        // Second slice: get elements 1..4 from the previous slice
        let slice2 = slice1.slice(0, 1..4).unwrap();
        assert_eq!(*slice2.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_mut() {
        // Test mutable range slicing
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![4, 2];
        let mut tensor = make_tensor(buf, shape);
        
        let mut slice = tensor.slice_mut(0, 1..3).unwrap(); // Rows 1 and 2
        assert_eq!(*slice.shape(), vec![2, 2]);
        
        // Modify values in the slice
        slice.set(&Idx::Coord(vec![0, 0]), 30).unwrap();
        slice.set(&Idx::Coord(vec![1, 1]), 80).unwrap();
        
        // Check original tensor was modified
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &tensor.view()).unwrap(), 30);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &tensor.view()).unwrap(), 80);
    }

    #[test]
    fn test_range_slice_full_dimension() {
        // Test slicing with full range (should be same as original)
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 0..2).unwrap(); // All rows
        assert_eq!(*slice.shape(), vec![2, 3]);
        assert_eq!(*slice.strides(), vec![3, 1]);
        
        // Should contain all original data
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice).unwrap(), 6);
    }

    #[test]
    fn test_range_slice_single_element() {
        // Test range with single element (n..n+1 is same as empty range n..n after indexing)
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 2..3).unwrap(); // Single element at index 2
        assert_eq!(*slice.shape(), vec![1]); // Dimension remains, but size is 1
        assert_eq!(*slice.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 3);
    }

    #[test]
    fn test_range_slice_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        // Range end beyond dimension size
        assert!(matches!(
            tensor.view().slice(0, 0..3),
            Err(TensorError::IdxOutOfBounds(_))
        ));
        
        // Range start beyond dimension size
        assert!(matches!(
            tensor.view().slice(0, 3..4),
            Err(TensorError::IdxOutOfBounds(_))
        ));
        
        // Range with start > end
        assert!(matches!(
            tensor.view().slice(0, 2..1),
            Err(TensorError::IdxOutOfBounds(_))
        ));
    }

    #[test]
    fn test_negative_step_basic() {
        // Test basic negative step slicing on a 1D tensor
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // Slice with step -1 (reverse)
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 1);
    }

    #[test]
    fn test_negative_step_with_range() {
        // Test negative step with explicit start and end
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // Slice from index 6 down to index 2 with step -1
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(6), Some(2), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 7); // index 6
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 5); // index 4
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 4); // index 3
    }

    #[test]
    fn test_negative_step_skip_elements() {
        // Test negative step -2 (every other element backwards)
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8); // index 7
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 4); // index 3
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 2); // index 1
    }

    #[test]
    fn test_negative_step_matrix_rows() {
        // Test negative step on matrix rows
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![4, 3];
        let tensor = make_tensor(buf, shape);
        
        // Reverse rows with step -1
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![4, 3]);
        
        // First row should be the last row of original
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice).unwrap(), 12);
        
        // Last row should be the first row of original
        assert_eq!(index_tensor(Idx::Coord(vec![3, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 1]), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 2]), &slice).unwrap(), 3);
    }

    #[test]
    fn test_negative_step_with_start_less_than_end() {
        // When step is negative and start < end, result should be empty
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(2), Some(5), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty slice
    }

    #[test]
    fn test_positive_step_with_start_greater_than_end() {
        // When step is positive and start > end, result should be empty
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(5), Some(2), 1)).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty slice
    }

    #[test]
    fn test_negative_step_partial_range() {
        // Test negative step with only start specified
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // From index 5 going backwards to the beginning
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(5), None, -1)).unwrap();
        assert_eq!(*slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 6); // index 5
        assert_eq!(index_tensor(Idx::At(5), &slice).unwrap(), 1); // index 0
    }

    #[test]
    fn test_negative_step_partial_end() {
        // Test negative step with only end specified
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![8];
        let tensor = make_tensor(buf, shape);
        
        // From the end going backwards to index 3
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(None, Some(3), -1)).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 8); // index 7
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 5); // index 4
    }

    #[test]
    fn test_negative_step_3d_tensor() {
        // Test negative step on 3D tensor
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        ];
        let shape = vec![4, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        // Reverse along the depth dimension
        let view = tensor.view();
        let slice = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![4, 2, 2]);
        
        // First depth slice should be the last one of original
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 13);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 16);
        
        // Last depth slice should be the first one of original
        assert_eq!(index_tensor(Idx::Coord(vec![3, 0, 0]), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![3, 1, 1]), &slice).unwrap(), 4);
    }

    #[test]
    fn test_negative_step_chained() {
        // Test chaining slices with negative steps
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // First reverse the whole array
        let view = tensor.view();
        let slice1 = view.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(index_tensor(Idx::At(0), &slice1).unwrap(), 10);
        
        // Then take every other element
        let slice2 = slice1.slice(0, Slice::full().step(2)).unwrap();
        assert_eq!(*slice2.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(4), &slice2).unwrap(), 2);
    }

    #[test]
    fn test_negative_step_funtax() {
        // Test chaining slices with negative steps
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // Then take every other element
        let view = tensor.view();
        let slice2 = view.slice(0, 9..=0).unwrap();
        assert_eq!(*slice2.shape(), vec![10]);
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 9);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 8);
    }

    #[test]
    fn test_negative_step_mut() {
        // Test mutable slicing with negative step
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let mut tensor = make_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::full().step(-1)).unwrap();
        
        // Modify reversed view
        slice.set(&Idx::At(0), 60).unwrap(); // Should modify index 5
        slice.set(&Idx::At(5), 10).unwrap(); // Should modify index 0
        
        // Check original tensor
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(5), &tensor.view()).unwrap(), 60);
    }

    #[test]
    fn test_negative_step_mut_funtax() {
        // Test mutable slicing with negative step
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![6];
        let mut tensor = make_tensor(buf, shape);
        
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        
        // Modify reversed view
        slice.set(&Idx::At(0), 60).unwrap(); // Should modify index 5
        slice.set(&Idx::At(5), 10).unwrap(); // Should modify index 0
        
        // Check original tensor
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(5), &tensor.view()).unwrap(), 60);
    }

    #[test]
    fn test_negative_step_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![6]);
        
        // Invalid start index with negative step
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(10), Some(2), -1)),
            Err(TensorError::IdxOutOfBounds(_))
        ));
        
        // Invalid end index with negative step
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(5), Some(10), -1)),
            Err(TensorError::IdxOutOfBounds(_))
        ));
        
        // Step of 0 should error
        assert!(matches!(
            tensor.view().slice(0, Slice::new(Some(2), Some(5), 0)),
            Err(TensorError::InvalidShape(_))
        ));
    }

    #[test]
    fn test_reverse_range_auto_negative_step() {
        // Test that reversed ranges automatically get negative step
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let tensor = make_tensor(buf, shape);
        
        // Exclusive range with start > end should auto-reverse
        let view = tensor.view();
        let slice = view.slice(0, 9..0).unwrap();
        assert_eq!(*slice.shape(), vec![9]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10); // index 9
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 9);  // index 8
        assert_eq!(index_tensor(Idx::At(8), &slice).unwrap(), 2);  // index 1
        
        // Inclusive range with start > end should auto-reverse
        let view = tensor.view();
        let slice = view.slice(0, 9..=0).unwrap();
        assert_eq!(*slice.shape(), vec![10]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10); // index 9
        assert_eq!(index_tensor(Idx::At(9), &slice).unwrap(), 1);  // index 0
    }

    #[test]
    fn test_custom_positive_step() {
        // Test slicing with custom positive step values (step > 1)
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_tensor(buf, shape);
        
        // Step by 2: take every other element from full range
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 14);
        
        // Step by 3: from index 1 to 8
        let slice2 = view.slice(0, Slice::from(1..8).step(3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]); // Indices 1, 4, 7
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 7);
        
        // Step by 4: from start to 12
        let slice3 = view.slice(0, Slice::from(..12).step(4)).unwrap();
        assert_eq!(*slice3.shape(), vec![3]); // Indices 0, 4, 8
        assert_eq!(index_tensor(Idx::At(0), &slice3).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &slice3).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice3).unwrap(), 8);
        
        // Step by 5: from 3 to end
        let slice4 = view.slice(0, Slice::from(3..).step(5)).unwrap();
        assert_eq!(*slice4.shape(), vec![3]); // Indices 3, 8, 13
        assert_eq!(index_tensor(Idx::At(0), &slice4).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(1), &slice4).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(2), &slice4).unwrap(), 13);
        
        // Large step: step by 10
        let slice5 = view.slice(0, Slice::from(2..).step(10)).unwrap();
        assert_eq!(*slice5.shape(), vec![2]); // Indices 2, 12
        assert_eq!(index_tensor(Idx::At(0), &slice5).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &slice5).unwrap(), 12);
    }

    #[test]
    fn test_custom_negative_step() {
        // Test slicing with custom negative step values (step < -1)
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_tensor(buf, shape);
        let view = tensor.view();
        
        // Step by -2: every other element, reversed
        let slice = view.slice(0, Slice::from(..).step(-2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 15); // Start from end
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 13);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(7), &slice).unwrap(), 1);
        
        // Step by -3: from index 10 to 1
        let slice2 = view.slice(0, Slice::from(10..1).step(-3)).unwrap();
        assert_eq!(*slice2.shape(), vec![3]); // Indices 10, 7, 4
        assert_eq!(index_tensor(Idx::At(0), &slice2).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice2).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(2), &slice2).unwrap(), 4);
        
        // Step by -4: from end to index 2
        let slice3 = view.slice(0, Slice::from(..2).step(-4)).unwrap();
        assert_eq!(*slice3.shape(), vec![4]); // Indices 15, 11, 7, 3
        assert_eq!(index_tensor(Idx::At(0), &slice3).unwrap(), 15);
        assert_eq!(index_tensor(Idx::At(1), &slice3).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(2), &slice3).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(3), &slice3).unwrap(), 3);
        
        // Step by -5: from index 14 to start
        let slice4 = view.slice(0, Slice::from(14..).step(-5)).unwrap();
        assert_eq!(*slice4.shape(), vec![3]); // Indices 14, 9, 4
        assert_eq!(index_tensor(Idx::At(0), &slice4).unwrap(), 14);
        assert_eq!(index_tensor(Idx::At(1), &slice4).unwrap(), 9);
        assert_eq!(index_tensor(Idx::At(2), &slice4).unwrap(), 4);
    }

    #[test]
    fn test_custom_positive_step_mut() {
        // Test mutable slicing with custom positive step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
        // Step by 2: modify every other element
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, Slice::from(..).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![8]);
        
        // Set values at step positions
        slice.set(&Idx::At(0), 100).unwrap(); // index 0
        slice.set(&Idx::At(1), 102).unwrap(); // index 2
        slice.set(&Idx::At(7), 114).unwrap(); // index 14
        
        // Verify changes in original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 100);
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 102);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 3); // Unchanged
        assert_eq!(index_tensor(Idx::At(14), &view).unwrap(), 114);
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 15); // Unchanged
    }

    #[test]
    fn test_custom_positive_step_mut_with_range() {
        // Test mutable slicing with step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
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
    fn test_custom_negative_step_mut() {
        // Test mutable slicing with custom negative step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
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
        assert_eq!(index_tensor(Idx::At(14), &view).unwrap(), 14); // Unchanged
        assert_eq!(index_tensor(Idx::At(15), &view).unwrap(), 115);
    }

    #[test]
    fn test_custom_negative_step_mut_with_range() {
        // Test mutable slicing with negative step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_tensor(buf, shape);
        
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
        assert_eq!(index_tensor(Idx::At(10), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::At(12), &view).unwrap(), 212);
    }

    #[test]
    fn slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);

        
        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // slice along depth, should give a view of shape [2, 2]
        assert_eq!(*slice.shape(), vec![2, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &slice).unwrap(), 9);

        let slice_of_slice = slice.slice(1, 0..0).unwrap(); // slice along row, should give a view of shape [2]
        assert_eq!(*slice_of_slice.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &slice_of_slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(1), &slice_of_slice).unwrap(), 8);

        // slice of slice of slice
        let slice_of_slice_of_slice = slice_of_slice.slice(0, 1..1).unwrap(); // slice along column, should give a view of shape []
        assert_eq!(*slice_of_slice_of_slice.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &slice_of_slice_of_slice).unwrap(), 8);
    }

    #[test]
    fn test_mut_slices() {
        // mut slice from owned tensor
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];

        let mut tensor = make_tensor(buf, shape);
        let mut view = tensor.view_mut();
        let mut slice = view.slice_mut(0, 1..1).unwrap(); // slice along rows, should give a view of shape [3]
        
        assert_eq!(*slice.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        slice.set(&Idx::At(1), 50).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &tensor.view()).unwrap(), 50);
    }

    #[test]
    fn test_column() {
    let tensor = Tensor::column(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![3]);
        assert_eq!(index_tensor(0, &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(1, &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(2, &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_row() {
    let tensor = Tensor::row(vec![1, 2, 3]);
    assert_eq!(*tensor.shape(), vec![1, 3]);
        assert_eq!(index_tensor((0, 0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor((0, 1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor((0, 2), &tensor.view()).unwrap(), 3);

        assert_eq!(tensor.view().get(&[0, 1]).unwrap(), 2);
    }

    #[test]
    fn test_scalar() {
        let buf = vec![42];
        let shape = vec![];
        let tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.is_scalar());
        assert_eq!(Tensor::scalar(42), tensor);
    }

    #[test]
    fn test_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_tensor(buf, shape);

        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);

        tensor.view_mut().set(&Idx::At(1), 1).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 1);
    }

    #[test]
    fn test_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

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
    fn test_cube() {
        //
        let buf = vec![
            1, 2,
            4, 5,

            6, 7,
            8, 9
        ];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor.view()).unwrap(), 1); // depth, row, column
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &tensor.view()).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &tensor.view()).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 1]), &tensor.view()).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 0]), &tensor.view()).unwrap(), 8);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &tensor.view()).unwrap(), 9);

        // modify
        tensor.view_mut().set(&Idx::Coord(vec![1, 0, 0]), 67).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &tensor.view()).unwrap(), 67);
    }

    #[test]
    fn test_view_as_owned_success() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        let v = tensor.view();
        let reshaped = v.view_as((3, 2)).unwrap();
        assert_eq!(*reshaped.shape(), vec![3, 2]);
        assert_eq!(*reshaped.strides(), vec![2, 1]);
        // Row-major sequence preserved
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &reshaped).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &reshaped).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &reshaped).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 1]), &reshaped).unwrap(), 6);
    }

    #[test]
    #[should_panic]
    fn test_view_as_owned_error() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        tensor.view_as(vec![4, 2]).unwrap();
    }

    #[test]
    fn test_view_as_slice_success() {
        let buf = vec![
            1, 2, 3, 
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);

        let view = tensor.view();
        let slice = view.slice(0, 1..1).unwrap(); // shape [3]
        assert_eq!(*slice.shape(), vec![3]);
        let reshaped = slice.view_as((1, 3)).unwrap();
        assert_eq!(*reshaped.shape(), vec![1, 3]);
        assert_eq!(*reshaped.strides(), vec![3, 1]);
        // Values should correspond to original slice elements 4,5,6
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &reshaped).unwrap(), 4);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &reshaped).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &reshaped).unwrap(), 6);
    }

    #[test]
    fn test_view_as_mut_view_modify() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let mut tensor = make_tensor(buf, shape);
        let mut view_mut = tensor.view_mut(); // shape [2,2]
        // Modify before reshaping to avoid borrow conflicts
        view_mut.set(&Idx::Coord(vec![1, 0]), 40).unwrap(); // coordinate [1,0] maps to linear index 2
        let reshaped = view_mut.view_as(vec![4]).unwrap(); // reshape to flat vector
        assert_eq!(*reshaped.shape(), vec![4]);
        assert_eq!(*reshaped.strides(), vec![1]);
        // Check reshaped view sees update at linear index 2
        assert_eq!(index_tensor(Idx::At(2), &reshaped).unwrap(), 40);
    }

    #[test]
    fn test_view_as_scalar() {
        let tensor = Tensor::scalar(99); // shape []
        let view1 = tensor.view();
        assert_eq!(*view1.shape(), vec![]);
        let reshaped = view1.view_as(vec![1]).unwrap();
        assert_eq!(*reshaped.shape(), vec![1]);
        assert_eq!(*reshaped.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &reshaped).unwrap(), 99);

        // view as [1, 1, 1]

        let r2 = reshaped.view_as(vec![1, 1, 1]).unwrap();
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &r2).unwrap(), 99);

    }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug, B: Backend>(index: impl Into<Idx>, tensor: &'a impl TensorAccess<T, B>) -> Result<T, TensorError> {
        let index = index.into();
        let r: Result<T, TensorError> = tensor.get(&index);
        let a = match r.as_ref() {
            Ok(v) => Ok(*v),
            Err(e) => return Err(e.clone()),
        };
        let b = match &index {
            Idx::At(i) => tensor.get(vec![*i]),
            Idx::Coord(idx) => tensor.get(Idx::Coord(idx.clone())),
            Idx::Item => tensor.item(),
        };
        assert_eq!(a, b);
        r
    }

    #[test]
    fn test_shape_to_stride() {
        let shape = vec![2, 2, 3];
        let stride: Strides = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![6, 3, 1]);
    }

    #[test]
    fn test_shape_to_stride_single_dim() {
        let shape = vec![4];
        let stride: Strides = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_shape_to_stride_empty() {
        let shape: Shape = Shape::empty();
        let stride: Strides = super::shape_to_stride(&shape);

        assert!(stride.is_empty());
    }

    #[test]
    fn test_shape_to_stride_ones() {
        let shape = vec![1, 1, 1];
        let stride: Strides = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![1, 1, 1]);
    }

    #[test]
    fn test_shape_to_stride_mixed() {
        let shape = vec![5, 1, 2];
        let stride: Strides = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![2, 2, 1]);
    }

    #[test]
    fn test_shape_to_stride_larger() {
        let shape = vec![3, 4, 5];
        let stride: Strides = super::shape_to_stride(&shape.into());

        assert_eq!(stride, vec![20, 5, 1]);
    }

    #[test]
    fn test_from_buf_error() {
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            Tensor::from_buf(buf, shape),
            Err(TensorError::InvalidShape(_))
        ));
    }

    #[test] 
    fn test_get_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            index_tensor(Idx::Coord(vec![0, 0, 0]), &tensor.view()),
            Err(TensorError::WrongDims(_))
        ));
        assert!(matches!(
            index_tensor(Idx::Coord(vec![2, 0]), &tensor.view()),
            Err(TensorError::IdxOutOfBounds(_))
        ));
    }

    #[test]
    fn test_slice_errors() {
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert!(matches!(
            tensor.view().slice(2, 0..0),
            Err(TensorError::InvalidDim(_))
        ));
        assert!(matches!(
            tensor.view().slice(0, 2..2),
            Err(TensorError::IdxOutOfBounds(_))
        ));
    }

    #[test]
    fn test_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);

        // Test Index on TensorOwned
        assert_eq!(tensor.view().get(&Idx::Coord(vec![0, 1])).unwrap(), 2);
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 6);

        // Test IndexMut on TensorOwned
        tensor.view_mut().set(vec![1, 1], 55).unwrap();
        assert_eq!(tensor.view().get(&Idx::Coord(vec![1, 1])).unwrap(), 55);
        assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 55);

        // Test on a slice (TensorView)
        let view = tensor.view();
        let view = view.slice(0, 1..1).unwrap(); // second row
        assert_eq!(view.get(vec![0]).unwrap(), 4);
        assert_eq!(view.get(vec![1]).unwrap(), 55);
        assert_eq!(view.get(vec![2]).unwrap(), 6);

        // Test on a mutable slice (TensorViewMut)
        let mut mut_view = tensor.view_mut();
        let mut mut_slice = mut_view.slice_mut(0, 0..0).unwrap(); // first row
        mut_slice.set(vec![2], 33).unwrap();
        assert_eq!(mut_slice.get(&Idx::Coord(vec![2])).unwrap(), 33);
        assert_eq!(mut_slice.get(vec![2]).unwrap(), 33);

        // Verify original tensor was changed
        assert_eq!(tensor.view().get(vec![0, 2]).unwrap(), 33);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_wrong_dims_panic() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let _ = tensor.view().get(vec![0, 0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_mut_out_of_bounds_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor.view_mut().set(vec![3], 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_mut_wrong_dims_panic() {
        let mut tensor = make_tensor(vec![1, 2, 3], vec![3]);
        tensor.view_mut().set(vec![0, 0], 4).unwrap();
    }

    // --- Additional coverage tests ---
    #[test]
    fn test_set_method() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tensor.view_mut().set(&Idx::Coord(vec![1, 2]), 99), Ok(())));
        assert_eq!(tensor.view().get(vec![1, 2]).unwrap(), 99);
    }

    #[test]
    fn test_is_row_and_is_column() {
        let row = Tensor::row(vec![1, 2, 3]);
        let col = Tensor::column(vec![1, 2, 3]);
        let scalar = Tensor::scalar(10);
        assert!(row.is_row());
        assert!(!row.is_column());
        assert!(!col.is_row());
        assert!(col.is_column());
        assert!(scalar.is_scalar());
        assert!(!scalar.is_row());
        assert!(!scalar.is_column());
    }

    #[test]
    fn test_slice_scalar_error() {
        let scalar = Tensor::scalar(5);
        assert!(matches!(scalar.view().slice(0, 0..0), Err(TensorError::InvalidDim(_))));
    }

    #[test]
    #[should_panic]
    fn test_view_as_slice_error() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap(); // shape [3]
        slice.view_as(vec![2, 2]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_view_mut_as_error() {
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        let view_mut = tensor.view_mut();
        view_mut.view_as(vec![3, 2]).unwrap();
    }

    #[test]
    fn test_item_wrong_dims_error() {
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert!(matches!(tensor.view().get(&Idx::Item), Err(TensorError::WrongDims(_))));
    }

    #[test]
    fn test_from_buf_empty_shape_error() {
        assert!(matches!(Tensor::from_buf(Vec::<i32>::new(), vec![]), Err(TensorError::InvalidShape(_))));
    }

    // #[test]
    // fn test_modify_after_reshape_reflects() {
    //     let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
    //     {
    //         let view_mut = tensor.view_mut();
    //         let mut reshaped = view_mut.view_as(vec![4]).unwrap();
    //         reshaped.set(Idx::At(3), 40).unwrap(); // modify last element
    //         assert_eq!(reshaped.get(&Idx::At(3)).unwrap(), 40);
    //     }
    //     assert_eq!(tensor.view().get(vec![1, 1]).unwrap(), 40);
    // }

    #[test]
    fn test_slice_single_dim_to_scalar() {
        let tensor = make_tensor(vec![42], vec![1]);

        let view = tensor.view();
        let slice = view.slice(0, 0..0).unwrap();
        assert_eq!(*slice.shape(), vec![]);
        assert_eq!(slice.get(&Idx::Item).unwrap(), 42);
        assert_eq!(slice.get(&Idx::Coord(vec![])).unwrap(), 42);
    }

    #[test]
    fn test_cube_slice_mut_chain() {
        let mut tensor = make_tensor(vec![
            1, 2,
            4, 5,
            6, 7,
            8, 9
        ], vec![2, 2, 2]);

        let mut view = tensor.view_mut();
        let mut depth_view = view.slice_mut(0, 1..1).unwrap(); // shape [2,2]
        assert_eq!(*depth_view.shape(), vec![2, 2]);
        let mut row_view = depth_view.slice_mut(0, 1..1).unwrap(); // shape [2]
        assert_eq!(*row_view.shape(), vec![2]);
        row_view.set(Idx::At(0), 800).unwrap(); // maps to original [1,1,0]
        assert_eq!(tensor.view().get(vec![1, 1, 0]).unwrap(), 800);
        assert_eq!(tensor.view().get(vec![1, 1, 1]).unwrap(), 9);
    }

    // --- Additional metadata coverage tests ---
    #[test]
    fn test_meta_dims_and_dim() {
        let scalar = Tensor::scalar(10);
        assert_eq!(scalar.dims(), 0);

        let vec = make_tensor(vec![1, 2, 3], vec![3]);
        assert_eq!(vec.dims(), 1);
        assert_eq!(vec.dim(0), 3);

        let mat = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        assert_eq!(mat.dims(), 2);
        assert_eq!(mat.dim(0), 2);
        assert_eq!(mat.dim(1), 2);

        let cube = make_tensor(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        assert_eq!(cube.dims(), 3);
        assert_eq!(cube.dim(0), 2);
        assert_eq!(cube.dim(1), 2);
        assert_eq!(cube.dim(2), 2);
    }

    #[test]
    fn test_meta_size() {
        let scalar = Tensor::scalar(42);
        assert_eq!(scalar.size(), 1);

        let vec = make_tensor(vec![1, 2, 3, 4], vec![4]);
        assert_eq!(vec.size(), 4);

        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert_eq!(mat.size(), 6);
    }

    #[test]
    fn test_meta_offsets_on_slices() {
        // 2x2 -> stride [2,1]
        let m = make_tensor(vec![1,2,3,4], vec![2,2]);
        let v_m = m.view();
        let s = v_m.slice(0, 1..1).unwrap(); // take second row

        assert_eq!(s.offset(), 2);

        // 2x2x2 -> stride [4,2,1]
        let c = make_tensor(vec![1,2,3,4,5,6,7,8], vec![2,2,2]);
        let v_c = c.view();
        let s1 = v_c.slice(0, 1..1).unwrap(); // offset +4

        assert_eq!(s1.offset(), 4);
        let s2 = s1.slice(1, 1..1).unwrap(); // now stride [2,1], add +1
        assert_eq!(s2.offset(), 5);
    }

    #[test]
    fn test_is_contiguous_owned_and_slices() {
        // Owned are contiguous
        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert!(mat.is_contiguous());

        let v1 = mat.view();
        let row = v1.slice(0, 1..1).unwrap();

    // Row slice -> contiguous (stride [1])
        assert!(row.is_contiguous());

        // Column-like slice -> non-contiguous (stride [3])
        let v2 = mat.view();
        let col_like = v2.slice(1, 0..0).unwrap();
        assert!(!col_like.is_contiguous());
    }

    // #[test]
    // fn test_reshape_noncontiguous_slice_contiguous() {
    //     // Start non-contiguous 1D view (stride [3])
    //     let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
    //     let v3 = mat.view();
    //     let col_like = v3.slice(1, 0..0).unwrap(); // shape [2], stride [3]
    //     assert!(!col_like.is_contiguous());

    //     // Reshape to [1,2] -> becomes contiguous
    //     let reshaped = col_like.view_as(vec![1, 2]).unwrap();
    //     assert!(reshaped.is_contiguous());
    //     assert_eq!(*reshaped.shape(), vec![1,2]);
    //     // Note: reshaping a non-contiguous slice uses underlying memory order
    //     // starting at the slice's offset; here it's [1, 2]
    //     assert_eq!(reshaped.get(vec![0,0]).unwrap(), 1);
    //     assert_eq!(reshaped.get(vec![0,1]).unwrap(), 2);
    // }

    #[test]
    fn test_tensor_meta_direct_impl() {
        // contiguous meta
        let shape = vec![2, 3];
        let stride = super::shape_to_stride(&shape.clone().into());
        let meta = MetaTensor::new(shape.clone(), stride.clone(), 0);
        assert_eq!(meta.shape(), &shape);
        assert_eq!(meta.strides(), &stride);
        assert_eq!(meta.offset(), 0);
        assert_eq!(meta.dims(), 2);
        assert_eq!(meta.dim(0), 2);
        assert_eq!(meta.dim(1), 3);
        assert_eq!(meta.size(), 6);
        assert!(!meta.is_scalar());
        assert!(!meta.is_row());
        assert!(!meta.is_column());
        assert!(meta.is_contiguous());

        // non-contiguous meta (e.g., a column-like view of a [2,3] matrix)
        let nc_meta = MetaTensor::new(vec![2], vec![3], 0);
        assert_eq!(nc_meta.dims(), 1);
        assert_eq!(nc_meta.size(), 2);
        assert!(!nc_meta.is_contiguous());
    }

    #[test]
    fn test_idx_at_vs_coord_on_sliced_view() {
        // Create a non-contiguous 1D view and ensure At == Coord mapping
        let mat = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        let v4 = mat.view();
        let col_like = v4.slice(1, 0..0).unwrap(); // [2], stride [3]
        assert_eq!(col_like.get(vec![0]).unwrap(), col_like.get(&Idx::Coord(vec![0])).unwrap());
        assert_eq!(col_like.get(vec![1]).unwrap(), col_like.get(&Idx::Coord(vec![1])).unwrap());
    }

    #[test]
    fn test_dims_equals_stride_len() {
        let owned = make_tensor(vec![1,2,3,4,5,6], vec![2,3]);
        assert_eq!(owned.dims(), owned.strides().len());

        let v5 = owned.view();
        let row = v5.slice(0, 1..1).unwrap();
        assert_eq!(row.dims(), row.strides().len());

        let v6 = owned.view();
        let col_like = v6.slice(1, 0..0).unwrap();
        assert_eq!(col_like.dims(), col_like.strides().len());
    }

    #[test]
    fn test_view_to_owned_contiguous() {
        // Test converting a contiguous view to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let owned = view.owned();
        
        // Should have same shape and values
        assert_eq!(owned.shape(), tensor.shape());
        assert_eq!(owned.buf, tensor.buf);
        assert!(owned.is_contiguous());
    }

    #[test]
    fn test_view_to_owned_noncontiguous_column() {
        // Test converting a non-contiguous column slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let col_slice = view.slice(1, 1..1).unwrap(); // Middle column: [2, 5]
        
        // This is non-contiguous (stride [3])
        assert!(!col_slice.is_contiguous());
        assert_eq!(*col_slice.shape(), vec![2]);
        
        // Convert to owned
        let owned = col_slice.owned();
        
        // Owned should be contiguous with correct values
        assert!(owned.is_contiguous());
        assert_eq!(*owned.shape(), vec![2]);
        assert_eq!(owned.buf, vec![2, 5].into_boxed_slice());
        assert_eq!(index_tensor(Idx::At(0), &owned.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &owned.view()).unwrap(), 5);
    }

    #[test]
    fn test_view_to_owned_row_slice() {
        // Test converting a row slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let row_slice = view.slice(0, 1..1).unwrap(); // Second row: [4, 5, 6]
        
        assert!(row_slice.is_contiguous());
        
        let owned = row_slice.owned();
        
        assert_eq!(*owned.shape(), vec![3]);
        assert_eq!(owned.buf, vec![4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_view_to_owned_reshaped() {
        // Test converting a reshaped view to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view = tensor.view();
        let reshaped = view.view_as(vec![3, 2]).unwrap();
        
        let owned = reshaped.owned();
        
        assert_eq!(*owned.shape(), vec![3, 2]);
        assert_eq!(owned.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
        assert!(owned.is_contiguous());
    }

    #[test]
    fn test_view_to_owned_3d_slice() {
        // Test converting a 3D slice to owned
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]);
        let view = tensor.view();
        let depth_slice = view.slice(0, 1..1).unwrap(); // Second depth: [5, 6, 7, 8]
        
        let owned = depth_slice.owned();
        
        assert_eq!(*owned.shape(), vec![2, 2]);
        assert_eq!(owned.buf, vec![5, 6, 7, 8].into_boxed_slice());
        assert!(owned.is_contiguous());
    }

    #[test]
    fn test_viewmut_to_owned() {
        // Test that mutable views can also be converted to owned
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let view_mut = tensor.view_mut();
        let owned = view_mut.owned();
        
        assert_eq!(*owned.shape(), vec![2, 3]);
        assert_eq!(owned.buf, vec![1, 2, 3, 4, 5, 6].into_boxed_slice());
    }

    #[test]
    fn test_instantiation() {
        let tensor = Tensor::<f32>::zeros((1, 1));

        assert_eq!(*tensor.shape(), vec![1, 1]);
        assert_eq!(tensor.buf.len(), 1);
        assert_eq!(tensor.buf[0], 0.0);

    }

    #[test]
    fn test_range_to_syntax() {
        // Test ..5 syntax (from start to 5)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, ..5).unwrap();
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 50);
        
        // Test mutation - add to each element
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, ..5).unwrap();
        slice_mut.set(&Idx::At(0), 11).unwrap();
        slice_mut.set(&Idx::At(2), 33).unwrap();
        slice_mut.set(&Idx::At(4), 55).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 33);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 55);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 60); // Unchanged
    }

    #[test]
    fn test_range_from_syntax() {
        // Test 2.. syntax (from 2 to end)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 2..).unwrap();
        assert_eq!(*slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(5), &slice).unwrap(), 80);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 2..).unwrap();
        slice_mut.set(&Idx::At(0), 333).unwrap();
        slice_mut.set(&Idx::At(5), 888).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 333);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 888);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 20); // Unchanged
    }

    #[test]
    fn test_range_full_syntax() {
        // Test .. syntax (full range)
        let buf = vec![10, 20, 30, 40];
        let shape = vec![4];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, ..).unwrap();
        assert_eq!(*slice.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 40);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, ..).unwrap();
        slice_mut.set(&Idx::At(0), 100).unwrap();
        slice_mut.set(&Idx::At(3), 400).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 100);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 400);
    }

    #[test]
    fn test_inclusive_range_syntax() {
        // Test 2..=5 syntax (inclusive end)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 2..=5).unwrap();
        assert_eq!(*slice.shape(), vec![4]); // Indices 2, 3, 4, 5
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 30);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 50);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 60);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 2..=5).unwrap();
        slice_mut.set(&Idx::At(0), 333).unwrap();
        slice_mut.set(&Idx::At(3), 666).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 333);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 666);
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 20); // Unchanged
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged
    }

    #[test]
    fn test_inclusive_range_zero_end() {
        // Test 0..=2 syntax (first 3 elements)
        let buf = vec![10, 20, 30, 40, 50];
        let shape = vec![5];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in slice
        let view = tensor.view();
        let slice = view.slice(0, 0..=2).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 0, 1, 2
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 10);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 0..=2).unwrap();
        slice_mut.set(&Idx::At(1), 222).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 222);
    }

    #[test]
    fn test_single_index_syntax() {
        // Test single index 3 (reduces dimension)
        let buf = vec![10, 20, 30, 40, 50, 60];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);
        
        // Slice along dimension 1 (columns) at index 1
        let view = tensor.view();
        let slice = view.slice(1, 1).unwrap();
        assert_eq!(*slice.shape(), vec![2]); // Dimension reduced
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 20); // First row, column 1
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50); // Second row, column 1
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(1, 1).unwrap();
        slice_mut.set(&Idx::At(0), 222).unwrap();
        slice_mut.set(&Idx::At(1), 555).unwrap();
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &view).unwrap(), 222);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1]), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &view).unwrap(), 10); // Unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &view).unwrap(), 30); // Unchanged
    }

    #[test]
    fn test_reverse_range_exclusive_with_values() {
        // Test 5..2 auto-reverse (step=-1)
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values in reversed slice
        let view = tensor.view();
        let slice = view.slice(0, 5..2).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 5, 4, 3
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 60); // Index 5
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 50); // Index 4
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 40); // Index 3
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 5..2).unwrap();
        slice_mut.set(&Idx::At(0), 666).unwrap(); // Should modify index 5
        slice_mut.set(&Idx::At(2), 444).unwrap(); // Should modify index 3
        
        // Verify mutations reflected in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 666);
        assert_eq!(index_tensor(Idx::At(3), &view).unwrap(), 444);
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 30); // Unchanged
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged
    }

    #[test]
    fn test_slice_from_with_step() {
        // Test Slice::from(8..).step(-1) - from index 8 to start, reversed
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(8..).step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![9]); // Indices 8, 7, 6, 5, 4, 3, 2, 1, 0
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 90);  // Index 8
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 80);  // Index 7
        assert_eq!(index_tensor(Idx::At(8), &slice).unwrap(), 10);  // Index 0
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(8..).step(-1)).unwrap();
        slice_mut.set(&Idx::At(0), 999).unwrap();  // Should modify index 8
        slice_mut.set(&Idx::At(8), 111).unwrap();  // Should modify index 0
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 111);
        assert_eq!(index_tensor(Idx::At(9), &view).unwrap(), 100); // Unchanged
    }

    #[test]
    fn test_slice_to_with_step() {
        // Test Slice::from(..5).step(-1) - from end down to index 5 (exclusive)
        // With negative step and end=5, start defaults to last index (7)
        // So this gives indices [7, 6, 5] going backwards = 3 elements
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(..5).step(-1)).unwrap();
        assert_eq!(*slice.shape(), vec![2]); // Indices 7, 6 (end=5 is exclusive)
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 80); // Index 7
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 70); // Index 6
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(..5).step(-1)).unwrap();
        slice_mut.set(&Idx::At(0), 888).unwrap(); // Should modify index 7
        slice_mut.set(&Idx::At(1), 777).unwrap(); // Should modify index 6
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 888);
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 777);
        assert_eq!(index_tensor(Idx::At(5), &view).unwrap(), 60); // Unchanged (end is exclusive)
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 50); // Unchanged
    }

    #[test]
    fn test_reverse_first_n_elements() {
        // To reverse first N elements, use Slice::new(Some(N-1), None, Some(-1))
        // This gets elements from index N-1 down to 0
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let shape = vec![8];
        let mut tensor = make_tensor(buf, shape);
        
        // Reverse first 5 elements: indices 4, 3, 2, 1, 0
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(4), None, -1)).unwrap();
        assert_eq!(*slice.shape(), vec![5]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 50); // Index 4
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 40); // Index 3
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 30); // Index 2
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 20); // Index 1
        assert_eq!(index_tensor(Idx::At(4), &slice).unwrap(), 10); // Index 0
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::new(Some(4), None, -1)).unwrap();
        slice_mut.set(&Idx::At(0), 555).unwrap(); // Should modify index 4
        slice_mut.set(&Idx::At(4), 111).unwrap(); // Should modify index 0
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 111);
    }

    #[test]
    fn test_manual_slice_construction_values() {
        // Test Slice::new(Some(8), Some(2), Some(-2)) - from 8 to 2, step -2
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Check values
        let view = tensor.view();
        let slice = view.slice(0, Slice::new(Some(8), Some(2), -2)).unwrap();
        assert_eq!(*slice.shape(), vec![3]); // Indices 8, 6, 4
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 90);  // Index 8
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 70);  // Index 6
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 50);  // Index 4
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::new(Some(8), Some(2), -2)).unwrap();
        slice_mut.set(&Idx::At(0), 999).unwrap();  // Should modify index 8
        slice_mut.set(&Idx::At(2), 555).unwrap();  // Should modify index 4
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(8), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::At(4), &view).unwrap(), 555);
        assert_eq!(index_tensor(Idx::At(6), &view).unwrap(), 70); // Unchanged until we check
    }

    #[test]
    fn test_empty_slice_same_indices() {
        // Test 5..5 (empty slice)
        let buf = vec![10, 20, 30, 40, 50, 60];
        let shape = vec![6];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let slice = view.slice(0, 5..5).unwrap();
        assert_eq!(*slice.shape(), vec![]); // Empty - dimension collapsed
    }

    #[test]
    fn test_matrix_column_slice_with_mutation() {
        // Test slicing a column from a matrix with value checks and mutation
        let buf = vec![
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12
        ];
        let shape = vec![3, 4]; // 3 rows, 4 columns
        let mut tensor = make_tensor(buf, shape);
        
        // Slice column 2 (third column)
        let view = tensor.view();
        let slice = view.slice(1, 2..3).unwrap();
        assert_eq!(*slice.shape(), vec![3, 1]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice).unwrap(), 3);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0]), &slice).unwrap(), 11);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(1, 2..3).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 0]), 33).unwrap();
        slice_mut.set(&Idx::Coord(vec![2, 0]), 111).unwrap();
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &view).unwrap(), 33);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2]), &view).unwrap(), 111);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &view).unwrap(), 7); // Unchanged
    }

    #[test]
    fn test_3d_depth_slice_with_mutation() {
        // Test 3D tensor depth slicing with value checks and mutation
        let buf = vec![
            // Depth 0
            1, 2,
            3, 4,
            // Depth 1
            5, 6,
            7, 8,
            // Depth 2
            9, 10,
            11, 12
        ];
        let shape = vec![3, 2, 2]; // 3 depth, 2 rows, 2 columns
        let mut tensor = make_tensor(buf, shape);
        
        // Slice depth 1..=1 (second depth slice)
        let view = tensor.view();
        let slice = view.slice(0, 1..=1).unwrap();
        assert_eq!(*slice.shape(), vec![1, 2, 2]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &slice).unwrap(), 5);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 0]), &slice).unwrap(), 7);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1, 1]), &slice).unwrap(), 8);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, 1..=1).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 0, 0]), 55).unwrap();
        slice_mut.set(&Idx::Coord(vec![0, 1, 1]), 88).unwrap();
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0, 0]), &view).unwrap(), 55);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &view).unwrap(), 88);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::Coord(vec![2, 0, 0]), &view).unwrap(), 9); // Unchanged
    }

    #[test]
    fn test_chained_slices_with_mutation() {
        // Test chaining different slice syntaxes with mutation
        let buf = vec![
            1,  2,  3,  4,  5,  6,
            7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24
        ];
        let shape = vec![4, 6]; // 4 rows, 6 columns
        let mut tensor = make_tensor(buf, shape);
        
        // First slice: rows 1..3
        let view = tensor.view();
        let slice1 = view.slice(0, 1..3).unwrap();
        assert_eq!(*slice1.shape(), vec![2, 6]);
        
        // Second slice: columns 2..=4 (inclusive)
        let slice2 = slice1.slice(1, 2..=4).unwrap();
        assert_eq!(*slice2.shape(), vec![2, 3]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &slice2).unwrap(), 9);   // Row 1, Col 2
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &slice2).unwrap(), 11);  // Row 1, Col 4
        assert_eq!(index_tensor(Idx::Coord(vec![1, 0]), &slice2).unwrap(), 15);  // Row 2, Col 2
        assert_eq!(index_tensor(Idx::Coord(vec![1, 2]), &slice2).unwrap(), 17);  // Row 2, Col 4
        
        // Test mutation through chained slices
        let mut view_mut = tensor.view_mut();
        let mut slice1_mut = view_mut.slice_mut(0, 1..3).unwrap();
        let mut slice2_mut = slice1_mut.slice_mut(1, 2..=4).unwrap();
        slice2_mut.set(&Idx::Coord(vec![0, 1]), 999).unwrap(); // Row 1, Col 3 in original
        slice2_mut.set(&Idx::Coord(vec![1, 2]), 1717).unwrap(); // Row 2, Col 4 in original
        
        // Verify mutations in original
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::Coord(vec![1, 3]), &view).unwrap(), 999);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 4]), &view).unwrap(), 1717);
    }

    #[test]
    fn test_step_2_with_mutation() {
        // Test custom step with value checks and mutation
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let shape = vec![10];
        let mut tensor = make_tensor(buf, shape);
        
        // Every other element starting from index 1
        let view = tensor.view();
        let slice = view.slice(0, Slice::from(1..9).step(2)).unwrap();
        assert_eq!(*slice.shape(), vec![4]); // Indices 1, 3, 5, 7
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(2), &slice).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(3), &slice).unwrap(), 8);
        
        // Test mutation
        let mut view_mut = tensor.view_mut();
        let mut slice_mut = view_mut.slice_mut(0, Slice::from(1..9).step(2)).unwrap();
        slice_mut.set(&Idx::At(0), 22).unwrap();  // Index 1
        slice_mut.set(&Idx::At(3), 88).unwrap();  // Index 7
        
        // Verify mutations
        let view = tensor.view();
        assert_eq!(index_tensor(Idx::At(1), &view).unwrap(), 22);
        assert_eq!(index_tensor(Idx::At(7), &view).unwrap(), 88);
        assert_eq!(index_tensor(Idx::At(0), &view).unwrap(), 1); // Unchanged
        assert_eq!(index_tensor(Idx::At(2), &view).unwrap(), 3); // Unchanged (skipped by step)
    }

    // --- Permute and Transpose Tests ---
    
    #[test]
    fn test_transpose_2d() {
        // Test basic 2D matrix transpose
        let buf = vec![
            1, 2, 3,
            4, 5, 6
        ];
        let shape = vec![2, 3]; // 2 rows, 3 columns
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let transposed = view.transpose();
        assert_eq!(*transposed.shape(), vec![3, 2]); // 3 rows, 2 columns
        assert_eq!(*transposed.strides(), vec![1, 3]); // stride is swapped
        
        // Check values are correctly transposed
        assert_eq!(index_tensor(coord![0, 0], &transposed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 1], &transposed).unwrap(), 4);
        assert_eq!(index_tensor(coord![1, 0], &transposed).unwrap(), 2);
        assert_eq!(index_tensor(coord![1, 1], &transposed).unwrap(), 5);
        assert_eq!(index_tensor(coord![2, 0], &transposed).unwrap(), 3);
        assert_eq!(index_tensor(coord![2, 1], &transposed).unwrap(), 6);
    }

    #[test]
    fn test_transpose_3d() {
        // Test 3D tensor transpose (reverses all dimensions)
        let buf = vec![
            1, 2,
            3, 4,
            
            5, 6,
            7, 8
        ];
        let shape = vec![2, 2, 2]; // depth, rows, columns
        let tensor = make_tensor(buf, shape);
        
        let transposed = tensor.transpose();
        assert_eq!(*transposed.shape(), vec![2, 2, 2]); // dimensions reversed
        assert_eq!(*transposed.strides(), vec![1, 2, 4]); // strides reversed
        
        // Original [0,0,0] -> 1, Transposed [0,0,0] -> 1
        assert_eq!(index_tensor(coord![0, 0, 0], &transposed).unwrap(), 1);
        // Original [0,0,1] -> 2, Transposed [1,0,0] -> 2
        assert_eq!(index_tensor(coord![1, 0, 0], &transposed).unwrap(), 2);
        // Original [0,1,0] -> 3, Transposed [0,1,0] -> 3
        assert_eq!(index_tensor(coord![0, 1, 0], &transposed).unwrap(), 3);
        // Original [1,0,0] -> 5, Transposed [0,0,1] -> 5
        assert_eq!(index_tensor(coord![0, 0, 1], &transposed).unwrap(), 5);
        // Original [1,1,1] -> 8, Transposed [1,1,1] -> 8
        assert_eq!(index_tensor(coord![1, 1, 1], &transposed).unwrap(), 8);
    }

    #[test]
    fn test_transpose_mut_2d() {
        // Test mutable transpose with write
        let buf = vec![
            1, 2, 3,
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);
        
        let mut transposed = tensor.transpose_mut();
        assert_eq!(*transposed.shape(), vec![3, 2]);
        
        // Modify through transposed view
        transposed.set(coord![0, 0], 10).unwrap(); // Original [0,0]
        transposed.set(coord![1, 1], 50).unwrap(); // Original [1,1]
        transposed.set(coord![2, 1], 60).unwrap(); // Original [1,2]
        
        // Check original tensor was modified
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0], &view).unwrap(), 10);
        assert_eq!(index_tensor(coord![1, 1], &view).unwrap(), 50);
        assert_eq!(index_tensor(coord![1, 2], &view).unwrap(), 60);
    }

    #[test]
    fn test_transpose_mut_3d() {
        // Test mutable 3D transpose with write
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);
        
        let mut transposed = tensor.transpose_mut();
        assert_eq!(*transposed.shape(), vec![2, 2, 2]);
        
        // Modify through transposed view
        transposed.set(coord![0, 0, 0], 100).unwrap(); // Original [0,0,0]
        transposed.set(coord![1, 0, 0], 200).unwrap(); // Original [0,0,1]
        transposed.set(coord![0, 0, 1], 500).unwrap(); // Original [1,0,0]
        
        // Check original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0, 0], &view).unwrap(), 100);
        assert_eq!(index_tensor(coord![0, 0, 1], &view).unwrap(), 200);
        assert_eq!(index_tensor(coord![1, 0, 0], &view).unwrap(), 500);
    }

    #[test]
    fn test_permute_2d_swap() {
        // Test permuting 2D tensor (same as transpose for 2D)
        let buf = vec![
            1, 2, 3,
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let permuted = tensor.permute(vec![1, 0]).unwrap();
        assert_eq!(*permuted.shape(), vec![3, 2]);
        assert_eq!(*permuted.strides(), vec![1, 3]);
        
        // Should match transpose
        assert_eq!(index_tensor(coord![0, 0], &permuted).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 0], &permuted).unwrap(), 2);
        assert_eq!(index_tensor(coord![2, 1], &permuted).unwrap(), 6);
    }

    #[test]
    fn test_permute_3d_rotate() {
        // Test 3D permute with rotation [0,1,2] -> [2,0,1]
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![2, 2, 3]; // depth, rows, columns
        let tensor = make_tensor(buf, shape);
        
        // Permute: [depth, rows, cols] -> [cols, depth, rows]
        let permuted = tensor.permute(vec![2, 0, 1]).unwrap();
        assert_eq!(*permuted.shape(), vec![3, 2, 2]); // [cols, depth, rows]
        
        // Original strides: [6, 3, 1]
        // Permuted strides: [1, 6, 3]
        assert_eq!(*permuted.strides(), vec![1, 6, 3]);
        
        // Original [0,0,0] -> 1, Permuted [0,0,0] -> 1
        assert_eq!(index_tensor(coord![0, 0, 0], &permuted).unwrap(), 1);
        // Original [0,0,1] -> 2, Permuted [1,0,0] -> 2
        assert_eq!(index_tensor(coord![1, 0, 0], &permuted).unwrap(), 2);
        // Original [0,1,0] -> 4, Permuted [0,0,1] -> 4
        assert_eq!(index_tensor(coord![0, 0, 1], &permuted).unwrap(), 4);
        // Original [1,1,2] -> 12, Permuted [2,1,1] -> 12
        assert_eq!(index_tensor(coord![2, 1, 1], &permuted).unwrap(), 12);
    }

    #[test]
    fn test_permute_3d_partial_swap() {
        // Test swapping only first two dimensions
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        // Permute: [0,1,2] -> [1,0,2] (swap first two dims)
        let permuted = tensor.permute(vec![1, 0, 2]).unwrap();
        assert_eq!(*permuted.shape(), vec![2, 2, 2]);
        assert_eq!(*permuted.strides(), vec![2, 4, 1]);
        
        // Original [0,0,0] -> 1, Permuted [0,0,0] -> 1
        assert_eq!(index_tensor(coord![0, 0, 0], &permuted).unwrap(), 1);
        // Original [0,1,0] -> 3, Permuted [1,0,0] -> 3
        assert_eq!(index_tensor(coord![1, 0, 0], &permuted).unwrap(), 3);
        // Original [1,0,1] -> 6, Permuted [0,1,1] -> 6
        assert_eq!(index_tensor(coord![0, 1, 1], &permuted).unwrap(), 6);
    }

    #[test]
    fn test_permute_mut_with_write() {
        // Test mutable permute with write operations
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![2, 2, 3];
        let mut tensor = make_tensor(buf, shape);
        
        let mut permuted = tensor.permute_mut(vec![2, 0, 1]).unwrap();
        assert_eq!(*permuted.shape(), vec![3, 2, 2]);
        
        // Write through permuted view
        permuted.set(coord![0, 0, 0], 100).unwrap(); // Original [0,0,0]
        permuted.set(coord![1, 0, 0], 200).unwrap(); // Original [0,0,1]
        permuted.set(coord![2, 1, 1], 1200).unwrap(); // Original [1,1,2]
        
        // Verify original tensor was modified
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0, 0], &view).unwrap(), 100);
        assert_eq!(index_tensor(coord![0, 0, 1], &view).unwrap(), 200);
        assert_eq!(index_tensor(coord![1, 1, 2], &view).unwrap(), 1200);
    }

    #[test]
    fn test_permute_identity() {
        // Test that permuting with identity permutation keeps everything the same
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let permuted = tensor.permute(vec![0, 1]).unwrap();
        assert_eq!(*permuted.shape(), vec![2, 3]);
        assert_eq!(*permuted.strides(), vec![3, 1]);
        
        // All values should be the same
        assert_eq!(index_tensor(coord![0, 0], &permuted).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &permuted).unwrap(), 6);
    }

    #[test]
    fn test_permute_4d() {
        // Test 4D permutation
        let buf: Vec<i32> = (1..=16).collect();
        let shape = vec![2, 2, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        // Permute: [0,1,2,3] -> [3,2,1,0] (reverse all)
        let permuted = tensor.permute(vec![3, 2, 1, 0]).unwrap();
        assert_eq!(*permuted.shape(), vec![2, 2, 2, 2]);
        assert_eq!(*permuted.strides(), vec![1, 2, 4, 8]);
        
        // Original [0,0,0,0] -> 1, Permuted [0,0,0,0] -> 1
        assert_eq!(index_tensor(coord![0, 0, 0, 0], &permuted).unwrap(), 1);
        // Original [0,0,0,1] -> 2, Permuted [1,0,0,0] -> 2
        assert_eq!(index_tensor(coord![1, 0, 0, 0], &permuted).unwrap(), 2);
        // Original [1,1,1,1] -> 16, Permuted [1,1,1,1] -> 16
        assert_eq!(index_tensor(coord![1, 1, 1, 1], &permuted).unwrap(), 16);
    }

    #[test]
    fn test_permute_on_slice() {
        // Test permuting a slice
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ];
        let shape = vec![3, 4];
        let tensor = make_tensor(buf, shape);
        
        // First take a slice of rows
        let slice = tensor.slice(0, 1..3).unwrap(); // Rows 1-2
        assert_eq!(*slice.shape(), vec![2, 4]);
        
        // Then permute the slice
        let permuted = slice.permute(vec![1, 0]).unwrap();
        assert_eq!(*permuted.shape(), vec![4, 2]);
        
        // Check values
        assert_eq!(index_tensor(coord![0, 0], &permuted).unwrap(), 5); // Original [1,0]
        assert_eq!(index_tensor(coord![3, 1], &permuted).unwrap(), 12); // Original [2,3]
    }

    #[test]
    fn test_transpose_on_slice() {
        // Test transposing a slice
        let buf = vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        ];
        let shape = vec![4, 3];
        let tensor = make_tensor(buf, shape);
        
        // Slice middle two rows
        let slice = tensor.slice(0, 1..3).unwrap();
        assert_eq!(*slice.shape(), vec![2, 3]);
        
        // Transpose the slice
        let transposed = slice.transpose();
        assert_eq!(*transposed.shape(), vec![3, 2]);
        
        assert_eq!(index_tensor(coord![0, 0], &transposed).unwrap(), 4);
        assert_eq!(index_tensor(coord![2, 1], &transposed).unwrap(), 9);
    }

    #[test]
    fn test_chained_permutations() {
        // Test chaining multiple permutations
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![2, 2, 2];
        let tensor = make_tensor(buf, shape);
        
        // First permute: [0,1,2] -> [2,0,1]
        let perm1 = tensor.permute(vec![2, 0, 1]).unwrap();
        assert_eq!(*perm1.shape(), vec![2, 2, 2]);
        
        // Second permute: [0,1,2] -> [1,2,0] on perm1
        // Composition: [2,0,1][1,2,0] = [0,1,2] (identity)
        let perm2 = perm1.permute(vec![1, 2, 0]).unwrap();
        assert_eq!(*perm2.shape(), vec![2, 2, 2]);
        
        // Verify composition - should be back to original layout
        assert_eq!(index_tensor(coord![0, 0, 0], &perm2).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 0, 0], &perm2).unwrap(), 5);
        assert_eq!(index_tensor(coord![0, 1, 0], &perm2).unwrap(), 3);
        assert_eq!(index_tensor(coord![1, 1, 1], &perm2).unwrap(), 8);
    }

    #[test]
    fn test_permute_mut_chained() {
        // Test chained mutable permutations with writes
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![2, 2, 2];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut perm1 = tensor.permute_mut(vec![1, 0, 2]).unwrap();
            let mut perm2 = perm1.permute_mut(vec![2, 0, 1]).unwrap();
            
            // Write through doubly-permuted view
            perm2.set(coord![0, 0, 0], 111).unwrap();
        }
        
        // Check original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0, 0], &view).unwrap(), 111);
    }

    #[test]
    fn test_permute_wrong_dims_error() {
        // Test error when permutation has wrong number of dimensions
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Too few dimensions
        assert!(matches!(
            tensor.view().permute(vec![0]),
            Err(TensorError::WrongDims(_))
        ));
        
        // Too many dimensions
        assert!(matches!(
            tensor.view().permute(vec![0, 1, 2]),
            Err(TensorError::WrongDims(_))
        ));
    }

    #[test]
    fn test_permute_invalid_dim_error() {
        // Test error when permutation contains invalid dimension index
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Dimension index out of range
        assert!(matches!(
            tensor.view().permute(vec![0, 3]),
            Err(TensorError::InvalidDim(_))
        ));
        
        assert!(matches!(
            tensor.view().permute(vec![2, 1]),
            Err(TensorError::InvalidDim(_))
        ));
    }

    #[test]
    fn test_transpose_then_slice() {
        // Test slicing after transpose
        let buf = vec![
            1, 2, 3,
            4, 5, 6
        ];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let transposed = tensor.transpose();
        assert_eq!(*transposed.shape(), vec![3, 2]);
        
        // Slice first row of transposed (which is first column of original)
        let slice = transposed.slice(0, 0..0).unwrap();
        assert_eq!(*slice.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &slice).unwrap(), 4);
    }

    #[test]
    fn test_slice_then_transpose_mut() {
        // Test transpose_mut after slicing with writes
        let buf = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ];
        let shape = vec![3, 4];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut slice_mut = tensor.slice_mut(0, 0..2).unwrap();
            let mut transposed = slice_mut.transpose_mut();
            
            // Modify through transposed slice
            transposed.set(coord![0, 0], 100).unwrap(); // Original [0,0]
            transposed.set(coord![3, 1], 800).unwrap(); // Original [1,3]
        }
        
        // Check original
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0], &view).unwrap(), 100);
        assert_eq!(index_tensor(coord![1, 3], &view).unwrap(), 800);
        assert_eq!(index_tensor(coord![2, 0], &view).unwrap(), 9); // Unchanged
    }

    #[test]
    fn test_permute_write_compatibility_complex() {
        // Complex test: slice -> permute -> write -> verify original
        let buf: Vec<i32> = (1..=24).collect();
        let shape = vec![2, 3, 4];
        let mut tensor = make_tensor(buf, shape);
        
        {
            // Take a slice
            let mut slice_mut = tensor.slice_mut(0, 1..2).unwrap();
            assert_eq!(*slice_mut.shape(), vec![1, 3, 4]);
            
            // Permute the slice
            let mut permuted = slice_mut.permute_mut(vec![2, 0, 1]).unwrap();
            assert_eq!(*permuted.shape(), vec![4, 1, 3]);
            
            // Write through permuted view
            permuted.set(coord![0, 0, 0], 1300).unwrap(); // Original [1,0,0]
            permuted.set(coord![3, 0, 2], 2400).unwrap(); // Original [1,2,3]
        }
        
        // Verify writes propagated to original
        let view = tensor.view();
        assert_eq!(index_tensor(coord![1, 0, 0], &view).unwrap(), 1300);
        assert_eq!(index_tensor(coord![1, 2, 3], &view).unwrap(), 2400);
        assert_eq!(index_tensor(coord![0, 0, 0], &view).unwrap(), 1); // Unchanged (different slice)
    }

    #[test]
    fn test_unsqueeze_scalar() {
        // Test unsqueezing a scalar tensor
        let buf = vec![42];
        let shape = vec![];  // scalar has empty shape
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed = view.unsqueeze();
        
        // Should have shape [1] after unsqueezing
        assert_eq!(*unsqueezed.shape(), vec![1]);
        assert_eq!(*unsqueezed.strides(), vec![1]);
        assert_eq!(index_tensor(Idx::At(0), &unsqueezed).unwrap(), 42);
    }

    #[test]
    fn test_unsqueeze_1d() {
        // Test unsqueezing a 1D tensor
        let buf = vec![1, 2, 3, 4, 5];
        let shape = vec![5];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed = view.unsqueeze();
        
        // Should have shape [1, 5] after unsqueezing
        assert_eq!(*unsqueezed.shape(), vec![1, 5]);
        assert_eq!(*unsqueezed.strides(), vec![5, 1]);
        
        // Verify data access
        assert_eq!(index_tensor(coord![0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 1], &unsqueezed).unwrap(), 2);
        assert_eq!(index_tensor(coord![0, 2], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![0, 3], &unsqueezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![0, 4], &unsqueezed).unwrap(), 5);
    }

    #[test]
    fn test_unsqueeze_2d() {
        // Test unsqueezing a 2D tensor
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed = view.unsqueeze();
        
        // Should have shape [1, 2, 3] after unsqueezing
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3]);
        assert_eq!(*unsqueezed.strides(), vec![6, 3, 1]);
        
        // Verify data access
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 1], &unsqueezed).unwrap(), 2);
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![0, 1, 0], &unsqueezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![0, 1, 1], &unsqueezed).unwrap(), 5);
        assert_eq!(index_tensor(coord![0, 1, 2], &unsqueezed).unwrap(), 6);
    }

    #[test]
    fn test_double_unsqueeze() {
        // Test unsqueezing twice
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed_once = view.unsqueeze();
        assert_eq!(*unsqueezed_once.shape(), vec![1, 3]);
        
        let unsqueezed_twice = unsqueezed_once.unsqueeze();
        assert_eq!(*unsqueezed_twice.shape(), vec![1, 1, 3]);
        assert_eq!(*unsqueezed_twice.strides(), vec![3, 3, 1]);
        
        // Verify data access
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed_twice).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 1], &unsqueezed_twice).unwrap(), 2);
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed_twice).unwrap(), 3);
    }

    #[test]
    fn test_unsqueeze_after_slice() {
        // Test unsqueezing after taking a slice
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        // Take a slice to get first row [1, 2, 3]
        let view = tensor.view();
        let sliced = view.slice(0, 0..0).unwrap();
        assert_eq!(*sliced.shape(), vec![3]);
        
        // Now unsqueeze the slice
        let unsqueezed = sliced.unsqueeze();
        assert_eq!(*unsqueezed.shape(), vec![1, 3]);
        assert_eq!(*unsqueezed.strides(), vec![3, 1]);
        
        // Verify data access
        assert_eq!(index_tensor(coord![0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 1], &unsqueezed).unwrap(), 2);
        assert_eq!(index_tensor(coord![0, 2], &unsqueezed).unwrap(), 3);
    }

    #[test]
    fn test_unsqueeze_after_column_slice() {
        // Test unsqueezing after taking a non-contiguous slice
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_tensor(buf, shape);
        
        // Take a column slice [1, 4]
        let view = tensor.view();
        let sliced = view.slice(1, 0..0).unwrap();
        assert_eq!(*sliced.shape(), vec![2]);
        assert_eq!(*sliced.strides(), vec![3]);  // Non-unit stride
        
        // Now unsqueeze the slice
        let unsqueezed = sliced.unsqueeze();
        assert_eq!(*unsqueezed.shape(), vec![1, 2]);
        assert_eq!(*unsqueezed.strides(), vec![6, 3]);  // Stride should be computed correctly
        
        // Verify data access
        assert_eq!(index_tensor(coord![0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 1], &unsqueezed).unwrap(), 4);
    }

    #[test]
    fn test_unsqueeze_mut_basic() {
        // Test the mutable version of unsqueeze
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut view = tensor.view_mut();
            let unsqueezed = view.unsqueeze_mut().unwrap();
            
            // Should have shape [1, 2, 2]
            assert_eq!(*unsqueezed.shape(), vec![1, 2, 2]);
            assert_eq!(*unsqueezed.strides(), vec![4, 2, 1]);
            
            // Verify we can read through the unsqueezed view
            assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
            assert_eq!(index_tensor(coord![0, 1, 1], &unsqueezed).unwrap(), 4);
        }
    }

    #[test]
    fn test_unsqueeze_mut_with_write() {
        // Test that we can write through an unsqueezed mutable view
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut view = tensor.view_mut();
            let mut unsqueezed = view.unsqueeze_mut().unwrap();
            
            assert_eq!(*unsqueezed.shape(), vec![1, 2, 3]);
            
            // Write through the unsqueezed view
            unsqueezed.set(coord![0, 0, 0], 100).unwrap();
            unsqueezed.set(coord![0, 1, 2], 600).unwrap();
        }
        
        // Verify writes propagated to original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(coord![0, 0], &view).unwrap(), 100);
        assert_eq!(index_tensor(coord![1, 2], &view).unwrap(), 600);
        // Unchanged values
        assert_eq!(index_tensor(coord![0, 1], &view).unwrap(), 2);
        assert_eq!(index_tensor(coord![1, 0], &view).unwrap(), 4);
    }

    #[test]
    fn test_unsqueeze_mut_after_slice() {
        // Test unsqueeze_mut after slicing
        let buf = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let shape = vec![2, 4];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut view = tensor.view_mut();
            let mut sliced = view.slice_mut(0, 1..1).unwrap();  // Get second row
            assert_eq!(*sliced.shape(), vec![4]);
            
            let mut unsqueezed = sliced.unsqueeze_mut().unwrap();
            assert_eq!(*unsqueezed.shape(), vec![1, 4]);
            
            // Write through the unsqueezed slice
            unsqueezed.set(coord![0, 0], 500).unwrap();
            unsqueezed.set(coord![0, 3], 800).unwrap();
        }
        
        // Verify writes propagated to original tensor
        let view = tensor.view();
        assert_eq!(index_tensor(coord![1, 0], &view).unwrap(), 500);
        assert_eq!(index_tensor(coord![1, 3], &view).unwrap(), 800);
        // First row should be unchanged
        assert_eq!(index_tensor(coord![0, 0], &view).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 3], &view).unwrap(), 4);
    }

    #[test]
    fn test_unsqueeze_scalar_mut() {
        // Test unsqueezing a scalar with the mutable version
        let buf = vec![99];
        let shape = vec![];
        let mut tensor = make_tensor(buf, shape);
        
        {
            let mut view = tensor.view_mut();
            let mut unsqueezed = view.unsqueeze_mut().unwrap();
            
            assert_eq!(*unsqueezed.shape(), vec![1]);
            assert_eq!(index_tensor(Idx::At(0), &unsqueezed).unwrap(), 99);
            
            // Write through unsqueezed view
            unsqueezed.set(Idx::At(0), 77).unwrap();
        }
        
        // Verify write propagated
        let view = tensor.view();
        assert_eq!(view.item().unwrap(), 77);
    }

    #[test]
    fn test_unsqueeze_indexing_comprehensive() {
        // Comprehensive test to verify all indices work correctly after unsqueeze
        let buf = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let shape = vec![3, 4];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed = view.unsqueeze();
        
        // Shape should be [1, 3, 4]
        assert_eq!(*unsqueezed.shape(), vec![1, 3, 4]);
        assert_eq!(*unsqueezed.strides(), vec![12, 4, 1]);
        
        // Verify every single element is accessible with correct indexing
        // Original [0, 0] -> Unsqueezed [0, 0, 0]
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 10);
        assert_eq!(index_tensor(coord![0, 0, 1], &unsqueezed).unwrap(), 20);
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed).unwrap(), 30);
        assert_eq!(index_tensor(coord![0, 0, 3], &unsqueezed).unwrap(), 40);
        
        // Original [1, 0] -> Unsqueezed [0, 1, 0]
        assert_eq!(index_tensor(coord![0, 1, 0], &unsqueezed).unwrap(), 50);
        assert_eq!(index_tensor(coord![0, 1, 1], &unsqueezed).unwrap(), 60);
        assert_eq!(index_tensor(coord![0, 1, 2], &unsqueezed).unwrap(), 70);
        assert_eq!(index_tensor(coord![0, 1, 3], &unsqueezed).unwrap(), 80);
        
        // Original [2, 0] -> Unsqueezed [0, 2, 0]
        assert_eq!(index_tensor(coord![0, 2, 0], &unsqueezed).unwrap(), 90);
        assert_eq!(index_tensor(coord![0, 2, 1], &unsqueezed).unwrap(), 100);
        assert_eq!(index_tensor(coord![0, 2, 2], &unsqueezed).unwrap(), 110);
        assert_eq!(index_tensor(coord![0, 2, 3], &unsqueezed).unwrap(), 120);
        
        // Verify that the original tensor still works
        assert_eq!(index_tensor(coord![0, 0], &view).unwrap(), 10);
        assert_eq!(index_tensor(coord![1, 2], &view).unwrap(), 70);
        assert_eq!(index_tensor(coord![2, 3], &view).unwrap(), 120);
    }

    #[test]
    fn test_unsqueeze_3d_indexing() {
        // Test unsqueezing a 3D tensor to verify stride calculation for higher dimensions
        let buf: Vec<i32> = (1..=24).collect();
        let shape = vec![2, 3, 4];
        let tensor = make_tensor(buf, shape);
        
        let view = tensor.view();
        let unsqueezed = view.unsqueeze();
        
        // Shape should be [1, 2, 3, 4]
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3, 4]);
        assert_eq!(*unsqueezed.strides(), vec![24, 12, 4, 1]);
        
        // Test a few key positions
        // Original [0, 0, 0] = 1 -> Unsqueezed [0, 0, 0, 0]
        assert_eq!(index_tensor(coord![0, 0, 0, 0], &unsqueezed).unwrap(), 1);
        
        // Original [0, 0, 3] = 4 -> Unsqueezed [0, 0, 0, 3]
        assert_eq!(index_tensor(coord![0, 0, 0, 3], &unsqueezed).unwrap(), 4);
        
        // Original [0, 2, 3] = 12 -> Unsqueezed [0, 0, 2, 3]
        assert_eq!(index_tensor(coord![0, 0, 2, 3], &unsqueezed).unwrap(), 12);
        
        // Original [1, 0, 0] = 13 -> Unsqueezed [0, 1, 0, 0]
        assert_eq!(index_tensor(coord![0, 1, 0, 0], &unsqueezed).unwrap(), 13);
        
        // Original [1, 2, 3] = 24 -> Unsqueezed [0, 1, 2, 3]
        assert_eq!(index_tensor(coord![0, 1, 2, 3], &unsqueezed).unwrap(), 24);
    }

    // ============================================================================
    // UNSQUEEZE_AT TESTS
    // ============================================================================

    #[test]
    fn test_unsqueeze_at_beginning() {
        // Test unsqueezing at dimension 0
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        let unsqueezed = tensor.unsqueeze_at(0).unwrap();
        
        // Shape should change from [2, 3] to [1, 2, 3]
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3]);
        assert_eq!(*unsqueezed.strides(), vec![6, 3, 1]);
        
        // Test data is still accessible
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![0, 1, 0], &unsqueezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![0, 1, 2], &unsqueezed).unwrap(), 6);
    }

    #[test]
    fn test_unsqueeze_at_middle() {
        // Test unsqueezing at dimension 1 (middle)
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        let unsqueezed = tensor.unsqueeze_at(1).unwrap();
        
        // Shape should change from [2, 3] to [2, 1, 3]
        assert_eq!(*unsqueezed.shape(), vec![2, 1, 3]);
        assert_eq!(*unsqueezed.strides(), vec![3, 3, 1]);
        
        // Test data is still accessible
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![1, 0, 0], &unsqueezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![1, 0, 2], &unsqueezed).unwrap(), 6);
    }

    #[test]
    fn test_unsqueeze_at_end() {
        // Test unsqueezing at dimension 2 (end)
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        let unsqueezed = tensor.unsqueeze_at(2).unwrap();
        
        // Shape should change from [2, 3] to [2, 3, 1]
        assert_eq!(*unsqueezed.shape(), vec![2, 3, 1]);
        assert_eq!(*unsqueezed.strides(), vec![3, 1, 1]);
        
        // Test data is still accessible
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 2, 0], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![1, 0, 0], &unsqueezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![1, 2, 0], &unsqueezed).unwrap(), 6);
    }

    #[test]
    fn test_unsqueeze_at_1d_tensor() {
        // Test unsqueezing a 1D tensor
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![4]);
        
        // Unsqueeze at beginning
        let unsqueezed = tensor.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 4]);
        assert_eq!(*unsqueezed.strides(), vec![4, 1]);
        
        // Unsqueeze at end
        let unsqueezed2 = tensor.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed2.shape(), vec![4, 1]);
        assert_eq!(*unsqueezed2.strides(), vec![1, 1]);
    }

    #[test]
    fn test_unsqueeze_at_3d_tensor() {
        // Test unsqueezing a 3D tensor at various positions
        let tensor = make_tensor(
            (1..=24).collect::<Vec<i32>>(),
            vec![2, 3, 4]
        );
        
        // Unsqueeze at position 0: [2, 3, 4] -> [1, 2, 3, 4]
        let unsqueezed = tensor.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3, 4]);
        assert_eq!(*unsqueezed.strides(), vec![24, 12, 4, 1]);
        
        // Unsqueeze at position 1: [2, 3, 4] -> [2, 1, 3, 4]
        let unsqueezed = tensor.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![2, 1, 3, 4]);
        assert_eq!(*unsqueezed.strides(), vec![12, 12, 4, 1]);
        
        // Unsqueeze at position 2: [2, 3, 4] -> [2, 3, 1, 4]
        let unsqueezed = tensor.unsqueeze_at(2).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![2, 3, 1, 4]);
        assert_eq!(*unsqueezed.strides(), vec![12, 4, 4, 1]);
        
        // Unsqueeze at position 3: [2, 3, 4] -> [2, 3, 4, 1]
        let unsqueezed = tensor.unsqueeze_at(3).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![2, 3, 4, 1]);
        assert_eq!(*unsqueezed.strides(), vec![12, 4, 1, 1]);
    }

    #[test]
    fn test_unsqueeze_at_multiple_sequential() {
        // Test multiple unsqueezes in sequence
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        // [2, 2] -> [1, 2, 2]
        let step1 = tensor.unsqueeze_at(0).unwrap();
        assert_eq!(*step1.shape(), vec![1, 2, 2]);
        
        // [1, 2, 2] -> [1, 1, 2, 2]
        let step2 = step1.unsqueeze_at(1).unwrap();
        assert_eq!(*step2.shape(), vec![1, 1, 2, 2]);
        
        // [1, 1, 2, 2] -> [1, 1, 1, 2, 2]
        let step3 = step2.unsqueeze_at(2).unwrap();
        assert_eq!(*step3.shape(), vec![1, 1, 1, 2, 2]);
        
        // Verify data is still accessible
        assert_eq!(index_tensor(coord![0, 0, 0, 0, 0], &step3).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 0, 1, 1], &step3).unwrap(), 4);
    }

    #[test]
    fn test_unsqueeze_at_multiple_different_positions() {
        // Test unsqueezing at different positions in sequence
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // [2, 3] -> [2, 3, 1] (add at end)
        let step1 = tensor.unsqueeze_at(2).unwrap();
        assert_eq!(*step1.shape(), vec![2, 3, 1]);
        
        // [2, 3, 1] -> [1, 2, 3, 1] (add at beginning)
        let step2 = step1.unsqueeze_at(0).unwrap();
        assert_eq!(*step2.shape(), vec![1, 2, 3, 1]);
        
        // [1, 2, 3, 1] -> [1, 2, 1, 3, 1] (add in middle)
        let step3 = step2.unsqueeze_at(2).unwrap();
        assert_eq!(*step3.shape(), vec![1, 2, 1, 3, 1]);
        
        // Verify data integrity
        assert_eq!(index_tensor(coord![0, 0, 0, 0, 0], &step3).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 0, 0, 2, 0], &step3).unwrap(), 3);
        assert_eq!(index_tensor(coord![0, 1, 0, 0, 0], &step3).unwrap(), 4);
        assert_eq!(index_tensor(coord![0, 1, 0, 2, 0], &step3).unwrap(), 6);
    }

    // ============================================================================
    // UNSQUEEZE_AT_MUT TESTS
    // ============================================================================

    #[test]
    fn test_unsqueeze_at_mut_basic() {
        // Test basic mutable unsqueeze
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        let mut unsqueezed = tensor.unsqueeze_at_mut(0).unwrap();
        
        // Shape should change from [2, 3] to [1, 2, 3]
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3]);
        
        // Modify data through the unsqueezed view
        unsqueezed.set(coord![0, 0, 0], 100).unwrap();
        unsqueezed.set(coord![0, 1, 2], 200).unwrap();
        
        // Drop the mutable view
        drop(unsqueezed);
        
        // Verify changes in original tensor
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 100);
        assert_eq!(tensor.get(coord![1, 2]).unwrap(), 200);
    }

    #[test]
    fn test_unsqueeze_at_mut_middle_dimension() {
        // Test mutable unsqueeze in the middle
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        let mut unsqueezed = tensor.unsqueeze_at_mut(1).unwrap();
        
        // Shape: [2, 3] -> [2, 1, 3]
        assert_eq!(*unsqueezed.shape(), vec![2, 1, 3]);
        
        // Modify multiple elements
        unsqueezed.set(coord![0, 0, 1], 10).unwrap();
        unsqueezed.set(coord![1, 0, 0], 20).unwrap();
        
        drop(unsqueezed);
        
        // Verify changes
        assert_eq!(tensor.get(coord![0, 1]).unwrap(), 10);
        assert_eq!(tensor.get(coord![1, 0]).unwrap(), 20);
    }

    #[test]
    fn test_unsqueeze_at_mut_end_dimension() {
        // Test mutable unsqueeze at the end
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        let mut unsqueezed = tensor.unsqueeze_at_mut(2).unwrap();
        
        // Shape: [2, 2] -> [2, 2, 1]
        assert_eq!(*unsqueezed.shape(), vec![2, 2, 1]);
        
        // Modify through the view
        unsqueezed.set(coord![0, 0, 0], 99).unwrap();
        unsqueezed.set(coord![1, 1, 0], 88).unwrap();
        
        drop(unsqueezed);
        
        // Verify changes
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 99);
        assert_eq!(tensor.get(coord![1, 1]).unwrap(), 88);
    }

    #[test]
    fn test_unsqueeze_at_mut_chained() {
        // Test chaining multiple mutable unsqueezes
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        // Chain: [2, 2] -> [1, 2, 2] -> [1, 1, 2, 2]
        let mut step1 = tensor.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*step1.shape(), vec![1, 2, 2]);
        
        let mut step2 = step1.unsqueeze_at_mut(1).unwrap();
        assert_eq!(*step2.shape(), vec![1, 1, 2, 2]);
        
        // Modify through deeply nested view
        step2.set(coord![0, 0, 0, 0], 111).unwrap();
        step2.set(coord![0, 0, 1, 1], 222).unwrap();
        
        drop(step2);
        drop(step1);
        
        // Verify changes in original tensor
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 111);
        assert_eq!(tensor.get(coord![1, 1]).unwrap(), 222);
    }

    #[test]
    fn test_unsqueeze_at_mut_multiple_dims_in_row() {
        // Test unsqueezing multiple dimensions one after another
        let mut tensor = make_tensor((1..=8).collect::<Vec<i32>>(), vec![2, 4]);
        
        // [2, 4] -> [2, 4, 1]
        let mut step1 = tensor.unsqueeze_at_mut(2).unwrap();
        assert_eq!(*step1.shape(), vec![2, 4, 1]);
        step1.set(coord![0, 0, 0], 100).unwrap();
        drop(step1);
        
        // [2, 4] -> [1, 2, 4]
        let mut step2 = tensor.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*step2.shape(), vec![1, 2, 4]);
        step2.set(coord![0, 1, 3], 200).unwrap();
        drop(step2);
        
        // [2, 4] -> [2, 1, 4]
        let mut step3 = tensor.unsqueeze_at_mut(1).unwrap();
        assert_eq!(*step3.shape(), vec![2, 1, 4]);
        step3.set(coord![1, 0, 2], 300).unwrap();
        drop(step3);
        
        // Verify all changes were applied
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 100);
        assert_eq!(tensor.get(coord![1, 3]).unwrap(), 200);
        assert_eq!(tensor.get(coord![1, 2]).unwrap(), 300);
    }

    #[test]
    fn test_unsqueeze_at_mut_complex_chaining() {
        // Test complex chaining with modifications at each step
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Step 1: [2, 3] -> [2, 3, 1]
        let mut view1 = tensor.unsqueeze_at_mut(2).unwrap();
        assert_eq!(*view1.shape(), vec![2, 3, 1]);
        view1.set(coord![0, 0, 0], 10).unwrap();
        
        // Step 2: [2, 3, 1] -> [1, 2, 3, 1]
        let mut view2 = view1.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*view2.shape(), vec![1, 2, 3, 1]);
        view2.set(coord![0, 0, 1, 0], 20).unwrap();
        
        // Step 3: [1, 2, 3, 1] -> [1, 2, 1, 3, 1]
        let mut view3 = view2.unsqueeze_at_mut(2).unwrap();
        assert_eq!(*view3.shape(), vec![1, 2, 1, 3, 1]);
        view3.set(coord![0, 1, 0, 2, 0], 30).unwrap();
        
        drop(view3);
        drop(view2);
        drop(view1);
        
        // Verify all modifications were applied to original tensor
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 10);
        assert_eq!(tensor.get(coord![0, 1]).unwrap(), 20);
        assert_eq!(tensor.get(coord![1, 2]).unwrap(), 30);
    }

    #[test]
    fn test_unsqueeze_at_mut_alternating_positions() {
        // Test alternating between different unsqueeze positions with modifications
        let mut tensor = make_tensor(vec![1, 2, 3, 4], vec![2, 2]);
        
        // Unsqueeze at end: [2, 2] -> [2, 2, 1]
        let mut v1 = tensor.unsqueeze_at_mut(2).unwrap();
        v1.set(coord![0, 0, 0], 100).unwrap();
        
        // Unsqueeze at beginning: [2, 2, 1] -> [1, 2, 2, 1]
        let mut v2 = v1.unsqueeze_at_mut(0).unwrap();
        v2.set(coord![0, 0, 1, 0], 200).unwrap();
        
        // Unsqueeze in middle: [1, 2, 2, 1] -> [1, 1, 2, 2, 1]
        let mut v3 = v2.unsqueeze_at_mut(1).unwrap();
        v3.set(coord![0, 0, 1, 0, 0], 300).unwrap();
        
        // Unsqueeze at another position: [1, 1, 2, 2, 1] -> [1, 1, 2, 1, 2, 1]
        let mut v4 = v3.unsqueeze_at_mut(3).unwrap();
        assert_eq!(*v4.shape(), vec![1, 1, 2, 1, 2, 1]);
        v4.set(coord![0, 0, 1, 0, 1, 0], 400).unwrap();
        
        drop(v4);
        drop(v3);
        drop(v2);
        drop(v1);
        
        // Verify all modifications
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 100);
        assert_eq!(tensor.get(coord![0, 1]).unwrap(), 200);
        assert_eq!(tensor.get(coord![1, 0]).unwrap(), 300);
        assert_eq!(tensor.get(coord![1, 1]).unwrap(), 400);
    }

    #[test]
    fn test_unsqueeze_at_preserves_data_access() {
        // Test that unsqueezing doesn't affect data access patterns
        let tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![3, 4]);
        
        // Test various unsqueeze positions and verify all data is still accessible
        for dim in 0..=2 {
            let unsqueezed = tensor.unsqueeze_at(dim).unwrap();
            
            // Verify a few sample values are correct regardless of unsqueeze position
            let original_val = tensor.get(coord![0, 0]).unwrap();
            let original_val2 = tensor.get(coord![2, 3]).unwrap();
            
            // Map original coordinates to unsqueezed coordinates
            match dim {
                0 => {
                    assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), original_val);
                    assert_eq!(index_tensor(coord![0, 2, 3], &unsqueezed).unwrap(), original_val2);
                }
                1 => {
                    assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), original_val);
                    assert_eq!(index_tensor(coord![2, 0, 3], &unsqueezed).unwrap(), original_val2);
                }
                2 => {
                    assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), original_val);
                    assert_eq!(index_tensor(coord![2, 3, 0], &unsqueezed).unwrap(), original_val2);
                }
                _ => {}
            }
        }
    }

    // ============================================================================
    // UNSQUEEZE_AT WITH SLICING TESTS
    // ============================================================================

    #[test]
    fn test_unsqueeze_at_after_slice() {
        // Test unsqueezing after slicing
        let tensor = make_tensor((1..=24).collect::<Vec<i32>>(), vec![4, 6]);
        
        // Slice to get middle rows: [4, 6] -> [2, 6]
        let sliced = tensor.slice(0, 1..3).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 6]);
        
        // Unsqueeze the sliced tensor
        let unsqueezed = sliced.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 6]);
        
        // Verify data: sliced[0, 0] should be original[1, 0] = 7
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 7);
        assert_eq!(index_tensor(coord![0, 0, 5], &unsqueezed).unwrap(), 12);
        assert_eq!(index_tensor(coord![0, 1, 0], &unsqueezed).unwrap(), 13);
        assert_eq!(index_tensor(coord![0, 1, 5], &unsqueezed).unwrap(), 18);
    }

    #[test]
    fn test_unsqueeze_at_after_column_slice() {
        // Test unsqueezing after column slicing
        let tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![3, 4]);
        
        // Slice columns: [3, 4] -> [3, 2]
        let sliced = tensor.slice(1, 1..3).unwrap();
        assert_eq!(*sliced.shape(), vec![3, 2]);
        
        // Unsqueeze at the end
        let unsqueezed = sliced.unsqueeze_at(2).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![3, 2, 1]);
        
        // Verify: sliced[0, 0] = original[0, 1] = 2
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 2);
        assert_eq!(index_tensor(coord![0, 1, 0], &unsqueezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![2, 1, 0], &unsqueezed).unwrap(), 11);
    }

    #[test]
    fn test_unsqueeze_at_with_negative_stride() {
        // Test unsqueezing with negative stride (reversed slice)
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![8]);
        
        // Reverse the tensor: [8] with negative stride
        let reversed = tensor.slice(0, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*reversed.shape(), vec![8]);
        assert_eq!(*reversed.strides(), vec![-1]);
        
        // Verify reversed order
        assert_eq!(index_tensor(coord![0], &reversed).unwrap(), 8);
        assert_eq!(index_tensor(coord![7], &reversed).unwrap(), 1);
        
        // Unsqueeze the reversed tensor
        let unsqueezed = reversed.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 8]);
        assert_eq!(*unsqueezed.strides(), vec![-8, -1]);
        
        // Verify data is still in reversed order
        assert_eq!(index_tensor(coord![0, 0], &unsqueezed).unwrap(), 8);
        assert_eq!(index_tensor(coord![0, 7], &unsqueezed).unwrap(), 1);
    }

    #[test]
    fn test_unsqueeze_at_negative_stride_2d() {
        // Test unsqueezing a 2D tensor with negative stride
        let tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![3, 4]);
        
        // Reverse rows: [3, 4] with negative stride on first dim
        let reversed = tensor.slice(0, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*reversed.shape(), vec![3, 4]);
        assert_eq!(*reversed.strides(), vec![-4, 1]);
        
        // Verify: first row should be last row of original
        assert_eq!(index_tensor(coord![0, 0], &reversed).unwrap(), 9);
        assert_eq!(index_tensor(coord![2, 3], &reversed).unwrap(), 4);
        
        // Unsqueeze in the middle
        let unsqueezed = reversed.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![3, 1, 4]);
        assert_eq!(*unsqueezed.strides(), vec![-4, 4, 1]);
        
        // Verify data integrity with negative stride
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 9);
        assert_eq!(index_tensor(coord![0, 0, 3], &unsqueezed).unwrap(), 12);
        assert_eq!(index_tensor(coord![2, 0, 0], &unsqueezed).unwrap(), 1);
    }

    #[test]
    fn test_unsqueeze_at_mut_after_slice() {
        // Test mutable unsqueeze after slicing with modifications
        let mut tensor = make_tensor((1..=20).collect::<Vec<i32>>(), vec![4, 5]);
        
        // Slice rows 1-3: [4, 5] -> [2, 5]
        let mut sliced = tensor.slice_mut(0, 1..3).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 5]);
        
        // Unsqueeze at beginning: [2, 5] -> [1, 2, 5]
        let mut unsqueezed = sliced.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 5]);
        
        // Modify through the unsqueezed view
        unsqueezed.set(coord![0, 0, 0], 100).unwrap();
        unsqueezed.set(coord![0, 1, 4], 200).unwrap();
        
        drop(unsqueezed);
        drop(sliced);
        
        // Verify changes propagated to original tensor
        // sliced[0, 0] = original[1, 0] = position 5
        assert_eq!(tensor.get(coord![1, 0]).unwrap(), 100);
        // sliced[1, 4] = original[2, 4] = position 14
        assert_eq!(tensor.get(coord![2, 4]).unwrap(), 200);
    }

    #[test]
    fn test_unsqueeze_at_mut_negative_stride() {
        // Test mutable unsqueeze with negative stride and modifications
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![6]);
        
        // Reverse the tensor
        let mut reversed = tensor.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*reversed.strides(), vec![-1]);
        assert_eq!(get!(reversed, 0).unwrap(), 6);
        
        // Unsqueeze it
        let mut unsqueezed = reversed.unsqueeze_at_mut(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![6, 1]);
        assert_eq!(*unsqueezed.strides(), vec![-1, 1]);
        
        // Modify: unsqueezed[0, 0] should be original[5] = 6
        unsqueezed.set(coord![0, 0], 100).unwrap();
        // Modify: unsqueezed[5, 0] should be original[0] = 1
        unsqueezed.set(coord![5, 0], 200).unwrap();
        
        drop(unsqueezed);
        drop(reversed);
        
        // Verify modifications
        assert_eq!(tensor.get(coord![5]).unwrap(), 100);
        assert_eq!(tensor.get(coord![0]).unwrap(), 200);
    }

    #[test]
    fn test_unsqueeze_at_mut_negative_stride_2d() {
        // Test mutable unsqueeze on 2D tensor with negative stride
        let mut tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![3, 4]);
        
        // Reverse both dimensions
        let mut reversed_rows = tensor.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        let mut reversed_both = reversed_rows.slice_mut(1, Slice::from(..).step(-1)).unwrap();
        assert_eq!(*reversed_both.strides(), vec![-4, -1]);
        
        // Unsqueeze at position 1
        let mut unsqueezed = reversed_both.unsqueeze_at_mut(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![3, 1, 4]);
        
        // Modify: unsqueezed[0, 0, 0] should be original[2, 3] = 12
        unsqueezed.set(coord![0, 0, 0], 999).unwrap();
        // Modify: unsqueezed[2, 0, 3] should be original[0, 0] = 1
        unsqueezed.set(coord![2, 0, 3], 888).unwrap();
        
        drop(unsqueezed);
        drop(reversed_both);
        drop(reversed_rows);
        
        // Verify modifications to original
        assert_eq!(tensor.get(coord![2, 3]).unwrap(), 999);
        assert_eq!(tensor.get(coord![0, 0]).unwrap(), 888);
    }

    #[test]
    fn test_unsqueeze_at_mut_complex_slice_pattern() {
        // Test with complex slicing pattern: skip elements, reverse, then unsqueeze
        let mut tensor = make_tensor((0..20).collect::<Vec<i32>>(), vec![4, 5]);
        
        // Take every other row in reverse: start at 3, go backwards with step -2
        let mut sliced = tensor.slice_mut(0, Slice::from(3..).step(-2)).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 5]);
        
        // Verify slice content: should be rows 3 and 1 (reversed order)
        assert_eq!(index_tensor(coord![0, 0], &sliced).unwrap(), 15); // row 3, col 0
        assert_eq!(index_tensor(coord![1, 0], &sliced).unwrap(), 5);  // row 1, col 0
        
        // Unsqueeze at end: [2, 5] -> [2, 5, 1]
        let mut unsqueezed = sliced.unsqueeze_at_mut(2).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![2, 5, 1]);
        
        // Modify through unsqueezed view
        unsqueezed.set(coord![0, 0, 0], 300).unwrap(); // should modify original[3, 0]
        unsqueezed.set(coord![1, 4, 0], 400).unwrap(); // should modify original[1, 4]
        
        drop(unsqueezed);
        drop(sliced);
        
        // Verify modifications
        assert_eq!(tensor.get(coord![3, 0]).unwrap(), 300);
        assert_eq!(tensor.get(coord![1, 4]).unwrap(), 400);
    }

    #[test]
    fn test_unsqueeze_at_mut_multiple_slices_then_unsqueeze() {
        // Test multiple slicing operations followed by unsqueeze and modification
        let mut tensor = make_tensor((1..=60).collect::<Vec<i32>>(), vec![5, 12]);
        
        // Slice rows: [5, 12] -> [3, 12]
        let mut step1 = tensor.slice_mut(0, 1..4).unwrap();
        
        // Slice columns: [3, 12] -> [3, 5]
        let mut step2 = step1.slice_mut(1, 2..7).unwrap();
        assert_eq!(*step2.shape(), vec![3, 5]);
        
        // Unsqueeze at position 1: [3, 5] -> [3, 1, 5]
        let mut unsqueezed = step2.unsqueeze_at_mut(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![3, 1, 5]);
        
        // Modify values
        unsqueezed.set(coord![0, 0, 0], 500).unwrap(); // original[1, 2]
        unsqueezed.set(coord![2, 0, 4], 600).unwrap(); // original[3, 6]
        
        drop(unsqueezed);
        drop(step2);
        drop(step1);
        
        // Verify: original[1, 2] = 1*12 + 2 = 14 + 1 = 15, now should be 500
        assert_eq!(tensor.get(coord![1, 2]).unwrap(), 500);
        // Verify: original[3, 6] = 3*12 + 6 = 36 + 1 + 6 = 43, now should be 600
        assert_eq!(tensor.get(coord![3, 6]).unwrap(), 600);
    }

    #[test]
    fn test_unsqueeze_at_mut_chained_with_negative_stride() {
        // Test chaining unsqueezes with negative strides
        let mut tensor = make_tensor((1..=24).collect::<Vec<i32>>(), vec![4, 6]);
        
        // Reverse rows: [4, 6] with negative stride
        let mut reversed = tensor.slice_mut(0, Slice::from(..).step(-1)).unwrap();
        
        // Unsqueeze at beginning: [4, 6] -> [1, 4, 6]
        let mut unsqueezed1 = reversed.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*unsqueezed1.shape(), vec![1, 4, 6]);
        
        // Unsqueeze at end: [1, 4, 6] -> [1, 4, 6, 1]
        let mut unsqueezed2 = unsqueezed1.unsqueeze_at_mut(3).unwrap();
        assert_eq!(*unsqueezed2.shape(), vec![1, 4, 6, 1]);
        
        // Modify: [0, 0, 0, 0] should map to reversed[0, 0] = original[3, 0] = 19
        unsqueezed2.set(coord![0, 0, 0, 0], 700).unwrap();
        // Modify: [0, 3, 5, 0] should map to reversed[3, 5] = original[0, 5] = 6
        unsqueezed2.set(coord![0, 3, 5, 0], 800).unwrap();
        
        drop(unsqueezed2);
        drop(unsqueezed1);
        drop(reversed);
        
        // Verify modifications
        assert_eq!(tensor.get(coord![3, 0]).unwrap(), 700);
        assert_eq!(tensor.get(coord![0, 5]).unwrap(), 800);
    }

    #[test]
    fn test_unsqueeze_at_preserves_strides_after_slice() {
        // Verify that stride calculations are correct after slicing and unsqueezing
        let tensor = make_tensor((1..=30).collect::<Vec<i32>>(), vec![5, 6]);
        
        // Slice with step: take every other column
        let sliced = tensor.slice(1, Slice::from(..).step(2)).unwrap();
        assert_eq!(*sliced.shape(), vec![5, 3]);
        assert_eq!(*sliced.strides(), vec![6, 2]); // stride doubled for columns
        
        // Unsqueeze at middle
        let unsqueezed = sliced.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![5, 1, 3]);
        // Stride should be [6, 6, 2] - middle stride inherits from outer dimension
        assert_eq!(*unsqueezed.strides(), vec![6, 6, 2]);
        
        // Verify data access still works correctly
        // sliced[0, 0] = original[0, 0] = 1
        assert_eq!(index_tensor(coord![0, 0, 0], &unsqueezed).unwrap(), 1);
        // sliced[0, 2] = original[0, 4] = 5
        assert_eq!(index_tensor(coord![0, 0, 2], &unsqueezed).unwrap(), 5);
        // sliced[4, 2] = original[4, 4] = 29
        assert_eq!(index_tensor(coord![4, 0, 2], &unsqueezed).unwrap(), 29);
    }

    // ============================================================================
    // UNSQUEEZE_AT ERROR CASES
    // ============================================================================

    #[test]
    #[should_panic]
    fn test_unsqueeze_at_out_of_bounds_too_large() {
        // Test that unsqueezing beyond valid dimension range fails
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Valid dimensions for a [2, 3] tensor are 0, 1, 2 (can insert at beginning, middle, or end)
        // Dimension 4 is out of bounds
        tensor.unsqueeze_at(4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_unsqueeze_at_out_of_bounds_1d() {
        // Test out of bounds on 1D tensor
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![4]);
        
        // Valid dimensions for a [4] tensor are 0, 1
        // Dimension 3 is out of bounds
        tensor.unsqueeze_at(3).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_unsqueeze_at_mut_out_of_bounds() {
        // Test that mutable unsqueeze fails with out of bounds dimension
        let mut tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![3, 4]);
        
        // Valid dimensions are 0, 1, 2 for a [3, 4] tensor
        // Dimension 5 is way out of bounds
        tensor.unsqueeze_at_mut(5).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_unsqueeze_at_out_of_bounds_after_slice() {
        // Test that unsqueeze fails even after slicing when dimension is invalid
        let tensor = make_tensor((1..=24).collect::<Vec<i32>>(), vec![4, 6]);
        
        // Slice to [2, 6]
        let sliced = tensor.slice(0, 1..3).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 6]);
        
        // Valid dimensions for [2, 6] are 0, 1, 2
        // Dimension 4 is out of bounds
        sliced.unsqueeze_at(4).unwrap();
    }

    // ============================================================================
    // SQUEEZE TESTS
    // ============================================================================

    #[test]
    fn test_squeeze_basic_2d() {
        // Test squeezing a single dimension from [1, 3] -> [3]
        let tensor = make_tensor(vec![1, 2, 3], vec![1, 3]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![3]);
        assert_eq!(*squeezed.strides(), vec![1]);
        assert!(!squeezed.is_scalar());
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &squeezed).unwrap(), 3);
    }

    #[test]
    fn test_squeeze_basic_3d() {
        // Test squeezing from [1, 2, 1, 3] -> [2, 3]
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 2, 1, 3]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 2], &squeezed).unwrap(), 3);
        assert_eq!(index_tensor(coord![1, 0], &squeezed).unwrap(), 4);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_all_dimensions() {
        // Test squeezing tensor with all size-1 dims [1, 1, 1] -> []
        let tensor = make_tensor(vec![42], vec![1, 1, 1]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![]);
        assert_eq!(*squeezed.strides(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &squeezed).unwrap(), 42);
    }

    #[test]
    fn test_squeeze_no_size_one_dims() {
        // Test that squeezing a tensor with no size-1 dims doesn't change it
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_at_specific_dim() {
        // Test squeezing a specific dimension
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3]);
        let squeezed = tensor.squeeze_at(0).unwrap();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_at_middle_dim() {
        // Test squeezing middle dimension [2, 1, 3] -> [2, 3]
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 1, 3]);
        let squeezed = tensor.squeeze_at(1).unwrap();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_at_end_dim() {
        // Test squeezing end dimension [2, 3, 1] -> [2, 3]
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3, 1]);
        let squeezed = tensor.squeeze_at(2).unwrap();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_after_slice() {
        // Test squeezing after slicing
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Slice to get a single row: shape [3]
        let sliced = tensor.slice(0, 0..0).unwrap();
        assert_eq!(*sliced.shape(), vec![3]);
        
        // Unsqueeze to [1, 3], then squeeze back to [3]
        let unsqueezed = sliced.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 3]);
        
        let squeezed = unsqueezed.squeeze();
        assert_eq!(*squeezed.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(2), &squeezed).unwrap(), 3);
    }

    #[test]
    fn test_squeeze_after_column_slice() {
        // Test squeezing after column slicing
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Slice to get a single column: shape [2]
        let sliced = tensor.slice(1, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &sliced).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &sliced).unwrap(), 5);
        
        // Unsqueeze to [2, 1], then squeeze
        let unsqueezed = sliced.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![2, 1]);
        
        let squeezed = unsqueezed.squeeze();
        assert_eq!(*squeezed.shape(), vec![2]);
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 5);
    }

    #[test]
    fn test_squeeze_after_3d_slice() {
        // Test squeezing after 3D slicing
        let tensor = make_tensor(
            (1..=24).collect::<Vec<i32>>(), 
            vec![2, 3, 4]
        );
        
        // Slice along first dimension to get [3, 4]
        let sliced = tensor.slice(0, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![3, 4]);
        
        // Unsqueeze to [1, 3, 4]
        let unsqueezed = sliced.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 3, 4]);
        
        // Squeeze back
        let squeezed = unsqueezed.squeeze();
        assert_eq!(*squeezed.shape(), vec![3, 4]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 13);
        assert_eq!(index_tensor(coord![2, 3], &squeezed).unwrap(), 24);
    }

    #[test]
    fn test_squeeze_mut_basic() {
        // Test mutable squeeze
        let mut tensor = make_tensor(vec![1, 2, 3], vec![1, 3]);
        let mut squeezed = tensor.squeeze_mut();
        
        assert_eq!(*squeezed.shape(), vec![3]);
        assert_eq!(*squeezed.strides(), vec![1]);
        
        // Modify through squeezed view
        squeezed.set(Idx::At(1), 20).unwrap();
        
        // Check modification persisted
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 20);
    }

    #[test]
    fn test_squeeze_mut_after_modifications() {
        // Test that modifications after squeezing work correctly
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3]);
        
        let mut squeezed = tensor.squeeze_mut();
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        
        // Modify several elements
        squeezed.set(coord![0, 0], 10).unwrap();
        squeezed.set(coord![0, 2], 30).unwrap();
        squeezed.set(coord![1, 1], 50).unwrap();
        
        // Verify modifications
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 10);
        assert_eq!(index_tensor(coord![0, 2], &squeezed).unwrap(), 30);
        assert_eq!(index_tensor(coord![1, 1], &squeezed).unwrap(), 50);
        
        // Original values unchanged where not modified
        assert_eq!(index_tensor(coord![0, 1], &squeezed).unwrap(), 2);
        assert_eq!(index_tensor(coord![1, 0], &squeezed).unwrap(), 4);
    }

    #[test]
    fn test_squeeze_at_mut_basic() {
        // Test mutable squeeze_at
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 1, 3]);
        let mut squeezed = tensor.squeeze_at_mut(1).unwrap();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        
        // Modify through squeezed view
        squeezed.set(coord![0, 0], 10).unwrap();
        squeezed.set(coord![1, 2], 60).unwrap();
        
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 10);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 60);
    }

    #[test]
    fn test_squeeze_mut_after_slice() {
        // Test mutable squeeze after slicing
        let mut tensor = make_tensor(
            (1..=12).collect::<Vec<i32>>(), 
            vec![3, 4]
        );
        
        // Slice to get middle row: [4]
        let mut sliced = tensor.slice_mut(0, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        
        // Unsqueeze to [1, 4]
        let mut unsqueezed = sliced.unsqueeze_at_mut(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 4]);
        
        // Squeeze back to [4]
        let mut squeezed = unsqueezed.squeeze_mut();
        assert_eq!(*squeezed.shape(), vec![4]);
        
        // Modify through squeezed view
        squeezed.set(Idx::At(0), 50).unwrap();
        squeezed.set(Idx::At(3), 80).unwrap();
        
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 50);
        assert_eq!(index_tensor(Idx::At(3), &squeezed).unwrap(), 80);
    }

    #[test]
    fn test_squeeze_mut_after_column_slice() {
        // Test mutable squeeze after column slicing
        let mut tensor = make_tensor(
            vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ], 
            vec![3, 3]
        );
        
        // Slice to get middle column: [3]
        let mut sliced = tensor.slice_mut(1, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &sliced).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &sliced).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &sliced).unwrap(), 8);
        
        // Unsqueeze and squeeze
        let mut unsqueezed = sliced.unsqueeze_at_mut(1).unwrap();
        let mut squeezed = unsqueezed.squeeze_mut();
        
        // Modify
        squeezed.set(Idx::At(0), 20).unwrap();
        squeezed.set(Idx::At(2), 80).unwrap();
        
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 20);
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &squeezed).unwrap(), 80);
    }

    #[test]
    fn test_squeeze_chained_with_slices() {
        // Test complex chaining: slice -> unsqueeze -> slice -> unsqueeze -> squeeze
        let tensor = make_tensor(
            (1..=24).collect::<Vec<i32>>(), 
            vec![2, 3, 4]
        );
        
        // Slice along dim 0: [3, 4]
        let slice1 = tensor.slice(0, 0..0).unwrap();
        assert_eq!(*slice1.shape(), vec![3, 4]);
        
        // Unsqueeze at dim 1: [3, 1, 4]
        let unsqueezed1 = slice1.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed1.shape(), vec![3, 1, 4]);
        
        // Slice along dim 2: [3, 1]
        let slice2 = unsqueezed1.slice(2, 2..2).unwrap();
        assert_eq!(*slice2.shape(), vec![3, 1]);
        
        // Unsqueeze at dim 0: [1, 3, 1]
        let unsqueezed2 = slice2.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed2.shape(), vec![1, 3, 1]);
        
        // Squeeze all: [3]
        let squeezed = unsqueezed2.squeeze();
        assert_eq!(*squeezed.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 7);
        assert_eq!(index_tensor(Idx::At(2), &squeezed).unwrap(), 11);
    }

    #[test]
    fn test_squeeze_preserves_data_after_negative_stride() {
        // Test that squeeze works correctly with negative strides
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![6]);
        
        // Reverse with negative stride
        let reversed = tensor.slice(0, Slice::full().step(-1)).unwrap();
        assert_eq!(index_tensor(Idx::At(0), &reversed).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(5), &reversed).unwrap(), 1);
        
        // Unsqueeze to [1, 6] with negative stride
        let unsqueezed = reversed.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 6]);
        
        // Squeeze back to [6]
        let squeezed = unsqueezed.squeeze();
        assert_eq!(*squeezed.shape(), vec![6]);
        
        // Verify data is still reversed
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 6);
        assert_eq!(index_tensor(Idx::At(5), &squeezed).unwrap(), 1);
    }

    #[test]
    fn test_squeeze_multiple_dims_2d() {
        // Test squeezing multiple dimensions at once
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 6, 1]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![6]);
        assert_eq!(*squeezed.strides(), vec![1]);
        for i in 0..6 {
            assert_eq!(index_tensor(Idx::At(i), &squeezed).unwrap(), (i + 1) as i32);
        }
    }

    #[test]
    fn test_squeeze_multiple_dims_4d() {
        // Test squeezing from [1, 2, 1, 3, 1] -> [2, 3]
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 2, 1, 3, 1]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    #[test]
    fn test_squeeze_then_modify_mut() {
        // Test that modifications work correctly after squeeze
        let mut tensor = make_tensor(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![1, 2, 4, 1]
        );
        
        let mut squeezed = tensor.squeeze_mut();
        assert_eq!(*squeezed.shape(), vec![2, 4]);
        
        // Modify all elements in first row
        for i in 0..4 {
            squeezed.set(coord![0, i], (i * 10) as i32).unwrap();
        }
        
        // Verify modifications
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 0);
        assert_eq!(index_tensor(coord![0, 1], &squeezed).unwrap(), 10);
        assert_eq!(index_tensor(coord![0, 2], &squeezed).unwrap(), 20);
        assert_eq!(index_tensor(coord![0, 3], &squeezed).unwrap(), 30);
        
        // Second row should be unchanged
        assert_eq!(index_tensor(coord![1, 0], &squeezed).unwrap(), 5);
        assert_eq!(index_tensor(coord![1, 3], &squeezed).unwrap(), 8);
    }

    #[test]
    fn test_squeeze_scalar_to_tensor() {
        // Test squeezing a scalar (no-op)
        let tensor = make_tensor(vec![42], vec![]);
        let squeezed = tensor.squeeze();
        
        assert_eq!(*squeezed.shape(), vec![]);
        assert_eq!(index_tensor(Idx::Item, &squeezed).unwrap(), 42);
    }

    #[test]
    fn test_squeeze_at_then_slice() {
        // Test slicing after squeeze_at
        let tensor = make_tensor((1..=12).collect::<Vec<i32>>(), vec![1, 3, 4]);
        let squeezed = tensor.squeeze_at(0).unwrap();
        
        assert_eq!(*squeezed.shape(), vec![3, 4]);
        
        // Now slice the squeezed tensor
        let sliced = squeezed.slice(0, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        assert_eq!(index_tensor(Idx::At(0), &sliced).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(3), &sliced).unwrap(), 8);
    }

    #[test]
    fn test_double_squeeze_at() {
        // Test applying squeeze_at multiple times
        let tensor = make_tensor(vec![1, 2, 3, 4], vec![1, 2, 1, 2]);
        
        let squeezed1 = tensor.squeeze_at(0).unwrap();
        assert_eq!(*squeezed1.shape(), vec![2, 1, 2]);
        
        let squeezed2 = squeezed1.squeeze_at(1).unwrap();
        assert_eq!(*squeezed2.shape(), vec![2, 2]);
        
        assert_eq!(index_tensor(coord![0, 0], &squeezed2).unwrap(), 1);
        assert_eq!(index_tensor(coord![0, 1], &squeezed2).unwrap(), 2);
        assert_eq!(index_tensor(coord![1, 0], &squeezed2).unwrap(), 3);
        assert_eq!(index_tensor(coord![1, 1], &squeezed2).unwrap(), 4);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        // Test that squeeze and unsqueeze are inverses
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        
        // Unsqueeze at various positions
        let unsqueezed = tensor.unsqueeze_at(0).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![1, 2, 3]);
        
        // Squeeze back
        let squeezed = unsqueezed.squeeze_at(0).unwrap();
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert_eq!(*squeezed.strides(), vec![3, 1]);
        
        // Verify data integrity
        assert_eq!(index_tensor(coord![0, 0], &squeezed).unwrap(), 1);
        assert_eq!(index_tensor(coord![1, 2], &squeezed).unwrap(), 6);
    }

    // ============================================================================
    // SQUEEZE ERROR CASES
    // ============================================================================

    #[test]
    #[should_panic]
    fn test_squeeze_at_non_singleton_dim() {
        // Test that squeeze_at fails when dimension size is not 1
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        tensor.squeeze_at(0).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_squeeze_at_invalid_dim() {
        // Test that squeeze_at fails with invalid dimension
        let tensor = make_tensor(vec![1, 2, 3], vec![1, 3]);
        tensor.squeeze_at(5).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_squeeze_at_mut_non_singleton_dim() {
        // Test that squeeze_at_mut fails when dimension size is not 1
        let mut tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        tensor.squeeze_at_mut(1).unwrap();
    }

    #[test]
    fn test_squeeze_preserves_contiguity() {
        // Test that squeeze preserves contiguous layout
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3]);
        assert!(tensor.is_contiguous());
        
        let squeezed = tensor.squeeze();
        assert_eq!(*squeezed.shape(), vec![2, 3]);
        assert!(squeezed.is_contiguous());
    }

    #[test]
    fn test_squeeze_with_non_contiguous_slice() {
        // Test squeezing a non-contiguous slice
        let tensor = make_tensor(
            vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            ], 
            vec![3, 3]
        );
        
        // Get column slice (non-contiguous)
        let col_slice = tensor.slice(1, 1..1).unwrap();
        assert_eq!(*col_slice.shape(), vec![3]);
        assert!(!col_slice.is_contiguous());
        
        // Unsqueeze it
        let unsqueezed = col_slice.unsqueeze_at(1).unwrap();
        assert_eq!(*unsqueezed.shape(), vec![3, 1]);
        
        // Squeeze it back
        let squeezed = unsqueezed.squeeze();
        assert_eq!(*squeezed.shape(), vec![3]);
        
        // Verify data
        assert_eq!(index_tensor(Idx::At(0), &squeezed).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(1), &squeezed).unwrap(), 5);
        assert_eq!(index_tensor(Idx::At(2), &squeezed).unwrap(), 8);
    }

    #[test]
    fn test_pad_1d() {
        let tensor = make_tensor(vec![1, 2, 3], vec![3]);
        let padded = tensor.pad(vec![2], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![7]);
        assert_eq!(index_tensor(Idx::At(0), &padded).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(2), &padded).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(4), &padded).unwrap(), 3);
        assert_eq!(index_tensor(Idx::At(6), &padded).unwrap(), 0);
    }

    #[test]
    fn test_pad_2d() {
        let tensor = make_tensor(
            vec![
                1, 2, 
                3, 4
            ], 
            vec![2, 2]
        );
        let padded = tensor.pad(vec![1, 2], &PaddingType::Zeros).unwrap();
        let expected = make_tensor(
            vec![
                0, 0, 0, 0, 0, 0,
                0, 0, 1, 2, 0, 0,
                0, 0, 3, 4, 0, 0,
                0, 0, 0, 0, 0, 0
            ], 
            vec![4, 6]
        );
        assert_eq!(padded, expected);
    }

    #[test]
    fn test_pad_3d() {
        // Test 3D tensor padding: [2, 2, 2] -> [4, 4, 4]
        let tensor = make_tensor(
            vec![
                1, 2,  // depth 0, row 0
                3, 4,  // depth 0, row 1
                5, 6,  // depth 1, row 0
                7, 8   // depth 1, row 1
            ], 
            vec![2, 2, 2]
        );
        let padded = tensor.pad(vec![1, 1, 1], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![4, 4, 4]);
        
        // Check corner is zero
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &padded).unwrap(), 0);
        // Check original data is preserved in the middle
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &padded).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 2]), &padded).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2, 2]), &padded).unwrap(), 8);
    }

    #[test]
    fn test_pad_4d() {
        // Test 4D tensor (batch, channel, height, width): [1, 2, 2, 2] -> [3, 4, 4, 4]
        let tensor = make_tensor(
            vec![
                1, 2,  3, 4,   // batch 0, channel 0
                5, 6,  7, 8    // batch 0, channel 1
            ], 
            vec![1, 2, 2, 2]
        );
        let padded = tensor.pad(vec![1, 1, 1, 1], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![3, 4, 4, 4]);
        
        // Check that padding is applied
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0, 0]), &padded).unwrap(), 0);
        // Check original data
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1, 1]), &padded).unwrap(), 1);
    }

    #[test]
    fn test_pad_asymmetric() {
        // Test asymmetric padding: different padding on different dimensions
        let tensor = make_tensor(
            vec![1, 2, 3, 4, 5, 6], 
            vec![2, 3]
        );
        let padded = tensor.pad(vec![3, 1], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![8, 5]); // [2+3*2, 3+1*2]
        
        // Check padding zones
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &padded).unwrap(), 0);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 4]), &padded).unwrap(), 0);
        // Check original data in the middle
        assert_eq!(index_tensor(Idx::Coord(vec![3, 1]), &padded).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![4, 3]), &padded).unwrap(), 6);
    }

    #[test]
    fn test_pad_zero_padding_some_dims() {
        // Test with zero padding on some dimensions
        let tensor = make_tensor(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2]
        );
        let padded = tensor.pad(vec![0, 2, 1], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![2, 6, 4]); // [2+0, 2+2*2, 2+1*2]
        
        // First dimension unchanged, original [0,0,0] -> [0, 2, 1]
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2, 1]), &padded).unwrap(), 1);
        // Check padding in other dimensions
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 1]), &padded).unwrap(), 0);
        // Original [1,1,1] has value 8, becomes [1, 3, 2] after padding
        assert_eq!(index_tensor(Idx::Coord(vec![1, 3, 2]), &padded).unwrap(), 8);
    }

    #[test]
    fn test_pad_large_padding() {
        // Test with large padding relative to tensor size
        let tensor = make_tensor(vec![42], vec![1]);
        let padded = tensor.pad(vec![5], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![11]); // 1 + 5*2
        
        // Check middle element
        assert_eq!(index_tensor(Idx::At(5), &padded).unwrap(), 42);
        // Check padding
        assert_eq!(index_tensor(Idx::At(0), &padded).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(10), &padded).unwrap(), 0);
    }

    #[test]
    fn test_pad_rectangular_2d() {
        // Test non-square 2D tensor
        let tensor = make_tensor(
            vec![1, 2, 3, 4, 5, 6], 
            vec![3, 2]
        );
        let padded = tensor.pad(vec![2, 3], &PaddingType::Zeros).unwrap();
        assert_eq!(*padded.shape(), vec![7, 8]); // [3+2*2, 2+3*2]
        
        let expected = make_tensor(
            vec![
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 2, 0, 0, 0,
                0, 0, 0, 3, 4, 0, 0, 0,
                0, 0, 0, 5, 6, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0
            ], 
            vec![7, 8]
        );
        assert_eq!(padded, expected);
    }

    #[test]
    fn test_pad_sliced_1d_row() {
        // Test padding a 1D slice from a 2D tensor
        // Compare: slice then pad vs. creating equivalent tensor and padding
        let tensor = make_tensor(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
            ],
            vec![3, 4]
        );
        
        // Slice to get first row: [1, 2, 3, 4]
        let view = tensor.view();
        let sliced = view.slice(0, 0..0).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![2], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor and pad it
        let full = make_tensor(vec![1, 2, 3, 4], vec![4]);
        let padded_full = full.pad(vec![2], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![8]);
        assert_eq!(index_tensor(Idx::At(0), &padded_slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(2), &padded_slice).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(5), &padded_slice).unwrap(), 4);
        assert_eq!(index_tensor(Idx::At(7), &padded_slice).unwrap(), 0);
    }

    #[test]
    fn test_pad_sliced_1d_column() {
        // Test padding a non-contiguous 1D column slice
        let tensor = make_tensor(
            vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
            ],
            vec![4, 3]
        );
        
        // Slice to get second column: [2, 5, 8, 11]
        let view = tensor.view();
        let sliced = view.slice(1, 1..1).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        assert_eq!(*sliced.strides(), vec![3]); // Non-contiguous
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![1], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor and pad it
        let full = make_tensor(vec![2, 5, 8, 11], vec![4]);
        let padded_full = full.pad(vec![1], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &padded_slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &padded_slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(4), &padded_slice).unwrap(), 11);
        assert_eq!(index_tensor(Idx::At(5), &padded_slice).unwrap(), 0);
    }

    #[test]
    fn test_pad_sliced_2d_submatrix() {
        // Test padding a 2D slice from a larger matrix
        let tensor = make_tensor(
            vec![
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
                11, 12, 13, 14, 15,
                16, 17, 18, 19, 20
            ],
            vec![4, 5]
        );
        
        // Slice to get center 2x3 submatrix: [[7,8,9], [12,13,14]]
        let view = tensor.view();
        let sliced_rows = view.slice(0, 1..3).unwrap();
        let sliced = sliced_rows.slice(1, 1..4).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3]);
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![1, 2], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor and pad it
        let full = make_tensor(vec![7, 8, 9, 12, 13, 14], vec![2, 3]);
        let padded_full = full.pad(vec![1, 2], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![4, 7]);
    }

    #[test]
    fn test_pad_sliced_3d_subcube() {
        // Test padding a 3D slice
        let tensor = make_tensor(
            vec![
                // Depth 0
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                // Depth 1
                17, 18, 19, 20,
                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 30, 31, 32,
                // Depth 2
                33, 34, 35, 36,
                37, 38, 39, 40,
                41, 42, 43, 44,
                45, 46, 47, 48
            ],
            vec![3, 4, 4]
        );
        
        // Slice to get a 2x2x2 subcube from the middle
        // Depth [1..3], Rows [1..3], Cols [1..3]
        let view = tensor.view();
        let d_slice = view.slice(0, 1..3).unwrap();
        let r_slice = d_slice.slice(1, 1..3).unwrap();
        let sliced = r_slice.slice(2, 1..3).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 2, 2]);
        
        // Extract the values: depth1[1,1], depth1[1,2], depth1[2,1], depth1[2,2],
        //                     depth2[1,1], depth2[1,2], depth2[2,1], depth2[2,2]
        // = [22, 23, 26, 27, 38, 39, 42, 43]
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![1, 1, 1], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor and pad it
        let full = make_tensor(vec![22, 23, 26, 27, 38, 39, 42, 43], vec![2, 2, 2]);
        let padded_full = full.pad(vec![1, 1, 1], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![4, 4, 4]);
        
        // Verify some values
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0, 0]), &padded_slice).unwrap(), 0); // padding
        assert_eq!(index_tensor(Idx::Coord(vec![1, 1, 1]), &padded_slice).unwrap(), 22); // original data
        assert_eq!(index_tensor(Idx::Coord(vec![2, 2, 2]), &padded_slice).unwrap(), 43); // original data
    }

    #[test]
    fn test_pad_reversed_slice() {
        // Test padding a slice with negative stride (reversed)
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![10]);
        
        // Slice with reverse: indices [7, 6, 5, 4, 3]
        let view = tensor.view();
        let sliced = view.slice(0, 7..2).unwrap();
        assert_eq!(*sliced.shape(), vec![5]);
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![2], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor (reversed order) and pad it
        let full = make_tensor(vec![8, 7, 6, 5, 4], vec![5]);
        let padded_full = full.pad(vec![2], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![9]);
        assert_eq!(index_tensor(Idx::At(2), &padded_slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(6), &padded_slice).unwrap(), 4);
    }

    #[test]
    fn test_pad_strided_slice() {
        // Test padding a slice with step > 1
        let tensor = make_tensor(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![10]);
        
        // Slice with step 2: indices [1, 3, 5, 7] = [2, 4, 6, 8]
        let view = tensor.view();
        let sliced = view.slice(0, Slice::from(1..9).step(2)).unwrap();
        assert_eq!(*sliced.shape(), vec![4]);
        
        // Pad the slice
        let padded_slice = sliced.pad(vec![1], &PaddingType::Zeros).unwrap();
        
        // Create equivalent full tensor and pad it
        let full = make_tensor(vec![2, 4, 6, 8], vec![4]);
        let padded_full = full.pad(vec![1], &PaddingType::Zeros).unwrap();
        
        // They should be equal
        assert_eq!(padded_slice, padded_full);
        assert_eq!(*padded_slice.shape(), vec![6]);
        assert_eq!(index_tensor(Idx::At(0), &padded_slice).unwrap(), 0);
        assert_eq!(index_tensor(Idx::At(1), &padded_slice).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(4), &padded_slice).unwrap(), 8);
        assert_eq!(index_tensor(Idx::At(5), &padded_slice).unwrap(), 0);
    }
}


