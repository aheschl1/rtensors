#![allow(non_snake_case)]

#[cfg(test)]
#[cfg(feature = "remote")]
mod tests {
    use std::sync::Once;
    use std::thread;
    use crate::{
        backend::{Backend, remote::{client::RemoteBackend, server::RemoteServer}},
        core::{
            idx::Idx, 
            primitives::{RemoteTensor, TensorBase}, 
            tensor::{AsView, AsViewMut, TensorAccess, TensorAccessMut, TensorError}, 
            value::TensorValue, 
            MetaTensor, 
            MetaTensorView, 
            Shape, 
            Slice,
            Tensor,
            value::types,
        },
        ops::reduction::TotalReductionOp,
        ops::{linalg::MatMul, reduction::ReductionOpTypes},
    };
    
    const SERVER_IP: &str = "127.0.0.1";
    const SERVER_PORT: u16 = 7879;
    
    static INIT: Once = Once::new();
    
    fn setup_server() {
        INIT.call_once(|| {
            crate::backend::remote::server::ensure_test_server(SERVER_IP.parse().unwrap(), SERVER_PORT);
        });
    }
    
    /// One connection shared by every test tensor. Buffer ids are scoped to a connection,
    /// so tensors used together in one op must come from the same backend.
    fn shared_backend() -> RemoteBackend {
        static BACKEND: std::sync::OnceLock<RemoteBackend> = std::sync::OnceLock::new();
        BACKEND.get_or_init(|| {
            setup_server();
            let mut backend = RemoteBackend::new_with_address(SERVER_IP.parse().unwrap(), SERVER_PORT).unwrap();
            backend.connect().unwrap();
            backend
        }).clone()
    }

    fn make_remote_tensor<T: TensorValue>(buf: Vec<T>, shape: impl Into<Shape>) -> Result<RemoteTensor<T>, TensorError> {
        let backend = shared_backend();
        
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
        
        let buffer = backend.alloc_from_slice(buf.into())?;
        let stride = crate::core::shape_to_stride(&shape);
        Ok(TensorBase::from_parts(backend, buffer, MetaTensor::new(shape, stride, 0), None))
    }

    fn index_tensor<'a, T: TensorValue + PartialEq + std::fmt::Debug>(
        index: Idx, 
        tensor: &'a impl TensorAccess<T, RemoteBackend>
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
    fn test_remote_backend_init() {
        setup_server();
        let backend = RemoteBackend::new_with_address(SERVER_IP.parse().unwrap(), SERVER_PORT);
        assert!(backend.is_ok(), "Failed to initialize Remote backend");
    }

    #[test]
    fn test_remote_scalar() {
        setup_server();
        let tensor = make_remote_tensor(vec![42], vec![]).unwrap();
        assert_eq!(index_tensor(Idx::Item, &tensor.view()).unwrap(), 42);
        assert!(tensor.meta.is_scalar());
    }

    #[test]
    fn test_remote_column() {
        setup_server();
        let tensor = make_remote_tensor(vec![1, 2, 3], vec![3]).unwrap();
        assert_eq!(*tensor.meta.shape(), vec![3]);
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_remote_row() {
        setup_server();
        let tensor = make_remote_tensor(vec![1, 2, 3], vec![1, 3]).unwrap();
        assert_eq!(*tensor.meta.shape(), vec![1, 3]);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 0]), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 1]), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::Coord(vec![0, 2]), &tensor.view()).unwrap(), 3);
    }

    #[test]
    fn test_remote_array() {
        let buf = vec![1, 2, 3];
        let shape = vec![3];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();

        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 2);
        assert_eq!(index_tensor(Idx::At(2), &tensor.view()).unwrap(), 3);

        tensor.set(&Idx::At(1), 10).unwrap();
        assert_eq!(index_tensor(Idx::At(1), &tensor.view()).unwrap(), 10);
    }

    #[test]
    fn test_remote_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();

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
    fn test_remote_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_slice_matrix() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_slice_of_slice() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_slice_of_slice_cube() {
        let buf = vec![1, 2, 4, 5, 6, 7, 8, 9];
        let shape = vec![2, 2, 2];
        let tensor = make_remote_tensor(buf, shape).unwrap();

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
    fn test_remote_mut_slices() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_from_buf_error() {
        setup_server();
        let buf = vec![1, 2, 3, 4];
        let shape = vec![2, 3];
        assert!(matches!(
            make_remote_tensor(buf, shape),
            Err(TensorError::InvalidShape(_))
        ));
    }

    #[test]
    fn test_remote_get_errors() {
        let tensor = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
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
    fn test_remote_slice_errors() {
        let tensor = make_remote_tensor(
            vec![1, 2, 3, 4],
            vec![2, 2]
        ).unwrap();
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
    fn test_remote_index_and_index_mut() {
        let buf = vec![1, 2, 3, 4, 5, 6];
        let shape = vec![2, 3];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();

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
    fn test_remote_index_out_of_bounds_panic() {
        let tensor = make_remote_tensor(vec![1, 2, 3], vec![3]).unwrap();
        let _ = tensor.view().get(vec![3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_remote_index_wrong_dims_panic() {
        let tensor = make_remote_tensor(vec![1, 2, 3], vec![3]).unwrap();
        let _ = tensor.view().get(vec![0, 0]).unwrap();
    }

    #[test]
    fn rediculously_large_remote_tensor() {
        let n = 10_000_000_usize;
        let buf: Vec<u8> = vec![1; n];
        let shape = vec![n];
        let tensor = make_remote_tensor(buf, shape).unwrap();
        assert_eq!(index_tensor(Idx::At(0), &tensor.view()).unwrap(), 1);
        assert_eq!(index_tensor(Idx::At(n - 1), &tensor.view()).unwrap(), 1);
    }

    #[test]
    fn test_remote_custom_positive_step() {
        // Test Remote slicing with custom positive step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_custom_negative_step() {
        // Test Remote slicing with custom negative step values
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let tensor = make_remote_tensor(buf, shape).unwrap();
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
    fn test_remote_custom_positive_step_mut() {
        // Test mutable Remote slicing with custom positive step
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_custom_positive_step_mut_with_range() {
        // Test mutable Remote slicing with step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_custom_negative_step_mut() {
        // Test mutable Remote slicing with custom negative step
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_custom_negative_step_mut_with_range() {
        // Test mutable Remote slicing with negative step on a range
        let buf = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let shape = vec![16];
        let mut tensor = make_remote_tensor(buf, shape).unwrap();
        
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
    fn test_remote_matmul_same_operand() {
        // x @ x used to panic on the server (overlapping keys in get_disjoint_mut)
        let a = make_remote_tensor(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = a.matmul(&a).unwrap();
        let expected = Tensor::<f32>::from_buf(vec![7.0, 10.0, 15.0, 22.0], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_broadcast_inplace() {
        // dst aliases left; the server must hand the backend a writable pointer
        let mut a = make_remote_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = make_remote_tensor(vec![10, 20, 30], vec![3]).unwrap();
        a += &b;
        let expected = Tensor::<i32>::from_buf(vec![11, 22, 33, 14, 25, 36], vec![2, 3]).unwrap();
        assert_eq!(a.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_scalar_ops_layouts() {
        // contiguous
        let mut t = make_remote_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        t += 1;
        t *= 2;
        t -= 4;
        assert_eq!(t.cpu().unwrap(), Tensor::<i32>::from_buf(vec![0, 2, 4, 6, 8, 10], vec![2, 3]).unwrap());

        // 1d strided: a single column of a row-major matrix
        let mut t = make_remote_tensor(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        {
            let mut col = t.slice_mut(1, 1).unwrap();
            col += 100;
        }
        assert_eq!(t.cpu().unwrap(), Tensor::<i32>::from_buf(vec![1, 102, 3, 4, 105, 6], vec![2, 3]).unwrap());

        // nd strided: transpose
        let mut t = make_remote_tensor(vec![2.0f64, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3]).unwrap();
        {
            let mut tr = t.transpose_mut();
            tr /= 2.0;
        }
        assert_eq!(t.cpu().unwrap(), Tensor::<f64>::from_buf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_remote_unsupported_op_errors() {
        let mut t = make_remote_tensor(vec![1.0f32, 4.0], vec![2]).unwrap();
        let mut out = t.backend.alloc::<f32>(2).unwrap();
        let meta = t.meta.clone();
        let config = crate::ops::linalg::ConvConfig2D::default();
        let err = t.backend.apply_conv_2d((&t.buf, &meta), (&t.buf, &meta), &mut out, &config).unwrap_err();
        assert!(matches!(err, TensorError::UnsupportedOperation(_)), "{err:?}");
    }

    fn assert_invalid(result: Result<(), TensorError>) {
        assert!(matches!(result, Err(TensorError::RemoteError(ref m)) if m.contains("invalid request")), "{result:?}");
    }

    /// A fresh connection, for tests that deliberately leave errors behind.
    fn own_backend() -> RemoteBackend {
        setup_server();
        let mut backend = RemoteBackend::new_with_address(SERVER_IP.parse().unwrap(), SERVER_PORT).unwrap();
        backend.connect().unwrap();
        backend
    }

    #[test]
    fn test_remote_deferred_error_surfaces_on_next_call() {
        let backend = own_backend();
        let mut buf = backend.alloc_from_slice::<i32>(vec![1, 2].into()).unwrap();
        // Pipelined: returns before the server has run it.
        backend.write(&mut buf, 10, 5).unwrap();
        let err = backend.read(&buf, 0).unwrap_err();
        assert!(matches!(err, TensorError::IdxOutOfBounds(_)), "{err:?}");
        // Reported once; the connection keeps working afterwards.
        assert_eq!(backend.read(&buf, 1).unwrap(), 2);
        backend.write(&mut buf, 10, 5).unwrap();
        assert!(backend.sync().is_err());
        backend.sync().unwrap();
    }

    #[test]
    fn test_remote_out_of_range_requests_are_rejected() {
        // These would index out of bounds in the backend (a panic, which aborts a release-built
        // server), so the server must reject them before they get there.
        let backend = own_backend();
        let mut buf = backend.alloc_from_slice::<f32>(vec![-1.0, 2.0].into()).unwrap();
        let other = backend.alloc_from_slice::<f32>(vec![0.0; 4].into()).unwrap();
        let attempts: Vec<Box<dyn Fn(&mut crate::backend::remote::client::RemoteBuf<f32>)>> = vec![
            Box::new(|b| backend.apply_relu_contiguous(b, 0, 1000).unwrap()),
            Box::new(|b| backend.apply_relu_1d_strided(b, 1, -2, 2).unwrap()),
            Box::new(|b| backend.scalar_apply_add_nd(b, 1.0, 0, &[2, 2], &[1, 1]).unwrap()),
            Box::new(|b| backend.copy_range_within(b, &other, 1, 0, 2).unwrap()),
        ];
        for (i, attempt) in attempts.iter().enumerate() {
            attempt(&mut buf);
            let err = backend.sync().unwrap_err();
            assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("invalid request")), "attempt {i}: {err:?}");
        }
        assert_eq!(&*backend.dump(&buf).unwrap(), &[-1.0, 2.0]);
    }

    #[test]
    fn test_remote_matmul_rejects_bad_extents() {
        let backend = own_backend();
        let lhs = backend.alloc_from_slice::<f32>(vec![1.0; 4].into()).unwrap();
        let rhs = backend.alloc_from_slice::<f32>(vec![1.0; 4].into()).unwrap();
        let mut dst = backend.alloc::<f32>(4).unwrap();
        let meta = MetaTensor::new(vec![2, 2], vec![2, 1], 0);
        // Claims 3x3 matrices backed by 4-element buffers.
        crate::backend::BackendMatMul::<f32>::matmul(
            &backend,
            (&lhs, &meta, crate::core::meta::ContiguityTypes::RowMajor),
            (&rhs, &meta, crate::core::meta::ContiguityTypes::RowMajor),
            &mut dst, 1, 3, 3, 3,
        ).unwrap();
        let err = backend.sync().unwrap_err();
        assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("invalid request")), "{err:?}");
        // Offset pushes a valid-looking 2x2 past the end of the buffer.
        let shifted = MetaTensor::new(vec![2, 2], vec![2, 1], 1);
        crate::backend::BackendMatMul::<f32>::matmul(
            &backend,
            (&lhs, &shifted, crate::core::meta::ContiguityTypes::RowMajor),
            (&rhs, &meta, crate::core::meta::ContiguityTypes::RowMajor),
            &mut dst, 1, 2, 2, 2,
        ).unwrap();
        assert!(backend.sync().is_err());
    }

    #[test]
    fn test_remote_deferred_errors_stay_with_their_thread() {
        let backend = own_backend();
        let mut bad = backend.alloc_from_slice::<i32>(vec![0; 2].into()).unwrap();
        backend.sync().unwrap();
        // This thread's pipelined write fails on the server...
        backend.write(&mut bad, 99, 1).unwrap();
        // ...but another thread's calls are unaffected, even after the failure has arrived.
        std::thread::scope(|scope| {
            scope.spawn(|| {
                backend.sync().unwrap();
                let buf = backend.alloc_from_slice::<i32>(vec![5, 6].into()).unwrap();
                assert_eq!(backend.read(&buf, 1).unwrap(), 6);
            });
        });
        let err = backend.read(&bad, 0).unwrap_err();
        assert!(matches!(err, TensorError::IdxOutOfBounds(_)), "{err:?}");
    }

    #[test]
    fn test_remote_concurrent_threads_share_a_connection() {
        let backend = own_backend();
        std::thread::scope(|scope| {
            for t in 0..8i64 {
                let backend = &backend;
                scope.spawn(move || {
                    for i in 0..200i64 {
                        let mut buf = backend.alloc_from_slice::<i64>(vec![t, i].into()).unwrap();
                        backend.scalar_apply_mul_contiguous(&mut buf, 3, 0, 2).unwrap();
                        backend.scalar_apply_add_contiguous(&mut buf, 1, 0, 2).unwrap();
                        assert_eq!(&*backend.dump(&buf).unwrap(), &[3 * t + 1, 3 * i + 1]);
                    }
                });
            }
        });
    }

    #[test]
    fn test_remote_rejects_buffer_from_other_connection() {
        let a = own_backend();
        let b = own_backend();
        let buf = a.alloc_from_slice::<i32>(vec![1, 2].into()).unwrap();
        let err = b.read(&buf, 0).unwrap_err();
        assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("different remote connection")), "{err:?}");
        assert_eq!(a.read(&buf, 0).unwrap(), 1);
    }

    #[test]
    fn test_remote_dtype_mismatch_is_rejected() {
        let backend = own_backend();
        let buf = backend.alloc_from_slice::<i32>(vec![1, 2].into()).unwrap();
        // Forge a handle with the same id but another dtype.
        let forged = crate::backend::remote::client::RemoteBuf::<f32> {
            id: buf.id,
            len: buf.len,
            owner: buf.owner.clone(),
            _marker: std::marker::PhantomData,
        };
        let err = backend.read(&forged, 0).unwrap_err();
        assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("has dtype")), "{err:?}");
        // Dropping the forged handle would free the real buffer.
        std::mem::forget(forged);
        assert_eq!(backend.read(&buf, 0).unwrap(), 1);
    }

    #[test]
    fn test_remote_handshake_version_mismatch() {
        use crate::backend::remote::protocol::{read_frame, write_frame, Op, Request, Response};
        setup_server();
        let mut stream = std::net::TcpStream::connect((SERVER_IP, SERVER_PORT)).unwrap();
        write_frame(&mut stream, &Request { id: 7, reply: true, op: Op::Hello { version: u32::MAX, device: crate::backend::remote::RemoteDevice::Cpu } }).unwrap();
        let response: Response = read_frame(&mut stream).unwrap().unwrap();
        assert_eq!(response.id, 7);
        assert!(response.result.is_err());
        // The server hangs up after a failed handshake.
        assert!(read_frame::<_, Response>(&mut stream).unwrap().is_none());
    }

    #[test]
    fn test_remote_unary_ops_match_cpu() {
        use crate::ops::unary::*;
        let data = vec![-0.9f64, -0.25, 0.0, 0.3, 0.75];
        macro_rules! check {
            ($($op:ident),+) => {$(
                let mut remote = make_remote_tensor(data.clone(), vec![5]).unwrap();
                let mut cpu = Tensor::<f64>::from_buf(data.clone(), vec![5]).unwrap();
                remote.$op();
                cpu.$op();
                let got = remote.cpu().unwrap();
                for i in 0..5 {
                    let (g, e) = (got.get(&Idx::At(i)).unwrap(), cpu.get(&Idx::At(i)).unwrap());
                    assert!((g.is_nan() && e.is_nan()) || g == e, "{}: index {i}: {g} != {e}", stringify!($op));
                }
            )+};
        }
        check!(
            neg_inplace, relu_inplace, sigmoid_inplace, silu_inplace, tanh_inplace, abs_inplace,
            sqrt_inplace, ln_inplace, expm1_inplace, ln1p_inplace, floor_inplace, ceil_inplace,
            round_inplace, trunc_inplace, sin_inplace, cos_inplace, tan_inplace, asin_inplace,
            acos_inplace, atan_inplace, sinh_inplace, cosh_inplace, asinh_inplace, acosh_inplace,
            atanh_inplace, rsqrt_inplace, reciprocal_inplace, square_inplace, cube_inplace,
            exp_inplace, sign_inplace
        );
    }

    #[test]
    fn test_remote_integer_unary_ops_match_cpu() {
        use crate::ops::unary::*;
        let data = vec![-7i32, -1, 0, 3, 12];
        macro_rules! check {
            ($($op:ident),+) => {$(
                let mut remote = make_remote_tensor(data.clone(), vec![5]).unwrap();
                let mut cpu = Tensor::<i32>::from_buf(data.clone(), vec![5]).unwrap();
                remote.$op();
                cpu.$op();
                assert_eq!(remote.cpu().unwrap(), cpu, "{}", stringify!($op));
            )+};
        }
        check!(neg_inplace, relu_inplace);
    }

    #[test]
    fn test_remote_scalar_ops_match_cpu() {
        use crate::ops::scalar::*;
        let data = vec![-0.9f32, 0.25, 1.5, 3.0];
        macro_rules! check {
            ($($op:ident($v:expr) on $d:expr),+) => {$(
                let mut remote = make_remote_tensor($d.clone(), vec![4]).unwrap();
                let mut cpu = Tensor::<f32>::from_buf($d.clone(), vec![4]).unwrap();
                remote.$op($v);
                cpu.$op($v);
                assert_eq!(remote.cpu().unwrap(), cpu, "{}", stringify!($op));
            )+};
        }
        let positive: Vec<f32> = data.iter().map(|x| x.abs()).collect();
        check!(
            log_inplace(10.0) on positive, log1p_inplace(10.0) on positive,
            leaky_relu_inplace(0.1) on data, elu_inplace(0.1) on data
        );
    }

    #[test]
    fn test_remote_fill_layouts() {
        use crate::core::tensor::TensorAccessMut;
        let mut t = make_remote_tensor(vec![0i32; 6], vec![2, 3]).unwrap();
        t.fill(4).unwrap();
        assert_eq!(t.cpu().unwrap(), Tensor::<i32>::from_buf(vec![4; 6], vec![2, 3]).unwrap());
        t.view_mut().slice_mut(1, 1).unwrap().fill(9).unwrap();
        assert_eq!(t.cpu().unwrap(), Tensor::<i32>::from_buf(vec![4, 9, 4, 4, 9, 4], vec![2, 3]).unwrap());
        t.transpose_mut().fill(1).unwrap();
        assert_eq!(t.cpu().unwrap(), Tensor::<i32>::from_buf(vec![1; 6], vec![2, 3]).unwrap());
    }

    #[test]
    fn test_remote_convert() {
        let t = make_remote_tensor(vec![1.7f32, -2.2, 3.0], vec![3]).unwrap();
        let as_int = t.into_dtype::<i64>().unwrap();
        let expected = Tensor::<f32>::from_buf(vec![1.7, -2.2, 3.0], vec![3]).unwrap().into_dtype::<i64>().unwrap();
        assert_eq!(as_int.cpu().unwrap(), expected);
        let as_bool = t.into_dtype::<types::boolean>().unwrap();
        let expected = Tensor::<f32>::from_buf(vec![1.7, -2.2, 3.0], vec![3]).unwrap().into_dtype::<types::boolean>().unwrap();
        assert_eq!(as_bool.cpu().unwrap(), expected);
    }

    #[test]
    fn test_remote_reductions_match_cpu() {
        use crate::ops::reduction::{ReductionOp, TotalReductionOp};
        let data: Vec<f64> = (0..24).map(|i| (i as f64 * 0.37).sin()).collect();
        let remote = make_remote_tensor(data.clone(), vec![2, 3, 4]).unwrap();
        let cpu = Tensor::<f64>::from_buf(data, vec![2, 3, 4]).unwrap();
        assert_eq!(remote.sum().unwrap().cpu().unwrap(), cpu.sum().unwrap());
        assert_eq!(remote.prod().unwrap().cpu().unwrap(), cpu.prod().unwrap());
        assert_eq!(remote.mean().unwrap().cpu().unwrap(), cpu.mean().unwrap());
        for dim in 0..3 {
            assert_eq!(remote.sum_at(dim).unwrap().cpu().unwrap(), cpu.sum_at(dim).unwrap(), "sum_at({dim})");
            assert_eq!(remote.max_at(dim).unwrap().cpu().unwrap(), cpu.max_at(dim).unwrap(), "max_at({dim})");
            assert_eq!(remote.min_at(dim).unwrap().cpu().unwrap(), cpu.min_at(dim).unwrap(), "min_at({dim})");
            assert_eq!(remote.mean_at(dim).unwrap().cpu().unwrap(), cpu.mean_at(dim).unwrap(), "mean_at({dim})");
        }
    }

    #[test]
    fn test_remote_unsupported_reductions_error_instead_of_crashing() {
        // The CPU server backend has no variance accumulator and no argmax; it must say so
        // rather than hit the panics/todo!()s behind them.
        let backend = own_backend();
        let src = backend.alloc_from_slice::<f32>(vec![1.0, 2.0, 3.0].into()).unwrap();
        let mut dst = backend.alloc::<f32>(1).unwrap();
        backend.apply_reduce_contiguous_flat(&src, &mut dst, 0, 3, ReductionOpTypes::Variance { unbiased: true }).unwrap();
        let err = backend.sync().unwrap_err();
        assert!(matches!(err, TensorError::UnsupportedOperation(_)), "{err:?}");
        let mut idx = backend.alloc::<u64>(1).unwrap();
        backend.apply_argmax_contiguous_flat(&src, &mut idx, 0, 3, ReductionOpTypes::ArgMax).unwrap();
        let err = backend.sync().unwrap_err();
        assert!(matches!(err, TensorError::UnsupportedOperation(_)), "{err:?}");
        // Reduction kinds can't be swapped between entry points.
        backend.apply_reduce_contiguous_flat(&src, &mut dst, 0, 3, ReductionOpTypes::ArgMax).unwrap();
        assert_invalid(backend.sync());
    }

    #[test]
    fn test_remote_reduction_rejects_bad_extents() {
        let backend = own_backend();
        let src = backend.alloc_from_slice::<f32>(vec![1.0; 6].into()).unwrap();
        let mut dst = backend.alloc::<f32>(1).unwrap();
        backend.apply_reduce_contiguous_flat(&src, &mut dst, 4, 5, ReductionOpTypes::Sum).unwrap();
        assert!(matches!(backend.sync().unwrap_err(), TensorError::RemoteError(ref m) if m.contains("invalid request")));
        // 2x3 summed over dim 1 needs 2 outputs.
        let meta = MetaTensor::new(vec![2, 3], vec![3, 1], 0);
        let out_meta = MetaTensor::new(vec![2], vec![1], 0);
        backend.apply_reduce_contiguous_nd((&src, &meta), (&mut dst, &out_meta), 1, ReductionOpTypes::Sum).unwrap();
        assert!(matches!(backend.sync().unwrap_err(), TensorError::RemoteError(ref m) if m.contains("invalid request")));
        // Transposed (non-row-major) source.
        let mut dst2 = backend.alloc::<f32>(3).unwrap();
        let transposed = MetaTensor::new(vec![3, 2], vec![1, 3], 0);
        let out_meta = MetaTensor::new(vec![3], vec![1], 0);
        backend.apply_reduce_contiguous_nd((&src, &transposed), (&mut dst2, &out_meta), 1, ReductionOpTypes::Sum).unwrap();
        assert_invalid(backend.sync());

        // Empty extent with a start past the end (kernels still slice start..start).
        backend.apply_reduce_contiguous_flat(&src, &mut dst, 100, 0, ReductionOpTypes::Sum).unwrap();
        assert_invalid(backend.sync());
        let mut fill_target = backend.alloc::<f32>(6).unwrap();
        backend.fill_contiguous(&mut fill_target, 1.0, 100, 0).unwrap();
        assert_invalid(backend.sync());
        // Missing strides.
        let no_strides = MetaTensor::new(vec![1, 1], vec![], 0);
        backend.apply_reduce_contiguous_nd((&src, &no_strides), (&mut dst, &out_meta), 0, ReductionOpTypes::Sum).unwrap();
        assert_invalid(backend.sync());
        // Shape whose element count overflows.
        let huge = MetaTensor::new(vec![4, 1 << 62], vec![1 << 62, 1], 0);
        backend.apply_reduce_contiguous_nd((&src, &huge), (&mut dst2, &out_meta), 1, ReductionOpTypes::Sum).unwrap();
        assert_invalid(backend.sync());
        // Extent arithmetic that would wrap.
        backend.fill_nd(&mut fill_target, 2.0, 0, &[usize::MAX, usize::MAX - 1], &[isize::MAX, 1000]).unwrap();
        assert_invalid(backend.sync());
        backend.fill_nd(&mut fill_target, 2.0, 0, &[3, 7], &[2, 1]).unwrap();
        assert_invalid(backend.sync());
        assert_eq!(&*backend.dump(&fill_target).unwrap(), &[0.0; 6]);
    }

    #[test]
    fn test_remote_empty_reductions() {
        use crate::ops::reduction::ReductionOp;
        // A zero-size trailing dim means no outputs; the kernels must not run at all.
        let remote = make_remote_tensor(Vec::<f32>::new(), vec![3, 0]).unwrap();
        // (The CPU backend panics on this locally; the server must not.)
        let out = remote.sum_at(0).unwrap();
        assert_eq!(out.size(), 0);
        assert_eq!(out.cpu().unwrap().size(), 0);
        // Reducing an empty dim gives one output per remaining element.
        let remote = make_remote_tensor(Vec::<f32>::new(), vec![0, 3]).unwrap();
        let cpu = Tensor::<f32>::from_buf(Vec::<f32>::new(), vec![0, 3]).unwrap();
        assert_eq!(remote.sum_at(0).unwrap().cpu().unwrap(), cpu.sum_at(0).unwrap());
    }

    #[test]
    fn test_remote_convert_validation() {
        let backend = own_backend();
        let src = backend.alloc_from_slice::<f32>(vec![1.0, 2.0].into()).unwrap();
        let mut short = backend.alloc::<i32>(1).unwrap();
        assert!(matches!(backend.convert(&src, &mut short), Err(TensorError::SizeMismatch(_))));
        let mut same = backend.alloc_from_slice::<f32>(vec![0.0, 0.0].into()).unwrap();
        backend.convert(&src, &mut same).unwrap();
        assert_eq!(&*backend.dump(&same).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_remote_reduce_rank1_and_offset_view() {
        use crate::ops::reduction::{ReductionOp, TotalReductionOp};
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let remote = make_remote_tensor(data.clone(), vec![6]).unwrap();
        let cpu = Tensor::<f64>::from_buf(data.clone(), vec![6]).unwrap();
        assert_eq!(remote.sum_at(0).unwrap().cpu().unwrap(), cpu.sum_at(0).unwrap());
        // A full reduction of an offset view goes through the flat path, which honours the offset.
        let remote = make_remote_tensor(data.clone(), vec![2, 3]).unwrap();
        let cpu = Tensor::<f64>::from_buf(data, vec![2, 3]).unwrap();
        let (rv, cv) = (remote.slice(0, 1).unwrap(), cpu.slice(0, 1).unwrap());
        assert_eq!(rv.sum().unwrap().cpu().unwrap(), cv.sum().unwrap());
    }


    #[test]
    fn test_remote_bool_roundtrip() {
        let t = make_remote_tensor(vec![types::boolean(true), types::boolean(false)], vec![2]).unwrap();
        assert_eq!(t.cpu().unwrap(), Tensor::from_buf(vec![types::boolean(true), types::boolean(false)], vec![2]).unwrap());
    }

    /// Starts a server with `config` on an ephemeral port and returns the port.
    fn spawn_server(config: crate::backend::remote::server::ServerConfig) -> u16 {
        let listener = std::net::TcpListener::bind((SERVER_IP, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let mut server = RemoteServer::new(SERVER_IP.parse().unwrap(), port).with_config(config);
            let _ = server.serve_on(listener);
        });
        port
    }

    #[test]
    fn test_remote_buffers_are_freed_on_drop() {
        use crate::backend::remote::server::ServerConfig;
        // 4 MiB budget; allocating 1 MiB fifty times only works if dropped buffers are freed.
        let port = spawn_server(ServerConfig { max_session_bytes: Some(4 << 20), ..Default::default() });
        let backend = RemoteBackend::connect_to(SERVER_IP.parse().unwrap(), port).unwrap();
        for i in 0..50u8 {
            let t = RemoteTensor::<u8>::from_buf_on(&backend, vec![i; 1 << 20], vec![1 << 20]).unwrap();
            assert_eq!(t.get(&Idx::At(7)).unwrap(), i);
        }
        // Temporaries created by ops are freed too.
        let a = RemoteTensor::<f32>::from_buf_on(&backend, vec![1.0; 1 << 18], vec![1 << 18]).unwrap();
        for _ in 0..20 {
            let b = &a + &a;
            assert_eq!(b.get(&Idx::At(0)).unwrap(), 2.0);
        }
        // Holding more than the budget fails cleanly instead of exhausting the server.
        let held: Vec<_> = (0..3).map(|_| backend.alloc::<u8>(1 << 20).unwrap()).collect();
        backend.alloc::<u8>(2 << 20).unwrap();
        let err = backend.sync().unwrap_err();
        assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("limit")), "{err:?}");
        drop(held);
        let _big = backend.alloc::<u8>(2 << 20).unwrap();
        backend.sync().unwrap();
        // Sizes that overflow are rejected rather than reaching the allocator.
        backend.alloc::<u64>(usize::MAX / 2).unwrap();
        assert!(backend.sync().is_err());
    }

    #[test]
    fn test_remote_small_queue_depth_still_works() {
        use crate::backend::remote::server::ServerConfig;
        let port = spawn_server(ServerConfig { queue_depth: 1, ..Default::default() });
        let backend = RemoteBackend::connect_to(SERVER_IP.parse().unwrap(), port).unwrap();
        let mut t = RemoteTensor::<i64>::from_buf_on(&backend, vec![0; 64], vec![64]).unwrap();
        for _ in 0..500 {
            t += 1;
        }
        assert_eq!(t.get(&Idx::At(63)).unwrap(), 500);
    }

    #[test]
    fn test_remote_call_timeout() {
        use crate::backend::remote::protocol::{read_frame, write_frame, Reply, Request, Response};
        use crate::core::primitives::DeviceType;
        // A server that completes the handshake and then never answers.
        let listener = std::net::TcpListener::bind((SERVER_IP, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let hello: Request = read_frame(&mut stream).unwrap().unwrap();
            write_frame(&mut stream, &Response { id: hello.id, result: Ok(Reply::DeviceType(DeviceType::Cpu)) }).unwrap();
            while let Ok(Some(_)) = read_frame::<_, Request>(&mut stream) {}
        });
        let backend = RemoteBackend::connect_to(SERVER_IP.parse().unwrap(), port).unwrap();
        backend.set_timeout(Some(std::time::Duration::from_millis(100)));
        let err = backend.sync().unwrap_err();
        assert!(matches!(err, TensorError::RemoteError(ref m) if m.contains("timed out")), "{err:?}");
    }

    #[test]
    fn test_remote_device_reports_its_server() {
        use crate::core::primitives::DeviceType;
        let backend = shared_backend();
        let t = make_remote_tensor(vec![1.0f32], vec![1]).unwrap();
        let expected = DeviceType::Remote { ip: SERVER_IP.parse().unwrap(), port: SERVER_PORT, remote_type: DeviceType::Cpu.into() };
        assert_eq!(backend.device(), expected);
        assert_eq!(t.device(), expected);
        assert!(format!("{t:?}").contains(&format!("device=remote({SERVER_IP}:{SERVER_PORT}, cpu)")));
    }

    #[test]
    fn test_remote_binary_ops_on_non_default_connection() {
        // Outputs of a + b etc. used to be allocated via B::new() (the default connection).
        let backend = own_backend();
        let a = RemoteTensor::<f64>::from_buf_on(&backend, vec![6.0, 8.0], vec![2]).unwrap();
        let b = RemoteTensor::<f64>::from_buf_on(&backend, vec![2.0, 4.0], vec![2]).unwrap();
        let (ca, cb) = (a.cpu().unwrap(), b.cpu().unwrap());
        assert_eq!((&a + &b).cpu().unwrap(), &ca + &cb);
        assert_eq!((&a - &b).cpu().unwrap(), &ca - &cb);
        assert_eq!((&a * &b).cpu().unwrap(), &ca * &cb);
        assert_eq!((&a / &b).cpu().unwrap(), &ca / &cb);
        assert_eq!(a.view().sum().unwrap().cpu().unwrap(), ca.view().sum().unwrap());
    }

    #[test]
    fn test_remote_to_remote_roundtrip() {
        let backend = shared_backend();
        let cpu = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let remote = cpu.to_remote(&backend).unwrap();
        assert_eq!(remote.cpu().unwrap(), cpu);
        // A non-contiguous source (transposed strides) is materialised first.
        let mut transposed = cpu.clone();
        transposed.meta = MetaTensor::new(vec![3, 2], vec![1, 3], 0);
        let remote = transposed.to_remote(&backend).unwrap();
        assert_eq!(remote.cpu().unwrap(), crate::core::tensor::AsTensor::owned(&transposed.view()));
    }

    #[test]
    fn test_remote_shared_connections_are_reused() {
        setup_server();
        let a = crate::backend::remote::try_use_remote_backend(SERVER_IP.parse().unwrap(), SERVER_PORT).unwrap();
        let b = crate::backend::remote::try_use_remote_backend(SERVER_IP.parse().unwrap(), SERVER_PORT).unwrap();
        let x = RemoteTensor::<i32>::from_buf_on(&a, vec![1, 2], vec![2]).unwrap();
        let y = RemoteTensor::<i32>::from_buf_on(&b, vec![3, 4], vec![2]).unwrap();
        assert_eq!((&x + &y).cpu().unwrap(), Tensor::<i32>::from_buf(vec![4, 6], vec![2]).unwrap());
        // with_remote uses the same shared connection.
        let z = RemoteTensor::<i32>::with_remote(SERVER_IP.parse().unwrap(), SERVER_PORT).unwrap();
        assert_eq!(z.backend.device(), a.device());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_remote_cuda_session_needs_cuda_server() {
        setup_server();
        let err = RemoteBackend::connect_device(SERVER_IP.parse().unwrap(), SERVER_PORT, crate::backend::remote::RemoteDevice::Cuda(0)).unwrap_err();
        assert!(err.to_string().contains("without CUDA"), "{err}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_remote_cuda_session() {
        use crate::core::primitives::DeviceType;
        use crate::ops::{reduction::{ReductionOp, TotalReductionOp}, unary::*};
        setup_server();
        let backend = RemoteBackend::connect_device(SERVER_IP.parse().unwrap(), SERVER_PORT, crate::backend::remote::RemoteDevice::Cuda(0)).unwrap();
        assert!(matches!(backend.device(), DeviceType::Remote { ref remote_type, .. } if **remote_type == DeviceType::Cuda(0)));
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
        let mut remote = RemoteTensor::<f32>::from_buf_on(&backend, data.clone(), vec![3, 4]).unwrap();
        let mut cpu = Tensor::<f32>::from_buf(data, vec![3, 4]).unwrap();
        remote.relu_inplace();
        cpu.relu_inplace();
        remote *= 2.0;
        cpu *= 2.0;
        assert_eq!(remote.cpu().unwrap(), cpu);
        assert_eq!(remote.sum().unwrap().cpu().unwrap(), cpu.sum().unwrap());
        assert_eq!(remote.sum_at(1).unwrap().cpu().unwrap(), cpu.sum_at(1).unwrap());
        let m = remote.matmul(&remote.transpose()).unwrap();
        assert_eq!(m.cpu().unwrap(), cpu.matmul(&cpu.transpose()).unwrap());
    }
}
