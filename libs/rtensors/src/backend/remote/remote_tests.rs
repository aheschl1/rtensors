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
        ops::linalg::MatMul,
    };
    
    const SERVER_IP: &str = "127.0.0.1";
    const SERVER_PORT: u16 = 7879;
    
    static INIT: Once = Once::new();
    
    fn setup_server() {
        INIT.call_once(|| {
            let mut server = RemoteServer::new(SERVER_IP.parse().unwrap(), SERVER_PORT);
            thread::spawn(move || {
                let _ = server.serve();
            });
            thread::sleep(std::time::Duration::from_millis(10));
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
        let t = make_remote_tensor(vec![1.0f32, 4.0], vec![2]).unwrap();
        let mut buf = t.backend.copy(&t.buf).unwrap();
        let err = t.backend.apply_sqrt_contiguous(&mut buf, 0, 2).unwrap_err();
        assert!(matches!(err, TensorError::UnsupportedOperation(_)), "{err:?}");
    }

    #[test]
    fn test_remote_bool_roundtrip() {
        let t = make_remote_tensor(vec![types::boolean(true), types::boolean(false)], vec![2]).unwrap();
        assert_eq!(t.cpu().unwrap(), Tensor::from_buf(vec![types::boolean(true), types::boolean(false)], vec![2]).unwrap());
    }
}
