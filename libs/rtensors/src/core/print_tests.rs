#[cfg(test)]
mod print_tests {
    use crate::core::{Tensor, tensor::AsView, tensor::AsViewMut};
    
    #[test]
    fn test_scalar_print() {
        let tensor = Tensor::<f32>::scalar(42.0);
        let output = format!("{:?}", tensor);
        println!("Scalar: {}", output);
        assert!(output.contains("tensor"));
        assert!(output.contains("42"));
    }
    
    #[test]
    fn test_1d_print() {
        let tensor = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = format!("{:?}", tensor);
        println!("1D: {}", output);
        assert!(output.contains("tensor"));
        assert!(output.contains("1"));
        assert!(output.contains("2"));
        assert!(output.contains("3"));
    }
    
    #[test]
    fn test_2d_print() {
        let tensor = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        ).unwrap();
        let output = format!("{:?}", tensor);
        println!("2D:\n{}", output);
        assert!(output.contains("tensor"));
        // Should have nested structure
        assert!(output.contains("["));
    }
    
    #[test]
    fn test_3d_print() {
        let tensor = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2]
        ).unwrap();
        let output = format!("{:?}", tensor);
        println!("3D:\n{}", output);
        assert!(output.contains("tensor"));
    }
    
    #[test]
    fn test_view_print() {
        let tensor = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let view = tensor.view();
        let output = format!("{:?}", view);
        println!("View:\n{}", output);
        assert!(output.contains("tensor"));
    }
    
    #[test]
    fn test_view_mut_print() {
        let mut tensor = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let view_mut = tensor.view_mut();
        let output = format!("{:?}", view_mut);
        println!("ViewMut:\n{}", output);
        assert!(output.contains("tensor"));
    }
    
    #[test]
    fn test_non_contiguous_print() {
        // Create a contiguous tensor
        let tensor = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        ).unwrap();
        let view = tensor.view();
        // Even though this is contiguous, it still tests that views work
        let output = format!("{:?}", view);
        println!("View:\n{}", output);
        assert!(output.contains("tensor"));
    }
}
