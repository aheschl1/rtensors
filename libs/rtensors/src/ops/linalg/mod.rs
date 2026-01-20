use crate::{backend::Backend, core::{tensor::TensorError, value::{TensorValue, WeightValue}}};

pub mod matmul;
mod conv;

pub use conv::*;

/// Linear algebra operations: matrix multiplication and dot product.
/// 
/// For f32 and f64 types, uses OpenBLAS on CPU and cuBLAS on CUDA for optimized performance.
pub trait MatMul<Rhs, T, B> 
where 
    T: TensorValue,
    B: Backend,
{
    type Output;
    /// Performs matrix multiplication.
    /// 
    /// # Examples
    /// ```ignore
    /// let a = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0, 4.0], (2, 2)).unwrap();
    /// let b = Tensor::<f32>::from_buf(vec![5.0, 6.0, 7.0, 8.0], (2, 2)).unwrap();
    /// let c = a.matmul(&b).unwrap();
    /// ```
    fn matmul(&self, rhs: &Rhs) -> Result<Self::Output, TensorError>;
    
    /// Computes the dot product of two 1-D tensors.
    /// 
    /// For higher-dimensional tensors, this performs dimension manipulation 
    /// and calls `matmul` internally.
    /// 
    /// # Examples
    /// ```ignore
    /// let a = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], (3,)).unwrap();
    /// let b = Tensor::<f32>::from_buf(vec![4.0, 5.0, 6.0], (3,)).unwrap();
    /// let result = a.dot(&b).unwrap(); // Scalar: 32.0
    /// ```
    fn dot(&self, rhs: &Rhs) -> Result<Self::Output, TensorError>;
}

pub trait Conv<K, T, B>
where
    T: WeightValue,
    B: Backend,
{
    type Output;


    fn conv2d(&self, kernel: &K, config: &ConvConfig2D) -> Result<Self::Output, TensorError>;

    fn conv3d(&self, kernel: &K, config: &ConvConfig3D) -> Result<Self::Output, TensorError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{primitives::Tensor, tensor::TensorAccess, MetaTensorView, Slice};

    #[test]
    fn test_matmul_2d_basic() {
        // Test basic 2x3 @ 3x2 matrix multiplication
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]
        // Expected: [[58, 64],
        //            [139, 154]]
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![7, 8, 9, 10, 11, 12], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2d_square() {
        // Test 2x2 @ 2x2 square matrix multiplication
        // A = [[1, 2],
        //      [3, 4]]
        // B = [[5, 6],
        //      [7, 8]]
        // Expected: [[19, 22],
        //            [43, 50]]
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2d_identity() {
        // Test multiplication with identity matrix
        // A = [[1, 2],
        //      [3, 4]]
        // I = [[1, 0],
        //      [0, 1]]
        // Expected: A unchanged
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let identity = Tensor::<i32>::from_buf(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        
        let result = a.matmul(&identity).unwrap();
        
        assert_eq!(result, a);
    }

    #[test]
    fn test_matmul_2d_float() {
        // Test with floating point values
        let a = Tensor::<f32>::from_buf(vec![1.5, 2.0, 3.5, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[1.5*2.0 + 2.0*3.0, 1.5*1.0 + 2.0*4.0],
        //            [3.5*2.0 + 4.0*3.0, 3.5*1.0 + 4.0*4.0]]
        //         = [[9.0, 9.5],
        //            [19.0, 19.5]]
        let expected = Tensor::<f32>::from_buf(vec![9.0, 9.5, 19.0, 19.5], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    // ============================================================================
    // F32 GEMM FAST PATH TESTS
    // ============================================================================

    #[test]
    fn test_matmul_f32_large_square() {
        // Test f32 GEMM with larger square matrices
        let size = 10;
        let mut a_data = vec![0.0f32; size * size];
        let mut b_data = vec![0.0f32; size * size];
        
        // Initialize with simple pattern
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = (i + j) as f32;
                b_data[i * size + j] = (i * j + 1) as f32;
            }
        }
        
        let a = Tensor::<f32>::from_buf(a_data, vec![size, size]).unwrap();
        let b = Tensor::<f32>::from_buf(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
        
        // Verify a specific element manually
        // result[0,0] should be sum of a[0,k] * b[k,0] for k in 0..size
        let expected_00: f32 = (0..size).map(|k| {
            let a_val = k as f32;
            let b_val = (k * 0 + 1) as f32;
            a_val * b_val
        }).sum();
        
        assert!((result.get(vec![0, 0]).unwrap() - expected_00).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_rectangular_large() {
        // Test f32 GEMM with rectangular matrices (M x K) @ (K x N)
        let m = 15;
        let k = 20;
        let n = 10;
        
        let a_data: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i % 7) as f32 + 0.5).collect();
        
        let a = Tensor::<f32>::from_buf(a_data, vec![m, k]).unwrap();
        let b = Tensor::<f32>::from_buf(b_data, vec![k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![m, n]);
    }

    #[test]
    fn test_matmul_f32_batched_gemm() {
        // Test batched f32 GEMM (3D tensors)
        let batch = 4;
        let m = 8;
        let k = 6;
        let n = 5;
        
        let a_data: Vec<f32> = (0..batch*m*k).map(|i| (i % 10) as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..batch*k*n).map(|i| (i % 8) as f32 + 1.0).collect();
        
        let a = Tensor::<f32>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = Tensor::<f32>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f32_precision() {
        // Test that f32 GEMM maintains reasonable precision
        let a = Tensor::<f32>::from_buf(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3]
        ).unwrap();
        let b = Tensor::<f32>::from_buf(
            vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed expected values
        // result[0,0] = 0.1*1.1 + 0.2*1.3 + 0.3*1.5 = 0.11 + 0.26 + 0.45 = 0.82
        // result[0,1] = 0.1*1.2 + 0.2*1.4 + 0.3*1.6 = 0.12 + 0.28 + 0.48 = 0.88
        // result[1,0] = 0.4*1.1 + 0.5*1.3 + 0.6*1.5 = 0.44 + 0.65 + 0.90 = 1.99
        // result[1,1] = 0.4*1.2 + 0.5*1.4 + 0.6*1.6 = 0.48 + 0.70 + 0.96 = 2.14
        
        assert!((result.get(vec![0, 0]).unwrap() - 0.82).abs() < 1e-5);
        assert!((result.get(vec![0, 1]).unwrap() - 0.88).abs() < 1e-5);
        assert!((result.get(vec![1, 0]).unwrap() - 1.99).abs() < 1e-5);
        assert!((result.get(vec![1, 1]).unwrap() - 2.14).abs() < 1e-5);
    }

    // ============================================================================
    // F64 GEMM FAST PATH TESTS
    // ============================================================================

    #[test]
    fn test_matmul_f64_basic() {
        // Test basic f64 GEMM
        let a = Tensor::<f64>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        let b = Tensor::<f64>::from_buf(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[58.0, 64.0], [139.0, 154.0]]
        let expected = Tensor::<f64>::from_buf(vec![58.0, 64.0, 139.0, 154.0], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_f64_large_square() {
        // Test f64 GEMM with larger square matrices
        let size = 12;
        let mut a_data = vec![0.0f64; size * size];
        let mut b_data = vec![0.0f64; size * size];
        
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = i as f64 + j as f64 * 0.5;
                b_data[i * size + j] = i as f64 * 0.3 + j as f64;
            }
        }
        
        let a = Tensor::<f64>::from_buf(a_data, vec![size, size]).unwrap();
        let b = Tensor::<f64>::from_buf(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
    }

    #[test]
    fn test_matmul_f64_rectangular() {
        // Test f64 GEMM with rectangular matrices
        let a = Tensor::<f64>::from_buf(
            (0..60).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            vec![10, 6]
        ).unwrap();
        let b = Tensor::<f64>::from_buf(
            (0..48).map(|i| i as f64 * 0.2).collect::<Vec<_>>(),
            vec![6, 8]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![10, 8]);
    }

    #[test]
    fn test_matmul_f64_batched() {
        // Test batched f64 GEMM
        let batch = 3;
        let m = 5;
        let k = 4;
        let n = 6;
        
        let a_data: Vec<f64> = (0..batch*m*k).map(|i| i as f64 * 0.25).collect();
        let b_data: Vec<f64> = (0..batch*k*n).map(|i| i as f64 * 0.33).collect();
        
        let a = Tensor::<f64>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = Tensor::<f64>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f64_high_precision() {
        // Test that f64 GEMM maintains high precision
        let a = Tensor::<f64>::from_buf(
            vec![
                0.123456789, 0.987654321,
                0.111111111, 0.999999999,
            ],
            vec![2, 2]
        ).unwrap();
        let b = Tensor::<f64>::from_buf(
            vec![
                0.314159265, 0.271828182,
                0.161803398, 0.577215664,
            ],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed with high precision
        // result[0,0] = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398
        let expected_00 = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398;
        assert!((result.get(vec![0, 0]).unwrap() - expected_00).abs() < 1e-10);
    }

    // ============================================================================
    // SLICED TENSOR TESTS (Testing non-contiguous memory)
    // ============================================================================

    #[test]
    fn test_matmul_f32_sliced_rows() {
        // Create a larger tensor and slice rows for matmul
        let full = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get rows 1-3 (shape [2, 3])
        let a = full.slice(0, 1..3).unwrap();
        assert_eq!(*a.shape(), vec![2, 3]);
        
        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 
                3.0, 4.0, 
                5.0, 6.0
            ],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<f32>::from_buf(vec![
            49.0, 64.0, 
            76.0, 100.0
        ], vec![2, 2]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_f32_sliced_cols() {
        // Test matmul with column-sliced tensor
        let full = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        // Slice columns 1-3 (shape [2, 2])
        let a = full.slice(1, 1..3).unwrap();
        
        let b = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // a is [[2,3], [6,7]]
        // result[0,0] = 2*1 + 3*3 = 2 + 9 = 11
        assert!((result.get(vec![0, 0]).unwrap() - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f64_sliced_both() {
        // Test matmul where both tensors are sliced
        let full_a = Tensor::<f64>::from_buf(
            (0..24).map(|i| i as f64).collect::<Vec<_>>(),
            vec![4, 6]
        ).unwrap();
        
        let full_b = Tensor::<f64>::from_buf(
            (0..20).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice a: rows 1-3, then cols 2-6 -> [2, 4]
        let a_step1 = full_a.slice(0, 1..3).unwrap();
        let a = a_step1.slice(1, 2..6).unwrap();
        // Slice b: rows 1-5, then cols 0-3 -> [4, 3]
        let b_step1 = full_b.slice(0, 1..5).unwrap();
        let b = b_step1.slice(1, 0..3).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 3]);
    }

    #[test]
    fn test_matmul_f32_sliced_batched() {
        // Test batched matmul with sliced tensors
        let full = Tensor::<f32>::from_buf(
            (0..120).map(|i| i as f32).collect::<Vec<_>>(),
            vec![5, 4, 6]
        ).unwrap();
        
        // Slice batches 1-4, keep all rows, slice cols 1-5 -> [3, 4, 4]
        let a_step1 = full.slice(0, 1..4).unwrap();
        let a = a_step1.slice(2, 1..5).unwrap();
        
        let b_data: Vec<f32> = (0..72).map(|i| i as f32 * 0.5).collect();
        let b = Tensor::<f32>::from_buf(b_data, vec![3, 4, 6]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![3, 4, 6]);
    }

    #[test]
    fn test_matmul_f32_sliced_rhs() {
        // Test where only the right-hand side is sliced
        let a = Tensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        let full_b = Tensor::<f32>::from_buf(
            (0..20).map(|i| i as f32 * 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice b: rows 1-4, cols 0-2 -> [3, 2]
        let b_step1 = full_b.slice(0, 1..4).unwrap();
        let b = b_step1.slice(1, 0..2).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f64_sliced_lhs() {
        // Test where only the left-hand side is sliced
        let full_a = Tensor::<f64>::from_buf(
            (0..30).map(|i| i as f64).collect::<Vec<_>>(),
            vec![5, 6]
        ).unwrap();
        
        // Slice rows 2-4, then cols 1-4 -> [2, 3]
        let a_step1 = full_a.slice(0, 2..4).unwrap();
        let a = a_step1.slice(1, 1..4).unwrap();
        
        let b = Tensor::<f64>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f32_sliced_identity() {
        // Test sliced tensor with identity-like behavior
        let full = Tensor::<f32>::from_buf(
            vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get the identity part [2, 2]
        let identity_step1 = full.slice(0, 1..3).unwrap();
        let identity_slice = identity_step1.slice(1, 1..3).unwrap();
        
        let test_matrix = Tensor::<f32>::from_buf(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2]
        ).unwrap();
        
        println!("Identity Slice: {:?}", identity_slice.meta);

        let result = identity_slice.matmul(&test_matrix).unwrap();
        
        // Should get test_matrix back
        assert_eq!(result, test_matrix);
    }

    #[test]
    fn test_matmul_f64_slice_then_transpose() {
        let full = Tensor::<f64>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4]
        ).unwrap();
        
        // Slice rows 0-2, cols 1-3 -> [2, 2]
        let sliced_step1 = full.slice(0, 0..2).unwrap();
        let sliced = sliced_step1.slice(1, 1..3).unwrap();
        // [[2,3],[6,7]]
        // Transpose it -> [2, 2]
        let transposed = sliced.transpose();
        // [[2,6],[3,7]]
        let b = Tensor::<f64>::from_buf(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2]
        ).unwrap();
        
        let result = transposed.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);

        let result2 = b.matmul(&transposed).unwrap();
        assert_eq!(result2, result);
        // sliced is [[2,3], [6,7]]
        // transposed is [[2,6], [3,7]]
        // result should be transposed unchanged (identity mult)
        let expected = Tensor::<f64>::from_buf(
            vec![2.0, 6.0, 3.0, 7.0],
            vec![2, 2]
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_3d_batched() {
        // Test batched 3D matrix multiplication
        // Shape: (2, 2, 2) @ (2, 2, 2) -> (2, 2, 2)
        // Batch 1: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
        // Batch 2: [[9, 10], [11, 12]] @ [[13, 14], [15, 16]]
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 9, 10, 11, 12], 
            vec![2, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![5, 6, 7, 8, 13, 14, 15, 16], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2]);
        // Batch 1: [[19, 22], [43, 50]]
        // Batch 2: [[267, 286], [323, 346]]
        let expected = Tensor::<i32>::from_buf(
            vec![19, 22, 43, 50, 267, 286, 323, 346], 
            vec![2, 2, 2]
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_3d_larger_batch() {
        // Test with larger batch dimension
        // Shape: (3, 2, 2) @ (3, 2, 2)
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 0, 0, 1,  // Batch 0: Identity
                2, 0, 0, 2,  // Batch 1: 2*Identity
                3, 0, 0, 3,  // Batch 2: 3*Identity
            ], 
            vec![3, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![
                5, 6, 7, 8,
                5, 6, 7, 8,
                5, 6, 7, 8,
            ], 
            vec![3, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![3, 2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                5, 6, 7, 8,      // 1*[[5,6],[7,8]]
                10, 12, 14, 16,  // 2*[[5,6],[7,8]]
                15, 18, 21, 24,  // 3*[[5,6],[7,8]]
            ], 
            vec![3, 2, 2]
        ).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // Test that mismatched dimensions produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch(_)));
    }

    #[test]
    fn test_matmul_rank_mismatch() {
        // Test that different ranks produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_batch_mismatch() {
        // Test that mismatched batch dimensions produce an error
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], vec![3, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch(_)));
    }

    #[test]
    fn test_matmul_2d_rectangular() {
        // Test with non-square matrices
        // 3x4 @ 4x2 -> 3x2
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 4]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![4, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![3, 2]);
        // Row 0: [1*1+2*3+3*5+4*7, 1*2+2*4+3*6+4*8] = [50, 60]
        // Row 1: [5*1+6*3+7*5+8*7, 5*2+6*4+7*6+8*8] = [114, 140]
        // Row 2: [9*1+10*3+11*5+12*7, 9*2+10*4+11*6+12*8] = [178, 220]
        let expected = Tensor::<i32>::from_buf(vec![50, 60, 114, 140, 178, 220], vec![3, 2]).unwrap();
        assert_eq!(result, expected);
    }

    // ============================================================================
    // FAILURE CASE TESTS
    // ============================================================================

    #[test]
    fn test_matmul_1d_array_fails() {
        // Test that 1D arrays (vectors) fail
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![4, 5, 6], vec![3]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_1d_with_2d_fails() {
        // Test that mixing 1D with 2D fails
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3], vec![3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![3, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_scalar_fails() {
        // Test that scalars (rank-0 tensors) fail
        let a = Tensor::<i32>::scalar(5);
        let b = Tensor::<i32>::scalar(10);
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_scalar_with_2d_fails() {
        // Test that scalar with 2D fails
        let a = Tensor::<i32>::scalar(5);
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_4d_should_work_with_batch_squashing() {
        // Test that 4D tensors should work by squashing batch dimensions
        // This test is expected to fail with current implementation
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2, 1]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 1, 2]
        ).unwrap();
        
        // Currently fails but should work in the future
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_matmul_inner_dimension_mismatch_2d() {
        // Test that (2,3) @ (2,2) fails (inner dimensions 3 != 2)
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch(_)));
    }

    #[test]
    fn test_matmul_inner_dimension_mismatch_3d() {
        // Test that (2,2,3) @ (2,2,2) fails (inner dimensions 3 != 2)
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![2, 2, 3]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch(_)));
    }

    #[test]
    fn test_matmul_batch_size_mismatch() {
        // Test that different batch sizes fail: (2,2,2) @ (3,2,2)
        let a = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8], 
            vec![2, 2, 2]
        ).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            vec![3, 2, 2]
        ).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::SizeMismatch(_)));
    }

    #[test]
    fn test_matmul_2d_3d_rank_mismatch() {
        // Test that 2D @ 3D fails
        let a = Tensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = Tensor::<i32>::from_buf(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 2, 2]).unwrap();
        
        let result = a.matmul(&b);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TensorError::InvalidShape(_)));
    }

    #[test]
    fn test_matmul_stride_rows() {
        // Test matmul with tensors that have row strides
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ]
            , vec![4, 4]
        ).unwrap();
        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                50.0, 60.0,
                178.0, 220.0,
            ],
            vec![2, 2]
        ).unwrap();

        assert_eq!(result, expected);

    }

    // Test stride on right-hand side matrix with f32
    #[test]
    fn test_matmul_stride_rhs_f32() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                19.0, 22.0, 25.0, 28.0,
                39.0, 46.0, 53.0, 60.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // Test stride on right-hand side matrix with i32
    #[test]
    fn test_matmul_stride_rhs_i32() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2,
                3, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22, 25, 28,
                39, 46, 53, 60,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // Test stride on both left and right-hand side with f32
    #[test]
    fn test_matmul_stride_both_f32() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // Test stride on both left and right-hand side with i32
    #[test]
    fn test_matmul_stride_both_i32() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // Test stride on left-hand side matrix with f32 (original test extended)
    #[test]
    fn test_matmul_stride_lhs_f32() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                50.0, 60.0,
                178.0, 220.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // Test stride on left-hand side matrix with i32
    #[test]
    fn test_matmul_stride_lhs_i32() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2,
                3, 4,
                5, 6,
                7, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                50, 60,
                178, 220,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_strided_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

        #[test]
    fn test_strided_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_strided_multi_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,

                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0,
                29.0, 210.0, 211.0, 212.0,
                213.0, 214.0, 215.0, 216.0,

                291.0, 292.0, 293.0, 294.0,
                295.0, 296.0, 297.0, 298.0,
                299.0, 2910.0, 2911.0, 2912.0,
                2913.0, 2914.0, 2915.0, 2916.0,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,

                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.,

                1210.0, 1300.0, 1390.0, 1480.0,
                10806.0, 11468.0, 12130.0, 12792.0,

                596020.0, 833710.0, 834880.0, 836050.0,
                5678286.0, 8044418.0, 8053450.0, 8062482.0,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_strided_multi_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,

                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 210, 211, 212,
                213, 214, 215, 216,

                291, 292, 293, 294,
                295, 296, 297, 298,
                299, 2910, 2911, 2912,
                2913, 2914, 2915, 2916,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,

                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682,

                1210, 1300, 1390, 1480,
                10806, 11468, 12130, 12792,
                596020, 833710, 834880, 836050,
                5678286, 8044418, 8053450, 8062482,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot() {
        let a = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;

        assert!(result.is_scalar());
        assert!((result - expected).item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_dot_with_zeros() {
        // Dot product with a zero vector should produce zero
        let a = Tensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![0.0, 0.0, 0.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        assert!(result.is_scalar());
        assert!((result.item().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_dot_orthogonal_vectors() {
        // Dot product of orthogonal vectors should be zero
        let a = Tensor::<f32>::from_buf(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![0.0, 1.0, 0.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        assert!(result.is_scalar());
        assert!((result.item().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn test_dot_single_element() {
        // Dot product of single-element vectors
        let a = Tensor::<f32>::from_buf(vec![5.0], vec![1]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![3.0], vec![1]).unwrap();

        let result = a.dot(&b).unwrap();
        let expected = 5.0 * 3.0;

        assert!(result.is_scalar());
        assert!((result.item().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_dot_negative_values() {
        // Dot product with negative values
        let a = Tensor::<f32>::from_buf(vec![-1.0, 2.0, -3.0], vec![3]).unwrap();
        let b = Tensor::<f32>::from_buf(vec![4.0, -5.0, 6.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        let expected = (-1.0)*4.0 + 2.0*(-5.0) + (-3.0)*6.0;

        assert!(result.is_scalar());
        assert!((result.item().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_both_column_major() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                2.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                5.0, 7.0,
                6.0, 8.0,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = Tensor::<f32>::from_buf(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }


    #[test]
    fn test_both_column_major_column_stride() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                8.0, 8.0,
                2.0, 4.0, 
                8.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                5.0,  7.0,
                122.0, 122.0,
                6.0, 8.0,
                678.0, 678.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = Tensor::<f32>::from_buf(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }



    #[test]
    fn test_both_column_major_i32() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 3,
                2, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                5, 7,
                6, 8,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }


    #[test]
    fn test_both_column_major_column_stride_i32() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 3,
                8, 8,
                2, 4, 
                8, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                5,  7,
                122, 122,
                6, 8,
                678, 678,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_batched_i32_strided_rows() {
        let a = Tensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 210, 211, 212,
                213, 214, 215, 216,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = Tensor::<i32>::from_buf(
            vec![
                1, 2,
                3, 4,
                5, 6,
                7, 8,

                11, 12,
                13, 14,
                15, 16,
                17, 18,
            ],
            vec![2, 4, 2]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                50, 60,
                178, 220,

                1270, 1360,
                9818, 10480
            ],
            vec![2, 2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    // #[test]
    // fn test_matmul_broadcast_batch() {
    //     let a = Tensor::<f32>::from_buf(
    //         vec![
    //             1.0, 2.0, 3.0,
    //             4.0, 5.0, 6.0,
    //         ],
    //         vec![1, 2, 3]
    //     ).unwrap();

    //     let b = Tensor::<f32>::from_buf(
    //         vec![
    //             7.0, 8.0,
    //             9.0, 10.0,
    //             11.0, 12.0,

    //             1.0, 2.0,
    //             3.0, 4.0,
    //             5.0, 6.0,
    //         ],
    //         vec![2, 3, 2]
    //     ).unwrap();

    //     let result = a.matmul(&b).unwrap();
        
    //     assert_eq!(*result.shape(), vec![2, 2, 2]);
    //     let expected = Tensor::<f32>::from_buf(
    //         vec![
    //             58.0, 64.0,
    //             139.0, 154.0,

    //             22.0, 28.0,
    //             49.0, 64.0,
    //         ],
    //         vec![2, 2, 2]
    //     ).unwrap();
        
    //     assert_eq!(result, expected);
    // }

    #[test]
    fn test_multibatch_non_contiguous_matmul() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,

                11.0, 12.0, 13.0,
                14.0, 15.0, 16.0,

                21.0, 22.0, 23.0,
                24.0, 25.0, 26.0,

                11.0, 12.0, 13.0,
                14.0, 15.0, 16.0,


                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,

                11.0, 12.0, 13.0,
                14.0, 15.0, 16.0,
            ],
            vec![3, 2, 2, 3]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,

                17.0, 18.0,
                19.0, 20.0,
                21.0, 22.0,

                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,

                17.0, 18.0,
                19.0, 20.0,
                21.0, 22.0,

                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,

                17.0, 18.0,
                19.0, 20.0,
                21.0, 22.0,
            ],
            vec![3, 2, 3, 2]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();

        assert_eq!(*aslice.shape(), vec![2, 2, 2, 3]);
        assert_eq!(*bslice.shape(), vec![2, 2, 3, 2]);
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 2]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                58.0, 64.0,
                139.0, 154.0,

                688.0, 724.0,
                859.0, 904.0,

                58.0, 64.0,
                139.0, 154.0,

                688.0, 724.0,
                859.0, 904.0,
            ],
            vec![2, 2, 2, 2]
        ).unwrap();
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_both_column_major_identity() {
        let a = Tensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                2.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = Tensor::<f32>::from_buf(
            vec![
                1., 0.,
                0., 1.,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = Tensor::<f32>::from_buf(
            vec![
                1., 2.,
                3., 4.,
            ],
            vec![2, 2]
        ).unwrap();

        assert_eq!(result, expected);
    }

}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::core::{primitives::{CudaTensor, Tensor}, tensor::TensorAccess, MetaTensorView, Slice};

    // ============================================================================
    // F32 GEMM TESTS
    // ============================================================================

    #[test]
    fn test_matmul_f32_2d_float() {
        // Test with floating point values
        let a = CudaTensor::<f32>::from_buf(vec![1.5, 2.0, 3.5, 4.0], vec![2, 2]).unwrap();
        let b = CudaTensor::<f32>::from_buf(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[1.5*2.0 + 2.0*3.0, 1.5*1.0 + 2.0*4.0],
        //            [3.5*2.0 + 4.0*3.0, 3.5*1.0 + 4.0*4.0]]
        //         = [[9.0, 9.5],
        //            [19.0, 19.5]]
        let expected = Tensor::<f32>::from_buf(vec![9.0, 9.5, 19.0, 19.5], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f32_large_square() {
        // Test f32 GEMM with larger square matrices
        let size = 10;
        let mut a_data = vec![0.0f32; size * size];
        let mut b_data = vec![0.0f32; size * size];
        
        // Initialize with simple pattern
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = (i + j) as f32;
                b_data[i * size + j] = (i * j + 1) as f32;
            }
        }
        
        let a = CudaTensor::<f32>::from_buf(a_data, vec![size, size]).unwrap();
        let b = CudaTensor::<f32>::from_buf(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
        
        // Verify a specific element manually
        // result[0,0] should be sum of a[0,k] * b[k,0] for k in 0..size
        let expected_00: f32 = (0..size).map(|k| {
            let a_val = k as f32;
            let b_val = (k * 0 + 1) as f32;
            a_val * b_val
        }).sum();
        
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.get(vec![0, 0]).unwrap() - expected_00).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_rectangular_large() {
        // Test f32 GEMM with rectangular matrices (M x K) @ (K x N)
        let m = 15;
        let k = 20;
        let n = 10;
        
        let a_data: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i % 7) as f32 + 0.5).collect();
        
        let a = CudaTensor::<f32>::from_buf(a_data, vec![m, k]).unwrap();
        let b = CudaTensor::<f32>::from_buf(b_data, vec![k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![m, n]);
    }

    #[test]
    fn test_matmul_f32_batched_gemm() {
        // Test batched f32 GEMM (3D tensors)
        let batch = 4;
        let m = 8;
        let k = 6;
        let n = 5;
        
        let a_data: Vec<f32> = (0..batch*m*k).map(|i| (i % 10) as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..batch*k*n).map(|i| (i % 8) as f32 + 1.0).collect();
        
        let a = CudaTensor::<f32>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = CudaTensor::<f32>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f32_precision() {
        // Test that f32 GEMM maintains reasonable precision
        let a = CudaTensor::<f32>::from_buf(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<f32>::from_buf(
            vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed expected values
        // result[0,0] = 0.1*1.1 + 0.2*1.3 + 0.3*1.5 = 0.11 + 0.26 + 0.45 = 0.82
        // result[0,1] = 0.1*1.2 + 0.2*1.4 + 0.3*1.6 = 0.12 + 0.28 + 0.48 = 0.88
        // result[1,0] = 0.4*1.1 + 0.5*1.3 + 0.6*1.5 = 0.44 + 0.65 + 0.90 = 1.99
        // result[1,1] = 0.4*1.2 + 0.5*1.4 + 0.6*1.6 = 0.48 + 0.70 + 0.96 = 2.14
        
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.get(vec![0, 0]).unwrap() - 0.82).abs() < 1e-5);
        assert!((result_cpu.get(vec![0, 1]).unwrap() - 0.88).abs() < 1e-5);
        assert!((result_cpu.get(vec![1, 0]).unwrap() - 1.99).abs() < 1e-5);
        assert!((result_cpu.get(vec![1, 1]).unwrap() - 2.14).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_sliced_rows() {
        // Create a larger tensor and slice rows for matmul
        let full = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get rows 1-3 (shape [2, 3])
        let a = full.slice(0, 1..3).unwrap();
        assert_eq!(*a.shape(), vec![2, 3]);
        
        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 
                3.0, 4.0, 
                5.0, 6.0
            ],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        let expected = Tensor::<f32>::from_buf(
            vec![
                49.0, 64.0,
                76.0, 100.0,
            ],
            vec![2, 2]
        ).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f32_sliced_cols() {
        // Test matmul with column-sliced tensor
        let full = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        // Slice columns 1-3 (shape [2, 2])
        let a = full.slice(1, 1..3).unwrap();
        
        let b = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // a is [[2,3], [6,7]]
        // result[0,0] = 2*1 + 3*3 = 2 + 9 = 11
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.get(vec![0, 0]).unwrap() - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_sliced_batched() {
        // Test batched matmul with sliced tensors
        let full = CudaTensor::<f32>::from_buf(
            (0..120).map(|i| i as f32).collect::<Vec<_>>(),
            vec![5, 4, 6]
        ).unwrap();
        
        // Slice batches 1-4, keep all rows, slice cols 1-5 -> [3, 4, 4]
        let a_step1 = full.slice(0, 1..4).unwrap();
        let a = a_step1.slice(2, 1..5).unwrap();
        
        let b_data: Vec<f32> = (0..72).map(|i| i as f32 * 0.5).collect();
        let b = CudaTensor::<f32>::from_buf(b_data, vec![3, 4, 6]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![3, 4, 6]);
    }

    #[test]
    fn test_matmul_f32_sliced_rhs() {
        // Test where only the right-hand side is sliced
        let a = CudaTensor::<f32>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        let full_b = CudaTensor::<f32>::from_buf(
            (0..20).map(|i| i as f32 * 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice b: rows 1-4, cols 0-2 -> [3, 2]
        let b_step1 = full_b.slice(0, 1..4).unwrap();
        let b = b_step1.slice(1, 0..2).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f32_sliced_identity() {
        // Test sliced tensor with identity-like behavior
        let full = CudaTensor::<f32>::from_buf(
            vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get the identity part [2, 2]
        let identity_step1 = full.slice(0, 1..3).unwrap();
        let identity_slice = identity_step1.slice(1, 1..3).unwrap();
        
        let test_matrix = CudaTensor::<f32>::from_buf(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2]
        ).unwrap();
        
        let result = identity_slice.matmul(&test_matrix).unwrap();
        
        // Should get test_matrix back
        assert_eq!(result.cpu().unwrap(), test_matrix.cpu().unwrap());
    }

    // ============================================================================
    // F64 GEMM TESTS
    // ============================================================================

    #[test]
    fn test_matmul_f64_basic() {
        // Test basic f64 GEMM
        let a = CudaTensor::<f64>::from_buf(
            vec![
                1.0, 2.0, 3.0, 
                4.0, 5.0, 6.0
            ],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<f64>::from_buf(
            vec![
                7.0, 8.0, 
                9.0, 10.0, 
                11.0, 12.0
            ],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[58.0, 64.0], [139.0, 154.0]]
        let expected = Tensor::<f64>::from_buf(vec![58.0, 64.0, 139.0, 154.0], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f64_large_square() {
        // Test f64 GEMM with larger square matrices
        let size = 12;
        let mut a_data = vec![0.0f64; size * size];
        let mut b_data = vec![0.0f64; size * size];
        
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = i as f64 + j as f64 * 0.5;
                b_data[i * size + j] = i as f64 * 0.3 + j as f64;
            }
        }
        
        let a = CudaTensor::<f64>::from_buf(a_data, vec![size, size]).unwrap();
        let b = CudaTensor::<f64>::from_buf(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
    }

    #[test]
    fn test_matmul_f64_rectangular() {
        // Test f64 GEMM with rectangular matrices
        let a = CudaTensor::<f64>::from_buf(
            (0..60).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            vec![10, 6]
        ).unwrap();
        let b = CudaTensor::<f64>::from_buf(
            (0..48).map(|i| i as f64 * 0.2).collect::<Vec<_>>(),
            vec![6, 8]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![10, 8]);
    }

    #[test]
    fn test_matmul_f64_batched() {
        // Test batched f64 GEMM
        let batch = 3;
        let m = 5;
        let k = 4;
        let n = 6;
        
        let a_data: Vec<f64> = (0..batch*m*k).map(|i| i as f64 * 0.25).collect();
        let b_data: Vec<f64> = (0..batch*k*n).map(|i| i as f64 * 0.33).collect();
        
        let a = CudaTensor::<f64>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = CudaTensor::<f64>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f64_high_precision() {
        // Test that f64 GEMM maintains high precision
        let a = CudaTensor::<f64>::from_buf(
            vec![
                0.123456789, 0.987654321,
                0.111111111, 0.999999999,
            ],
            vec![2, 2]
        ).unwrap();
        let b = CudaTensor::<f64>::from_buf(
            vec![
                0.314159265, 0.271828182,
                0.161803398, 0.577215664,
            ],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed with high precision
        // result[0,0] = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398
        let expected_00 = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398;
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.get(vec![0, 0]).unwrap() - expected_00).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_f64_sliced_both() {
        // Test matmul where both tensors are sliced
        let full_a = CudaTensor::<f64>::from_buf(
            (0..24).map(|i| i as f64).collect::<Vec<_>>(),
            vec![4, 6]
        ).unwrap();
        
        let full_b = CudaTensor::<f64>::from_buf(
            (0..20).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice a: rows 1-3, then cols 2-6 -> [2, 4]
        let a_step1 = full_a.slice(0, 1..3).unwrap();
        let a = a_step1.slice(1, 2..6).unwrap();
        // Slice b: rows 1-5, then cols 0-3 -> [4, 3]
        let b_step1 = full_b.slice(0, 1..5).unwrap();
        let b = b_step1.slice(1, 0..3).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 3]);
    }

    #[test]
    fn test_matmul_f64_sliced_lhs() {
        // Test where only the left-hand side is sliced
        let full_a = CudaTensor::<f64>::from_buf(
            (0..30).map(|i| i as f64).collect::<Vec<_>>(),
            vec![5, 6]
        ).unwrap();
        
        // Slice rows 2-4, then cols 1-4 -> [2, 3]
        let a_step1 = full_a.slice(0, 2..4).unwrap();
        let a = a_step1.slice(1, 1..4).unwrap();
        
        let b = CudaTensor::<f64>::from_buf(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f64_slice_then_transpose() {
        // Test slicing followed by transpose for matmul
        let full = CudaTensor::<f64>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4]
        ).unwrap();
        
        // Slice rows 0-2, cols 1-3 -> [2, 2]
        let sliced_step1 = full.slice(0, 0..2).unwrap();
        let sliced = sliced_step1.slice(1, 1..3).unwrap();
        
        // Transpose it -> [2, 2]
        let transposed = sliced.transpose();
        
        let b = CudaTensor::<f64>::from_buf(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2]
        ).unwrap();
        
        let result = transposed.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);

        let result2 = b.matmul(&transposed).unwrap();
        assert_eq!(result2.cpu().unwrap(), result.cpu().unwrap());
        // sliced is [[2,3], [6,7]]
        // transposed is [[2,6], [3,7]]
        // result should be transposed unchanged (identity mult)
        let expected = Tensor::<f64>::from_buf(
            vec![2.0, 6.0, 3.0, 7.0],
            vec![2, 2]
        ).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // ============================================================================
    // INTEGER TYPE TESTS (i32, i64, u32, u64, etc.)
    // ============================================================================

    #[test]
    fn test_matmul_i32_basic() {
        // Test basic i32 matrix multiplication
        let a = CudaTensor::<i32>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<i32>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[58, 64], [139, 154]]
        let expected = Tensor::<i32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i32_square() {
        // Test i32 square matrix multiplication
        let a = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = CudaTensor::<i32>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i32_identity() {
        // Test i32 multiplication with identity matrix
        let a = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let identity = CudaTensor::<i32>::from_buf(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        
        let result = a.matmul(&identity).unwrap();
        
        assert_eq!(result.cpu().unwrap(), a.cpu().unwrap());
    }

    #[test]
    fn test_matmul_i32_batched() {
        // Test batched i32 GEMM (3D tensors)
        let batch = 3;
        let m = 4;
        let k = 5;
        let n = 3;
        
        let a_data: Vec<i32> = (0..batch*m*k).map(|i| (i % 10) as i32).collect();
        let b_data: Vec<i32> = (0..batch*k*n).map(|i| (i % 8) as i32 + 1).collect();
        
        let a = CudaTensor::<i32>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = CudaTensor::<i32>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_i32_rectangular_large() {
        // Test i32 with larger rectangular matrices
        let m = 10;
        let k = 15;
        let n = 8;
        
        let a_data: Vec<i32> = (0..m*k).map(|i| (i % 20) as i32).collect();
        let b_data: Vec<i32> = (0..k*n).map(|i| (i % 15) as i32 + 1).collect();
        
        let a = CudaTensor::<i32>::from_buf(a_data, vec![m, k]).unwrap();
        let b = CudaTensor::<i32>::from_buf(b_data, vec![k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![m, n]);
    }

    #[test]
    fn test_matmul_i64_basic() {
        // Test basic i64 matrix multiplication
        let a = CudaTensor::<i64>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<i64>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i64>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i64_large_values() {
        // Test i64 with large values that would overflow i32
        let a = CudaTensor::<i64>::from_buf(
            vec![1000000, 2000000, 3000000, 4000000],
            vec![2, 2]
        ).unwrap();
        let b = CudaTensor::<i64>::from_buf(
            vec![5000000, 6000000, 7000000, 8000000],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 1000000 * 5000000 + 2000000 * 7000000 = 5e12 + 14e12 = 19e12
        let result_cpu = result.cpu().unwrap();
        assert_eq!(result_cpu.get(vec![0, 0]).unwrap(), 19000000000000i64);
    }

    #[test]
    fn test_matmul_i64_batched() {
        // Test batched i64 matrix multiplication
        let batch = 2;
        let m = 3;
        let k = 4;
        let n = 3;
        
        let a_data: Vec<i64> = (0..batch*m*k).map(|i| i as i64).collect();
        let b_data: Vec<i64> = (0..batch*k*n).map(|i| (i + 1) as i64).collect();
        
        let a = CudaTensor::<i64>::from_buf(a_data, vec![batch, m, k]).unwrap();
        let b = CudaTensor::<i64>::from_buf(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_u32_basic() {
        // Test basic u32 matrix multiplication
        let a = CudaTensor::<u32>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<u32>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u32_square() {
        // Test u32 square matrix multiplication
        let a = CudaTensor::<u32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = CudaTensor::<u32>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u64_basic() {
        // Test basic u64 matrix multiplication
        let a = CudaTensor::<u64>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<u64>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u64>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u64_large_values() {
        // Test u64 with very large values
        let a = CudaTensor::<u64>::from_buf(
            vec![10000000000, 20000000000, 30000000000, 40000000000],
            vec![2, 2]
        ).unwrap();
        let b = CudaTensor::<u64>::from_buf(
            vec![5, 6, 7, 8],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 10000000000 * 5 + 20000000000 * 7 = 50000000000 + 140000000000 = 190000000000
        let result_cpu = result.cpu().unwrap();
        assert_eq!(result_cpu.get(vec![0, 0]).unwrap(), 190000000000u64);
    }

    #[test]
    fn test_matmul_i16_basic() {
        // Test basic i16 matrix multiplication
        let a = CudaTensor::<i16>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<i16>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i16>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u16_basic() {
        // Test basic u16 matrix multiplication
        let a = CudaTensor::<u16>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<u16>::from_buf(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u16>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i8_basic() {
        // Test basic i8 matrix multiplication
        let a = CudaTensor::<i8>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<i8>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        let expected = Tensor::<i8>::from_buf(vec![22, 28, 49, 64], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u8_basic() {
        // Test basic u8 matrix multiplication
        let a = CudaTensor::<u8>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = CudaTensor::<u8>::from_buf(
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u8>::from_buf(vec![22, 28, 49, 64], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_mixed_integer_sizes() {
        // Test that different integer sizes produce correct results
        // Compare i32 vs i64 for same operation
        let a_i32 = CudaTensor::<i32>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b_i32 = CudaTensor::<i32>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let a_i64 = CudaTensor::<i64>::from_buf(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b_i64 = CudaTensor::<i64>::from_buf(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result_i32 = a_i32.matmul(&b_i32).unwrap().cpu().unwrap();
        let result_i64 = a_i64.matmul(&b_i64).unwrap().cpu().unwrap();
        
        // Both should produce [19, 22, 43, 50]
        assert_eq!(result_i32.get(vec![0, 0]).unwrap(), 19);
        assert_eq!(result_i64.get(vec![0, 0]).unwrap(), 19);
        assert_eq!(result_i32.get(vec![1, 1]).unwrap(), 50);
        assert_eq!(result_i64.get(vec![1, 1]).unwrap(), 50);
    }

    // ============================================================================
    // STRIDE TESTS FOR CUDA TENSORS
    // ============================================================================

    // Test stride on left-hand side matrix with f32
    #[test]
    fn test_matmul_cuda_stride_lhs_f32() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                50.0, 60.0,
                178.0, 220.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on left-hand side matrix with i32
    #[test]
    fn test_matmul_cuda_stride_lhs_i32() {
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = CudaTensor::<i32>::from_buf(
            vec![
                1, 2,
                3, 4,
                5, 6,
                7, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                50, 60,
                178, 220,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on right-hand side matrix with f32
    #[test]
    fn test_matmul_cuda_stride_rhs_f32() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                19.0, 22.0, 25.0, 28.0,
                39.0, 46.0, 53.0, 60.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on right-hand side matrix with i32
    #[test]
    fn test_matmul_cuda_stride_rhs_i32() {
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 2,
                3, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22, 25, 28,
                39, 46, 53, 60,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on both left and right-hand side with f32
    #[test]
    fn test_matmul_cuda_stride_both_f32() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on both left and right-hand side with i32
    #[test]
    fn test_matmul_cuda_stride_both_i32() {
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

        #[test]
    fn test_strided_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = CudaTensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu(), expected.cpu());
    }

        #[test]
    fn test_strided_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = CudaTensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu(), expected.cpu());
    }

    #[test]
    fn test_strided_multi_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,

                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0,
                29.0, 210.0, 211.0, 212.0,
                213.0, 214.0, 215.0, 216.0,

                291.0, 292.0, 293.0, 294.0,
                295.0, 296.0, 297.0, 298.0,
                299.0, 2910.0, 2911.0, 2912.0,
                2913.0, 2914.0, 2915.0, 2916.0,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,

                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.,

                1210.0, 1300.0, 1390.0, 1480.0,
                10806.0, 11468.0, 12130.0, 12792.0,

                596020.0, 833710.0, 834880.0, 836050.0,
                5678286.0, 8044418.0, 8053450.0, 8062482.0,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_strided_multi_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,

                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 210, 211, 212,
                213, 214, 215, 216,

                291, 292, 293, 294,
                295, 296, 297, 298,
                299, 2910, 2911, 2912,
                2913, 2914, 2915, 2916,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,

                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = CudaTensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682,

                1210, 1300, 1390, 1480,
                10806, 11468, 12130, 12792,
                596020, 833710, 834880, 836050,
                5678286, 8044418, 8053450, 8062482,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }


    // ============================================================================
    // DOT PRODUCT TESTS (CUDA)
    // ============================================================================

    #[test]
    fn test_cuda_dot_basic() {
        // Basic dot product on CUDA
        let a = CudaTensor::<f32>::from_buf(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = CudaTensor::<f32>::from_buf(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;

        assert!(result.is_scalar());
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.item().unwrap() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cuda_dot_larger_vectors() {
        // Dot product with larger vectors
        let size = 100;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

        let a = CudaTensor::<f32>::from_buf(a_data.clone(), vec![size]).unwrap();
        let b = CudaTensor::<f32>::from_buf(b_data.clone(), vec![size]).unwrap();

        let result = a.dot(&b).unwrap();
        
        // Expected: sum of i * (i*2) for i in 0..100
        let expected: f32 = a_data.iter().zip(&b_data).map(|(x, y)| x * y).sum();

        assert!(result.is_scalar());
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.item().unwrap() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_cuda_dot_orthogonal_vectors() {
        // Dot product of orthogonal unit vectors should be zero
        let a = CudaTensor::<f32>::from_buf(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        let b = CudaTensor::<f32>::from_buf(vec![0.0, 1.0, 0.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        assert!(result.is_scalar());
        let result_cpu = result.cpu().unwrap();
        assert!(result_cpu.item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_both_column_major() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                2.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                5.0, 7.0,
                6.0, 8.0,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = CudaTensor::<f32>::from_buf(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }


    #[test]
    fn test_both_column_major_column_stride() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                8.0, 8.0,
                2.0, 4.0, 
                8.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                5.0,  7.0,
                122.0, 122.0,
                6.0, 8.0,
                678.0, 678.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = CudaTensor::<f32>::from_buf(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }

    #[test]
    fn test_both_column_major_i32() {
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 3,
                2, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                5, 7,
                6, 8,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }


    #[test]
    fn test_both_column_major_column_stride_i32() {
        let a = CudaTensor::<i32>::from_buf(
            vec![
                1, 3,
                8, 8,
                2, 4, 
                8, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let b = CudaTensor::<i32>::from_buf(
            vec![
                5,  7,
                122, 122,
                6, 8,
                678, 678,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_both_column_major_identity() {
        let a = CudaTensor::<f32>::from_buf(
            vec![
                1.0, 3.0,
                2.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = CudaTensor::<f32>::from_buf(
            vec![
                1., 0.,
                0., 1.,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = CudaTensor::<f32>::from_buf(
            vec![
                1., 2.,
                3., 4.,
            ],
            vec![2, 2]
        ).unwrap();

        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }


}

#[cfg(all(test, feature = "remote"))]
mod remote_tests {
    use super::*;
    use std::sync::OnceLock;
    use std::thread;
    use crate::backend::remote::get_backend_default;
    use crate::{
        backend::{remote::{client::RemoteBackend, server::RemoteServer}, Backend},
        core::{
            primitives::{RemoteTensor, TensorBase}, 
            tensor::{TensorAccess, TensorError}, 
            value::TensorValue, 
            MetaTensor, 
            MetaTensorView, 
            Shape, 
            Slice, Tensor
        }
    };
    
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
            let backend = get_backend_default().unwrap();
            
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
        
        Ok(TensorBase::from_parts(tensor_backend, buffer, MetaTensor::new(shape, stride, 0)))
    }

    #[test]
    fn test_matmul_f32_2d_float() {
        // Test with floating point values
        let a = make_remote_tensor(vec![1.5, 2.0, 3.5, 4.0], vec![2, 2]).unwrap();
        let b = make_remote_tensor(vec![2.0, 1.0, 3.0, 4.0], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[1.5*2.0 + 2.0*3.0, 1.5*1.0 + 2.0*4.0],
        //            [3.5*2.0 + 4.0*3.0, 3.5*1.0 + 4.0*4.0]]
        //         = [[9.0, 9.5],
        //            [19.0, 19.5]]
        let expected = Tensor::<f32>::from_buf(vec![9.0, 9.5, 19.0, 19.5], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f32_large_square() {
        // Test f32 GEMM with larger square matrices
        let size = 10;
        let mut a_data = vec![0.0f32; size * size];
        let mut b_data = vec![0.0f32; size * size];
        
        // Initialize with simple pattern
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = (i + j) as f32;
                b_data[i * size + j] = (i * j + 1) as f32;
            }
        }
        
        let a = make_remote_tensor(a_data, vec![size, size]).unwrap();
        let b = make_remote_tensor(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
        
        // Verify a specific element manually
        // result[0,0] should be sum of a[0,k] * b[k,0] for k in 0..size
        let expected_00: f32 = (0..size).map(|k| {
            let a_val = k as f32;
            let b_val = (k * 0 + 1) as f32;
            a_val * b_val
        }).sum();
        
        let result_cpu = result;
        assert!((result_cpu.get(vec![0, 0]).unwrap() - expected_00).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_rectangular_large() {
        // Test f32 GEMM with rectangular matrices (M x K) @ (K x N)
        let m = 15;
        let k = 20;
        let n = 10;
        
        let a_data: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i % 7) as f32 + 0.5).collect();
        
        let a = make_remote_tensor(a_data, vec![m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![m, n]);
    }

    #[test]
    fn test_matmul_f32_batched_gemm() {
        // Test batched f32 GEMM (3D tensors)
        let batch = 4;
        let m = 8;
        let k = 6;
        let n = 5;
        
        let a_data: Vec<f32> = (0..batch*m*k).map(|i| (i % 10) as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..batch*k*n).map(|i| (i % 8) as f32 + 1.0).collect();
        
        let a = make_remote_tensor(a_data, vec![batch, m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f32_precision() {
        // Test that f32 GEMM maintains reasonable precision
        let a = make_remote_tensor(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed expected values
        // result[0,0] = 0.1*1.1 + 0.2*1.3 + 0.3*1.5 = 0.11 + 0.26 + 0.45 = 0.82
        // result[0,1] = 0.1*1.2 + 0.2*1.4 + 0.3*1.6 = 0.12 + 0.28 + 0.48 = 0.88
        // result[1,0] = 0.4*1.1 + 0.5*1.3 + 0.6*1.5 = 0.44 + 0.65 + 0.90 = 1.99
        // result[1,1] = 0.4*1.2 + 0.5*1.4 + 0.6*1.6 = 0.48 + 0.70 + 0.96 = 2.14
        
        let result_cpu = result.cpu().unwrap();
        assert!((result_cpu.get(vec![0, 0]).unwrap() - 0.82) < 1e-5);
        assert!((result_cpu.get(vec![0, 1]).unwrap() - 0.88) < 1e-5);
        assert!((result_cpu.get(vec![1, 0]).unwrap() - 1.99) < 1e-5);
        assert!((result_cpu.get(vec![1, 1]).unwrap() - 2.14) < 1e-5);
    }

    #[test]
    fn test_matmul_f32_sliced_rows() {
        // Create a larger tensor and slice rows for matmul
        let full = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get rows 1-3 (shape [2, 3])
        let a = full.slice(0, 1..3).unwrap();
        assert_eq!(*a.shape(), vec![2, 3]);
        
        let b = make_remote_tensor(
            vec![
                1.0, 2.0, 
                3.0, 4.0, 
                5.0, 6.0
            ],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        let expected = Tensor::<f32>::from_buf(
            vec![
                49.0, 64.0,
                76.0, 100.0,
            ],
            vec![2, 2]
        ).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f32_sliced_cols() {
        // Test matmul with column-sliced tensor
        let full = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        // Slice columns 1-3 (shape [2, 2])
        let a = full.slice(1, 1..3).unwrap();
        
        let b = make_remote_tensor(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // a is [[2,3], [6,7]]
        // result[0,0] = 2*1 + 3*3 = 2 + 9 = 11
        let result_cpu = result;
        assert!((result_cpu.get(vec![0, 0]).unwrap() - 11.0) < 1e-5);
    }

    #[test]
    fn test_matmul_f32_sliced_batched() {
        // Test batched matmul with sliced tensors
        let full = make_remote_tensor(
            (0..120).map(|i| i as f32).collect::<Vec<_>>(),
            vec![5, 4, 6]
        ).unwrap();
        
        // Slice batches 1-4, keep all rows, slice cols 1-5 -> [3, 4, 4]
        let a_step1 = full.slice(0, 1..4).unwrap();
        let a = a_step1.slice(2, 1..5).unwrap();
        
        let b_data: Vec<f32> = (0..72).map(|i| i as f32 * 0.5).collect();
        let b = make_remote_tensor(b_data, vec![3, 4, 6]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![3, 4, 6]);
    }

    #[test]
    fn test_matmul_f32_sliced_rhs() {
        // Test where only the right-hand side is sliced
        let a = make_remote_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        ).unwrap();
        
        let full_b = make_remote_tensor(
            (0..20).map(|i| i as f32 * 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice b: rows 1-4, cols 0-2 -> [3, 2]
        let b_step1 = full_b.slice(0, 1..4).unwrap();
        let b = b_step1.slice(1, 0..2).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f32_sliced_identity() {
        // Test sliced tensor with identity-like behavior
        let full = make_remote_tensor(
            vec![
                0.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 0.0,
            ],
            vec![4, 3]
        ).unwrap();
        
        // Slice to get the identity part [2, 2]
        let identity_step1 = full.slice(0, 1..3).unwrap();
        let identity_slice = identity_step1.slice(1, 1..3).unwrap();
        
        let test_matrix = make_remote_tensor(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2]
        ).unwrap();
        
        let result = identity_slice.matmul(&test_matrix).unwrap();
        
        // Should get test_matrix back
        assert_eq!(result.cpu().unwrap(), test_matrix.cpu().unwrap());
    }

    // ============================================================================
    // F64 GEMM TESTS
    // ============================================================================

    #[test]
    fn test_matmul_f64_basic() {
        // Test basic f64 GEMM
        let a = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 
                4.0, 5.0, 6.0
            ],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![
                7.0, 8.0, 
                9.0, 10.0, 
                11.0, 12.0
            ],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[58.0, 64.0], [139.0, 154.0]]
        let expected = Tensor::<f64>::from_buf(vec![58.0, 64.0, 139.0, 154.0], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_f64_large_square() {
        // Test f64 GEMM with larger square matrices
        let size = 12;
        let mut a_data = vec![0.0f64; size * size];
        let mut b_data = vec![0.0f64; size * size];
        
        for i in 0..size {
            for j in 0..size {
                a_data[i * size + j] = i as f64 + j as f64 * 0.5;
                b_data[i * size + j] = i as f64 * 0.3 + j as f64;
            }
        }
        
        let a = make_remote_tensor(a_data, vec![size, size]).unwrap();
        let b = make_remote_tensor(b_data, vec![size, size]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![size, size]);
    }

    #[test]
    fn test_matmul_f64_rectangular() {
        // Test f64 GEMM with rectangular matrices
        let a = make_remote_tensor(
            (0..60).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            vec![10, 6]
        ).unwrap();
        let b = make_remote_tensor(
            (0..48).map(|i| i as f64 * 0.2).collect::<Vec<_>>(),
            vec![6, 8]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![10, 8]);
    }

    #[test]
    fn test_matmul_f64_batched() {
        // Test batched f64 GEMM
        let batch = 3;
        let m = 5;
        let k = 4;
        let n = 6;
        
        let a_data: Vec<f64> = (0..batch*m*k).map(|i| i as f64 * 0.25).collect();
        let b_data: Vec<f64> = (0..batch*k*n).map(|i| i as f64 * 0.33).collect();
        
        let a = make_remote_tensor(a_data, vec![batch, m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_f64_high_precision() {
        // Test that f64 GEMM maintains high precision
        let a = make_remote_tensor(
            vec![
                0.123456789, 0.987654321,
                0.111111111, 0.999999999,
            ],
            vec![2, 2]
        ).unwrap();
        let b = make_remote_tensor(
            vec![
                0.314159265, 0.271828182,
                0.161803398, 0.577215664,
            ],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        // Manually computed with high precision
        // result[0,0] = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398
        let expected_00 = 0.123456789 * 0.314159265 + 0.987654321 * 0.161803398;
        let result_cpu = result;
        assert!((result_cpu.get(vec![0, 0]).unwrap() - expected_00) < 1e-10);
    }

    #[test]
    fn test_matmul_f64_sliced_both() {
        // Test matmul where both tensors are sliced
        let full_a = make_remote_tensor(
            (0..24).map(|i| i as f64).collect::<Vec<_>>(),
            vec![4, 6]
        ).unwrap();
        
        let full_b = make_remote_tensor(
            (0..20).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
            vec![5, 4]
        ).unwrap();
        
        // Slice a: rows 1-3, then cols 2-6 -> [2, 4]
        let a_step1 = full_a.slice(0, 1..3).unwrap();
        let a = a_step1.slice(1, 2..6).unwrap();
        // Slice b: rows 1-5, then cols 0-3 -> [4, 3]
        let b_step1 = full_b.slice(0, 1..5).unwrap();
        let b = b_step1.slice(1, 0..3).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 3]);
    }

    #[test]
    fn test_matmul_f64_sliced_lhs() {
        // Test where only the left-hand side is sliced
        let full_a = make_remote_tensor(
            (0..30).map(|i| i as f64).collect::<Vec<_>>(),
            vec![5, 6]
        ).unwrap();
        
        // Slice rows 2-4, then cols 1-4 -> [2, 3]
        let a_step1 = full_a.slice(0, 2..4).unwrap();
        let a = a_step1.slice(1, 1..4).unwrap();
        
        let b = make_remote_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
    }

    #[test]
    fn test_matmul_f64_slice_then_transpose() {
        // Test slicing followed by transpose for matmul
        let full = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            vec![3, 4]
        ).unwrap();
        
        // Slice rows 0-2, cols 1-3 -> [2, 2]
        let sliced_step1 = full.slice(0, 0..2).unwrap();
        let sliced = sliced_step1.slice(1, 1..3).unwrap();
        
        // Transpose it -> [2, 2]
        let transposed = sliced.transpose();
        
        let b = make_remote_tensor(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2]
        ).unwrap();
        
        let result = transposed.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![2, 2]);
        
        // sliced is [[2,3], [6,7]]
        // transposed is [[2,6], [3,7]]
        // result should be transposed unchanged (identity mult)
        let result_cpu = result;
        assert_eq!(result_cpu.get(vec![0, 0]).unwrap(), 2.0);
        assert_eq!(result_cpu.get(vec![0, 1]).unwrap(), 6.0);
    }

    // ============================================================================
    // INTEGER TYPE TESTS (i32, i64, u32, u64, etc.)
    // ============================================================================

    #[test]
    fn test_matmul_i32_basic() {
        // Test basic i32 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // Expected: [[58, 64], [139, 154]]
        let expected = Tensor::<i32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i32_square() {
        // Test i32 square matrix multiplication
        let a = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = make_remote_tensor(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i32_identity() {
        // Test i32 multiplication with identity matrix
        let a = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let identity = make_remote_tensor(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        
        let result = a.matmul(&identity).unwrap();
        
        assert_eq!(result.cpu().unwrap(), a.cpu().unwrap());
    }

    #[test]
    fn test_matmul_i32_batched() {
        // Test batched i32 GEMM (3D tensors)
        let batch = 3;
        let m = 4;
        let k = 5;
        let n = 3;
        
        let a_data: Vec<i32> = (0..batch*m*k).map(|i| (i % 10) as i32).collect();
        let b_data: Vec<i32> = (0..batch*k*n).map(|i| (i % 8) as i32 + 1).collect();
        
        let a = make_remote_tensor(a_data, vec![batch, m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_i32_rectangular_large() {
        // Test i32 with larger rectangular matrices
        let m = 10;
        let k = 15;
        let n = 8;
        
        let a_data: Vec<i32> = (0..m*k).map(|i| (i % 20) as i32).collect();
        let b_data: Vec<i32> = (0..k*n).map(|i| (i % 15) as i32 + 1).collect();
        
        let a = make_remote_tensor(a_data, vec![m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![m, n]);
    }

    #[test]
    fn test_matmul_i64_basic() {
        // Test basic i64 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i64>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i64_large_values() {
        // Test i64 with large values that would overflow i32
        let a = make_remote_tensor(
            vec![1000000, 2000000, 3000000, 4000000],
            vec![2, 2]
        ).unwrap();
        let b = make_remote_tensor(
            vec![5000000, 6000000, 7000000, 8000000],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 1000000 * 5000000 + 2000000 * 7000000 = 5e12 + 14e12 = 19e12
        let result_cpu = result;
        assert_eq!(result_cpu.get(vec![0, 0]).unwrap(), 19000000000000i64);
    }

    #[test]
    fn test_matmul_i64_batched() {
        // Test batched i64 matrix multiplication
        let batch = 2;
        let m = 3;
        let k = 4;
        let n = 3;
        
        let a_data: Vec<i64> = (0..batch*m*k).map(|i| i as i64).collect();
        let b_data: Vec<i64> = (0..batch*k*n).map(|i| (i + 1) as i64).collect();
        
        let a = make_remote_tensor(a_data, vec![batch, m, k]).unwrap();
        let b = make_remote_tensor(b_data, vec![batch, k, n]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(*result.shape(), vec![batch, m, n]);
    }

    #[test]
    fn test_matmul_u32_basic() {
        // Test basic u32 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u32>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u32_square() {
        // Test u32 square matrix multiplication
        let a = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b = make_remote_tensor(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u32>::from_buf(vec![19, 22, 43, 50], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u64_basic() {
        // Test basic u64 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u64>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u64_large_values() {
        // Test u64 with very large values
        let a = make_remote_tensor(
            vec![10000000000, 20000000000, 30000000000, 40000000000],
            vec![2, 2]
        ).unwrap();
        let b = make_remote_tensor(
            vec![5, 6, 7, 8],
            vec![2, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 10000000000 * 5 + 20000000000 * 7 = 50000000000 + 140000000000 = 190000000000
        let result_cpu = result;
        assert_eq!(result_cpu.get(vec![0, 0]).unwrap(), 190000000000u64);
    }

    #[test]
    fn test_matmul_i16_basic() {
        // Test basic i16 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i16>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u16_basic() {
        // Test basic u16 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![7, 8, 9, 10, 11, 12],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u16>::from_buf(vec![58, 64, 139, 154], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_i8_basic() {
        // Test basic i8 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        // result[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        let expected = Tensor::<i8>::from_buf(vec![22, 28, 49, 64], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_u8_basic() {
        // Test basic u8 matrix multiplication
        let a = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![2, 3]
        ).unwrap();
        let b = make_remote_tensor(
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2]
        ).unwrap();
        
        let result = a.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<u8>::from_buf(vec![22, 28, 49, 64], vec![2, 2]).unwrap();
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_matmul_mixed_integer_sizes() {
        // Test that different integer sizes produce correct results
        // Compare i32 vs i64 for same operation
        let a_i32 = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b_i32 = make_remote_tensor(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let a_i64 = make_remote_tensor(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let b_i64 = make_remote_tensor(vec![5, 6, 7, 8], vec![2, 2]).unwrap();
        
        let result_i32 = a_i32.matmul(&b_i32).unwrap();
        let result_i64 = a_i64.matmul(&b_i64).unwrap();
        
        // Both should produce [19, 22, 43, 50]
        assert_eq!(result_i32.get(vec![0, 0]).unwrap(), 19);
        assert_eq!(result_i64.get(vec![0, 0]).unwrap(), 19);
        assert_eq!(result_i32.get(vec![1, 1]).unwrap(), 50);
        assert_eq!(result_i64.get(vec![1, 1]).unwrap(), 50);
    }

    // ============================================================================
    // STRIDE TESTS FOR CUDA TENSORS
    // ============================================================================

    // Test stride on left-hand side matrix with f32
    #[test]
    fn test_matmul_cuda_stride_lhs_f32() {
        let a = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = make_remote_tensor(
            vec![
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                50.0, 60.0,
                178.0, 220.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on left-hand side matrix with i32
    #[test]
    fn test_matmul_cuda_stride_lhs_i32() {
        let a = make_remote_tensor(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let b = make_remote_tensor(
            vec![
                1, 2,
                3, 4,
                5, 6,
                7, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let result = aslice.matmul(&b).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                50, 60,
                178, 220,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on right-hand side matrix with f32
    #[test]
    fn test_matmul_cuda_stride_rhs_f32() {
        let a = make_remote_tensor(
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                19.0, 22.0, 25.0, 28.0,
                39.0, 46.0, 53.0, 60.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on right-hand side matrix with i32
    #[test]
    fn test_matmul_cuda_stride_rhs_i32() {
        let a = make_remote_tensor(
            vec![
                1, 2,
                3, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        let result = a.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22, 25, 28,
                39, 46, 53, 60,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on both left and right-hand side with f32
    #[test]
    fn test_matmul_cuda_stride_both_f32() {
        let a = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    // Test stride on both left and right-hand side with i32
    #[test]
    fn test_matmul_cuda_stride_both_i32() {
        let a = make_remote_tensor(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            ],
            vec![4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
            ],
            vec![8, 4]
        ).unwrap();

        let aslice = a.slice(0, Slice::full().step(2)).unwrap();
        let bslice = b.slice(0, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 4]);
        let expected = Tensor::<i32>::from_buf(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,
            ],
            vec![2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

        #[test]
    fn test_strided_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = make_remote_tensor(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }

        #[test]
    fn test_strided_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = make_remote_tensor(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,
            ],
            vec![2, 4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,
                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 8, 4]
        ).unwrap();

        let aslice = a.slice(1, Slice::full().step(2)).unwrap();
        let bslice = b.slice(1, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 4]);
        let expected = make_remote_tensor(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682
            ],
            vec![2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }

    #[test]
    fn test_strided_multi_batched_matmul() {
        // Test batched matmul with strided tensors
        let a = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,

                91.0, 92.0, 93.0, 94.0,
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0,
                913.0, 914.0, 915.0, 916.0,

                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0,
                29.0, 210.0, 211.0, 212.0,
                213.0, 214.0, 215.0, 216.0,

                291.0, 292.0, 293.0, 294.0,
                295.0, 296.0, 297.0, 298.0,
                299.0, 2910.0, 2911.0, 2912.0,
                2913.0, 2914.0, 2915.0, 2916.0,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,

                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0,

                91.0, 92.0, 93.0, 94.0, 
                95.0, 96.0, 97.0, 98.0,
                99.0, 910.0, 911.0, 912.0, 
                913.0, 914.0, 915.0, 916.0,
                917.0, 918.0, 919.0, 920.0, 
                921.0, 922.0, 923.0, 924.0,
                925.0, 926.0, 927.0, 928.0, 
                929.0, 930.0, 931.0, 932.0,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = Tensor::<f32>::from_buf(
            vec![
                170.0, 180.0, 190.0, 200.0,
                586.0, 628.0, 670.0, 712.0,

                189620.,  264510.,  264880.,  265250.,
                1778086., 2518018., 2520850., 2523682.,

                1210.0, 1300.0, 1390.0, 1480.0,
                10806.0, 11468.0, 12130.0, 12792.0,

                596020.0, 833710.0, 834880.0, 836050.0,
                5678286.0, 8044418.0, 8053450.0, 8062482.0,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }

    #[test]
    fn test_strided_multi_batched_matmul_i32() {
        // Test batched matmul with strided tensors
        let a = make_remote_tensor(
            vec![
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,

                91, 92, 93, 94,
                95, 96, 97, 98,
                99, 910, 911, 912,
                913, 914, 915, 916,

                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 210, 211, 212,
                213, 214, 215, 216,

                291, 292, 293, 294,
                295, 296, 297, 298,
                299, 2910, 2911, 2912,
                2913, 2914, 2915, 2916,
            ],
            vec![2, 2, 4, 4]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,

                1, 2, 3, 4, 
                5, 6, 7, 8,
                9, 10, 11, 12, 
                13, 14, 15, 16,
                17, 18, 19, 20, 
                21, 22, 23, 24,
                25, 26, 27, 28, 
                29, 30, 31, 32,

                91, 92, 93, 94, 
                95, 96, 97, 98,
                99, 910, 911, 912, 
                913, 914, 915, 916,
                917, 918, 919, 920, 
                921, 922, 923, 924,
                925, 926, 927, 928, 
                929, 930, 931, 932,
            ],
            vec![2, 2, 8, 4]
        ).unwrap();

        let aslice = a.slice(2, Slice::full().step(2)).unwrap();
        let bslice = b.slice(2, Slice::full().step(2)).unwrap();
        
        let result = aslice.matmul(&bslice).unwrap();
        
        assert_eq!(*result.shape(), vec![2, 2, 2, 4]);
        let expected = make_remote_tensor(
            vec![
                170, 180, 190, 200,
                586, 628, 670, 712,

                189620,  264510,  264880,  265250,
                1778086, 2518018, 2520850, 2523682,

                1210, 1300, 1390, 1480,
                10806, 11468, 12130, 12792,
                596020, 833710, 834880, 836050,
                5678286, 8044418, 8053450, 8062482,
            ],
            vec![2, 2, 2, 4]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }

    // ============================================================================
    // DOT PRODUCT TESTS (CUDA)
    // ============================================================================

    #[test]
    fn test_cuda_dot_basic() {
        // Basic dot product on CUDA
        let a = make_remote_tensor(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = make_remote_tensor(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = a.dot(&b).unwrap();
        let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0;

        assert!(result.is_scalar());
        let result_cpu = result;
        assert!((result_cpu.item().unwrap() - expected) < 1e-5);
    }

    #[test]
    fn test_cuda_dot_larger_vectors() {
        // Dot product with larger vectors
        let size = 100;
        let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

        let a = make_remote_tensor(a_data.clone(), vec![size]).unwrap();
        let b = make_remote_tensor(b_data.clone(), vec![size]).unwrap();

        let result = a.dot(&b).unwrap();
        
        // Expected: sum of i * (i*2) for i in 0..100
        let expected: f32 = a_data.iter().zip(&b_data).map(|(x, y)| x * y).sum();

        assert!(result.is_scalar());
        let result_cpu = result;
        assert!((result_cpu.item().unwrap() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_remote_dot_orthogonal_vectors() {
        // Dot product of orthogonal unit vectors should be zero
        let a = make_remote_tensor(vec![1.0, 0.0, 0.0], vec![3]).unwrap();
        let b = make_remote_tensor(vec![0.0, 1.0, 0.0], vec![3]).unwrap();
        println!("buffer a on device: {:?}", a.buf);
        println!("buffer b on device: {:?}", b.buf);
        let result = a.dot(&b).unwrap();
        println!("Dot Product Result: {:?}", result.cpu().unwrap());
        assert!(result.is_scalar());
        let result_cpu = result;
        assert!(result_cpu.item().unwrap() < 1e-6);
    }

    #[test]
    fn test_both_column_major() {
        let a = make_remote_tensor(
            vec![
                1.0, 3.0,
                2.0, 4.0,
            ],
            vec![2, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                5.0, 7.0,
                6.0, 8.0,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = make_remote_tensor(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }


    #[test]
    fn test_both_column_major_column_stride() {
        let a = make_remote_tensor(
            vec![
                1.0, 3.0,
                8.0, 8.0,
                2.0, 4.0, 
                8.0, 8.0,
            ],
            vec![4, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                5.0,  7.0,
                122.0, 122.0,
                6.0, 8.0,
                678.0, 678.0,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = make_remote_tensor(
            vec![
                19.0, 22.0,
                43.0,  50.0,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected.cpu().unwrap());
    }

    #[test]
    fn test_both_column_major_i32() {
        let a = make_remote_tensor(
            vec![
                1, 3,
                2, 4,
            ],
            vec![2, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                5, 7,
                6, 8,
            ],
            vec![2, 2]
        ).unwrap();

        let result = a.transpose().matmul(&b.transpose()).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }


    #[test]
    fn test_both_column_major_column_stride_i32() {
        let a = make_remote_tensor(
            vec![
                1, 3,
                8, 8,
                2, 4, 
                8, 8,
            ],
            vec![4, 2]
        ).unwrap();

        let b = make_remote_tensor(
            vec![
                5,  7,
                122, 122,
                6, 8,
                678, 678,
            ],
            vec![4, 2]
        ).unwrap();

        let result = a
            .slice(0, Slice::from(0..=2).step(2))
            .unwrap()
            .transpose()
            .matmul(
                &b
                .slice(0, Slice::from(0..=2).step(2))
                .unwrap()
                .transpose()
            ).unwrap();

        let expected = Tensor::<i32>::from_buf(
            vec![
                19, 22,
                43,  50,
            ],
            vec![2, 2]
        ).unwrap();
        
        assert_eq!(result.cpu().unwrap(), expected);
    }
    
}

