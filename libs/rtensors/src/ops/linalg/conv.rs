use crate::{backend::{BackendMatMul}, core::{primitives::TensorBase, shape_to_stride, tensor::{AsTensor, AsView, AsViewMut, TensorAccess, TensorAccessMut}, value::WeightValue, MetaTensor, MetaTensorView, Shape, TensorView, TensorViewMut}, ops::linalg::{Conv, MatMul}};

#[derive(Clone, Copy, Debug)]
pub enum PaddingType {
    Zeros
}

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig2D {
    pub stride: ConvStrides2D,
    pub padding: Padding2D,
    pub padding_type: PaddingType,
}

impl ConvConfig2D {
    pub fn new(stride: impl Into<ConvStrides2D>, padding: impl Into<Padding2D>, padding_type: PaddingType) -> Self {
        ConvConfig2D {
            stride: stride.into(),
            padding: padding.into(),
            padding_type,
        }
    }

    pub fn stride(mut self, stride: impl Into<ConvStrides2D>) -> Self {
        self.stride = stride.into();
        self
    }

    pub fn padding(mut self, padding: impl Into<Padding2D>) -> Self {
        self.padding = padding.into();
        self
    }

    pub fn padding_type(mut self, padding_type: PaddingType) -> Self {
        self.padding_type = padding_type;
        self
    }
}

impl Default for ConvConfig2D {
    fn default() -> Self {
        ConvConfig2D {
            stride: ConvStrides2D(1, 1),
            padding: Padding2D(0, 0),
            padding_type: PaddingType::Zeros,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Padding2D(usize, usize);

impl From<usize> for Padding2D {
    fn from(value: usize) -> Self {
        Padding2D(value, value)
    }
}

impl From<(usize, usize)> for Padding2D {
    fn from(value: (usize, usize)) -> Self {
        Padding2D(value.0, value.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvStrides2D(usize, usize);

impl From<usize> for ConvStrides2D {
    fn from(value: usize) -> Self {
        ConvStrides2D(value, value)
    }
}

impl From<(usize, usize)> for ConvStrides2D {
    fn from(value: (usize, usize)) -> Self {
        ConvStrides2D(value.0, value.1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig3D {
    pub stride: ConvStrides3D,
    pub padding: Padding3D,
    pub padding_type: PaddingType,
}

impl ConvConfig3D {
    pub fn new(stride: impl Into<ConvStrides3D>, padding: impl Into<Padding3D>, padding_type: PaddingType) -> Self {
        ConvConfig3D {
            stride: stride.into(),
            padding: padding.into(),
            padding_type,
        }
    }

    pub fn stride(mut self, stride: impl Into<ConvStrides3D>) -> Self {
        self.stride = stride.into();
        self
    }

    pub fn padding(mut self, padding: impl Into<Padding3D>) -> Self {
        self.padding = padding.into();
        self
    }

    pub fn padding_type(mut self, padding_type: PaddingType) -> Self {
        self.padding_type = padding_type;
        self
    }
}

impl Default for ConvConfig3D {
    fn default() -> Self {
        ConvConfig3D {
            stride: ConvStrides3D(1, 1, 1),
            padding: Padding3D(0, 0, 0),
            padding_type: PaddingType::Zeros,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Padding3D(usize, usize, usize);

impl From<usize> for Padding3D {
    fn from(value: usize) -> Self {
        Padding3D(value, value, value)
    }
}

impl From<(usize, usize, usize)> for Padding3D {
    fn from(value: (usize, usize, usize)) -> Self {
        Padding3D(value.0, value.1, value.2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConvStrides3D(usize, usize, usize);

impl From<usize> for ConvStrides3D {
    fn from(value: usize) -> Self {
        ConvStrides3D(value, value, value)
    }
}

impl From<(usize, usize, usize)> for ConvStrides3D {
    fn from(value: (usize, usize, usize)) -> Self {
        ConvStrides3D(value.0, value.1, value.2)
    }
}

impl<K, A, T, B> Conv<K, T, B> for A
where
    K: AsView<T, B>,
    A: AsView<T, B>,
    T: WeightValue,
    B: BackendMatMul<T>,
{
    type Output = TensorBase<T, B>;

    fn conv2d(&self, kernel: &K, config: &ConvConfig2D) -> Result<Self::Output, crate::core::tensor::TensorError> {
        let kernel_view = kernel.view();
        let mut self_view = self.view();
        // check dims. expecting [B, C, H, W] for input and [K, C, KH, KW] for kernel C must match
        // if the shape of the input is [C, H, W] expand to [1, C, H, W]
        // if the shape of the input is [H, W] expand to [1, 1, H, W]
        // the kernel has no flexibility, it must be [K, C, KH, KW]
        while self_view.rank() < 4 {
            self_view.unsqueeze_inplace();
        }

        if self_view.rank() != 4 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input must be no more than a 4D tensor [B, C, H, W], got rank {}",
                self_view.rank()
            )));
        }

        // validate channel match
        if self_view.shape()[1] != kernel_view.shape()[1] {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input channels ({}) do not match kernel channels ({})",
                self_view.shape()[1],
                kernel_view.shape()[1]
            )));
        }

        // validate kernel size
        if kernel_view.rank() != 4 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Kernel must be 4D tensor [K, C, KH, KW], got rank {}",
                kernel_view.rank()
            )));
        }

        if config.stride.0 == 0 || config.stride.1 == 0 {
            return Err(crate::core::tensor::TensorError::UnsupportedOperation(
                "Stride values must be greater than 0".to_string(),
            ));
        }

        let out_shape = compute_output_convolution_shape_2d(
            &self_view.shape(),
            &kernel_view.shape(),
            &config.stride,
            &config.padding,
        );

        let out_buffer = self_view.backend.alloc::<T>(out_shape.size())?;
        let out_backend = self_view.backend.clone();
        // out_backend.conv_2d

        // out_backend.apply_conv_2d(
        //     (&self_view.buf, self_view.meta()),
        //     (&kernel_view.buf, kernel_view.meta()), 
        //     &mut out_buffer, 
        //     config
        // )?;

        // temp_conv_slowpath()
        let out_strides = shape_to_stride(&out_shape);
        let out_meta = MetaTensor::new(out_shape, out_strides, 0);
        let mut out_tensor = TensorBase::<T, B>::from_parts(
            out_backend, 
            out_buffer, 
            out_meta, 
            None
        );
        temp_conv2d_fallback(
            self_view,
            kernel_view,
            config,
            &mut out_tensor.view_mut(),
        )?;
        Ok(out_tensor)
    }
    
    fn conv3d(&self, kernel: &K, config: &ConvConfig3D) -> Result<Self::Output, crate::core::tensor::TensorError> {
        let kernel_view = kernel.view();
        let mut self_view = self.view();
        // check dims. expecting [B, C, H, W, D] for input and [K, C, KH, KW, KD] for kernel C must match
        // if the shape of the input is [C, H, W] expand to [1, C, H, W]
        // if the shape of the input is [H, W] expand to [1, 1, H, W]
        // the kernel has no flexibility, it must be [K, C, KH, KW, KD]
        while self_view.rank() < 5 {
            self_view.unsqueeze_inplace();
        }

        if self_view.rank() != 5 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input must be no more than a 5D tensor [B, C, H, W, D], got rank {}",
                self_view.rank()
            )));
        }

        // validate channel match
        if self_view.shape()[1] != kernel_view.shape()[1] {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Input channels ({}) do not match kernel channels ({})",
                self_view.shape()[1],
                kernel_view.shape()[1]
            )));
        }

        // validate kernel size
        if kernel_view.rank() != 5 {
            return Err(crate::core::tensor::TensorError::InvalidShape(format!(
                "Kernel must be 5D tensor [K, C, KH, KW, KD], got rank {}",
                kernel_view.rank()
            )));
        }

        if config.stride.0 == 0 || config.stride.1 == 0 || config.stride.2 == 0 {
            return Err(crate::core::tensor::TensorError::UnsupportedOperation(
                "Stride values must be greater than 0".to_string(),
            ));
        }

        let out_shape = compute_output_convolution_shape_3d(
            &self_view.shape(),
            &kernel_view.shape(),
            &config.stride,
            &config.padding,
        );

        let out_buffer = self_view.backend.alloc::<T>(out_shape.size())?;
        let out_backend = self_view.backend.clone();
        // out_backend.conv_3d

        // out_backend.apply_conv_3d(
        //     (&self_view.buf, self_view.meta()),
        //     (&kernel_view.buf, kernel_view.meta()), 
        //     &mut out_buffer, 
        //     config
        // )?;

        // temp_conv_slowpath()
        let out_strides = shape_to_stride(&out_shape);
        let out_meta = MetaTensor::new(out_shape, out_strides, 0);
        let mut out_tensor = TensorBase::<T, B>::from_parts(out_backend, out_buffer, out_meta, None);
        temp_conv3d_fallback(
            self_view,
            kernel_view,
            config,
            &mut out_tensor.view_mut(),
        )?;
        Ok(out_tensor)
    }
}

#[inline]
fn temp_conv2d_fallback<T: WeightValue, B: BackendMatMul<T>>(
    input: TensorView<T, B>,
    kernel: TensorView<T, B>,
    config: &ConvConfig2D,
    output: &mut TensorViewMut<T, B>,
) -> Result<(), crate::core::tensor::TensorError> {
    // start by making a padded input
    let padding = config.padding;
    let mut _storage = None;

    let input_view = if padding.0 > 0 || padding.1 > 0 {
        _storage = Some(input.pad(
            (0, 0, padding.0, padding.1), 
            &config.padding_type
        )?);
        _storage.as_ref().unwrap().view()
    } else {
        input
    };

    // kernel is shape [K, C, KH, KW] which is [in channels, out channels, kernel height, kernel width]
    let kernel_width = kernel.shape()[3];
    let kernel_height = kernel.shape()[2];

    // now we have a padded input. perform convolution using a naive sliding window approach
    for coord in input_view.iter_coords() {
        // anchor top left, not center
        if coord[2] + kernel_height > input_view.shape()[2] {
            continue;
        }
        if coord[3] + kernel_width > input_view.shape()[3] {
            continue;
        }
        // only process positions that align with stride
        if coord[2] % config.stride.0 != 0 || coord[3] % config.stride.1 != 0 {
            continue;
        }

        // input is shape [B, C, H, W]
        let input_slice = input_view.slice(2, coord[2].. coord[2] + kernel_height)?;
        let input_slice = input_slice.slice(3, coord[3].. coord[3] + kernel_width)?;
        let input_slice = input_slice.slice(0, coord[0])?; // batch slice
        // shape is now [C, KH, KW]
        // now iterate through each output channel
        for oc in 0..kernel.shape()[0] {
            let output_coord = (
                coord[0], // batch
                oc,             // out channel will be iterated over
                coord[2] / config.stride.0,
                coord[3] / config.stride.1,
            );
            let kernel_oc = kernel.slice(0, oc)?;
            // shape is now [C, KH, KW]
            // flatten both and do dot product
            let in_shape = input_slice.size();
            println!("Input slice shape: {:?}", in_shape);
            let input_flat = input_slice.reshape((in_shape,))?;
            println!("Input flat shape: {:?}", input_flat.shape());
            println!("Kernel oc shape: {:?}", kernel_oc.shape());
            let kernel_flat = kernel_oc.reshape((in_shape,))?;
            println!("Kernel flat shape: {:?}", kernel_flat.shape());
            let dot = input_flat.dot(&kernel_flat)?;
            output.set(
                output_coord.clone(), 
                dot.item()?
            ).expect(format!("Failed to set output value at coord {:?}", output_coord).as_str());
        }
    }

    Ok(())

}

#[inline]
fn temp_conv3d_fallback<T: WeightValue, B: BackendMatMul<T>>(
    input: TensorView<T, B>,
    kernel: TensorView<T, B>,
    config: &ConvConfig3D,
    output: &mut TensorViewMut<T, B>,
) -> Result<(), crate::core::tensor::TensorError> {
    // start by making a padded input
    let padding = config.padding;
    let mut _storage = None;

    let input_view = if padding.0 > 0 || padding.1 > 0 || padding.2 > 0 {
        _storage = Some(input.pad(
            (0, 0, padding.0, padding.1, padding.2), 
            &config.padding_type
        )?);
        _storage.as_ref().unwrap().view()
    } else {
        input
    };

    // kernel is shape [K, C, KH, KW, KD] which is [out channels, in channels, kernel height, kernel width, kernel depth]
    let kernel_height = kernel.shape()[2];
    let kernel_width = kernel.shape()[3];
    let kernel_depth = kernel.shape()[4];

    // now we have a padded input. perform convolution using a naive sliding window approach
    // anchor top left, not center
    for coord in input_view.iter_coords() {
        if coord[2] + kernel_height > input_view.shape()[2] {
            continue;
        }
        if coord[3] + kernel_width > input_view.shape()[3] {
            continue;
        }
        if coord[4] + kernel_depth > input_view.shape()[4] {
            continue;
        }
        // only process positions that align with stride
        if coord[2] % config.stride.0 != 0 || coord[3] % config.stride.1 != 0 || coord[4] % config.stride.2 != 0 {
            continue;
        }
        
        // input is shape [B, C, H, W, D]
        let input_slice = input_view.slice(2, coord[2]..coord[2] + kernel_height)?;
        let input_slice = input_slice.slice(3, coord[3]..coord[3] + kernel_width)?;
        let input_slice = input_slice.slice(4, coord[4]..coord[4] + kernel_depth)?;
        let input_slice = input_slice.slice(0, coord[0])?; // batch slice
        // shape is now [C, KH, KW, KD]
        // now iterate through each output channel
        for oc in 0..kernel.shape()[0] {
            let output_coord = (
                coord[0], // batch
                oc,       // out channel will be iterated over
                coord[2] / config.stride.0,
                coord[3] / config.stride.1,
                coord[4] / config.stride.2,
            );
            let kernel_oc = kernel.slice(0, oc)?;
            // shape is now [C, KH, KW, KD]
            // flatten both and do dot product
            let in_shape = input_slice.size();
            let input_flat = input_slice.reshape((in_shape,))?;
            let kernel_flat = kernel_oc.reshape((in_shape,))?;
            let dot = input_flat.dot(&kernel_flat)?;
            output.set(
                output_coord.clone(), 
                dot.item()?
            ).expect(format!("Failed to set output value at coord {:?}", output_coord).as_str());
        }
    }

    Ok(())

}

#[inline]
fn compute_output_convolution_shape_2d(
    input_shape: &Shape,
    kernel_shape: &Shape,
    stride: &ConvStrides2D,
    padding: &Padding2D,
) -> Shape {
    if input_shape.len() != 4 || kernel_shape.len() != 4 {
        panic!("Input and kernel must be 4D tensors");
    }
    let batch_size = input_shape[0];
    let out_channels = kernel_shape[0];
    // input_shape: [B, C, H, W]
    // kernel_shape: [K, C, KH, KW]
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];

    let out_height = (in_height + 2*padding.0 - (kernel_height - 1) - 1) / stride.0 + 1;
    let out_width = (in_width + 2*padding.1 - (kernel_width - 1) - 1) / stride.1 + 1;

    Shape::from((batch_size, out_channels, out_height, out_width))
}

#[inline]
fn compute_output_convolution_shape_3d(
    input_shape: &Shape,
    kernel_shape: &Shape,
    stride: &ConvStrides3D,
    padding: &Padding3D,
) -> Shape {
    if input_shape.len() != 5 || kernel_shape.len() != 5 {
        panic!("Input and kernel must be 5D tensors");
    }
    let batch_size = input_shape[0];
    let out_channels = kernel_shape[0];
    // input_shape: [B, C, H, W, D]
    // kernel_shape: [K, C, KH, KW, KD]
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    let in_depth = input_shape[4];
    
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];
    let kernel_depth = kernel_shape[4];

    let out_height = (in_height + 2*padding.0 - (kernel_height - 1) - 1) / stride.0 + 1;
    let out_width = (in_width + 2*padding.1 - (kernel_width - 1) - 1) / stride.1 + 1;
    let out_depth = (in_depth + 2*padding.2 - (kernel_depth - 1) - 1) / stride.2 + 1;

    Shape::from((batch_size, out_channels, out_height, out_width, out_depth))
}

#[cfg(test)]
mod tests {
    use crate::{core::{Tensor, MetaTensorView, Shape}, ops::linalg::{conv::ConvConfig2D, Conv}};

    fn assert_tensor_close(actual: &Tensor<f32>, expected: &Tensor<f32>, tolerance: f32) {
        assert_eq!(actual.shape(), expected.shape(), "Shape mismatch");
        
        let actual_data: Vec<f32> = actual.buf.iter().map(|&x| x).collect();
        let expected_data: Vec<f32> = expected.buf.iter().map(|&x| x).collect();
        
        for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i, a, e, (a - e).abs()
            );
        }
    }

    #[test]
    fn test_conv2d_basic_no_padding_single_channel() {
        // Test 1: No padding, single batch, single input/output channel
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 
        ], (1, 1, 4, 4)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 
        ], (1, 1, 3, 3)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            -6.0, -6.0, -6.0, -6.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // Test 2: With padding=1, single batch, single input/output channel
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 
        ], (1, 1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 0.0, 4.0, 6.0, 8.0, 3.0, 7.0, 12.0, 
            14.0, 6.0, 0.0, 7.0, 8.0, 9.0, 
        ], (1, 1, 4, 4)).unwrap();

        let config = ConvConfig2D::default().padding(1);
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_multi_batch() {
        // Test 3: Multi-batch (2), single input/output channel, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 
            8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 
        ], (2, 1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            6.0, 8.0, 12.0, 14.0, 14.0, 12.0, 8.0, 6.0, 
        ], (2, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_multi_input_channels() {
        // Test 4: Single batch, multi input channels (2), single output channel, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 2, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            10.0, 12.0, 16.0, 18.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_multi_output_channels() {
        // Test 5: Single batch, single input channel, multi output channels (2), no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 
        ], (1, 1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 
        ], (2, 1, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            6.0, 8.0, 12.0, 14.0, 6.0, 8.0, 12.0, 14.0, 
        ], (1, 2, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_complex_multi_everything_with_padding() {
        // Test 6: Multi batch (2), multi input (2), multi output (2), padding=1
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 
            6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 
        ], (2, 2, 2, 2)).unwrap(); // [batch, in channels, height, width]

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (2, 2, 2, 2)).unwrap(); // [out channels, in channels, KH, KW]

        let expected = Tensor::<f32>::from_buf(vec![
            1.0, 6.0, 4.0, 7.0, 13.0, 6.0, 4.0, 7.0, 4.0, 6.0, 
            13.0, 6.0, 15.0, 31.0, 16.0, 7.0, 18.0, 12.0, 8.0, 3.0, 
            -4.0, 2.0, 5.0, 3.0, -4.0, 2.0, 5.0, 12.0, 14.0, 3.0, 
            12.0, 23.0, 11.0, 2.0, 9.0, 6.0, 
        ], (2, 2, 3, 3)).unwrap();

        let config = ConvConfig2D::default().padding(1);
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Test 7: Single batch/channel, stride=2, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 
            21.0, 22.0, 23.0, 24.0, 25.0, 
        ], (1, 1, 5, 5)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 
        ], (1, 1, 3, 3)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            -6.0, -6.0, -6.0, -6.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default().stride(2);
        let output = input.conv2d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv2d_asymmetric_padding() {
        // Test with different padding values for height and width
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 
        ], (1, 1, 2, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 
            0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default().padding((1, 2));
        let output = input.conv2d(&kernel, &config);
        
        assert!(output.is_ok());
        let output = output.unwrap();
        // Output shape should be (1, 1, 3, 6) with padding (1, 2)
        assert_eq!(*output.shape(), vec![1, 1, 3, 6]);
    }

    #[test]
    fn test_conv2d_asymmetric_stride() {
        // Test with different stride values for height and width
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 
            6.0, 7.0, 8.0, 9.0, 10.0, 
            11.0, 12.0, 13.0, 14.0, 15.0, 
            16.0, 17.0, 18.0, 19.0, 20.0, 
        ], (1, 1, 4, 5)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 
            0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default().stride((2, 1));
        let output = input.conv2d(&kernel, &config);
        
        assert!(output.is_ok());
        let output = output.unwrap();
        // Output shape should be (1, 1, 2, 4) with stride (2, 1)
        assert_eq!(*output.shape(), vec![1, 1, 2, 4]);
    }

    #[test]
    fn test_conv2d_shape_inference_from_3d() {
        // Test that 3D input [C, H, W] is automatically expanded to [1, C, H, W]
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0, 
        ], (1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 
            0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config);
        
        assert!(output.is_ok());
    }

    #[test]
    fn test_conv2d_invalid_channel_mismatch() {
        // Test that mismatched channels between input and kernel produces an error
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0, 
        ], (1, 1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 
            0.0, 1.0,
            1.0, 0.0, 
            0.0, 1.0, 
        ], (1, 2, 2, 2)).unwrap();

        let config = ConvConfig2D::default();
        let output = input.conv2d(&kernel, &config);
        
        assert!(output.is_err());
    }

    #[test]
    fn test_conv2d_zero_stride_error() {
        // Test that stride of 0 produces an error
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0, 
        ], (1, 1, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 
            0.0, 1.0, 
        ], (1, 1, 2, 2)).unwrap();

        let config = ConvConfig2D::default().stride((0, 1));
        let output = input.conv2d(&kernel, &config);
        
        assert!(output.is_err());
    }

    // ========== Conv3D Tests ==========

    #[test]
    fn test_conv3d_basic_no_padding_single_channel() {
        // Test 1: No padding, single batch, single input/output channel
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 
        ], (1, 1, 3, 3, 3)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            15.0, 17.0, 21.0, 23.0, 33.0, 35.0, 39.0, 41.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default();
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_with_padding() {
        // Test 2: With padding=1, single batch, single input/output channel
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0, 6.0, 
            13.0, 6.0, 15.0, 31.0, 16.0, 7.0, 18.0, 12.0, 5.0, 6.0, 
            0.0, 7.0, 13.0, 6.0, 0.0, 7.0, 8.0, 
        ], (1, 1, 3, 3, 3)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default().padding(1);
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_multi_batch() {
        // Test 3: Multi-batch (2), single input/output channel, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 
            6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 
        ], (2, 1, 2, 2, 2)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            31.0, 23.0, 
        ], (2, 1, 1, 1, 1)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default();
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_multi_input_channels() {
        // Test 4: Single batch, multi input channels (2), single output channel, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 2, 2, 2, 2)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 
        ], (1, 2, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            37.0, 
        ], (1, 1, 1, 1, 1)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default();
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_multi_output_channels() {
        // Test 5: Single batch, single input channel, multi output channels (2), no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 
        ], (2, 1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            31.0, 18.0, 
        ], (1, 2, 1, 1, 1)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default();
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_with_stride() {
        // Test 6: Single batch/channel, stride=2, no padding
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 
            41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 
            51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 
            61.0, 62.0, 63.0, 64.0, 
        ], (1, 1, 4, 4, 4)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            85.0, 97.0, 133.0, 145.0, 277.0, 289.0, 325.0, 337.0, 
        ], (1, 1, 2, 2, 2)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default().stride(2);
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }

    #[test]
    fn test_conv3d_complex_multi_everything_with_padding() {
        // Test 7: Multi batch (2), multi input (2), multi output (2), padding=1
        // Validated against PyTorch
        let input = Tensor::<f32>::from_buf(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 
            4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 
            5.0, 4.0, 3.0, 2.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 
            2.0, 1.0, 
        ], (2, 2, 2, 2, 2)).unwrap();

        let kernel = Tensor::<f32>::from_buf(vec![
            1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 
            -1.0, 1.0, 
        ], (2, 2, 2, 2, 2)).unwrap();

        let expected = Tensor::<f32>::from_buf(vec![
            2.0, 6.0, 5.0, 7.0, 19.0, 12.0, 7.0, 13.0, 5.0, 9.0, 
            22.0, 11.0, 25.0, 50.0, 25.0, 12.0, 28.0, 18.0, 11.0, 8.0, 
            -6.0, 10.0, 15.0, 5.0, -7.0, 7.0, 17.0, 2.0, 2.0, -1.0, 
            5.0, 8.0, 2.0, 3.0, 8.0, 5.0, 9.0, 12.0, 1.0, 19.0, 
            34.0, 15.0, 6.0, 22.0, 18.0, 11.0, 18.0, 6.0, 14.0, 34.0, 
            21.0, -1.0, 14.0, 17.0, 8.0, 24.0, 15.0, 23.0, 41.0, 18.0, 
            13.0, 17.0, 5.0, 21.0, 18.0, -1.0, 15.0, 30.0, 15.0, -2.0, 
            12.0, 12.0, 9.0, 2.0, -4.0, 0.0, 5.0, 5.0, -3.0, 3.0, 
            3.0, 8.0, 8.0, 1.0, 15.0, 22.0, 8.0, 7.0, 12.0, 5.0, 
            21.0, 28.0, 9.0, 21.0, 46.0, 25.0, 4.0, 18.0, 12.0, 9.0, 
            12.0, 4.0, 6.0, 16.0, 9.0, 1.0, 6.0, 3.0, 
        ], (2, 2, 3, 3, 3)).unwrap();

        let config = crate::ops::linalg::conv::ConvConfig3D::default().padding(1);
        let output = input.conv3d(&kernel, &config).unwrap();

        assert_tensor_close(&output, &expected, 1e-5);
    }
}