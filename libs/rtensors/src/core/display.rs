use std::fmt;

use crate::{backend::Backend, core::{MetaTensor, TensorView, TensorViewMut, primitives::TensorBase, value::TensorValue}};
use std::fmt::Debug;

const MAX_DISPLAY_SIZE: usize = 10;

/// Calculate total number of rows in the tensor (product of all dimensions except the last)
fn calculate_total_rows(shape: &[usize]) -> usize {
    if shape.len() <= 1 {
        return 1;
    }
    shape[..shape.len() - 1].iter().product()
}

fn format_tensor_recursive<T: TensorValue>(
    data: &[T],
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    indent: usize,
    f: &mut fmt::Formatter<'_>,
    row_indices: &[usize],
    row_counter: &mut usize,
) -> fmt::Result {
    if shape.is_empty() {
        // Scalar (0-dimensional tensor)
        write!(f, "{:?}", data[offset])?;
        return Ok(());
    }

    if shape.len() == 1 {
        // 1D tensor: [a, b, c, ...] - this is a row
        let current_row = *row_counter;
        *row_counter += 1;
        
        // Check if this row should be displayed
        if !row_indices.contains(&current_row) {
            return Ok(());
        }
        
        write!(f, "[")?;
        
        let size = shape[0];
        if size <= MAX_DISPLAY_SIZE {
            // Show all elements
            for i in 0..size {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let idx = offset + (i as isize * strides[0]) as usize;
                write!(f, "{:?}", data[idx])?;
            }
        } else {
            // Show first 5 and last 5 with ellipsis in between
            for i in 0..5 {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let idx = offset + (i as isize * strides[0]) as usize;
                write!(f, "{:?}", data[idx])?;
            }
            write!(f, ", ...")?;
            for i in (size - 5)..size {
                write!(f, ", ")?;
                let idx = offset + (i as isize * strides[0]) as usize;
                write!(f, "{:?}", data[idx])?;
            }
        }
        
        write!(f, "]")?;
        return Ok(());
    }

    // Multi-dimensional tensor
    write!(f, "[")?;
    
    let size = shape[0];
    let mut items_printed = 0;
    let mut ellipsis_printed = false;
    
    // For dimensions > 1, we need to check each sub-tensor to see if it contains any displayable rows
    let rows_per_subtensor = if shape.len() > 1 {
        shape[1..shape.len()-1].iter().product::<usize>()
    } else {
        1
    };
    
    for i in 0..size {
        let start_row = *row_counter;
        let end_row = start_row + rows_per_subtensor;
        
        // Check if any rows in this sub-tensor should be displayed
        let has_displayable_rows = row_indices.iter().any(|&r| r >= start_row && r < end_row);
        
        // Handle truncation at this dimension level
        if size > MAX_DISPLAY_SIZE && i >= 5 && i < size - 5 {
            // Skip this entire sub-tensor
            *row_counter = end_row;
            if !ellipsis_printed {
                if items_printed > 0 {
                    write!(f, ",\n{}", " ".repeat(indent + 1))?;
                }
                write!(f, "...")?;
                ellipsis_printed = true;
                items_printed += 1;
            }
            continue;
        }
        
        if !has_displayable_rows {
            *row_counter = end_row;
            continue;
        }
        
        if items_printed > 0 {
            // Add newline and indentation for subsequent elements
            write!(f, ",\n{}", " ".repeat(indent + 1))?;
            
            // Add extra newline between outer dimension elements for 3D+ tensors
            if shape.len() >= 3 {
                write!(f, "\n{}", " ".repeat(indent + 1))?;
            }
        }
        
        let new_offset = offset + (i as isize * strides[0]) as usize;
        format_tensor_recursive(
            data,
            &shape[1..],
            &strides[1..],
            new_offset,
            indent + 1,
            f,
            row_indices,
            row_counter,
        )?;
        
        items_printed += 1;
    }
    
    write!(f, "]")?;
    Ok(())
}

/// Format tensor for Debug output in PyTorch style
fn format_tensor_data<T: TensorValue, B: Backend>(
    backend: &B,
    buf: &B::Buf<T>,
    meta: &MetaTensor,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    // Dump data from backend (handles both CPU and CUDA)
    let data = match backend.dump(buf) {
        Ok(d) => d,
        Err(_) => {
            // If dump fails, fall back to showing metadata only
            return write!(f, "Tensor<{}>({:?})", std::any::type_name::<T>(), meta.shape());
        }
    };

    // Get dtype string
    let dtype_str = match T::DTYPE {
        crate::core::value::DType::U8 => "u8",
        crate::core::value::DType::I8 => "i8",
        crate::core::value::DType::U16 => "u16",
        crate::core::value::DType::I16 => "i16",
        crate::core::value::DType::U32 => "u32",
        crate::core::value::DType::U128 => "u128",
        crate::core::value::DType::I32 => "i32",
        crate::core::value::DType::U64 => "u64",
        crate::core::value::DType::I64 => "i64",
        crate::core::value::DType::I128 => "i128",
        crate::core::value::DType::F32 => "f32",
        crate::core::value::DType::F64 => "f64",
        crate::core::value::DType::BOOL => "bool",
    };
    
    // Get device string
    let device_type = B::device_type();
    
    write!(f, "tensor(")?;
    
    let total_rows = calculate_total_rows(meta.shape().as_slice());
    
    // Calculate which row indices to display
    let row_indices: Vec<usize> = if total_rows <= MAX_DISPLAY_SIZE {
        (0..total_rows).collect()
    } else {
        // Show first 5 and last 5 rows
        (0..5).chain((total_rows - 5)..total_rows).collect()
    };
    
    let mut row_counter = 0;
    
    format_tensor_recursive(
        &data,
        meta.shape().as_slice(),
        meta.strides().as_slice(),
        meta.offset(),
        6, // indent for "tensor("
        f,
        &row_indices,
        &mut row_counter,
    )?;
    
    // Format device string
    let device_str = match device_type {
        crate::core::primitives::DeviceType::Cpu => "cpu".to_string(),
        #[cfg(feature = "cuda")]
        crate::core::primitives::DeviceType::Cuda(id) => format!("cuda:{}", id),
        #[cfg(feature = "remote")]
        crate::core::primitives::DeviceType::Remote { .. } => "remote".to_string(),
    };
    
    write!(f, ", dtype={}, device={})", dtype_str, device_str)?;
    Ok(())
}

impl<T: TensorValue, B: Backend> Debug for TensorBase<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_data(&self.backend, &self.buf, &self.meta, f)
    }
}

impl<'a, T: TensorValue, B: Backend> Debug for TensorView<'a, T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_data(self.backend, self.buf, &self.meta, f)
    }
}

impl<'a, T: TensorValue, B: Backend> Debug for TensorViewMut<'a, T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor_data(self.backend, self.buf, &self.meta, f)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::Tensor;

    #[test]
    fn test_tensor_debug() {
        println!("=== 2D tensor (15x15) ===");
        let tensor_2d = Tensor::<f32>::ones((15, 15));
        println!("{:?}\n", tensor_2d);
        
        println!("=== 3D tensor (10x5x5) - 50 total rows ===");
        let tensor_3d = Tensor::<f32>::ones((10, 5, 5));
        println!("{:?}\n", tensor_3d);
        
        println!("=== 1D tensor (20) ===");
        let tensor_1d = Tensor::<f32>::ones((20,));
        println!("{:?}\n", tensor_1d);

        println!("=== 1D tensor (20) ===");
        let tensor_1d = Tensor::<f32>::ones((20, 20));
        println!("{:?}\n", tensor_1d);
    }
}