
use image::{DynamicImage, ImageBuffer, Rgb};

use crate::{backend::Backend, core::{MetaTensorView, primitives::TensorBase, tensor::TensorError}};

macro_rules! image_to_tensor_u8 {
    ($img:ident, ($($shape:expr),*)) => {
        // the memory is row major [H, W, C]
        // it is vec u8, great, we can just tensor base this
        TensorBase::<u8, B>::from_buf(
            $img.into_raw(),
            ($($shape),*),
        ).expect("Malformated image buffer")   
    };
}

impl<B: Backend> From<ImageBuffer<Rgb<u8>, Vec<u8>>> for TensorBase<u8, B> {
    fn from(value: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
        let height = value.height() as usize;
        let width = value.width() as usize;
        image_to_tensor_u8!(value, (height, width, 3))
    }
}

impl<B: Backend> From<DynamicImage> for TensorBase<u8, B> {
    fn from(value: DynamicImage) -> Self {
        match value {
            DynamicImage::ImageRgb8(image_buffer) => {
                image_buffer.into()
            },
            DynamicImage::ImageRgba8(image_buffer) => {
                let height = image_buffer.height() as usize;
                let width = image_buffer.width() as usize;
                image_to_tensor_u8!(image_buffer, (height, width, 4))
            },
            DynamicImage::ImageLuma8(image_buffer) => {
                let height = image_buffer.height() as usize;
                let width = image_buffer.width() as usize;
                image_to_tensor_u8!(image_buffer, (height, width, 1))
            },
            DynamicImage::ImageLumaA8(image_buffer) => {
                let height = image_buffer.height() as usize;
                let width = image_buffer.width() as usize;
                image_to_tensor_u8!(image_buffer, (height, width, 2))
            },
            _ => {
                panic!("Unsupported image format for conversion to TensorBase<u8, B>");
            }
        }
    }
}

pub trait IntoImage {
    fn into_rgb_image(&self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, TensorError>;
}

impl<B: Backend> IntoImage for TensorBase<u8, B> {
    fn into_rgb_image(&self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, TensorError> {
        let shape = self.shape();
        if self.rank() != 3 || shape[2] != 3 {
            return Err(TensorError::InvalidShape(
                "Tensor must have shape [height, width, 3] to convert to RGB image".to_string(),
            ));
        }

        let height = shape[0];
        let width = shape[1];

        let data = self.backend.dump(&self.buf)?.into_vec();

        ImageBuffer::from_vec(width as u32, height as u32, data).ok_or_else(|| {
            TensorError::ConversionError("Failed to convert tensor data to ImageBuffer".to_string())
        })
    }
}

pub trait ImageLoader {
    fn load_image<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TensorError>
    where
        Self: Sized;
}

impl<B: Backend> ImageLoader for TensorBase<u8, B> {
    fn load_image<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TensorError> {
        let img = image::open(path).map_err(|e| {
            TensorError::ConversionError(format!("Failed to load image: {}", e))
        })?;
        Ok(TensorBase::<u8, B>::from(img))
    }
}

pub trait ImageWriter {
    fn write_image<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TensorError>;
}

impl<B: Backend> ImageWriter for TensorBase<u8, B> {
    fn write_image<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TensorError> {
        let shape = self.shape();
        let rank = self.rank();
        
        // Handle different image formats based on shape
        match rank {
            2 => {
                // [H, W] - grayscale image
                let height = shape[0];
                let width = shape[1];
                let data = self.backend.dump(&self.buf)?.into_vec();
                
                let image = image::GrayImage::from_vec(width as u32, height as u32, data)
                    .ok_or_else(|| {
                        TensorError::ConversionError("Failed to convert tensor data to GrayImage".to_string())
                    })?;
                
                image.save(path).map_err(|e| {
                    TensorError::ConversionError(format!("Failed to save image: {}", e))
                })
            },
            3 => {
                let channels = shape[2];
                match channels {
                    1 => {
                        // [H, W, 1] - grayscale image
                        let height = shape[0];
                        let width = shape[1];
                        let data = self.backend.dump(&self.buf)?.into_vec();
                        
                        let image = image::GrayImage::from_vec(width as u32, height as u32, data)
                            .ok_or_else(|| {
                                TensorError::ConversionError("Failed to convert tensor data to GrayImage".to_string())
                            })?;
                        
                        image.save(path).map_err(|e| {
                            TensorError::ConversionError(format!("Failed to save image: {}", e))
                        })
                    },
                    3 => {
                        // [H, W, 3] - RGB image
                        let image = self.into_rgb_image()?;
                        image.save(path).map_err(|e| {
                            TensorError::ConversionError(format!("Failed to save image: {}", e))
                        })
                    },
                    _ => {
                        Err(TensorError::InvalidShape(
                            format!("Tensor must have 1 or 3 channels, got {} channels", channels)
                        ))
                    }
                }
            },
            _ => {
                Err(TensorError::InvalidShape(
                    "Tensor must have shape [height, width] or [height, width, channels] to save as image".to_string()
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{core::Tensor, io::image::{ImageLoader, ImageWriter, IntoImage}};

    #[test]
    fn test_image_cycle() {
        let width = 64;
        let height = 64;
        let buffer = vec![255u8; width * height * 3]; // White RGB image
        let tensor = Tensor::<u8>::from_buf(buffer, (height, width, 3)).unwrap();
        let image = tensor.into_rgb_image().unwrap();
        let tensor2 = Tensor::<u8>::from(image);
        assert_eq!(tensor, tensor2);
    }
}