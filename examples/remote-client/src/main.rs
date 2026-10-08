use rtensors::{backend::remote::{self, RemoteDevice}, ops::linalg::MatMul};
use std::error::Error;
use rtensors::core::primitives::{Tensor, RemoteTensor};

fn main() -> Result<(), Box<dyn Error>>{
    let backend = remote::try_use_remote_device("127.0.0.1".parse()?, 7878, RemoteDevice::Cpu)?;

    let a = RemoteTensor::<f32>::from_buf_on(&backend, vec![1.0, 2.0, 3.0, 4.0], (2, 2))?;
    let b = Tensor::<f32>::ones((2, 2)).to_remote(&backend)?;
    let mut c = a.matmul(&b)?;
    c += 1.0; 
    let local = c.cpu()?;
    println!("Result: {:?}", local);
    Ok(())
}
