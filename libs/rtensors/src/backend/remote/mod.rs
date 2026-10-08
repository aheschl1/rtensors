pub mod server;
pub mod client;
pub mod protocol;
pub use protocol::RemoteDevice;
#[cfg(test)]
mod remote_tests;


pub mod remote {
    use std::net::IpAddr;
    use std::{collections::HashMap, sync::Mutex};
    use std::sync::OnceLock;

    use crate::backend::remote::client::RemoteBackend;
    use crate::backend::remote::protocol::RemoteDevice;
    use crate::core::tensor::TensorError;

    type Key = (IpAddr, u16, RemoteDevice);

    /// One shared connection per server and device.
    static SHARED: OnceLock<Mutex<HashMap<Key, RemoteBackend>>> = OnceLock::new();
    /// What `RemoteBackend::new()` (and so `RemoteTensor::zeros` etc.) uses.
    static DEFAULT: Mutex<Option<RemoteBackend>> = Mutex::new(None);

    /// Returns the shared connection to `ip:port` with a session on `device`, connecting on
    /// first use (or again if the cached connection was lost). The first shared connection also
    /// becomes the default backend if none is set, whatever its device.
    pub fn try_use_remote_device(ip: IpAddr, port: u16, device: RemoteDevice) -> Result<RemoteBackend, TensorError> {
        let map = SHARED.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().unwrap();
        if let Some(backend) = guard.get(&(ip, port, device)) {
            // Reconnect transparently if the server went away (e.g. was restarted).
            if !backend.is_closed() {
                return Ok(backend.clone());
            }
        }
        let backend = RemoteBackend::connect_device(ip, port, device)?;
        guard.insert((ip, port, device), backend.clone());
        let mut default = DEFAULT.lock().unwrap();
        if default.as_ref().is_none_or(|d| d.is_closed()) {
            *default = Some(backend.clone());
        }
        Ok(backend)
    }

    /// [`try_use_remote_device`] with a CPU session.
    pub fn try_use_remote_backend(ip: IpAddr, port: u16) -> Result<RemoteBackend, TensorError> {
        try_use_remote_device(ip, port, RemoteDevice::Cpu)
    }

    /// Panicking version of [`try_use_remote_backend`].
    pub fn use_remote_backend(ip: IpAddr, port: u16) -> RemoteBackend {
        try_use_remote_backend(ip, port).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Makes `backend` the one `RemoteBackend::new()` returns, replacing any previous default.
    pub fn set_default_backend(backend: RemoteBackend) {
        *DEFAULT.lock().unwrap() = Some(backend);
    }

    /// The current default backend, without connecting anywhere.
    pub fn default_backend() -> Option<RemoteBackend> {
        DEFAULT.lock().unwrap().clone()
    }

    /// The default backend, falling back to a CPU session on 127.0.0.1:7878.
    pub fn get_backend_default() -> Option<RemoteBackend> {
        default_backend().or_else(|| try_use_remote_backend("127.0.0.1".parse().unwrap(), 7878).ok())
    }
}

pub use remote::*;


#[cfg(test)]
mod tests {

    use crate::{backend::{remote::client::RemoteBackend, Backend}, core::{primitives::TensorBase, tensor::TensorAccess, MetaTensor}};

    #[test]
    fn remote_basic() {
        let server_ip = "127.0.0.1";
        let server_port = 7880;
        let server_addr = format!("{}:{}", server_ip, server_port);
        println!("Server address: {}", server_addr);

        crate::backend::remote::server::ensure_test_server(server_ip.parse().unwrap(), server_port);

        let mut backend = RemoteBackend::new_with_address(server_ip.parse().unwrap(), server_port).unwrap();
        backend.connect().unwrap();
        let mut buffer = backend.alloc::<f32>(100).unwrap();
        println!("Allocated remote buffer: {:?}", buffer);
        backend.copy_from_slice(&mut buffer, vec![1.0f32; 100].as_slice()).unwrap();
        println!("Copied data to remote buffer");

        println!("Reading data back from remote buffer...");
        let res = backend.read(&buffer, 0).unwrap();
        println!("Read data from remote buffer: {:?}", res);

        let mut tensor = TensorBase::from_parts(
            backend, 
            buffer,
            MetaTensor::new(vec![10, 10], vec![10, 1], 0),
            None,
        );


        println!("Created remote tensor: {:?}", tensor);

        tensor += 1.0;

        let x = tensor.get((0, 0)).unwrap();
        assert_eq!(x, 2.0);

        tensor *= 2.0;
        assert_eq!(tensor.get((0, 0)).unwrap(), 4.0);
        
    }
}