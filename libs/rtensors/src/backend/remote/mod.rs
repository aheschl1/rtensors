pub mod server;
pub mod client;
pub mod protocol;
#[cfg(test)]
mod remote_tests;


pub mod remote {
    use std::net::IpAddr;
    use std::{collections::HashMap, sync::Mutex};
    use std::sync::OnceLock;

    use crate::backend::remote::client::RemoteBackend;

    static REMOTE_BACKENDS: OnceLock<Mutex<HashMap<String, RemoteBackend>>> = OnceLock::new();

    pub fn use_remote_backend(ip: IpAddr, port: u16) -> RemoteBackend {
        let key = format!("{ip}:{port}");

        let map = REMOTE_BACKENDS
            .get_or_init(|| Mutex::new(HashMap::new()));

        let mut guard = map.lock().unwrap();

        guard.entry(key.clone()).or_insert_with(|| {
            let mut backend = RemoteBackend::new_with_address(ip, port).unwrap();
            backend.connect().unwrap();
            backend
        });

        guard.get(&key).unwrap().clone()
    }

    pub fn get_backend_default() -> Option<RemoteBackend> {
        if REMOTE_BACKENDS.get().is_none() {
            use_remote_backend("127.0.0.1".parse().unwrap(), 7878);
        }
        REMOTE_BACKENDS
            .get()
            .and_then(|m| m.lock().ok())
            .and_then(|map| map.values().next().cloned())
    }
}

pub use remote::*;


#[cfg(test)]
mod tests {
    use std::thread;

    use crate::{backend::{remote::{client::RemoteBackend, server::RemoteServer}, Backend}, core::{primitives::TensorBase, tensor::TensorAccess, MetaTensor}};

    #[test]
    fn remote_basic() {
        let server_ip = "127.0.0.1";
        let server_port = 7879;
        let server_addr = format!("{}:{}", server_ip, server_port);
        println!("Server address: {}", server_addr);

        let mut server = RemoteServer::new(server_ip.parse().unwrap(), server_port);
        thread::spawn(move || {
            server.serve().unwrap();
        });
        println!("Server started, waiting for client...");
        thread::sleep(std::time::Duration::from_millis(10));

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