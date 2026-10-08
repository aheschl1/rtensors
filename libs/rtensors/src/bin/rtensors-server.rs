//! Standalone remote tensor server.
//!
//! ```text
//! cargo run --release -p rtensors --features remote --bin rtensors-server -- --host 0.0.0.0 --port 7878
//! ```

use std::{net::IpAddr, process::ExitCode};

use rtensors::backend::remote::server::{RemoteServer, ServerConfig};

const USAGE: &str = "\
usage: rtensors-server [options]

  --host <ip>                 address to listen on (default 127.0.0.1)
  --port <port>               port to listen on (default 7878)
  --max-session-bytes <n>     cap on live buffer bytes per connection (default: unlimited)
  --queue-depth <n>           requests buffered per connection before backpressure (default 1024)
  -h, --help                  show this help

There is no authentication or encryption: only listen on trusted networks.";

fn parse() -> Result<(IpAddr, u16, ServerConfig), String> {
    let mut host: IpAddr = "127.0.0.1".parse().unwrap();
    let mut port = 7878u16;
    let mut config = ServerConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--host" => host = value("--host")?.parse().map_err(|e| format!("--host: {e}"))?,
            "--port" => port = value("--port")?.parse().map_err(|e| format!("--port: {e}"))?,
            "--max-session-bytes" => {
                config.max_session_bytes = Some(value("--max-session-bytes")?.parse().map_err(|e| format!("--max-session-bytes: {e}"))?)
            }
            "--queue-depth" => config.queue_depth = value("--queue-depth")?.parse().map_err(|e| format!("--queue-depth: {e}"))?,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok((host, port, config))
}

fn main() -> ExitCode {
    let (host, port, config) = match parse() {
        Ok(parsed) => parsed,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprintln!("{USAGE}");
            return if e.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE };
        }
    };
    eprintln!("rtensors-server listening on {host}:{port} ({config:?})");
    match RemoteServer::new(host, port).with_config(config).serve() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
