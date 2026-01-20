use std::{env, path::PathBuf, process::Command};

fn main() {
    #[cfg(feature = "cuda")]
    {
        detect_and_warn_cuda_version();
        build_cuda_kernels();
    }

    build_openblas();

    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "cuda")]
fn detect_and_warn_cuda_version() {
    let cuda_version = detect_cuda_version();
    
    if let Some(version) = cuda_version {
        let major = version / 1000;
        let minor = (version % 1000) / 10;
        
        println!("cargo:warning=Detected CUDA version: {}.{}", major, minor);
        
        // Check if user might need a different cudarc feature
        if version >= 13000 {
            println!("cargo:warning=");
            println!("cargo:warning=Note: You have CUDA 13.x installed.");
            println!("cargo:warning=If you encounter runtime errors, you may need to change Cargo.toml:");
            println!("cargo:warning=  cudarc = {{ version = \"0.18.1\", features = [\"cuda-13000\"], optional = true }}");
            println!("cargo:warning=");
            println!("cargo:warning=Currently using cuda-12000 feature (backward compatible)");
        } else if version >= 12000 {
            println!("cargo:warning=✓ Using CUDA 12.x - cudarc configured correctly");
        } else {
            println!("cargo:warning=  CUDA version {}.{} may not be fully supported.", major, minor);
            println!("cargo:warning=Recommended: CUDA 12.0 or later");
        }
    } else {
        println!("cargo:warning=Could not auto-detect CUDA version");
        println!("cargo:warning=Assuming CUDA 12.x (default)");
    }
}

#[cfg(feature = "cuda")]
fn detect_cuda_version() -> Option<u32> {
    use std::process::Command;
    
    // Try to get CUDA version from nvcc
    if let Some(nvcc) = find_nvcc() {
        if let Ok(output) = Command::new(&nvcc).arg("--version").output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse version from output like "release 12.0, V12.0.140"
                if let Some(line) = stdout.lines().find(|l| l.contains("release")) {
                    // Extract version number (e.g., "12.0")
                    if let Some(version_str) = line.split("release").nth(1) {
                        if let Some(version_part) = version_str.trim().split(',').next() {
                            // Parse "12.0" into 12000
                            let parts: Vec<&str> = version_part.split('.').collect();
                            if parts.len() >= 2 {
                                if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                                    return Some(major * 1000 + minor * 10);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    None
}

fn build_openblas() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_dir = out_dir.join("OpenBLAS-0.3.30");
    let zip_path = out_dir.join("OpenBLAS-0.3.30.zip");
    let install_dir = out_dir.join("openblas-install");
    let lib_dir = install_dir.join("lib");
    let include_dir = install_dir.join("include");

    let lib_openblas = lib_dir.join("libopenblas.a");
    let cblas_h = include_dir.join("cblas.h");

    if lib_openblas.exists() && cblas_h.exists() {
        println!("cargo:warning=Using cached OpenBLAS at {}", install_dir.display());
    } else {
        let url = "https://github.com/aheschl1/rtensors/releases/download/blas/OpenBLAS-0.3.30.zip";

        if !zip_path.exists() {
            assert!(
                Command::new("curl")
                    .args(["-L", "-o"])
                    .arg(&zip_path)
                    .arg(url)
                    .status()
                    .expect("curl failed")
                    .success(),
                "failed to download OpenBLAS"
            );
        }

        if !src_dir.exists() {
            assert!(
                Command::new("unzip")
                    .args(["-q"])
                    .arg(&zip_path)
                    .arg("-d")
                    .arg(&out_dir)
                    .status()
                    .expect("unzip failed")
                    .success(),
                "failed to extract OpenBLAS"
            );
        }

        let jobs = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_else(|_| "4".into());

        let make = |extra: &[&str]| {
            let mut cmd = Command::new("make");
            cmd.current_dir(&src_dir)
                .arg(format!("-j{}", jobs))
                .args([
                    "NO_SHARED=1",
                    "NO_LAPACK=1",
                    "NOFORTRAN=1",
                    "USE_OPENMP=0",
                    "DYNAMIC_ARCH=0",
                ])
                .args(extra);
            cmd
        };

        if !make(&["libs"]).status().unwrap().success() {
            assert!(
                make(&["TARGET=GENERIC", "libs"])
                    .status()
                    .unwrap()
                    .success(),
                "OpenBLAS build failed"
            );
        }

        std::fs::create_dir_all(&lib_dir).unwrap();
        std::fs::create_dir_all(&include_dir).unwrap();

        let built_lib = src_dir.join("libopenblas.a");
        assert!(built_lib.exists(), "libopenblas.a not produced");
        std::fs::copy(&built_lib, &lib_openblas).unwrap();

        for entry in std::fs::read_dir(&src_dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("h") {
                std::fs::copy(&p, include_dir.join(p.file_name().unwrap())).unwrap();
            }
        }
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");

    generate_openblas_bindings(&include_dir);
}

fn generate_openblas_bindings(include_dir: &PathBuf) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cblas = include_dir.join("cblas.h");

    let bindings = bindgen::Builder::default()
        .header(cblas.to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("openblas_.*")
        .allowlist_function("goto_.*")
        .allowlist_function("cblas_.*")
        .allowlist_type("CBLAS_.*")
        .allowlist_type("blasint")
        .allowlist_var("OPENBLAS_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_partialeq(true)
        .generate()
        .expect("Unable to generate OpenBLAS bindings");

    bindings
        .write_to_file(out_dir.join("openblas_bindings.rs"))
        .unwrap();
}

#[cfg(feature = "cuda")]
fn find_nvcc() -> Option<PathBuf> {
    use std::process::Command;
    
    // 1. Check NVCC_PATH environment variable
    if let Ok(nvcc_path) = env::var("NVCC_PATH") {
        let path = PathBuf::from(nvcc_path);
        if path.exists() {
            println!("cargo:warning=Using nvcc from NVCC_PATH: {}", path.display());
            return Some(path);
        } else {
            println!("cargo:warning=NVCC_PATH is set but file doesn't exist: {}", path.display());
        }
    }
    
    // 2. Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_path = PathBuf::from(&cuda_home).join("bin/nvcc");
        if nvcc_path.exists() {
            println!("cargo:warning=Using nvcc from CUDA_HOME: {}", nvcc_path.display());
            return Some(nvcc_path);
        }
    }
    
    // 3. Try to find nvcc in PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                let path = PathBuf::from(path_str);
                println!("cargo:warning=Found nvcc in PATH: {}", path.display());
                return Some(path);
            }
        }
    }
    
    // 4. Common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/usr/local/cuda-11.0/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
    ];
    
    for path in &common_paths {
        let nvcc_path = PathBuf::from(path);
        if nvcc_path.exists() {
            println!("cargo:warning=Found nvcc at: {}", nvcc_path.display());
            return Some(nvcc_path);
        }
    }
    
    None
}

#[cfg(feature = "cuda")]
fn find_cuda_lib_path() -> Option<PathBuf> {
    // 1. Check CUDA_LIB_DIR environment variable
    if let Ok(cuda_lib_dir) = env::var("CUDA_LIB_DIR") {
        let path = PathBuf::from(cuda_lib_dir);
        if path.exists() {
            println!("cargo:warning=Using CUDA libraries from CUDA_LIB_DIR: {}", path.display());
            return Some(path);
        } else {
            println!("cargo:warning=CUDA_LIB_DIR is set but directory doesn't exist: {}", path.display());
        }
    }
    
    // 2. Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        for subdir in &["lib64", "lib"] {
            let lib_path = PathBuf::from(&cuda_home).join(subdir);
            if lib_path.join("libcudart.so").exists() {
                println!("cargo:warning=Using CUDA libraries from CUDA_HOME: {}", lib_path.display());
                return Some(lib_path);
            }
        }
    }
    
    // 3. Common CUDA library paths
    let common_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/opt/cuda/lib64",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.0/lib64",
        "/usr/local/cuda-13.0/lib64",
    ];
    
    for path in &common_paths {
        let lib_path = PathBuf::from(path);
        // Check if libcudart.so or libcudart.so.* exists in this path
        if lib_path.join("libcudart.so").exists() || 
           lib_path.read_dir().ok()
               .and_then(|entries| entries
                   .filter_map(|e| e.ok())
                   .find(|e| e.file_name().to_str().unwrap_or("").starts_with("libcudart.so")))
               .is_some() {
            println!("cargo:warning=Found CUDA libraries at: {}", lib_path.display());
            return Some(lib_path);
        }
    }
    
    None
}

#[cfg(feature = "cuda")]
fn find_cuda_kernel_files() -> Vec<PathBuf> {
    use std::fs;
    
    let kernels_dir = PathBuf::from("cuda/kernels");
    let mut cu_files = Vec::new();
    
    // Recursively find all .cu files, excluding legacy folder
    fn find_cu_files_recursive(dir: &PathBuf, cu_files: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                
                // Skip legacy directory
                if path.is_dir() {
                    if path.file_name().and_then(|s| s.to_str()) != Some("legacy") {
                        find_cu_files_recursive(&path, cu_files);
                    }
                } else if path.extension().and_then(|s| s.to_str()) == Some("cu") {
                    cu_files.push(path);
                }
            }
        }
    }
    
    find_cu_files_recursive(&kernels_dir, &mut cu_files);
    cu_files.sort();
    cu_files
}

#[cfg(feature = "cuda")]
fn build_cuda_kernels() {
    use std::process::Command;
    use std::env;
    
    // Find nvcc
    let nvcc = find_nvcc().expect(
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════╗\n\
        ║                    NVCC NOT FOUND                                ║\n\
        ╚══════════════════════════════════════════════════════════════════╝\n\
        \n\
        Could not find nvcc (NVIDIA CUDA Compiler).\n\
        \n\
        Please install the CUDA toolkit or specify the path using one of:\n\
        - Set NVCC_PATH environment variable to the full path to nvcc\n\
        - Set CUDA_HOME environment variable to your CUDA installation directory\n\
        - Add nvcc to your PATH\n\
        \n\
        Download CUDA toolkit from: https://developer.nvidia.com/cuda-downloads\n\
        "
    );
    
    println!("cargo:warning=Found nvcc at: {}", nvcc.display());
    
    let cuda_lib_path = find_cuda_lib_path().expect(
        "\n\n\
        ╔══════════════════════════════════════════════════════════════════╗\n\
        ║                CUDA LIBRARIES NOT FOUND                          ║\n\
        ╚══════════════════════════════════════════════════════════════════╝\n\
        \n\
        Could not find CUDA runtime libraries (libcudart.so).\n\
        \n\
        Please install the CUDA toolkit or specify the path using:\n\
        - Set CUDA_LIB_DIR environment variable to the directory containing libcudart.so\n\
        - Set CUDA_HOME environment variable to your CUDA installation directory\n\
        \n\
        Download CUDA toolkit from: https://developer.nvidia.com/cuda-downloads\n\
        "
    );
    
    println!("cargo:warning=Found CUDA libraries at: {}", cuda_lib_path.display());

    let kernel_files = find_cuda_kernel_files();
    
    if kernel_files.is_empty() {
        panic!("No CUDA kernel files (.cu) found in cuda/kernels/");
    }
    
    println!("cargo:warning=Found {} CUDA kernel file(s)", kernel_files.len());
    
    println!("cargo:rerun-if-changed=cuda/include/kernels.h");
    println!("cargo:rerun-if-changed=cuda/include/common.h");
    println!("cargo:rerun-if-changed=cuda/include/scalar.h");
    println!("cargo:rerun-if-changed=cuda/include/binary.h");
    // println!("cargo:rerun-if-changed=cuda/include/elementwise.h");
    for kernel_file in &kernel_files {
        println!("cargo:rerun-if-changed={}", kernel_file.display());
    }
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Get target architecture(s) from environment variable or use default
    let archs = if let Ok(cuda_archs) = env::var("CUDA_ARCHS") {
        println!("cargo:warning=Using CUDA architectures from CUDA_ARCHS: {}", cuda_archs);
        cuda_archs.split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<String>>()
    } else {
        // Default to sm_86 (RTX 30xx series)
        // Users can override with: export CUDA_ARCHS="sm_75,sm_80,sm_86"
        println!("cargo:warning=Using default CUDA architecture: sm_86");
        println!("cargo:warning=To target multiple architectures, set CUDA_ARCHS env var (e.g., CUDA_ARCHS=\"sm_75,sm_80,sm_86\")");
        vec!["sm_86".to_string()]
    };

    // Build all .cu files to object files (for static linking)
    let mut object_files = Vec::new();
    
    for kernel_file in &kernel_files {
        
        // Create a unique object file name by including the parent directory
        // This prevents conflicts when multiple .cu files have the same name
        // (e.g., scalar/contiguous.cu and unary/contiguous.cu)
        let parent_name = kernel_file.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        
        let file_stem = kernel_file.file_stem()
            .and_then(|s| s.to_str())
            .expect("Invalid kernel filename");
        
        let obj_file = out_dir.join(format!("{}_{}.o", parent_name, file_stem));
        
        println!("cargo:warning=Compiling {} to object file...", kernel_file.display());

        // Determine if this is a release build
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let is_release = profile == "release";

        // Build compilation command with multiple architectures if specified
        let mut compile_cmd = Command::new(&nvcc);
        compile_cmd
            .arg("-c")  // Compile only (create object file)
            .arg("-o")
            .arg(&obj_file)
            .arg(kernel_file)
            .arg("-I")
            .arg("cuda/include")
            .arg("--compiler-options")
            .arg("-fPIC");
        
        // Add optimization flags for release builds
        if is_release {
            compile_cmd
                .arg("-O3")                             // Maximum optimization
                .arg("--use_fast_math")                 // Fast math (trade precision for speed)
                .arg("--extra-device-vectorization")    // Enable extra vectorization
                .arg("--fmad=true")                     // Fused multiply-add
                .arg("--prec-div=false")                // Use fast division
                .arg("--prec-sqrt=false")               // Use fast sqrt
                .arg("--maxrregcount=128")              // Increase register usage for better perf
                .arg("-Xptxas=-O3");                    // Maximum PTX optimization
        } else {
            compile_cmd
                .arg("-g")                     // Debug info for debug builds
                .arg("-G");                    // Generate debug info for device code
        }
        
        // Add architecture flags
        for arch in &archs {
            compile_cmd.arg(format!("-arch={}", arch));
        }

        let nvcc_compile_status = compile_cmd.status().unwrap();

        assert!(
            nvcc_compile_status.success(),
            "\n\n\
            ╔══════════════════════════════════════════════════════════════════╗\n\
            ║              CUDA KERNEL COMPILATION FAILED                      ║\n\
            ╚══════════════════════════════════════════════════════════════════╝\n\
            \n\
            Failed to compile: {}\n\
            \n\
            This may be due to:\n\
            - Unsupported CUDA architecture (current: {:?})\n\
            - Incompatible CUDA version\n\
            - Syntax errors in CUDA kernel code\n\
            \n\
            To specify a different architecture, set CUDA_ARCHS environment variable.\n\
            For example: export CUDA_ARCHS=\"sm_75,sm_80,sm_86\"\n\
            ", 
            kernel_file.display(),
            archs
        );
        
        object_files.push(obj_file);
    }

    if !object_files.is_empty() {
        let kernels_lib = out_dir.join("libcudakernels.a");
        
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("crus").arg(&kernels_lib);
        
        for obj in &object_files {
            ar_cmd.arg(obj);
        }
        
        let ar_status = ar_cmd.status().unwrap();

        assert!(
            ar_status.success(),
            "Failed to create static library from CUDA object files"
        );
        
        println!("cargo:warning=Created CUDA kernels library at: {}", kernels_lib.display());
    }

    // Tell cargo where to find our library and link it
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cudakernels");
    
    // Add CUDA library search path and link against CUDA libraries
    println!("cargo:rustc-link-search=native={}", cuda_lib_path.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    
    // Link against C++ standard library (needed for CUDA C++ code)
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("cuda/include/kernels.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // we use "no_copy" and "no_debug" here because we don't know if we can safely generate them for our structs in C code (they may contain raw pointers)
        .no_copy("*")
        .no_debug("*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // we need to make modifications to the generated code
    let generated_bindings = bindings.to_string();

    // Regex to find raw pointers to float and replace them with CudaSlice<f32>
    // You can copy this regex to add/modify other types of pointers, for example "*mut i32"
    // let pointer_regex = Regex::new(r"\*mut f32").unwrap();
    // let modified_bindings = pointer_regex.replace_all(&generated_bindings, "CudaSlice<f32>");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::write(out_path.join("bindings.rs"), generated_bindings.as_bytes())
        .expect("Failed to write bindings");
}