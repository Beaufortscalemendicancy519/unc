// build.rs — compile C runtime shim and Metal shaders at build time.

fn main() {
    // Compile C runtime shim
    let runtime_files = [
        "runtime/runtime.c",
        "runtime/api_server.c",
        "runtime/repl.c",
    ];
    let existing: Vec<_> = runtime_files
        .iter()
        .filter(|f| std::path::Path::new(f).exists())
        .cloned()
        .collect();

    if !existing.is_empty() {
        cc::Build::new()
            .files(&existing)
            .include("runtime/")
            .compile("unc_runtime");
    }

    // Re-run if any runtime file changes
    for f in &runtime_files {
        println!("cargo:rerun-if-changed={f}");
    }

    // On macOS: compile our reference Metal kernels into unc_kernels.metallib
    #[cfg(target_os = "macos")]
    compile_metal_shaders();
}

#[cfg(target_os = "macos")]
fn compile_metal_shaders() {
    use std::process::Command;

    // Prefer the hand-written reference kernels in unc_kernels/
    let unc_kernel_dir = std::path::Path::new("kernel_sources/metal/unc_kernels");
    let kernel_dir = if unc_kernel_dir.exists() {
        unc_kernel_dir
    } else {
        println!("cargo:warning=kernel_sources/metal/unc_kernels not found — skipping Metal shader compilation");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metallib_path = format!("{out_dir}/unc_kernels.metallib");

    // Collect .metal files
    let metal_files: Vec<_> = std::fs::read_dir(kernel_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "metal"))
        .map(|e| e.path())
        .collect();

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in kernel_sources/metal/upstream_mlx");
        return;
    }

    // Compile each .metal → .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_string_lossy();
        let air_path = format!("{out_dir}/{stem}.air");

        let status = Command::new("xcrun")
            .args([
                "-sdk", "macosx", "metal",
                "-c", metal_file.to_str().unwrap(),
                "-I", kernel_dir.to_str().unwrap(),
                "-I", "kernel_sources/metal/upstream_mlx",
                "-I", "mlx",  // MLX source root for "mlx/backend/metal/kernels/..." includes
                "-std=metal3.0",
                "-o", &air_path,
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                air_files.push(air_path);
            }
            Ok(s) => {
                println!("cargo:warning=metal compile failed for {} (exit {s})", metal_file.display());
            }
            Err(e) => {
                println!("cargo:warning=xcrun metal not available: {e}");
                return;
            }
        }

        println!("cargo:rerun-if-changed={}", metal_file.display());
    }

    if air_files.is_empty() {
        return;
    }

    // Link .air files → .metallib
    let mut metallib_cmd = Command::new("xcrun");
    metallib_cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        metallib_cmd.arg(air);
    }
    metallib_cmd.args(["-o", &metallib_path]);

    match metallib_cmd.status() {
        Ok(s) if s.success() => {
            println!("cargo:rustc-env=UNC_METALLIB_PATH={metallib_path}");
            println!("cargo:warning=Compiled metallib: {metallib_path}");
        }
        Ok(s) => println!("cargo:warning=metallib link failed (exit {s})"),
        Err(e) => println!("cargo:warning=xcrun metallib failed: {e}"),
    }
}
