use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::runtime::cache;
use crate::runtime::loader::mmap_weights;
use crate::runtime::tokenizer::UNCTokenizer;
use crate::unc_format::deserialize::UNCBundle;

// ---------------------------------------------------------------------------
// FFI signatures that match unc_forward.m
// unc_init: void return (uses global unc_state internally)
// unc_forward: no state arg (uses global unc_state)
// ---------------------------------------------------------------------------

type FnUncInit = unsafe extern "C" fn(
    metallib_path: *const std::os::raw::c_char,
    weight_ptr: *mut std::os::raw::c_void,
    weight_size: usize,
    weights_already_converted: u8,
);

type FnUncForward = unsafe extern "C" fn(
    token_ids: *const u32,
    n_tokens: u32,
    position: u32,
    out_logits: *mut f32,
);

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_generation(
    bundle: &UNCBundle,
    metallib_path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<()> {
    // 1. Resolve JIT cache directory keyed by hash of the orchestrator source
    let src = &bundle.orchestrator_source;
    let orch_hash = crc32fast::hash(src.as_bytes());
    let hash = format!("{:08x}", orch_hash);
    let jit_cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("unc")
        .join("jit")
        .join(&hash);
    std::fs::create_dir_all(&jit_cache_dir)?;

    let dylib_path = jit_cache_dir.join("unc_forward.dylib");
    let src_path = jit_cache_dir.join("unc_forward.m");

    // 2. JIT-compile if not cached
    if !dylib_path.exists() {
        log::info!("JIT compiling orchestrator → {}", dylib_path.display());
        std::fs::write(&src_path, src.as_bytes())?;

        let status = Command::new("clang")
            .args([
                "-fobjc-arc",
                "-framework", "Metal",
                "-framework", "Foundation",
                "-shared",
                "-fPIC",
                "-O2",
                "-o", dylib_path.to_str().unwrap(),
                src_path.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| anyhow::anyhow!("clang not found: {e}"))?;

        if !status.success() {
            anyhow::bail!("JIT compile failed (clang exit {})", status);
        }
        log::info!("JIT compile succeeded → {}", dylib_path.display());
    } else {
        log::info!("JIT cache hit → {}", dylib_path.display());
    }

    // 3. dlopen + resolve symbols
    let lib = unsafe { libloading::Library::new(&dylib_path) }
        .map_err(|e| anyhow::anyhow!("dlopen failed: {e}"))?;

    let unc_init: libloading::Symbol<FnUncInit> = unsafe { lib.get(b"unc_init\0") }
        .map_err(|e| anyhow::anyhow!("unc_init symbol not found: {e}"))?;
    let unc_forward: libloading::Symbol<FnUncForward> = unsafe { lib.get(b"unc_forward\0") }
        .map_err(|e| anyhow::anyhow!("unc_forward symbol not found: {e}"))?;

    // 4. Check weight cache for pre-converted F16 weights
    let weight_paths = &bundle.graph.metadata.weight_file_paths;
    let cache_key = cache::compute_cache_key(weight_paths, orch_hash);
    let cached_weights_path = cache::find_cached_weights(&cache_key);
    let weights_already_converted = cached_weights_path.is_some();

    let weights = if let Some(ref cached_path) = cached_weights_path {
        log::info!("Weight cache hit → {}", cached_path.display());
        mmap_weights(&[cached_path.to_string_lossy().into_owned()])?
    } else {
        log::info!("Weight cache miss — will convert and cache after init");
        mmap_weights(weight_paths)?
    };

    // 5. Initialize Metal state
    let metallib_cstr = std::ffi::CString::new(metallib_path.to_str().unwrap())?;
    let converted_flag: u8 = if weights_already_converted { 1 } else { 0 };
    unsafe {
        unc_init(
            metallib_cstr.as_ptr(),
            weights.ptr as *mut std::os::raw::c_void,
            weights.size,
            converted_flag,
        );
    }

    // 6. Save converted weights to cache (if we just converted them)
    if !weights_already_converted {
        // The weight buffer was converted in-place by unc_init on the GPU.
        // We can't easily read back from the Metal buffer here since it's inside the dylib.
        // Instead, we save from the mmap'd region which was copied into the Metal buffer.
        // For now, the GPU conversion happens in the Metal buffer (not the mmap'd source),
        // so we skip disk caching until we add a getBytes callback.
        // TODO: Add a unc_get_weight_buf() FFI to read back converted weights.
        log::info!("Weight conversion done (disk cache deferred until unc_get_weight_buf is available)");
    }

    // 7. Tokenize prompt
    let tok_path = bundle.graph.metadata.tokenizer_path.as_deref()
        .ok_or_else(|| anyhow::anyhow!("no tokenizer_path in .unc bundle"))?;
    let tokenizer = UNCTokenizer::from_file(Path::new(tok_path))?;

    let mut token_ids: Vec<u32> = tokenizer.encode(prompt)?;
    if token_ids.is_empty() {
        anyhow::bail!("prompt encoded to empty token list");
    }

    // Prepend BOS if available
    if let Some(bos) = bundle.graph.metadata.bos_token_id {
        token_ids.insert(0, bos);
    }

    // Determine EOS
    let eos = bundle.graph.metadata.eos_token_id
        .or_else(|| tokenizer.eos_token_id());

    let vocab_size = bundle.graph.metadata.params.vocab_size;
    let mut logits: Vec<f32> = vec![0.0f32; vocab_size];

    // 8. Prefill
    let n_prompt = token_ids.len() as u32;
    unsafe {
        unc_forward(
            token_ids.as_ptr(),
            n_prompt,
            n_prompt,
            logits.as_mut_ptr(),
        );
    }

    // 9. Autoregressive decode loop
    let mut next = argmax(&logits);
    let mut generated_ids: Vec<u32> = token_ids.clone();
    let decode_start = std::time::Instant::now();
    let mut decode_tokens = 0u32;
    for step in 0..max_tokens {
        generated_ids.push(next);
        let decoded = tokenizer.decode_incremental(&generated_ids)?;
        print!("{decoded}");
        std::io::stdout().flush()?;
        decode_tokens += 1;

        if Some(next) == eos {
            break;
        }

        let position = n_prompt + step as u32 + 1;
        unsafe {
            unc_forward(&next as *const u32, 1, position, logits.as_mut_ptr());
        }
        next = argmax(&logits);
    }
    let decode_elapsed = decode_start.elapsed();
    let tok_per_sec = if decode_elapsed.as_secs_f64() > 0.0 {
        decode_tokens as f64 / decode_elapsed.as_secs_f64()
    } else { 0.0 };
    println!();
    eprintln!("\n--- {decode_tokens} tokens in {:.2}s = {tok_per_sec:.1} tok/s ---", decode_elapsed.as_secs_f64());

    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
