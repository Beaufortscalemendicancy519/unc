use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Compute a cache key from weight file paths, sizes, and orchestrator source hash.
pub fn compute_cache_key(weight_paths: &[String], orchestrator_hash: u32) -> String {
    let mut hasher = crc32fast::Hasher::new();
    for p in weight_paths {
        hasher.update(p.as_bytes());
        if let Ok(meta) = fs::metadata(p) {
            hasher.update(&meta.len().to_le_bytes());
        }
    }
    hasher.update(&orchestrator_hash.to_le_bytes());
    format!("{:08x}", hasher.finalize())
}

/// Return the cache directory for a given cache key.
pub fn cache_dir(key: &str) -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("unc")
        .join("models")
        .join(key)
}

/// Check if converted weights exist in the cache.
pub fn find_cached_weights(key: &str) -> Option<PathBuf> {
    let path = cache_dir(key).join("weights.f16");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Save converted weights from a pointer to disk cache.
pub fn save_converted_weights(
    key: &str,
    data: *const u8,
    size: usize,
) -> anyhow::Result<PathBuf> {
    let dir = cache_dir(key);
    fs::create_dir_all(&dir)?;
    let path = dir.join("weights.f16");

    let slice = unsafe { std::slice::from_raw_parts(data, size) };
    let mut file = fs::File::create(&path)?;
    file.write_all(slice)?;

    let meta = serde_json::json!({ "size": size });
    fs::write(dir.join("meta.json"), meta.to_string())?;

    log::info!("Cached converted weights: {} ({} bytes)", path.display(), size);
    Ok(path)
}
