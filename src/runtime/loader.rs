use memmap2::Mmap;
use std::fs::File;

pub struct WeightMapping {
    pub ptr: *const u8,
    pub size: usize,
    _storage: WeightStorage,
}

#[allow(dead_code)]
enum WeightStorage {
    SingleMmap(Mmap),
    Concatenated(Vec<u8>),
}

// Safety: the Mmap/Vec backing the pointer is kept alive inside the struct.
unsafe impl Send for WeightMapping {}
unsafe impl Sync for WeightMapping {}

pub fn mmap_weights(weight_paths: &[String]) -> anyhow::Result<WeightMapping> {
    if weight_paths.is_empty() {
        anyhow::bail!("no weight files found in .unc bundle");
    }

    if weight_paths.len() == 1 {
        // Single shard — zero-copy mmap
        let path = &weight_paths[0];
        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("cannot open weight file {path}: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| anyhow::anyhow!("mmap failed for {path}: {e}"))?;
        let ptr = mmap.as_ptr();
        let size = mmap.len();
        Ok(WeightMapping { ptr, size, _storage: WeightStorage::SingleMmap(mmap) })
    } else {
        // Multi-shard — mmap each file, concatenate into one contiguous buffer.
        // The weight binding pass computes cumulative offsets across shards.
        let mut mmaps = Vec::new();
        let mut total_size = 0usize;
        for path in weight_paths {
            let file = File::open(path)
                .map_err(|e| anyhow::anyhow!("cannot open weight file {path}: {e}"))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| anyhow::anyhow!("mmap failed for {path}: {e}"))?;
            total_size += mmap.len();
            mmaps.push(mmap);
        }

        log::info!(
            "Concatenating {} weight shards ({:.1} GB total)",
            mmaps.len(),
            total_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let mut buf = vec![0u8; total_size];
        let mut offset = 0;
        for mmap in &mmaps {
            buf[offset..offset + mmap.len()].copy_from_slice(mmap);
            offset += mmap.len();
        }

        let ptr = buf.as_ptr();
        let size = buf.len();
        Ok(WeightMapping { ptr, size, _storage: WeightStorage::Concatenated(buf) })
    }
}
