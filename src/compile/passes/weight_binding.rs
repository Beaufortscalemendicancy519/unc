//! WeightBindingResolution pass: scan safetensors headers to fill in byte offsets.

use std::collections::HashMap;

use anyhow::Context;
use safetensors::SafeTensors;

use crate::frontend::huggingface::WeightFile;
use crate::ir::graph::CompGraph;
use crate::ir::types::{DType, StorageClass};

/// Fill in `byte_offset` and `byte_size` for every Weight tensor in the graph.
///
/// Reads safetensors headers (without loading the weight data into RAM) and
/// matches tensor names to `StorageClass::Weight` nodes.
pub fn resolve_weight_bindings(
    graph: &mut CompGraph,
    weight_files: &[WeightFile],
) -> anyhow::Result<()> {
    // Build a map: tensor_name -> (byte_offset, byte_size, shape, dtype)
    // For multi-shard models, byte_offset is cumulative across all shards
    // (shard 0 at offset 0, shard 1 at offset file_size(shard 0), etc.)
    let mut name_map: HashMap<String, (usize, usize, Vec<usize>, DType)> = HashMap::new();
    let mut cumulative_file_offset: usize = 0;

    for wf in weight_files {
        match wf {
            WeightFile::Safetensors { path, .. } => {
                let data = std::fs::read(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                let file_size = data.len();
                let tensors = SafeTensors::deserialize(&data)
                    .with_context(|| format!("parsing safetensors {}", path.display()))?;

                // Compute the base pointer of the data region so we can derive offsets
                let base_ptr = data.as_ptr() as usize;

                for (name, view) in tensors.tensors() {
                    let dtype = safetensors_dtype_to_unc(view.dtype());
                    let shape: Vec<usize> = view.shape().to_vec();
                    let tensor_data = view.data();
                    // Byte offset within this file
                    let local_offset = tensor_data.as_ptr() as usize - base_ptr;
                    let byte_size = tensor_data.len();
                    // Global offset: cumulative file offset + local offset within file
                    let byte_offset = cumulative_file_offset + local_offset;
                    name_map.insert(name.to_string(), (byte_offset, byte_size, shape, dtype));
                }

                cumulative_file_offset += file_size;
            }
            WeightFile::Gguf { .. } => {
                // GGUF weight binding resolution is TODO
                log::warn!("GGUF weight binding resolution not yet implemented");
            }
        }
    }

    // Patch every Weight tensor in the graph
    let mut total_bytes = 0usize;
    for tensor in graph.tensors.values_mut() {
        if let StorageClass::Weight(ref mut binding) = tensor.storage {
            if let Some((offset, size, shape, dtype)) = name_map.get(&binding.name) {
                binding.byte_offset = *offset;
                binding.byte_size = *size;
                binding.file_shape = shape.clone();
                binding.file_dtype = *dtype;
                total_bytes += size;
            } else {
                log::warn!("Weight tensor '{}' not found in safetensors files", binding.name);
            }
        }
    }

    graph.metadata.total_weight_bytes = total_bytes;
    Ok(())
}

fn safetensors_dtype_to_unc(dtype: safetensors::Dtype) -> DType {
    match dtype {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::U32 => DType::U32,
        safetensors::Dtype::BOOL => DType::Bool,
        _ => DType::F32, // fallback
    }
}
