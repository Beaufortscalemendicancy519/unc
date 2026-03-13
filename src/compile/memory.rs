//! Static memory allocation plan with lifetime-based buffer aliasing.

use std::collections::HashMap;

use crate::ir::graph::CompGraph;
use crate::ir::types::{BufferId, DType, StorageClass, TensorId};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MemoryPlan {
    pub activation_buffers: Vec<BufferAllocation>,
    pub kv_cache: KVCacheLayout,
    pub total_activation_bytes: usize,
    pub total_kv_cache_bytes: usize,
    pub total_weight_bytes: usize,
    /// Decode regions: each is a separate MTLBuffer for concurrent dispatch.
    /// Maps 1:1 to activation_buffers (same BufferId).
    pub decode_regions: Vec<DecodeRegion>,
    /// TensorId → decode region index (into decode_regions vec).
    pub decode_tensor_region: HashMap<TensorId, u32>,
}

#[derive(Debug, Clone)]
pub struct DecodeRegion {
    pub region_id: u32,
    /// Size in bytes for decode (seq_len=1) — max of all tensors aliased into this region.
    pub size_bytes: usize,
    pub tensors: Vec<TensorId>,
}

#[derive(Debug, Clone)]
pub struct BufferAllocation {
    pub id: BufferId,
    pub size_bytes: usize,
    pub offset: usize,
    pub aliases: Vec<TensorId>,
    pub lifetime: BufferLifetime,
}

#[derive(Debug, Clone, Copy)]
pub struct BufferLifetime {
    pub first_write: usize,
    pub last_read: usize,
}

impl BufferLifetime {
    pub fn overlaps(&self, other: &BufferLifetime) -> bool {
        self.first_write <= other.last_read && other.first_write <= self.last_read
    }
}

#[derive(Debug)]
pub struct KVCacheLayout {
    pub layers: Vec<KVCacheLayerLayout>,
    pub total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct KVCacheLayerLayout {
    pub layer_idx: usize,
    pub key_offset: usize,
    pub key_size_bytes: usize,
    pub value_offset: usize,
    pub value_size_bytes: usize,
    pub dtype: DType,
}

// ---------------------------------------------------------------------------
// Memory planning algorithm
// ---------------------------------------------------------------------------

/// Compute a static memory plan for the graph.
///
/// Uses greedy interval-graph coloring to alias activation buffers
/// whose lifetimes don't overlap.
pub fn plan_memory(graph: &CompGraph) -> MemoryPlan {
    let total_weight_bytes = compute_weight_bytes(graph);
    let kv_cache = plan_kv_cache(graph);

    // Compute liveness for activation tensors
    let liveness = compute_liveness(graph);

    // Sort activations by size descending for better packing
    let mut activations: Vec<(TensorId, usize, BufferLifetime)> = liveness
        .into_iter()
        .map(|(tid, lt)| {
            let t = graph.tensor(tid);
            let size = t.max_size_bytes();
            (tid, size, lt)
        })
        .collect();
    activations.sort_by(|a, b| b.1.cmp(&a.1));

    // Greedy interval coloring: try to reuse existing buffers
    let mut buffers: Vec<BufferAllocation> = Vec::new();
    let mut tensor_to_buffer: HashMap<TensorId, BufferId> = HashMap::new();

    for (tid, size, lt) in &activations {
        // Find an existing buffer we can reuse (non-overlapping lifetime, sufficient size)
        let reused = buffers.iter_mut().find(|b| {
            !b.lifetime.overlaps(lt) && b.size_bytes >= *size
        });

        if let Some(buf) = reused {
            // Reuse — extend lifetime to cover both uses
            buf.lifetime.first_write = buf.lifetime.first_write.min(lt.first_write);
            buf.lifetime.last_read = buf.lifetime.last_read.max(lt.last_read);
            buf.aliases.push(*tid);
            tensor_to_buffer.insert(*tid, buf.id);
        } else {
            // Allocate a new buffer
            let id = BufferId(buffers.len() as u32);
            let offset = buffers.iter().map(|b| b.offset + b.size_bytes).max().unwrap_or(0);
            // Align to 16 bytes
            let offset = (offset + 15) & !15;
            buffers.push(BufferAllocation {
                id,
                size_bytes: *size,
                offset,
                aliases: vec![*tid],
                lifetime: *lt,
            });
            tensor_to_buffer.insert(*tid, id);
        }
    }

    // Assign BufferId back to tensors
    // (This would mutate graph.tensors[tid].storage — we do it in the pass runner)

    let total_activation_bytes = buffers.iter()
        .map(|b| b.offset + b.size_bytes)
        .max()
        .unwrap_or(0);

    // Build decode regions from buffer allocations.
    // Each BufferAllocation becomes one decode region (separate MTLBuffer).
    let mut decode_regions = Vec::with_capacity(buffers.len());
    let mut decode_tensor_region: HashMap<TensorId, u32> = HashMap::new();

    for buf in &buffers {
        let region_id = buf.id.0;
        let mut max_decode_bytes = 0usize;
        for &tid in &buf.aliases {
            let t = graph.tensor(tid);
            let decode_bytes = t.decode_size_bytes();
            max_decode_bytes = max_decode_bytes.max(decode_bytes);
            decode_tensor_region.insert(tid, region_id);
        }
        // Align to 16 bytes
        max_decode_bytes = (max_decode_bytes + 15) & !15;
        decode_regions.push(DecodeRegion {
            region_id,
            size_bytes: max_decode_bytes,
            tensors: buf.aliases.clone(),
        });
    }

    log::info!(
        "Decode regions: {} regions, total {} bytes",
        decode_regions.len(),
        decode_regions.iter().map(|r| r.size_bytes).sum::<usize>(),
    );

    MemoryPlan {
        activation_buffers: buffers,
        kv_cache,
        total_activation_bytes,
        total_kv_cache_bytes: graph.metadata.params.max_position_embeddings
            * graph.metadata.params.num_kv_heads
            * graph.metadata.params.head_dim
            * 2 // key + value
            * graph.metadata.params.num_hidden_layers
            * 2, // f16 = 2 bytes
        total_weight_bytes,
        decode_regions,
        decode_tensor_region,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_liveness(graph: &CompGraph) -> HashMap<TensorId, BufferLifetime> {
    let mut liveness: HashMap<TensorId, BufferLifetime> = HashMap::new();

    for (node_idx, node) in graph.nodes.iter().enumerate() {
        // The output tensor is first written here
        let tid = node.output;
        if let Some(t) = graph.tensors.get(&tid) {
            if matches!(t.storage, StorageClass::Activation { .. }) {
                let entry = liveness.entry(tid).or_insert(BufferLifetime {
                    first_write: node_idx,
                    last_read: node_idx,
                });
                entry.first_write = entry.first_write.min(node_idx);
            }
        }

        // Input tensors are last read at this node
        for input_tid in &node.inputs {
            if let Some(t) = graph.tensors.get(input_tid) {
                if matches!(t.storage, StorageClass::Activation { .. }) {
                    let entry = liveness.entry(*input_tid).or_insert(BufferLifetime {
                        first_write: node_idx,
                        last_read: node_idx,
                    });
                    entry.last_read = entry.last_read.max(node_idx);
                }
            }
        }
    }

    liveness
}

fn plan_kv_cache(graph: &CompGraph) -> KVCacheLayout {
    let p = &graph.metadata.params;
    let num_layers = p.num_hidden_layers;
    let kv_heads = p.num_kv_heads;
    let head_dim = p.head_dim;
    let max_seq = p.max_position_embeddings;
    let dtype = DType::F16;
    let bytes_per_elem = 2usize;

    let per_layer_key_bytes = kv_heads * max_seq * head_dim * bytes_per_elem;
    let per_layer_val_bytes = kv_heads * max_seq * head_dim * bytes_per_elem;
    let _per_layer_total = per_layer_key_bytes + per_layer_val_bytes;

    let mut layers = Vec::with_capacity(num_layers);
    let mut offset = 0usize;

    for layer_idx in 0..num_layers {
        let key_offset = offset;
        offset += per_layer_key_bytes;
        let value_offset = offset;
        offset += per_layer_val_bytes;
        layers.push(KVCacheLayerLayout {
            layer_idx,
            key_offset,
            key_size_bytes: per_layer_key_bytes,
            value_offset,
            value_size_bytes: per_layer_val_bytes,
            dtype,
        });
    }

    KVCacheLayout {
        total_bytes: offset,
        layers,
    }
}

fn compute_weight_bytes(graph: &CompGraph) -> usize {
    graph.tensors.values()
        .filter(|t| matches!(t.storage, StorageClass::Weight(ref w) if w.byte_size > 0))
        .map(|t| {
            if let StorageClass::Weight(ref w) = t.storage {
                w.byte_size
            } else {
                0
            }
        })
        .sum()
}
