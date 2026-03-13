//! Barrier analysis for concurrent Metal dispatch.
//!
//! Determines which dispatches need memory barriers and which specific
//! resources (decode region buffers) need to be barriered.
//!
//! Uses data-flow dependency tracking: a barrier is needed when a dispatch
//! reads a tensor that was written by a previous dispatch since the last barrier.

use std::collections::{HashMap, HashSet};

use crate::ir::types::TensorId;

/// Result of barrier analysis: which dispatch units need barriers and on which resources.
#[derive(Debug)]
pub struct BarrierPlan {
    /// Set of dispatch-unit indices that need a barrier emitted before them.
    pub barrier_before: HashSet<usize>,
    /// Per-dispatch: which region IDs need to be barriered before this dispatch.
    /// Only populated for dispatches in `barrier_before`.
    pub barrier_regions: HashMap<usize, Vec<u32>>,
    pub num_dispatches: usize,
    pub num_barriers: usize,
}

/// A single dispatch unit (may represent a fused group of nodes).
#[derive(Debug)]
pub struct DispatchUnit {
    /// The primary node index (used to correlate with emitter).
    pub primary_node_idx: usize,
    /// TensorIds this dispatch reads.
    pub reads: Vec<TensorId>,
    /// TensorIds this dispatch writes.
    pub writes: Vec<TensorId>,
}

/// Build the barrier plan using data-flow dependency tracking.
///
/// When `tensor_to_region` is provided, tracks per-region barriers:
/// only clears pending writes for the specific regions being barriered,
/// allowing unrelated regions to continue without barriers.
pub fn analyze_barriers(
    units: &[DispatchUnit],
    tensor_to_region: Option<&HashMap<TensorId, u32>>,
) -> BarrierPlan {
    let mut barrier_before = HashSet::new();
    let mut barrier_regions: HashMap<usize, Vec<u32>> = HashMap::new();

    // Track pending writes: TensorId → which dispatch wrote it
    let mut pending_writes: HashSet<TensorId> = HashSet::new();
    // Track which regions have pending writes
    let mut pending_region_writes: HashSet<u32> = HashSet::new();

    for (i, unit) in units.iter().enumerate() {
        // Check if any read of this dispatch was written since last barrier
        let conflicting_reads: Vec<&TensorId> = unit.reads.iter()
            .filter(|read_tid| pending_writes.contains(read_tid))
            .collect();

        if !conflicting_reads.is_empty() {
            barrier_before.insert(i);

            if let Some(t2r) = tensor_to_region {
                // Per-resource barrier: find which regions need barriering
                let mut regions_to_barrier: HashSet<u32> = HashSet::new();
                for read_tid in &conflicting_reads {
                    if let Some(&region) = t2r.get(read_tid) {
                        regions_to_barrier.insert(region);
                    }
                }
                let regions_vec: Vec<u32> = regions_to_barrier.iter().copied().collect();

                // Only clear pending writes for the barriered regions
                pending_writes.retain(|tid| {
                    if let Some(&r) = t2r.get(tid) {
                        !regions_to_barrier.contains(&r)
                    } else {
                        true // non-activation tensors (weights, KV) keep their pending state
                    }
                });
                for &r in &regions_to_barrier {
                    pending_region_writes.remove(&r);
                }

                barrier_regions.insert(i, regions_vec);
            } else {
                // Scope barrier: clear all pending writes
                pending_writes.clear();
                pending_region_writes.clear();
            }
        }

        // Add this dispatch's writes to pending set
        for write_tid in &unit.writes {
            pending_writes.insert(*write_tid);
            if let Some(t2r) = tensor_to_region {
                if let Some(&region) = t2r.get(write_tid) {
                    pending_region_writes.insert(region);
                }
            }
        }
    }

    let num_barriers = barrier_before.len();
    let num_dispatches = units.len();
    BarrierPlan {
        barrier_before,
        barrier_regions,
        num_dispatches,
        num_barriers,
    }
}
