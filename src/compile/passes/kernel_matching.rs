//! Kernel matching pass: assign a kernel from the registry to each graph node.

use std::collections::HashSet;

use crate::ir::graph::{CompGraph, ExecutionPath, KernelAssignment};
use crate::ir::ops::Op;
use crate::ir::types::KernelId;
use crate::kernel::registry::KernelRegistry;
use crate::target::Target;

/// Match kernels for all nodes. Returns the number of unique kernels used.
pub fn match_kernels(
    graph: &mut CompGraph,
    registry: &KernelRegistry,
    target: &Target,
) -> anyhow::Result<usize> {
    let mut matched = 0usize;
    let mut unmatched = 0usize;
    let mut unique_kernels: HashSet<KernelId> = HashSet::new();

    // Collect node IDs first to avoid borrow issues
    let node_ids: Vec<_> = graph.nodes.iter().map(|n| n.id).collect();

    for node_id in node_ids {
        let node = graph.node(node_id);

        // Skip zero-compute ops (reshape, transpose) — no kernel needed
        if node.op.is_zero_compute() {
            continue;
        }

        // Gather input tensors
        let input_ids = node.inputs.clone();
        let output_id = node.output;
        let op = node.op.clone();

        let input_tensors: Vec<&crate::ir::types::TensorRef> = input_ids
            .iter()
            .filter_map(|id| graph.tensors.get(id))
            .collect();

        let output_shape = graph.tensors.get(&output_id).map(|t| t.shape.clone());
        let output_shape = match output_shape {
            Some(s) => s,
            None => continue,
        };

        let mp = Some(&graph.metadata.params);
        if let Some(mut km) = registry.find_best(&op, &input_tensors, &output_shape, target, mp) {
            unique_kernels.insert(km.kernel.id);

            // For matmul-class ops, create a Dual path assignment
            let path = if is_gemm_op(&op) {
                // Try to find the corresponding GEMV kernel
                let gemv_op = to_gemv_op(&op);
                if let Some(mut decode_km) =
                    gemv_op.and_then(|gop| registry.find_best(&gop, &input_tensors, &output_shape, target, mp))
                {
                    // Dynamic shared memory: GEMV uses K*2 bytes for activation cache
                    if decode_km.dispatch.shared_memory_bytes > 0 {
                        let k_dim = input_tensors.first()
                            .and_then(|t| t.shape.0.first())
                            .map(|d| d.max_value())
                            .unwrap_or(0);
                        decode_km.dispatch.shared_memory_bytes = (k_dim * 2) as u32;
                    }
                    unique_kernels.insert(decode_km.kernel.id);
                    ExecutionPath::Dual {
                        prefill_kernel: km.kernel.id,
                        prefill_kernel_name: km.kernel.name.clone(),
                        prefill_dispatch: Box::new(km.dispatch.clone()),
                        decode_kernel: decode_km.kernel.id,
                        decode_kernel_name: decode_km.kernel.name.clone(),
                        decode_dispatch: Box::new(decode_km.dispatch),
                    }
                } else {
                    ExecutionPath::Prefill
                }
            } else {
                // Dynamic shared memory for standalone GEMV
                if matches!(op, Op::MatVec | Op::QuantizedMatVec { .. }) && km.dispatch.shared_memory_bytes > 0 {
                    let k_dim = input_tensors.first()
                        .and_then(|t| t.shape.0.first())
                        .map(|d| d.max_value())
                        .unwrap_or(0);
                    km.dispatch.shared_memory_bytes = (k_dim * 2) as u32;
                }
                ExecutionPath::Unified
            };

            let kernel_name = km.kernel.name.clone();
            let node = graph.node_mut(node_id);
            node.kernel = Some(KernelAssignment {
                kernel_id: km.kernel.id,
                kernel_name,
                dispatch: km.dispatch,
                path,
            });
            node.estimated_cost = Some(km.estimated_cost);
            matched += 1;
        } else {
            log::warn!("No kernel found for op {:?} on target {}", op, target.name());
            unmatched += 1;
        }
    }

    log::info!(
        "Kernel matching: {matched} matched, {unmatched} unmatched, {} unique kernels",
        unique_kernels.len()
    );

    Ok(unique_kernels.len())
}

fn is_gemm_op(op: &Op) -> bool {
    matches!(op, Op::MatMul | Op::BatchMatMul { .. } | Op::LogitProjection)
}

fn to_gemv_op(op: &Op) -> Option<Op> {
    match op {
        Op::MatMul | Op::LogitProjection => Some(Op::MatVec),
        Op::QuantizedMatMul { weight_dtype } => Some(Op::QuantizedMatVec { weight_dtype: *weight_dtype }),
        _ => None,
    }
}
