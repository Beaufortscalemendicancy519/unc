//! Dual-path insertion pass.
//!
//! Matmul ops must work for both prefill (seq_len > 1, compute-bound GEMM)
//! and decode (seq_len = 1, memory-bound GEMV) modes. This pass tags nodes
//! so the emitter generates a runtime branch on actual seq_len.

use crate::ir::graph::CompGraph;
use crate::ir::ops::Op;
use crate::kernel::registry::KernelRegistry;
use crate::target::Target;

/// Insert dual-path markers for all matmul-class nodes.
pub fn insert_dual_paths(graph: &mut CompGraph, _registry: &KernelRegistry, _target: &Target) {
    for node in graph.nodes.iter_mut() {
        if !is_gemm_op(&node.op) {
            continue;
        }

        // Look up the decode (GEMV) kernel for this op
        // The actual lookup happens in kernel_matching.rs; here we just mark the node
        // as needing a dual path by setting a placeholder.
        // The kernel_matching pass will fill in the real kernel IDs.
        if node.kernel.is_none() {
            // Mark as needing dual-path resolution
            node.estimated_cost = Some(f64::MAX); // will be replaced
        }
    }
}

fn is_gemm_op(op: &Op) -> bool {
    matches!(
        op,
        Op::MatMul | Op::BatchMatMul { .. } | Op::QuantizedMatMul { .. } | Op::LogitProjection
    )
}
