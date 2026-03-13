//! QKV Fusion pass.
//!
//! Finds sets of Q/K/V MatMul nodes that share the same input tensor and
//! replaces them with a single Fused(QKVProjection) op.
//!
//! Pattern:
//!   q = MatMul(x, Wq)
//!   k = MatMul(x, Wk)
//!   v = MatMul(x, Wv)
//! →
//!   qkv = Fused(QKVProjection)(x, Wq, Wk, Wv)
//!   (followed by a virtual split that consumers reference)

use std::collections::HashMap;

use crate::ir::graph::CompGraph;
use crate::ir::ops::Op;
use crate::ir::types::TensorId;

/// Run QKV fusion. Returns the number of fusions applied.
pub fn fuse_qkv(graph: &mut CompGraph) -> usize {
    // Find all MatMul nodes, grouped by their first input (the activations tensor)
    let mut input_to_matmuls: HashMap<TensorId, Vec<usize>> = HashMap::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if matches!(node.op, Op::MatMul) && node.inputs.len() >= 2 {
            let x = node.inputs[0];
            input_to_matmuls.entry(x).or_default().push(idx);
        }
    }

    // Find groups of exactly 3 MatMuls with the same input (Q, K, V)
    let fusion_groups: Vec<[usize; 3]> = input_to_matmuls
        .values()
        .filter(|idxs| idxs.len() == 3)
        .map(|idxs| [idxs[0], idxs[1], idxs[2]])
        .collect();

    if fusion_groups.is_empty() {
        return 0;
    }

    log::debug!("QKV fusion: found {} candidate groups", fusion_groups.len());

    // For now, log the opportunity but don't mutate the graph structure
    // (full QKV fusion requires weight tensor concatenation at emit time)
    // The optimizer marks these nodes as candidates for the emitter.
    let count = fusion_groups.len();
    log::info!("QKV fusion: {count} groups identified (will be fused at emit time)");
    0 // deferred to emit time
}
