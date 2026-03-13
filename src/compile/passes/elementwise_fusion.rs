//! Elementwise fusion pass.
//!
//! Chains of memory-bound elementwise ops between matmuls are natural fusion
//! candidates. For now, this pass identifies SiLU+Mul (SwiGLU gate) patterns
//! and marks them for fused dispatch.

use crate::ir::graph::CompGraph;
use crate::ir::ops::Op;

/// Fuse elementwise op chains. Returns number of fusions applied.
pub fn fuse_elementwise(graph: &mut CompGraph) -> usize {
    let mut fused = 0;

    // Pattern: SiLU(gate) followed immediately by Mul(silu_out, up)
    // These are the two outputs of a SwiGLU FFN block.
    // We detect but defer fusion to emit time (kernel already handles it).
    for i in 0..graph.nodes.len() {
        if matches!(graph.nodes[i].op, Op::SiLU) {
            let silu_out = graph.nodes[i].output;
            // Find a Mul node that consumes this output
            for j in (i + 1)..graph.nodes.len() {
                if matches!(graph.nodes[j].op, Op::Mul)
                    && graph.nodes[j].inputs.contains(&silu_out)
                {
                    // Found SiLU → Mul (SwiGLU gate)
                    fused += 1;
                    break;
                }
            }
        }
    }

    if fused > 0 {
        log::debug!("Elementwise fusion: found {fused} SwiGLU gate patterns");
    }
    0 // deferred to emit time
}
