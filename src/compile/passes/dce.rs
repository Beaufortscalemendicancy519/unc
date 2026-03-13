//! Dead Code Elimination pass.
//!
//! Removes graph nodes whose outputs are not consumed by any other node
//! and are not graph outputs.

use std::collections::HashSet;

use crate::ir::graph::CompGraph;
use crate::ir::types::TensorId;

pub fn eliminate_dead_code(graph: &mut CompGraph) {
    // Collect all tensor IDs that are used as inputs to some node or are graph outputs
    let mut used: HashSet<TensorId> = HashSet::new();

    for tid in &graph.outputs {
        used.insert(*tid);
    }
    for node in &graph.nodes {
        for input in &node.inputs {
            used.insert(*input);
        }
    }

    // Remove nodes whose primary output is not used
    let before = graph.nodes.len();
    graph.nodes.retain(|node| used.contains(&node.output));

    // Rebuild node_map
    graph.node_map.clear();
    for (idx, node) in graph.nodes.iter().enumerate() {
        graph.node_map.insert(node.id, idx);
    }

    let removed = before - graph.nodes.len();
    if removed > 0 {
        log::debug!("DCE: removed {removed} dead nodes");
    }
}
