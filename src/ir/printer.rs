//! Human-readable IR dump utilities.

use crate::ir::graph::CompGraph;
use crate::ir::types::StorageClass;

/// Print the graph in a human-readable format for debugging.
pub fn dump_graph(graph: &CompGraph) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "// CompGraph: {} ({} nodes, {} tensors)\n",
        graph.metadata.model_id,
        graph.nodes.len(),
        graph.tensors.len()
    ));
    out.push_str(&format!(
        "// Architecture: {:?}, {} layers\n\n",
        graph.metadata.architecture, graph.metadata.params.num_hidden_layers
    ));

    for node in &graph.nodes {
        let output = graph.tensor(node.output);
        out.push_str(&format!("%{} = {:?}(", node.output.0, node.op));
        for (i, input) in node.inputs.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let t = graph.tensor(*input);
            match &t.storage {
                StorageClass::Weight(w) => {
                    out.push_str(&format!("weight={}", w.name));
                }
                _ => {
                    out.push_str(&format!("%{}", input.0));
                }
            }
        }
        out.push_str(&format!(") : {} {}\n", output.shape, output.dtype));

        if let Some(ref ka) = node.kernel {
            out.push_str(&format!(
                "  [kernel: {:?}, path: {:?}]\n",
                ka.kernel_id, ka.path
            ));
        }
    }
    out
}
