//! Write .unc binary files.

use std::io::Write;
use std::path::Path;

use anyhow::Context;

use crate::compile::{CompilationResult, CompiledArtifact};
use crate::unc_format::{FORMAT_VERSION, MAGIC};

/// Write a compiled model to a .unc file.
pub fn write_unc(
    result: &CompilationResult,
    artifact: &CompiledArtifact,
    out_path: &Path,
) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(out_path)
        .with_context(|| format!("creating {}", out_path.display()))?;

    // Magic + version
    file.write_all(MAGIC)?;
    file.write_all(&FORMAT_VERSION.to_le_bytes())?;

    // Serialise the graph (bincode)
    let graph_bytes = bincode::serialize(&result.graph)
        .context("serialising CompGraph")?;
    let graph_len = graph_bytes.len() as u64;
    file.write_all(&graph_len.to_le_bytes())?;
    file.write_all(&graph_bytes)?;

    // Serialise memory plan stats as JSON (human-readable section)
    let stats_json = serde_json::json!({
        "nodes_before": result.stats.nodes_before,
        "nodes_after": result.stats.nodes_after,
        "unique_kernels": result.stats.unique_kernels,
        "fusions_applied": result.stats.fusions_applied,
        "peak_activation_bytes": result.stats.peak_activation_bytes,
        "kernel_launches_per_forward": result.stats.kernel_launches_per_forward,
        "total_activation_bytes": result.memory_plan.total_activation_bytes,
        "total_kv_cache_bytes": result.memory_plan.total_kv_cache_bytes,
        "total_weight_bytes": result.memory_plan.total_weight_bytes,
    });
    let stats_bytes = stats_json.to_string().into_bytes();
    let stats_len = stats_bytes.len() as u64;
    file.write_all(&stats_len.to_le_bytes())?;
    file.write_all(&stats_bytes)?;

    // Artifact-specific data
    match artifact {
        CompiledArtifact::Metal { metallib_path, orchestrator_source, weight_layout: _ } => {
            file.write_all(b"METAL")?;
            // Orchestrator source
            let orch_bytes = orchestrator_source.as_bytes();
            file.write_all(&(orch_bytes.len() as u64).to_le_bytes())?;
            file.write_all(orch_bytes)?;
            // Metallib path (for runtime to load)
            let mp_bytes = metallib_path.as_bytes();
            file.write_all(&(mp_bytes.len() as u64).to_le_bytes())?;
            file.write_all(mp_bytes)?;
        }
    }

    log::info!("Written: {}", out_path.display());
    Ok(())
}
