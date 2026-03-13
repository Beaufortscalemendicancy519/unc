//! Read .unc binary files.

use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Context};

use crate::ir::graph::CompGraph;
use crate::unc_format::{FORMAT_VERSION, MAGIC};

pub struct UNCBundle {
    pub graph: CompGraph,
    pub stats: serde_json::Value,
    pub target_tag: String,   // "METAL" or "CPU  "
    pub orchestrator_source: String,
    pub metallib_path: Option<String>,
}

pub fn read_unc(path: &Path) -> anyhow::Result<UNCBundle> {
    let data = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut cursor = std::io::Cursor::new(&data);

    // Magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(anyhow!("not a .unc file (bad magic)"));
    }

    // Version
    let mut ver_bytes = [0u8; 4];
    cursor.read_exact(&mut ver_bytes)?;
    let version = u32::from_le_bytes(ver_bytes);
    if version != FORMAT_VERSION {
        return Err(anyhow!("unsupported .unc format version {version}"));
    }

    // Graph
    let mut len_bytes = [0u8; 8];
    cursor.read_exact(&mut len_bytes)?;
    let graph_len = u64::from_le_bytes(len_bytes) as usize;
    let mut graph_bytes = vec![0u8; graph_len];
    cursor.read_exact(&mut graph_bytes)?;
    let graph: CompGraph = bincode::deserialize(&graph_bytes)
        .context("deserialising CompGraph")?;

    // Stats
    cursor.read_exact(&mut len_bytes)?;
    let stats_len = u64::from_le_bytes(len_bytes) as usize;
    let mut stats_bytes = vec![0u8; stats_len];
    cursor.read_exact(&mut stats_bytes)?;
    let stats: serde_json::Value = serde_json::from_slice(&stats_bytes)
        .context("deserialising stats JSON")?;

    // Target tag
    let mut tag = [0u8; 5];
    cursor.read_exact(&mut tag)?;
    let target_tag = String::from_utf8_lossy(&tag).into_owned();

    // Orchestrator source
    cursor.read_exact(&mut len_bytes)?;
    let orch_len = u64::from_le_bytes(len_bytes) as usize;
    let mut orch_bytes = vec![0u8; orch_len];
    cursor.read_exact(&mut orch_bytes)?;
    let orchestrator_source = String::from_utf8(orch_bytes)
        .context("decoding orchestrator source")?;

    let metallib_path = if target_tag.trim() == "METAL" {
        cursor.read_exact(&mut len_bytes)?;
        let mp_len = u64::from_le_bytes(len_bytes) as usize;
        let mut mp_bytes = vec![0u8; mp_len];
        cursor.read_exact(&mut mp_bytes)?;
        Some(String::from_utf8(mp_bytes).context("decoding metallib path")?)
    } else {
        None
    };

    Ok(UNCBundle {
        graph,
        stats,
        target_tag,
        orchestrator_source,
        metallib_path,
    })
}
