//! Compilation pipeline: optimization passes → kernel matching → memory planning.

pub mod passes;
pub mod memory;

use anyhow::Context;

use crate::frontend::huggingface::WeightFile;
use crate::ir::graph::CompGraph;
use crate::ir::types::DType;
use crate::kernel::registry::KernelRegistry;
use crate::target::Target;

// ---------------------------------------------------------------------------
// Pass pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PassKind {
    WeightBindingResolution,
    DeadCodeElimination,
    QKVFusion,
    ElementwiseFusion,
    AttentionFusion,
    DualPathInsertion,
    KernelMatching,
    LayoutOptimization,
    MemoryPlanning,
    Scheduling,
}

pub struct PassPipeline {
    pub passes: Vec<PassKind>,
}

impl Default for PassPipeline {
    fn default() -> Self {
        PassPipeline {
            passes: vec![
                PassKind::WeightBindingResolution,
                PassKind::DeadCodeElimination,
                PassKind::QKVFusion,
                PassKind::ElementwiseFusion,
                PassKind::AttentionFusion,
                PassKind::DeadCodeElimination,
                PassKind::DualPathInsertion,
                PassKind::KernelMatching,
                PassKind::LayoutOptimization,
                PassKind::MemoryPlanning,
                PassKind::Scheduling,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Compilation result
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CompilationResult {
    pub graph: CompGraph,
    pub memory_plan: memory::MemoryPlan,
    pub target: Target,
    pub stats: CompilationStats,
}

#[derive(Debug, Default)]
pub struct CompilationStats {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub unique_kernels: usize,
    pub fusions_applied: usize,
    pub estimated_flops_per_token: f64,
    pub peak_activation_bytes: usize,
    pub kernel_launches_per_forward: usize,
}

// ---------------------------------------------------------------------------
// Compiled artifact (code emission output)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum CompiledArtifact {
    Metal {
        metallib_path: String,
        orchestrator_source: String,
        weight_layout: WeightFileLayout,
    },
}

#[derive(Debug)]
pub struct WeightFileLayout {
    pub strategy: WeightStrategy,
    pub tensors: Vec<WeightTensorLayout>,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum WeightStrategy {
    MmapSafetensors,
    MmapRepacked,
    Embedded,
}

#[derive(Debug)]
pub struct WeightTensorLayout {
    pub name: String,
    pub offset: usize,
    pub size_bytes: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Pipeline runner
// ---------------------------------------------------------------------------

impl PassPipeline {
    pub fn run(
        &self,
        mut graph: CompGraph,
        registry: &KernelRegistry,
        target: &Target,
        weight_files: &[WeightFile],
    ) -> anyhow::Result<CompilationResult> {
        let nodes_before = graph.num_nodes();
        let mut stats = CompilationStats {
            nodes_before,
            ..Default::default()
        };

        for pass in &self.passes {
            log::debug!("Running pass: {:?}", pass);
            match pass {
                PassKind::WeightBindingResolution => {
                    passes::weight_binding::resolve_weight_bindings(&mut graph, weight_files)
                        .context("WeightBindingResolution")?;
                }
                PassKind::DeadCodeElimination => {
                    passes::dce::eliminate_dead_code(&mut graph);
                }
                PassKind::QKVFusion => {
                    let n = passes::qkv_fusion::fuse_qkv(&mut graph);
                    stats.fusions_applied += n;
                }
                PassKind::ElementwiseFusion => {
                    let n = passes::elementwise_fusion::fuse_elementwise(&mut graph);
                    stats.fusions_applied += n;
                }
                PassKind::AttentionFusion => {
                    // Attention is already lowered as ScaledDotProductAttention by templates
                    // This pass is a no-op for now (patterns were already in template)
                }
                PassKind::DualPathInsertion => {
                    passes::dual_path::insert_dual_paths(&mut graph, registry, target);
                }
                PassKind::KernelMatching => {
                    let n = passes::kernel_matching::match_kernels(&mut graph, registry, target)
                        .context("KernelMatching")?;
                    stats.unique_kernels = n;
                }
                PassKind::LayoutOptimization => {
                    // Layout optimization: ensure tensors are in expected layout for kernels.
                    // For Metal, all tensors are row-major by default — no-op for now.
                }
                PassKind::MemoryPlanning => {
                    // Done after kernel matching, results stored in CompilationResult
                }
                PassKind::Scheduling => {
                    // The graph is already in topological order from template construction
                }
            }
        }

        let memory_plan = memory::plan_memory(&graph);
        stats.nodes_after = graph.num_nodes();
        stats.peak_activation_bytes = memory_plan.total_activation_bytes;
        stats.kernel_launches_per_forward = graph.nodes.iter()
            .filter(|n| n.kernel.is_some())
            .count();

        Ok(CompilationResult {
            graph,
            memory_plan,
            target: target.clone(),
            stats,
        })
    }
}
