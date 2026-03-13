//! Kernel registry: the bridge between the IR and compiled kernel code.
//!
//! The registry holds `KernelEntry` descriptors for every available kernel on every
//! target. The kernel matching pass walks the CompIR graph and queries the registry
//! to find the best kernel for each node.

use std::collections::HashMap;

use crate::ir::graph::DispatchConfig;
use crate::ir::ops::{FusionPattern, Op};
use crate::ir::types::*;
use crate::target::Target;

// ---------------------------------------------------------------------------
// Kernel entry
// ---------------------------------------------------------------------------

/// A single kernel available in the library.
#[derive(Debug, Clone)]
pub struct KernelEntry {
    pub id: KernelId,
    pub name: String,
    pub op_pattern: OpPattern,
    pub input_constraints: Vec<TensorConstraint>,
    pub output_dtype: DType,
    pub target: Target,
    pub source: KernelSource,
    pub dispatch_template: DispatchTemplate,
    pub perf_model: PerfModel,
}

/// What operation pattern a kernel implements.
#[derive(Debug, Clone, PartialEq)]
pub enum OpPattern {
    Single(Op),
    Fused(FusionPattern),
    FlashAttention {
        supported_head_dims: Vec<usize>,
        causal: bool,
        max_gqa_factor: usize,
    },
    GEMV {
        quantized: bool,
        weight_dtype: Option<DType>,
    },
    GEMM {
        tile_m: u32,
        tile_n: u32,
        tile_k: u32,
    },
}

/// Constraints on an input tensor for a kernel to be applicable.
#[derive(Debug, Clone)]
pub struct TensorConstraint {
    pub dtypes: Vec<DType>,
    pub shape_ranges: Vec<Option<DimRange>>,
    pub layout: LayoutRequirement,
    pub alignment: usize,
}

impl TensorConstraint {
    /// Unconstrained: accept any dtype from the list, any shape, any layout.
    pub fn any(dtypes: Vec<DType>) -> Self {
        TensorConstraint {
            dtypes,
            shape_ranges: Vec::new(),
            layout: LayoutRequirement::Any,
            alignment: 1,
        }
    }
}

/// Range constraint on a single dimension.
#[derive(Debug, Clone, Copy)]
pub struct DimRange {
    pub min: usize,
    pub max: usize,
    pub divisible_by: Option<usize>,
}

/// Memory layout requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutRequirement {
    Any,
    RowMajor,
    ColMajor,
    Contiguous,
}

/// Where this kernel comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelSource {
    MLX { file: String, variant: String },
    Custom { file: String },
}

// ---------------------------------------------------------------------------
// Dispatch template
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DispatchTemplate {
    pub threadgroup: [u32; 3],
    pub grid_rules: [GridRule; 3],
    pub shared_memory_bytes: u32,
}

#[derive(Debug, Clone)]
pub enum GridRule {
    Constant(u32),
    CeilDivOutputDim { axis: usize, tile_size: u32 },
    CeilDivInputDim { input_idx: usize, axis: usize, tile_size: u32 },
    ModelParam(String),
    RuntimeParam(ParamName),
}

// ---------------------------------------------------------------------------
// Performance model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PerfModel {
    ComputeBound { utilization: f32 },
    MemoryBound { utilization: f32 },
    Roofline { compute_utilization: f32, memory_utilization: f32 },
    Fixed(f64),
}

// ---------------------------------------------------------------------------
// Kernel registry
// ---------------------------------------------------------------------------

pub struct KernelRegistry {
    entries: HashMap<KernelId, KernelEntry>,
    /// target_name -> op_class -> kernel IDs
    index: HashMap<String, HashMap<OpClass, Vec<KernelId>>>,
    next_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpClass {
    GEMM,
    GEMV,
    Attention,
    Normalization,
    Activation,
    Elementwise,
    Positional,
    Quantized,
    Reshape,
    KVCache,
    Fused,
    Other,
}

impl OpClass {
    pub fn from_op(op: &Op) -> Self {
        match op {
            Op::MatMul | Op::BatchMatMul { .. } | Op::LogitProjection => OpClass::GEMM,
            Op::MatVec => OpClass::GEMV,
            Op::QuantizedMatMul { .. } | Op::QuantizedMatVec { .. } => OpClass::Quantized,
            Op::ScaledDotProductAttention { .. } => OpClass::Attention,
            Op::RMSNorm { .. } | Op::LayerNorm { .. } | Op::QKNorm { .. } => OpClass::Normalization,
            Op::SiLU | Op::GELU | Op::ReLU => OpClass::Activation,
            Op::Mul | Op::Add | Op::Scale { .. } | Op::Softmax { .. } => OpClass::Elementwise,
            Op::RoPE { .. } => OpClass::Positional,
            Op::Reshape { .. } | Op::Transpose { .. } | Op::Concatenate { .. } | Op::Split { .. } => {
                OpClass::Reshape
            }
            Op::KVCacheAppend { .. } => OpClass::KVCache,
            Op::Gather { .. } | Op::CausalMask => OpClass::Other,
            Op::Fused(_) => OpClass::Fused,
        }
    }

    fn from_op_pattern(pattern: &OpPattern) -> Self {
        match pattern {
            OpPattern::Single(op) => OpClass::from_op(op),
            OpPattern::Fused(_) => OpClass::Fused,
            OpPattern::FlashAttention { .. } => OpClass::Attention,
            OpPattern::GEMV { quantized, .. } => {
                if *quantized { OpClass::Quantized } else { OpClass::GEMV }
            }
            OpPattern::GEMM { .. } => OpClass::GEMM,
        }
    }
}

/// Result of querying the registry for a matching kernel.
#[derive(Debug)]
pub struct KernelMatch {
    pub kernel: KernelEntry,
    pub estimated_cost: f64,
    pub dispatch: DispatchConfig,
}

impl KernelRegistry {
    pub fn new() -> Self {
        KernelRegistry {
            entries: HashMap::new(),
            index: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a kernel; returns its assigned ID.
    pub fn register(&mut self, mut entry: KernelEntry) -> KernelId {
        let id = KernelId(self.next_id);
        self.next_id += 1;
        entry.id = id;

        let target_key = entry.target.name().to_string();
        let op_class = OpClass::from_op_pattern(&entry.op_pattern);
        self.index
            .entry(target_key)
            .or_default()
            .entry(op_class)
            .or_default()
            .push(id);

        self.entries.insert(id, entry);
        id
    }

    /// Find the best matching kernel for a given op on the given target.
    pub fn find_best(
        &self,
        op: &Op,
        input_tensors: &[&TensorRef],
        output_shape: &Shape,
        target: &Target,
        model_params: Option<&crate::ir::graph::ModelParams>,
    ) -> Option<KernelMatch> {
        let target_key = target.name();
        let op_class = OpClass::from_op(op);

        let candidates = self.index.get(target_key)?.get(&op_class)?;

        let mut best: Option<KernelMatch> = None;

        for &kid in candidates {
            let entry = &self.entries[&kid];

            if !entry.op_pattern.matches(op) {
                continue;
            }
            if !self.check_constraints(&entry.input_constraints, input_tensors) {
                continue;
            }

            let dispatch = entry.dispatch_template.resolve(input_tensors, output_shape, model_params);
            let cost = self.estimate_cost(&entry.perf_model, input_tensors, output_shape);

            let is_better = best.as_ref().map_or(true, |b| cost < b.estimated_cost);
            if is_better {
                best = Some(KernelMatch {
                    kernel: entry.clone(),
                    estimated_cost: cost,
                    dispatch,
                });
            }
        }

        best
    }

    fn check_constraints(&self, constraints: &[TensorConstraint], inputs: &[&TensorRef]) -> bool {
        if constraints.len() > inputs.len() {
            return false;
        }
        for (constraint, tensor) in constraints.iter().zip(inputs.iter()) {
            if !constraint.dtypes.contains(&tensor.dtype) {
                return false;
            }
            for (i, range_opt) in constraint.shape_ranges.iter().enumerate() {
                if let Some(range) = range_opt {
                    if i >= tensor.shape.0.len() {
                        return false;
                    }
                    let dim_max = tensor.shape.0[i].max_value();
                    if dim_max < range.min || dim_max > range.max {
                        return false;
                    }
                    if let Some(div) = range.divisible_by {
                        if tensor.shape.0[i].is_static() && dim_max % div != 0 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn estimate_cost(&self, model: &PerfModel, inputs: &[&TensorRef], output_shape: &Shape) -> f64 {
        match model {
            PerfModel::ComputeBound { utilization } => {
                output_shape.max_numel() as f64 / *utilization as f64
            }
            PerfModel::MemoryBound { utilization } => {
                let total: f64 = inputs.iter().map(|t| t.max_size_bytes() as f64).sum();
                total / *utilization as f64
            }
            PerfModel::Roofline { compute_utilization, memory_utilization } => {
                let compute = output_shape.max_numel() as f64 / *compute_utilization as f64;
                let memory: f64 = inputs
                    .iter()
                    .map(|t| t.max_size_bytes() as f64)
                    .sum::<f64>()
                    / *memory_utilization as f64;
                compute.max(memory)
            }
            PerfModel::Fixed(cost) => *cost,
        }
    }

    pub fn get(&self, id: KernelId) -> Option<&KernelEntry> {
        self.entries.get(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Op pattern matching
// ---------------------------------------------------------------------------

impl OpPattern {
    pub fn matches(&self, op: &Op) -> bool {
        match (self, op) {
            (OpPattern::Single(pattern_op), op) => {
                std::mem::discriminant(pattern_op) == std::mem::discriminant(op)
            }
            (OpPattern::Fused(pattern_fusion), Op::Fused(op_fusion)) => {
                pattern_fusion == op_fusion
            }
            (
                OpPattern::FlashAttention { supported_head_dims, causal: pattern_causal, max_gqa_factor },
                Op::ScaledDotProductAttention { causal, gqa_factor, head_dim, .. },
            ) => {
                supported_head_dims.contains(head_dim)
                    && (*pattern_causal || !*causal) // causal kernel can handle causal; non-causal pattern requires non-causal op
                    && *gqa_factor <= *max_gqa_factor
            }
            (OpPattern::GEMV { quantized: false, .. }, Op::MatVec) => true,
            (OpPattern::GEMV { quantized: true, weight_dtype }, Op::QuantizedMatVec { weight_dtype: wd }) => {
                weight_dtype.as_ref() == Some(wd)
            }
            (OpPattern::GEMM { .. }, Op::MatMul | Op::BatchMatMul { .. } | Op::LogitProjection) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch resolution
// ---------------------------------------------------------------------------

impl DispatchTemplate {
    pub fn resolve(&self, inputs: &[&TensorRef], output_shape: &Shape, model_params: Option<&crate::ir::graph::ModelParams>) -> DispatchConfig {
        use crate::ir::graph::DispatchDim;

        let resolve_rule = |rule: &GridRule| -> DispatchDim {
            match rule {
                GridRule::Constant(v) => DispatchDim::Static(*v),
                GridRule::CeilDivOutputDim { axis, tile_size } => {
                    match &output_shape.0[*axis] {
                        Dim::Static(v) => {
                            DispatchDim::Static((*v as u32 + tile_size - 1) / tile_size)
                        }
                        Dim::Param(p) => DispatchDim::CeilDiv { param: p.name, divisor: *tile_size },
                    }
                }
                GridRule::CeilDivInputDim { input_idx, axis, tile_size } => {
                    let dim = &inputs[*input_idx].shape.0[*axis];
                    match dim {
                        Dim::Static(v) => {
                            DispatchDim::Static((*v as u32 + tile_size - 1) / tile_size)
                        }
                        Dim::Param(p) => DispatchDim::CeilDiv { param: p.name, divisor: *tile_size },
                    }
                }
                GridRule::ModelParam(name) => {
                    if let Some(mp) = model_params {
                        let val = match name.as_str() {
                            "num_heads" => mp.num_attention_heads as u32,
                            "num_kv_heads" => mp.num_kv_heads as u32,
                            "hidden_size" => mp.hidden_size as u32,
                            "head_dim" => mp.head_dim as u32,
                            "num_layers" => mp.num_hidden_layers as u32,
                            "vocab_size" => mp.vocab_size as u32,
                            _ => 1,
                        };
                        DispatchDim::Static(val)
                    } else {
                        DispatchDim::Static(1)
                    }
                }
                GridRule::RuntimeParam(param) => DispatchDim::CeilDiv { param: *param, divisor: 1 },
            }
        };

        DispatchConfig {
            grid: [
                resolve_rule(&self.grid_rules[0]),
                resolve_rule(&self.grid_rules[1]),
                resolve_rule(&self.grid_rules[2]),
            ],
            threadgroup: self.threadgroup,
            shared_memory_bytes: self.shared_memory_bytes,
        }
    }
}
