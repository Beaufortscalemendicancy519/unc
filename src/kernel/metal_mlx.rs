//! Registration of MLX Metal kernels into the KernelRegistry.
//!
//! Each `KernelEntry` describes one Metal shader variant with its constraints,
//! dispatch template, and performance model. The kernel matching pass selects
//! the best entry for each graph node.

use crate::ir::ops::{FusionPattern, Op};
use crate::ir::types::DType;
use crate::kernel::registry::{
    DispatchTemplate, GridRule, KernelEntry, KernelRegistry, KernelSource,
    OpPattern, PerfModel, TensorConstraint,
};
use crate::target::{AppleGPUFamily, MetalTarget, Target};

/// Register all MLX Metal kernels for the given Metal target.
pub fn register_metal_mlx_kernels(registry: &mut KernelRegistry, target: &MetalTarget) {
    let t = Target::Metal(target.clone());
    let (tile_m, tile_n, tile_k) = target.gpu_family.preferred_gemm_tile_f16();

    // -----------------------------------------------------------------------
    // 1. GEMM (prefill path — compute-bound)
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: format!("steel_gemm_f16_{}x{}x{}", tile_m, tile_n, tile_k),
        op_pattern: OpPattern::GEMM { tile_m, tile_n, tile_k },
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "steel/gemm/kernels/steel_gemm.metal".to_string(),
            variant: format!("steel_gemm_f16_{tile_m}x{tile_n}x{tile_k}"),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [tile_m, 1, 1],
            grid_rules: [
                GridRule::CeilDivOutputDim { axis: 0, tile_size: tile_m },
                GridRule::CeilDivOutputDim { axis: 1, tile_size: tile_n },
                GridRule::Constant(1),
            ],
            shared_memory_bytes: tile_m * tile_k * 2 + tile_k * tile_n * 2, // f16 tiles
        },
        perf_model: PerfModel::ComputeBound { utilization: 0.65 },
    });

    // BF16 GEMM (Apple9+ only)
    if target.gpu_family >= AppleGPUFamily::Apple9 {
        registry.register(KernelEntry {
            id: Default::default(),
            name: "steel_gemm_bf16_64x64x32".to_string(),
            op_pattern: OpPattern::GEMM { tile_m: 64, tile_n: 64, tile_k: 32 },
            input_constraints: vec![
                TensorConstraint::any(vec![DType::BF16]),
                TensorConstraint::any(vec![DType::BF16]),
            ],
            output_dtype: DType::BF16,
            target: t.clone(),
            source: KernelSource::MLX {
                file: "steel/gemm/kernels/steel_gemm.metal".to_string(),
                variant: "steel_gemm_bf16_64x64x32".to_string(),
            },
            dispatch_template: DispatchTemplate {
                threadgroup: [64, 1, 1],
                grid_rules: [
                    GridRule::CeilDivOutputDim { axis: 0, tile_size: 64 },
                    GridRule::CeilDivOutputDim { axis: 1, tile_size: 64 },
                    GridRule::Constant(1),
                ],
                shared_memory_bytes: 64 * 32 * 2 + 32 * 64 * 2,
            },
            perf_model: PerfModel::ComputeBound { utilization: 0.65 },
        });
    }

    // -----------------------------------------------------------------------
    // 2. FlashAttention (steel_attn — prefill path)
    // -----------------------------------------------------------------------
    let attn_head_dims = vec![64usize, 80, 96, 128];
    registry.register(KernelEntry {
        id: Default::default(),
        name: "steel_attention_causal_f16".to_string(),
        op_pattern: OpPattern::FlashAttention {
            supported_head_dims: attn_head_dims.clone(),
            causal: true,
            max_gqa_factor: 8,
        },
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]), // Q
            TensorConstraint::any(vec![DType::F16]), // K
            TensorConstraint::any(vec![DType::F16]), // V
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "steel/attn/kernels/steel_attention.metal".to_string(),
            variant: "steel_attention_causal_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [32, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::ModelParam("num_heads".to_string()),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0, // uses simd_sum, no threadgroup memory needed
        },
        perf_model: PerfModel::ComputeBound { utilization: 0.70 },
    });

    // Non-causal variant
    registry.register(KernelEntry {
        id: Default::default(),
        name: "steel_attention_f16".to_string(),
        op_pattern: OpPattern::FlashAttention {
            supported_head_dims: attn_head_dims,
            causal: false,
            max_gqa_factor: 8,
        },
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "steel/attn/kernels/steel_attention.metal".to_string(),
            variant: "steel_attention_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [32, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::ModelParam("num_heads".to_string()),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0, // uses simd_sum, no threadgroup memory needed
        },
        perf_model: PerfModel::ComputeBound { utilization: 0.70 },
    });

    // -----------------------------------------------------------------------
    // 3. GEMV (decode path — memory-bound, seq_len = 1)
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "gemv_f16".to_string(),
        op_pattern: OpPattern::GEMV { quantized: false, weight_dtype: None },
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]), // x (vector)
            TensorConstraint::any(vec![DType::F16]), // W (matrix)
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "gemv.metal".to_string(),
            variant: "gemv_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],  // GEMV_TG_SIZE=256, 1D dispatch
            grid_rules: [
                GridRule::CeilDivOutputDim { axis: 0, tile_size: 64 },  // GEMV_BM=8*8=64
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            // Shared-memory-free GEMV: reads activation directly from device memory.
            // With compact decode buffer (act_stride=1), reads are contiguous and cache well.
            shared_memory_bytes: 0,
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.75 },
    });

    // -----------------------------------------------------------------------
    // 4. Quantized GEMV
    // -----------------------------------------------------------------------
    for (quant_dtype, variant) in [
        (DType::Q4_0, "quantized_gemv_q4_0"),
        (DType::Q8_0, "quantized_gemv_q8_0"),
    ] {
        registry.register(KernelEntry {
            id: Default::default(),
            name: variant.to_string(),
            op_pattern: OpPattern::GEMV { quantized: true, weight_dtype: Some(quant_dtype) },
            input_constraints: vec![
                TensorConstraint::any(vec![DType::F16]),       // x
                TensorConstraint::any(vec![quant_dtype]),       // W (quantized)
            ],
            output_dtype: DType::F16,
            target: t.clone(),
            source: KernelSource::MLX {
                file: "quantized.metal".to_string(),
                variant: variant.to_string(),
            },
            dispatch_template: DispatchTemplate {
                threadgroup: [32, 4, 1],
                grid_rules: [
                    GridRule::CeilDivOutputDim { axis: 0, tile_size: 4 },
                    GridRule::Constant(1),
                    GridRule::Constant(1),
                ],
                shared_memory_bytes: 0,
            },
            perf_model: PerfModel::MemoryBound { utilization: 0.80 },
        });
    }

    // -----------------------------------------------------------------------
    // 5. RMSNorm
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "rms_norm_f16".to_string(),
        op_pattern: OpPattern::Single(Op::RMSNorm { eps: 1e-5 }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]),
            TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "rms_norm.metal".to_string(),
            variant: "rms_norm_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 1024, // 256 floats for parallel reduction
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.80 },
    });

    // -----------------------------------------------------------------------
    // 5b. QKNorm (per-head RMSNorm for Q/K — Qwen3)
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "qk_norm_f16".to_string(),
        op_pattern: OpPattern::Single(Op::QKNorm { eps: 1e-6, num_heads: 32, head_dim: 128 }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]),
            TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::Custom {
            file: "unc_kernels.metal".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [128, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 512, // 128 floats for parallel reduction
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.80 },
    });

    // -----------------------------------------------------------------------
    // 6. RoPE
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "rope_f16".to_string(),
        op_pattern: OpPattern::Single(Op::RoPE { base: 10000.0, interleaved: false }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16, DType::BF16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "rope.metal".to_string(),
            variant: "rope_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0,
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.85 },
    });

    // -----------------------------------------------------------------------
    // 7. Softmax
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "softmax_f16".to_string(),
        op_pattern: OpPattern::Single(Op::Softmax { axis: -1 }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "softmax.metal".to_string(),
            variant: "softmax_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 1024, // 256 floats for parallel reduction
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.80 },
    });

    // -----------------------------------------------------------------------
    // 8. Elementwise ops (SiLU, GELU, Add, Mul)
    // -----------------------------------------------------------------------
    let elementwise_pairs = [
        (Op::SiLU, "silu_f16"),
        (Op::GELU, "gelu_f16"),
        (Op::ReLU, "relu_f16"),
        (Op::Add, "add_f16"),
        (Op::Mul, "mul_f16"),
    ];
    for (op, name) in elementwise_pairs {
        let n_inputs = match &op {
            Op::Add | Op::Mul => 2,
            _ => 1,
        };
        registry.register(KernelEntry {
            id: Default::default(),
            name: name.to_string(),
            op_pattern: OpPattern::Single(op),
            input_constraints: vec![
                TensorConstraint::any(vec![DType::F16, DType::BF16, DType::F32]);
                n_inputs
            ],
            output_dtype: DType::F16,
            target: t.clone(),
            source: KernelSource::MLX {
                file: "unary.metal".to_string(),
                variant: name.to_string(),
            },
            dispatch_template: DispatchTemplate {
                threadgroup: [256, 1, 1],
                grid_rules: [
                    GridRule::CeilDivOutputDim { axis: 0, tile_size: 256 },
                    GridRule::Constant(1),
                    GridRule::Constant(1),
                ],
                shared_memory_bytes: 0,
            },
            perf_model: PerfModel::MemoryBound { utilization: 0.90 },
        });
    }

    // -----------------------------------------------------------------------
    // 9. Gather (embedding lookup)
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "gather_f16".to_string(),
        op_pattern: OpPattern::Single(Op::Gather { axis: 0 }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::U32, DType::I32]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::MLX {
            file: "binary.metal".to_string(),
            variant: "gather_f16".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0,
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.75 },
    });

    // -----------------------------------------------------------------------
    // 10. Fused: SwiGLU
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "swiglu_f16".to_string(),
        op_pattern: OpPattern::Fused(FusionPattern::SwiGLU),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::Custom {
            file: "custom/swiglu.metal".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::CeilDivOutputDim { axis: 0, tile_size: 256 },
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0,
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.85 },
    });

    // -----------------------------------------------------------------------
    // 11. Fused: RMSNorm + Linear
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "rms_norm_linear_f16".to_string(),
        op_pattern: OpPattern::Fused(FusionPattern::RMSNormLinear),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::Custom {
            file: "custom/rms_norm_linear.metal".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [tile_m, 1, 1],
            grid_rules: [
                GridRule::CeilDivOutputDim { axis: 0, tile_size: tile_m },
                GridRule::CeilDivOutputDim { axis: 1, tile_size: tile_n },
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 4096,
        },
        perf_model: PerfModel::ComputeBound { utilization: 0.65 },
    });

    // -----------------------------------------------------------------------
    // 12. KV cache append (zero-compute, just a memory copy)
    // -----------------------------------------------------------------------
    registry.register(KernelEntry {
        id: Default::default(),
        name: "kv_cache_append_f16".to_string(),
        op_pattern: OpPattern::Single(Op::KVCacheAppend {
            layer: 0, // pattern-matched by discriminant, layer is ignored
            role: crate::ir::types::KVRole::Key,
        }),
        input_constraints: vec![
            TensorConstraint::any(vec![DType::F16]),
            TensorConstraint::any(vec![DType::F16]),
        ],
        output_dtype: DType::F16,
        target: t.clone(),
        source: KernelSource::Custom {
            file: "custom/kv_cache.metal".to_string(),
        },
        dispatch_template: DispatchTemplate {
            threadgroup: [256, 1, 1],
            grid_rules: [
                GridRule::RuntimeParam(crate::ir::types::ParamName::SeqLen),
                GridRule::Constant(1),
                GridRule::Constant(1),
            ],
            shared_memory_bytes: 0,
        },
        perf_model: PerfModel::MemoryBound { utilization: 0.90 },
    });
}

// ---------------------------------------------------------------------------
// Default KernelId for construction
// ---------------------------------------------------------------------------

impl Default for crate::ir::types::KernelId {
    fn default() -> Self {
        crate::ir::types::KernelId(u32::MAX) // sentinel — overwritten by register()
    }
}
