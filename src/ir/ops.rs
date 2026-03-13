//! Operation types for the CompIR.
//!
//! Ops live at the right abstraction level: above individual multiply-adds
//! (too low for pattern matching) but below "TransformerBlock" (too high
//! for kernel selection). Each op maps cleanly to one or more kernel launches.

use crate::ir::types::DType;

/// The set of operations in the computation graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Op {
    // -----------------------------------------------------------------------
    // Matmul / GEMM
    // -----------------------------------------------------------------------
    MatMul,
    BatchMatMul { transpose_b: bool },
    MatVec,

    // -----------------------------------------------------------------------
    // Quantized operations
    // -----------------------------------------------------------------------
    QuantizedMatMul { weight_dtype: DType },
    QuantizedMatVec { weight_dtype: DType },

    // -----------------------------------------------------------------------
    // Normalization
    // -----------------------------------------------------------------------
    RMSNorm { eps: f32 },
    LayerNorm { eps: f32 },
    /// Per-head RMSNorm for Q/K projections (Qwen3-style QK normalization).
    /// Normalizes each head_dim-sized chunk independently using a shared weight [head_dim].
    QKNorm { eps: f32, num_heads: usize, head_dim: usize },

    // -----------------------------------------------------------------------
    // Positional encoding
    // -----------------------------------------------------------------------
    RoPE { base: f64, interleaved: bool },

    // -----------------------------------------------------------------------
    // Activations (elementwise, unary)
    // -----------------------------------------------------------------------
    SiLU,
    GELU,
    ReLU,

    // -----------------------------------------------------------------------
    // Elementwise binary
    // -----------------------------------------------------------------------
    Mul,
    Add,
    Scale { factor: f32 },

    // -----------------------------------------------------------------------
    // Reduction
    // -----------------------------------------------------------------------
    Softmax { axis: i32 },

    // -----------------------------------------------------------------------
    // Attention
    // -----------------------------------------------------------------------
    ScaledDotProductAttention {
        causal: bool,
        gqa_factor: usize,
        head_dim: usize,
        scale: f32,
        has_mask: bool,
        has_sinks: bool,
    },
    CausalMask,

    // -----------------------------------------------------------------------
    // Memory / reshape
    // -----------------------------------------------------------------------
    Gather { axis: usize },
    Reshape { target_shape: Vec<i64> },
    Transpose { perm: Vec<usize> },
    Concatenate { axis: usize },
    Split { axis: usize, sizes: Vec<usize> },

    // -----------------------------------------------------------------------
    // KV cache operations
    // -----------------------------------------------------------------------
    KVCacheAppend {
        layer: usize,
        role: crate::ir::types::KVRole,
    },

    // -----------------------------------------------------------------------
    // Output / sampling
    // -----------------------------------------------------------------------
    LogitProjection,

    // -----------------------------------------------------------------------
    // Fused operations (created by optimization passes only)
    // -----------------------------------------------------------------------
    Fused(FusionPattern),
}

/// Recognized fusion patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FusionPattern {
    RMSNormLinear,
    SwiGLU,
    QKVProjection { q_size: usize, k_size: usize, v_size: usize },
    LinearActivation { activation: ActivationType, has_bias: bool },
    LinearResidualAdd,
    Custom { name: String, ops: Vec<String> },
}

/// Activation types for fused linear+activation kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ActivationType {
    SiLU,
    GELU,
    ReLU,
    None,
}

impl Op {
    /// Returns true if this op is compute-bound (matmul-class).
    pub fn is_compute_bound(&self) -> bool {
        matches!(
            self,
            Op::MatMul
                | Op::BatchMatMul { .. }
                | Op::QuantizedMatMul { .. }
                | Op::ScaledDotProductAttention { .. }
                | Op::LogitProjection
                | Op::Fused(FusionPattern::QKVProjection { .. })
                | Op::Fused(FusionPattern::SwiGLU)
                | Op::Fused(FusionPattern::RMSNormLinear)
        )
    }

    /// Returns true if this op is memory-bound (elementwise / reduction).
    pub fn is_memory_bound(&self) -> bool {
        matches!(
            self,
            Op::RMSNorm { .. }
                | Op::LayerNorm { .. }
                | Op::QKNorm { .. }
                | Op::SiLU
                | Op::GELU
                | Op::ReLU
                | Op::Mul
                | Op::Add
                | Op::Scale { .. }
                | Op::Softmax { .. }
                | Op::RoPE { .. }
        )
    }

    /// Returns true if this op involves no computation (just a reinterpretation).
    pub fn is_zero_compute(&self) -> bool {
        matches!(self, Op::Reshape { .. } | Op::Transpose { .. } | Op::Split { .. })
    }
}
