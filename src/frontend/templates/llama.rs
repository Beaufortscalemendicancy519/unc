//! LLaMA/LLaMA-2/LLaMA-3 graph template.
//!
//! Constructs a CompGraph from ModelParams. Weight byte offsets are left as 0
//! (placeholders) — the WeightBindingResolution pass fills them in later.

use std::collections::HashMap;

use crate::frontend::huggingface::WeightFile;
use crate::ir::graph::{
    ArchitectureFamily, CompGraph, FFNType, GraphMetadata, ModelParams,
};
use crate::ir::ops::Op;
use crate::ir::types::{
    Dim, DType, KVRole, ParamDim, ParamName, Shape, StorageClass, WeightBinding,
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Lower LLaMA model params into a CompGraph.
///
/// The graph is in "pre-optimization" form: weights have placeholder byte offsets,
/// no kernels are assigned, and ops are not yet fused.
pub fn lower_llama(
    params: &ModelParams,
    weight_files: &[WeightFile],
    model_id: &str,
    tokenizer_path: Option<&str>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
) -> CompGraph {
    let weight_file_paths: Vec<String> = weight_files
        .iter()
        .map(|w| w.path().to_string_lossy().into_owned())
        .collect();

    let metadata = GraphMetadata {
        model_id: model_id.to_string(),
        architecture: ArchitectureFamily::LLaMA,
        params: params.clone(),
        weight_files: weight_file_paths.clone(),
        total_weight_bytes: 0, // filled in by WeightBindingResolution
        tokenizer_path: tokenizer_path.map(str::to_owned),
        bos_token_id: bos_token_id.or(Some(1)),
        eos_token_id: eos_token_id.or(Some(2)),
        weight_file_paths,
        quant: "f16".to_string(),
    };

    let mut b = GraphBuilder::new(params, metadata);
    b.build();
    b.graph
}

// ---------------------------------------------------------------------------
// Graph builder helpers
// ---------------------------------------------------------------------------

struct GraphBuilder<'a> {
    params: &'a ModelParams,
    graph: CompGraph,
    /// seq_len parameter dimension (dynamic)
    seq_dim: Dim,
    /// total_seq_len (seq + kv cache) for attention
    total_seq_dim: Dim,
}

impl<'a> GraphBuilder<'a> {
    fn new(params: &'a ModelParams, metadata: GraphMetadata) -> Self {
        let max_seq = params.max_position_embeddings;
        let seq_dim = Dim::Param(ParamDim {
            name: ParamName::SeqLen,
            min: 1,
            max: max_seq,
            alignment: 1,
        });
        let total_seq_dim = Dim::Param(ParamDim {
            name: ParamName::TotalSeqLen,
            min: 1,
            max: max_seq,
            alignment: 1,
        });
        GraphBuilder {
            params,
            graph: CompGraph::new(metadata),
            seq_dim,
            total_seq_dim,
        }
    }

    /// Create a static shape (all dims known at compile time).
    fn static_shape(&self, dims: &[usize]) -> Shape {
        Shape(dims.iter().map(|&d| Dim::Static(d)).collect())
    }

    /// Create a weight tensor with a placeholder byte offset.
    fn weight(&mut self, name: &str, shape: &[usize], dtype: DType) -> crate::ir::types::TensorId {
        let s = self.static_shape(shape);
        let binding = WeightBinding {
            name: name.to_string(),
            byte_offset: 0,
            byte_size: 0,
            file_shape: shape.to_vec(),
            file_dtype: dtype,
        };
        self.graph.new_tensor(s, dtype, StorageClass::Weight(binding))
    }

    /// Create an activation tensor with dynamic seq_len.
    fn activation_seq(&mut self, shape_prefix: &[usize], dtype: DType) -> crate::ir::types::TensorId {
        let mut dims: Vec<Dim> = shape_prefix.iter().map(|&d| Dim::Static(d)).collect();
        dims.push(self.seq_dim);
        let s = Shape(dims);
        self.graph.new_tensor(s, dtype, StorageClass::Activation { buffer: None })
    }

    /// Add an op node.
    fn node(
        &mut self,
        op: Op,
        inputs: Vec<crate::ir::types::TensorId>,
        output: crate::ir::types::TensorId,
    ) -> crate::ir::types::NodeId {
        self.graph.add_node(op, inputs, output, HashMap::new())
    }

    fn build(&mut self) {
        let p = self.params;
        let h = p.hidden_size;
        let heads = p.num_attention_heads;
        let kv_heads = p.num_kv_heads;
        let head_dim = p.head_dim;
        let layers = p.num_hidden_layers;
        let vocab = p.vocab_size;
        let ffn_h = p.intermediate_size;
        let eps = p.rms_norm_eps as f32;
        let rope_base = p.rope_theta;

        // -- Input token IDs: [seq] --
        let token_ids = {
            let s = Shape(vec![self.seq_dim]);
            self.graph.new_tensor(s, DType::U32, StorageClass::Activation { buffer: None })
        };
        self.graph.inputs.push(token_ids);

        // -- Embedding table: [vocab, hidden] --
        let embed_table = self.weight("model.embed_tokens.weight", &[vocab, h], DType::F16);

        // -- Embedding lookup: [seq, hidden] --
        let embed_out = self.activation_seq(&[h], DType::F16);
        self.node(Op::Gather { axis: 0 }, vec![embed_table, token_ids], embed_out);

        let mut x = embed_out;

        // -- Transformer layers --
        for layer in 0..layers {
            let ln = format!("model.layers.{layer}");

            // === Self-attention ===

            // Input layernorm weight: [hidden]
            let norm_w = self.weight(&format!("{ln}.input_layernorm.weight"), &[h], DType::F16);

            // Normed: [seq, hidden]
            let normed = self.activation_seq(&[h], DType::F16);
            self.node(Op::RMSNorm { eps }, vec![x, norm_w], normed);

            // Q projection: [hidden, q_dim]  where q_dim = heads * head_dim
            let q_dim = heads * head_dim;
            let k_dim = kv_heads * head_dim;
            let v_dim = kv_heads * head_dim;

            let wq = self.weight(&format!("{ln}.self_attn.q_proj.weight"), &[q_dim, h], DType::F16);
            let wk = self.weight(&format!("{ln}.self_attn.k_proj.weight"), &[k_dim, h], DType::F16);
            let wv = self.weight(&format!("{ln}.self_attn.v_proj.weight"), &[v_dim, h], DType::F16);
            let wo = self.weight(&format!("{ln}.self_attn.o_proj.weight"), &[h, q_dim], DType::F16);

            // Q: [seq, q_dim]
            let q_raw = self.activation_seq(&[q_dim], DType::F16);
            self.node(Op::MatMul, vec![normed, wq], q_raw);

            // K: [seq, k_dim]
            let k_raw = self.activation_seq(&[k_dim], DType::F16);
            self.node(Op::MatMul, vec![normed, wk], k_raw);

            // V: [seq, v_dim]
            let v = self.activation_seq(&[v_dim], DType::F16);
            self.node(Op::MatMul, vec![normed, wv], v);

            // Per-head Q/K normalization (Qwen3)
            let (q, k) = if p.qk_norm {
                let wq_norm = self.weight(&format!("{ln}.self_attn.q_norm.weight"), &[head_dim], DType::F16);
                let wk_norm = self.weight(&format!("{ln}.self_attn.k_norm.weight"), &[head_dim], DType::F16);
                let q_normed = self.activation_seq(&[q_dim], DType::F16);
                self.node(Op::QKNorm { eps, num_heads: heads, head_dim }, vec![q_raw, wq_norm], q_normed);
                let k_normed = self.activation_seq(&[k_dim], DType::F16);
                self.node(Op::QKNorm { eps, num_heads: kv_heads, head_dim }, vec![k_raw, wk_norm], k_normed);
                (q_normed, k_normed)
            } else {
                (q_raw, k_raw)
            };

            // RoPE on Q and K
            let q_rope = self.activation_seq(&[q_dim], DType::F16);
            self.node(
                Op::RoPE { base: rope_base, interleaved: false },
                vec![q],
                q_rope,
            );
            let k_rope = self.activation_seq(&[k_dim], DType::F16);
            self.node(
                Op::RoPE { base: rope_base, interleaved: false },
                vec![k],
                k_rope,
            );

            // KV cache append
            let k_cache = {
                let s = Shape(vec![
                    Dim::Static(kv_heads),
                    self.total_seq_dim,
                    Dim::Static(head_dim),
                ]);
                self.graph.new_tensor(
                    s,
                    DType::F16,
                    StorageClass::KVCache { layer, head_group: 0, role: KVRole::Key },
                )
            };
            let v_cache = {
                let s = Shape(vec![
                    Dim::Static(kv_heads),
                    self.total_seq_dim,
                    Dim::Static(head_dim),
                ]);
                self.graph.new_tensor(
                    s,
                    DType::F16,
                    StorageClass::KVCache { layer, head_group: 0, role: KVRole::Value },
                )
            };
            self.node(Op::KVCacheAppend { layer, role: KVRole::Key }, vec![k_cache, k_rope], k_cache);
            self.node(Op::KVCacheAppend { layer, role: KVRole::Value }, vec![v_cache, v], v_cache);

            // Scaled dot-product attention → [seq, q_dim]
            let gqa_factor = heads / kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let attn_out = self.activation_seq(&[q_dim], DType::F16);
            self.node(
                Op::ScaledDotProductAttention {
                    causal: true,
                    gqa_factor,
                    head_dim,
                    scale,
                    has_mask: false,
                    has_sinks: false,
                },
                vec![q_rope, k_cache, v_cache],
                attn_out,
            );

            // Output projection: [seq, hidden]
            let attn_proj = self.activation_seq(&[h], DType::F16);
            self.node(Op::MatMul, vec![attn_out, wo], attn_proj);

            // Residual add
            let x_after_attn = self.activation_seq(&[h], DType::F16);
            self.node(Op::Add, vec![x, attn_proj], x_after_attn);

            // === FFN ===

            let ffn_norm_w =
                self.weight(&format!("{ln}.post_attention_layernorm.weight"), &[h], DType::F16);
            let ffn_normed = self.activation_seq(&[h], DType::F16);
            self.node(Op::RMSNorm { eps }, vec![x_after_attn, ffn_norm_w], ffn_normed);

            x = match p.ffn_type {
                FFNType::SwiGLU => {
                    self.build_swiglu_ffn(&ln, ffn_normed, h, ffn_h, x_after_attn)
                }
                FFNType::GeGLU => {
                    self.build_geglu_ffn(&ln, ffn_normed, h, ffn_h, x_after_attn)
                }
                FFNType::Standard => {
                    self.build_standard_ffn(&ln, ffn_normed, h, ffn_h, x_after_attn)
                }
            };
        }

        // -- Final norm --
        let final_norm_w = self.weight("model.norm.weight", &[h], DType::F16);
        let normed_final = self.activation_seq(&[h], DType::F16);
        self.node(Op::RMSNorm { eps }, vec![x, final_norm_w], normed_final);

        // -- LM head / logit projection --
        let lm_head_w = if p.tie_word_embeddings {
            // Reuse embed table (same weight tensor id as embed_table)
            embed_table
        } else {
            self.weight("lm_head.weight", &[vocab, h], DType::F16)
        };

        // logits: [seq, vocab]  (only last token matters at decode, but we emit full)
        let logits = self.activation_seq(&[vocab], DType::F16);
        self.node(Op::LogitProjection, vec![normed_final, lm_head_w], logits);

        self.graph.outputs.push(logits);
    }

    fn build_swiglu_ffn(
        &mut self,
        layer_prefix: &str,
        normed: crate::ir::types::TensorId,
        h: usize,
        ffn_h: usize,
        residual: crate::ir::types::TensorId,
    ) -> crate::ir::types::TensorId {
        let w_gate =
            self.weight(&format!("{layer_prefix}.mlp.gate_proj.weight"), &[ffn_h, h], DType::F16);
        let w_up =
            self.weight(&format!("{layer_prefix}.mlp.up_proj.weight"), &[ffn_h, h], DType::F16);
        let w_down =
            self.weight(&format!("{layer_prefix}.mlp.down_proj.weight"), &[h, ffn_h], DType::F16);

        let gate = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::MatMul, vec![normed, w_gate], gate);

        let up = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::MatMul, vec![normed, w_up], up);

        let gate_act = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::SiLU, vec![gate], gate_act);

        let gated = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::Mul, vec![gate_act, up], gated);

        let down = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::MatMul, vec![gated, w_down], down);

        let out = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::Add, vec![residual, down], out);
        out
    }

    fn build_geglu_ffn(
        &mut self,
        layer_prefix: &str,
        normed: crate::ir::types::TensorId,
        h: usize,
        ffn_h: usize,
        residual: crate::ir::types::TensorId,
    ) -> crate::ir::types::TensorId {
        let w_gate =
            self.weight(&format!("{layer_prefix}.mlp.gate_proj.weight"), &[ffn_h, h], DType::F16);
        let w_up =
            self.weight(&format!("{layer_prefix}.mlp.up_proj.weight"), &[ffn_h, h], DType::F16);
        let w_down =
            self.weight(&format!("{layer_prefix}.mlp.down_proj.weight"), &[h, ffn_h], DType::F16);

        let gate = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::MatMul, vec![normed, w_gate], gate);

        let up = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::MatMul, vec![normed, w_up], up);

        let gate_act = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::GELU, vec![gate], gate_act);

        let gated = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::Mul, vec![gate_act, up], gated);

        let down = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::MatMul, vec![gated, w_down], down);

        let out = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::Add, vec![residual, down], out);
        out
    }

    fn build_standard_ffn(
        &mut self,
        layer_prefix: &str,
        normed: crate::ir::types::TensorId,
        h: usize,
        ffn_h: usize,
        residual: crate::ir::types::TensorId,
    ) -> crate::ir::types::TensorId {
        let w_up = self.weight(&format!("{layer_prefix}.mlp.dense_h_to_4h.weight"), &[ffn_h, h], DType::F16);
        let w_down = self.weight(&format!("{layer_prefix}.mlp.dense_4h_to_h.weight"), &[h, ffn_h], DType::F16);

        let up = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::MatMul, vec![normed, w_up], up);

        let act = self.activation_seq(&[ffn_h], DType::F16);
        self.node(Op::GELU, vec![up], act);

        let down = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::MatMul, vec![act, w_down], down);

        let out = self.activation_seq(&[self.params.hidden_size], DType::F16);
        self.node(Op::Add, vec![residual, down], out);
        out
    }
}
