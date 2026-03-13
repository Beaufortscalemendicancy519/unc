//! Parse HuggingFace config.json into UNC model parameters.

use std::path::Path;

use anyhow::{anyhow, Context};
use serde::Deserialize;

use crate::ir::graph::{
    ArchitectureFamily, AttentionType, FFNType, ModelParams, RoPEScaling,
};

// ---------------------------------------------------------------------------
// Raw config.json shape (serde)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RawConfig {
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    intermediate_size: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Value>,
    tie_word_embeddings: Option<bool>,
    hidden_act: Option<String>,
    // Phi-specific
    partial_rotary_factor: Option<f64>,
    // Head dim override (some models set this explicitly)
    head_dim: Option<usize>,
    // Token IDs
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse `config.json` at the given path into `(ArchitectureFamily, ModelParams)`.
/// Returns `(architecture, params, bos_token_id, eos_token_id)`.
pub fn parse_config(path: &Path) -> anyhow::Result<(ArchitectureFamily, ModelParams, Option<u32>, Option<u32>)> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let raw: RawConfig = serde_json::from_str(&json)
        .with_context(|| format!("parsing {}", path.display()))?;

    let arch_str = raw
        .architectures
        .as_ref()
        .and_then(|v| v.first())
        .map(|s| s.as_str())
        .unwrap_or("unknown");

    let architecture = map_architecture(arch_str)?;

    let hidden_size = raw
        .hidden_size
        .ok_or_else(|| anyhow!("missing hidden_size in config.json"))?;
    let num_hidden_layers = raw
        .num_hidden_layers
        .ok_or_else(|| anyhow!("missing num_hidden_layers"))?;
    let num_attention_heads = raw
        .num_attention_heads
        .ok_or_else(|| anyhow!("missing num_attention_heads"))?;
    let num_kv_heads = raw.num_key_value_heads.unwrap_or(num_attention_heads);
    let intermediate_size = raw
        .intermediate_size
        .ok_or_else(|| anyhow!("missing intermediate_size"))?;
    let vocab_size = raw
        .vocab_size
        .ok_or_else(|| anyhow!("missing vocab_size"))?;
    let max_position_embeddings = raw.max_position_embeddings.unwrap_or(4096);

    let head_dim = raw.head_dim.unwrap_or(hidden_size / num_attention_heads);

    let rms_norm_eps = raw
        .rms_norm_eps
        .or(raw.layer_norm_eps)
        .or(raw.norm_eps)
        .unwrap_or(1e-5);

    let rope_theta = raw.rope_theta.unwrap_or(10000.0);

    let rope_scaling = raw.rope_scaling.as_ref().and_then(parse_rope_scaling);

    let tie_word_embeddings = raw.tie_word_embeddings.unwrap_or(false);

    let attention_type = if num_kv_heads == 1 {
        AttentionType::MQA
    } else if num_kv_heads < num_attention_heads {
        AttentionType::GQA
    } else {
        AttentionType::MHA
    };

    let ffn_type = map_ffn_type(raw.hidden_act.as_deref(), architecture);

    // Qwen3 has per-head Q/K normalization
    let qk_norm = arch_str.starts_with("Qwen3");

    let params = ModelParams {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        rope_scaling,
        attention_type,
        ffn_type,
        tie_word_embeddings,
        qk_norm,
    };

    Ok((architecture, params, raw.bos_token_id, raw.eos_token_id))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_architecture(arch: &str) -> anyhow::Result<ArchitectureFamily> {
    match arch {
        "LlamaForCausalLM" | "LLaMAForCausalLM" => Ok(ArchitectureFamily::LLaMA),
        "MistralForCausalLM" | "MixtralForCausalLM" => Ok(ArchitectureFamily::Mistral),
        "Qwen2ForCausalLM" | "QWenLMHeadModel" | "Qwen2MoeForCausalLM"
        | "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => {
            Ok(ArchitectureFamily::Qwen)
        }
        "PhiForCausalLM" | "Phi3ForCausalLM" | "PhiMoEForCausalLM" => Ok(ArchitectureFamily::Phi),
        "GemmaForCausalLM" | "Gemma2ForCausalLM" => Ok(ArchitectureFamily::Gemma),
        "GPTNeoXForCausalLM" => Ok(ArchitectureFamily::GPTNeoX),
        other => Err(anyhow!("Unsupported architecture: {other}. Supported: LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, PhiForCausalLM, GemmaForCausalLM, GPTNeoXForCausalLM")),
    }
}

fn map_ffn_type(hidden_act: Option<&str>, arch: ArchitectureFamily) -> FFNType {
    match hidden_act {
        Some("silu") => FFNType::SwiGLU,
        Some("gelu") | Some("gelu_new") | Some("gelu_fast") | Some("gelu_pytorch_tanh") => {
            // GeGLU only if the architecture has a gating mechanism
            match arch {
                ArchitectureFamily::Gemma => FFNType::GeGLU,
                _ => FFNType::Standard,
            }
        }
        _ => {
            // Default by architecture
            match arch {
                ArchitectureFamily::LLaMA | ArchitectureFamily::Mistral | ArchitectureFamily::Qwen => {
                    FFNType::SwiGLU
                }
                _ => FFNType::Standard,
            }
        }
    }
}

fn parse_rope_scaling(v: &serde_json::Value) -> Option<RoPEScaling> {
    let obj = v.as_object()?;
    let scaling_type = obj
        .get("type")
        .or_else(|| obj.get("rope_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("linear")
        .to_string();
    let factor = obj
        .get("factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    Some(RoPEScaling { scaling_type, factor })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_config(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_llama_config() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 128256,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "hidden_act": "silu"
        }"#;
        let f = write_config(json);
        let (arch, params, _, _) = parse_config(f.path()).unwrap();
        assert_eq!(arch, ArchitectureFamily::LLaMA);
        assert_eq!(params.num_hidden_layers, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.attention_type, AttentionType::GQA);
        assert_eq!(params.ffn_type, FFNType::SwiGLU);
    }
}
