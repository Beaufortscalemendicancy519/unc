//! Mistral/Mixtral graph template.
//! Mistral is nearly identical to LLaMA but with sliding window attention.
//! For now, we lower it using the LLaMA template (SWA support TODO).

use crate::frontend::huggingface::WeightFile;
use crate::ir::graph::{CompGraph, ModelParams};

pub fn lower_mistral(params: &ModelParams, weight_files: &[WeightFile], model_id: &str, tokenizer_path: Option<&str>, bos_token_id: Option<u32>, eos_token_id: Option<u32>) -> CompGraph {
    // Mistral uses same weight naming as LLaMA; SWA is a runtime concern
    super::llama::lower_llama(params, weight_files, model_id, tokenizer_path, bos_token_id, eos_token_id)
}
