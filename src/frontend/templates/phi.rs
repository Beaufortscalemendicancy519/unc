//! Phi-2/Phi-3 graph template.

use crate::frontend::huggingface::WeightFile;
use crate::ir::graph::{CompGraph, ModelParams};

pub fn lower_phi(params: &ModelParams, weight_files: &[WeightFile], model_id: &str, tokenizer_path: Option<&str>, bos_token_id: Option<u32>, eos_token_id: Option<u32>) -> CompGraph {
    super::llama::lower_llama(params, weight_files, model_id, tokenizer_path, bos_token_id, eos_token_id)
}
