//! HuggingFace Hub model downloading and weight file discovery.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use indicatif::{ProgressBar, ProgressStyle};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Downloader/loader for HuggingFace models.
pub struct ModelLoader {
    api: hf_hub::api::sync::Api,
    _cache_dir: PathBuf,
}

/// All local file paths for a downloaded model.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config_json: PathBuf,
    pub tokenizer_json: Option<PathBuf>,
    pub tokenizer_config_json: Option<PathBuf>,
    pub weight_files: Vec<WeightFile>,
    pub model_id: String,
}

/// A single weight file, either safetensors or GGUF.
#[derive(Debug, Clone)]
pub enum WeightFile {
    /// A safetensors shard. `shard_index` is `Some(i)` for multi-shard models.
    Safetensors { path: PathBuf, shard_index: Option<usize> },
    /// A GGUF file.
    Gguf { path: PathBuf },
}

impl WeightFile {
    pub fn path(&self) -> &Path {
        match self {
            WeightFile::Safetensors { path, .. } => path,
            WeightFile::Gguf { path } => path,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelLoader implementation
// ---------------------------------------------------------------------------

impl ModelLoader {
    /// Create a loader using the default HF cache directory.
    pub fn new() -> anyhow::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .context("initialising HuggingFace API client")?;
        let cache_dir = hf_hub::Cache::default().path().to_path_buf();
        Ok(ModelLoader { api, _cache_dir: cache_dir })
    }

    /// Create a loader with an explicit cache directory and optional token.
    pub fn with_cache(cache_dir: PathBuf, token: Option<String>) -> anyhow::Result<Self> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(tok) = token {
            builder = builder.with_token(Some(tok));
        }
        let api = builder.build().context("initialising HuggingFace API client")?;
        Ok(ModelLoader { api, _cache_dir: cache_dir })
    }

    /// Download and resolve all files for `model_id` (e.g. `"meta-llama/Llama-3-8B"`).
    pub fn load(&self, model_id: &str) -> anyhow::Result<ModelFiles> {
        self.load_revision(model_id, "main")
    }

    /// Download and resolve files for a specific revision/branch.
    pub fn load_revision(&self, model_id: &str, revision: &str) -> anyhow::Result<ModelFiles> {
        log::info!("Downloading model: {} @ {}", model_id, revision);

        let repo = self.api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        // -- config.json (required) --
        let config_json = repo
            .get("config.json")
            .with_context(|| format!("downloading config.json for {model_id}"))?;

        // -- tokenizer files (optional) --
        let tokenizer_json = repo.get("tokenizer.json").ok();
        let tokenizer_config_json = repo.get("tokenizer_config.json").ok();

        // -- weight files --
        let weight_files = self.resolve_weights(&repo, model_id)?;

        if weight_files.is_empty() {
            return Err(anyhow!(
                "No weight files found for {model_id}. \
                 Expected model.safetensors, model.safetensors.index.json, or model.gguf"
            ));
        }

        Ok(ModelFiles {
            config_json,
            tokenizer_json,
            tokenizer_config_json,
            weight_files,
            model_id: model_id.to_string(),
        })
    }

    fn resolve_weights(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        model_id: &str,
    ) -> anyhow::Result<Vec<WeightFile>> {
        // 1. Try single-shard model.safetensors
        if let Ok(path) = repo.get("model.safetensors") {
            log::info!("Found single-shard safetensors");
            return Ok(vec![WeightFile::Safetensors { path, shard_index: None }]);
        }

        // 2. Try multi-shard via model.safetensors.index.json
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            return self.resolve_safetensors_shards(repo, &index_path, model_id);
        }

        // 3. Try GGUF
        for candidate in &["model.gguf", "ggml-model.gguf", "model-q4_0.gguf"] {
            if let Ok(path) = repo.get(candidate) {
                log::info!("Found GGUF: {candidate}");
                return Ok(vec![WeightFile::Gguf { path }]);
            }
        }

        Ok(Vec::new())
    }

    fn resolve_safetensors_shards(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        index_path: &Path,
        model_id: &str,
    ) -> anyhow::Result<Vec<WeightFile>> {
        #[derive(serde::Deserialize)]
        struct ShardIndex {
            weight_map: std::collections::HashMap<String, String>,
        }

        let json = std::fs::read_to_string(index_path)?;
        let index: ShardIndex = serde_json::from_str(&json)
            .context("parsing model.safetensors.index.json")?;

        // Collect unique shard filenames in sorted order
        let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
        shard_files.sort();
        shard_files.dedup();

        log::info!("Found {} safetensors shards for {}", shard_files.len(), model_id);

        let pb = ProgressBar::new(shard_files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
                .unwrap(),
        );

        let mut weight_files = Vec::new();
        for (i, shard_name) in shard_files.iter().enumerate() {
            pb.set_message(format!("downloading {shard_name}"));
            let path = repo
                .get(shard_name)
                .with_context(|| format!("downloading shard {shard_name}"))?;
            weight_files.push(WeightFile::Safetensors {
                path,
                shard_index: Some(i),
            });
            pb.inc(1);
        }
        pb.finish_with_message("done");

        Ok(weight_files)
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new().expect("failed to create ModelLoader")
    }
}
