use std::path::Path;

pub struct UNCTokenizer {
    inner: tokenizers::Tokenizer,
}

impl UNCTokenizer {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        Ok(UNCTokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode_single(&self, id: u32) -> anyhow::Result<String> {
        let s = self
            .inner
            .decode(&[id], false)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
        Ok(s)
    }

    /// Decode a prefix of tokens to get proper spacing, then return
    /// just the last token's text (with correct leading space).
    pub fn decode_incremental(&self, all_ids: &[u32]) -> anyhow::Result<String> {
        if all_ids.len() <= 1 {
            return self.decode_single(*all_ids.last().unwrap_or(&0));
        }
        let full = self.inner.decode(all_ids, false)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        let prefix = self.inner.decode(&all_ids[..all_ids.len() - 1], false)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;
        Ok(full[prefix.len()..].to_string())
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .get_added_tokens_decoder()
            .iter()
            .find(|(_, t)| t.special && (t.content == "</s>" || t.content == "<|endoftext|>" || t.content == "<eos>"))
            .map(|(id, _)| *id)
    }
}
