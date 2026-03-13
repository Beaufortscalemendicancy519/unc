//! AOT Metal binary emission.
//!
//! Produces a standalone Mach-O binary that embeds:
//! - The Metal orchestrator (unc_init / unc_forward)
//! - The compiled metallib (as a C byte array)
//! - A minimal BPE tokenizer (vocab + merges as C arrays)
//! - Model weights (appended after the Mach-O, mmap'd at runtime)
//!
//! Binary layout: [Mach-O code][weights][8B weight_offset][8B weight_size][8B magic]
//! Usage: `./binary --prompt "Hello" --max-tokens 50`

use std::path::Path;
use std::process::Command;

use anyhow::Context;

use crate::compile::CompilationResult;
use crate::frontend::huggingface::ModelFiles;

/// Magic trailer: "UNCW8TSX" (8 bytes)
const TRAILER_MAGIC: &[u8; 8] = b"UNCW8TSX";

/// Emit a standalone Mach-O binary for the given compilation result.
pub fn emit_metal_aot(
    result: &CompilationResult,
    model_files: &ModelFiles,
    output_binary: &Path,
) -> anyhow::Result<()> {
    let build_dir = output_binary.with_extension("aot_build");
    std::fs::create_dir_all(&build_dir)?;

    // 1. Generate orchestrator source (reuse existing emit)
    let orchestrator_src = super::metal::generate_orchestrator(result, false);

    // 2. Read the compiled metallib
    let metallib_bytes = read_metallib()?;

    // 3. Generate embedded tokenizer C code
    let tokenizer_path = model_files.tokenizer_json.as_deref()
        .ok_or_else(|| anyhow::anyhow!("no tokenizer.json found — required for AOT"))?;
    let tokenizer_src = generate_embedded_tokenizer(tokenizer_path)?;

    // 4. Generate the complete AOT main.m
    let vocab_size = result.graph.metadata.params.vocab_size;
    let bos_id = result.graph.metadata.bos_token_id.unwrap_or(1);
    let eos_id = result.graph.metadata.eos_token_id.unwrap_or(2);
    let main_src = generate_aot_main(
        &orchestrator_src,
        &metallib_bytes,
        &tokenizer_src,
        vocab_size as u32,
        bos_id,
        eos_id,
    );

    let src_path = build_dir.join("unc_main.m");
    std::fs::write(&src_path, &main_src)
        .context("writing AOT main.m")?;

    // 5. Prepare weight blob
    let weight_blob_path = build_dir.join("weights.bin");
    prepare_weight_blob(&model_files.weight_files, &weight_blob_path)?;
    let weight_size = std::fs::metadata(&weight_blob_path)?.len();

    log::info!("AOT build dir: {}", build_dir.display());
    log::info!("  main.m: {} bytes", main_src.len());
    log::info!("  weights.bin: {} bytes", weight_size);

    // 6. Compile
    let obj_path = build_dir.join("unc_main.o");
    let status = Command::new("clang")
        .args([
            "-c", "-fobjc-arc", "-O2",
            "-Wno-format",
            src_path.to_str().unwrap(),
            "-o", obj_path.to_str().unwrap(),
        ])
        .status()
        .context("running clang (compile step)")?;
    if !status.success() {
        anyhow::bail!("AOT compile failed (clang -c exit {})", status);
    }

    // 7. Link (small binary — no embedded weights)
    let code_binary_path = build_dir.join("unc_code");
    let status = Command::new("clang")
        .args([
            &obj_path.to_string_lossy().to_string(),
            "-framework", "Metal",
            "-framework", "Foundation",
            "-o", &code_binary_path.to_string_lossy().to_string(),
        ])
        .status()
        .context("running clang (link step)")?;
    if !status.success() {
        anyhow::bail!("AOT link failed (clang link exit {})", status);
    }

    // 8. Append weights + trailer to produce final binary
    //    Layout: [Mach-O code][padding to 16KB][weights][8B offset][8B size][8B magic]
    //    mmap requires page-aligned offset (16KB on Apple Silicon)
    let code_bytes = std::fs::read(&code_binary_path)
        .context("reading code binary")?;
    let code_size = code_bytes.len();
    let page_size: usize = 16384; // 16KB pages on Apple Silicon
    let weight_offset = (code_size + page_size - 1) & !(page_size - 1); // round up
    let padding = weight_offset - code_size;

    let mut final_binary = code_bytes;
    final_binary.extend(std::iter::repeat(0u8).take(padding));      // pad to page boundary
    let weight_data = std::fs::read(&weight_blob_path)
        .context("reading weight blob")?;
    final_binary.extend_from_slice(&weight_data);
    final_binary.extend_from_slice(&(weight_offset as u64).to_le_bytes()); // weight offset
    final_binary.extend_from_slice(&weight_size.to_le_bytes());            // weight size
    final_binary.extend_from_slice(TRAILER_MAGIC);                         // magic

    std::fs::write(output_binary, &final_binary)
        .context("writing final binary")?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(output_binary, perms)?;
    }

    let final_size = std::fs::metadata(output_binary)?.len();
    log::info!("AOT binary: {} ({:.1} MB code + {:.1} MB weights)",
        output_binary.display(),
        code_size as f64 / (1024.0 * 1024.0),
        weight_size as f64 / (1024.0 * 1024.0),
    );
    println!("  binary: {:.1} MB total", final_size as f64 / (1024.0 * 1024.0));

    Ok(())
}

// ---------------------------------------------------------------------------
// Metallib reading
// ---------------------------------------------------------------------------

fn read_metallib() -> anyhow::Result<Vec<u8>> {
    let metallib_path = std::env::var("UNC_METALLIB_PATH")
        .ok()
        .or_else(|| option_env!("UNC_METALLIB_PATH").map(String::from))
        .ok_or_else(|| anyhow::anyhow!("UNC_METALLIB_PATH not set — build with cargo build --release first"))?;

    std::fs::read(&metallib_path)
        .with_context(|| format!("reading metallib from {metallib_path}"))
}

// ---------------------------------------------------------------------------
// Weight blob preparation
// ---------------------------------------------------------------------------

fn prepare_weight_blob(
    weight_files: &[crate::frontend::huggingface::WeightFile],
    output: &Path,
) -> anyhow::Result<()> {
    // Embed the raw safetensors file. The GPU handles BF16→F16 + quantization at init.
    if weight_files.len() != 1 {
        anyhow::bail!("AOT currently supports single-shard models only ({} shards found)", weight_files.len());
    }
    let src_path = weight_files[0].path();
    std::fs::copy(src_path, output)
        .with_context(|| format!("copying weights from {}", src_path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedded metallib as C byte array
// ---------------------------------------------------------------------------

fn emit_metallib_bytes(metallib: &[u8]) -> String {
    let mut s = String::with_capacity(metallib.len() * 5 + 200);
    s.push_str("static const unsigned char unc_metallib_data[] = {\n");
    for (i, chunk) in metallib.chunks(16).enumerate() {
        s.push_str("    ");
        for (j, byte) in chunk.iter().enumerate() {
            s.push_str(&format!("0x{:02x}", byte));
            if i * 16 + j + 1 < metallib.len() {
                s.push(',');
            }
        }
        s.push('\n');
    }
    s.push_str("};\n");
    s.push_str(&format!(
        "static const unsigned int unc_metallib_size = {};\n\n",
        metallib.len()
    ));
    s
}

// ---------------------------------------------------------------------------
// Embedded BPE tokenizer
// ---------------------------------------------------------------------------

fn generate_embedded_tokenizer(tokenizer_path: &Path) -> anyhow::Result<String> {
    let data = std::fs::read_to_string(tokenizer_path)
        .with_context(|| format!("reading tokenizer from {}", tokenizer_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&data)
        .context("parsing tokenizer.json")?;

    // Extract vocab: model.vocab is a dict { "token_string": id }
    let vocab_obj = json.pointer("/model/vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json missing model.vocab"))?;

    let mut vocab: Vec<(u32, String)> = Vec::new();
    for (token, id_val) in vocab_obj {
        let id = id_val.as_u64().unwrap_or(0) as u32;
        vocab.push((id, token.clone()));
    }
    vocab.sort_by_key(|(id, _)| *id);

    let vocab_size = vocab.last().map(|(id, _)| *id + 1).unwrap_or(0) as usize;

    // Extract merges: model.merges is a list of "piece_a piece_b" strings
    let merges = json.pointer("/model/merges")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json missing model.merges"))?;

    let mut s = String::new();

    // Emit vocab array: unc_vocab[id] = "token_string" (with C escaping)
    s.push_str(&format!("#define UNC_TOK_VOCAB_SIZE {vocab_size}\n\n"));
    s.push_str("static const char* unc_vocab[] = {\n");

    let mut id_to_token: Vec<String> = vec![String::new(); vocab_size];
    for (id, token) in &vocab {
        if (*id as usize) < vocab_size {
            id_to_token[*id as usize] = token.clone();
        }
    }
    for token in id_to_token.iter() {
        let escaped = c_escape_string(token);
        s.push_str(&format!("    \"{escaped}\",\n"));
    }
    s.push_str("};\n\n");

    // Emit merge pairs as string pairs
    s.push_str(&format!("#define UNC_TOK_N_MERGES {}\n\n", merges.len()));
    s.push_str("static const char* unc_merges[][2] = {\n");
    for merge_val in merges {
        let merge_str = merge_val.as_str().unwrap_or("");
        if let Some((a, b)) = merge_str.split_once(' ') {
            let a_esc = c_escape_string(a);
            let b_esc = c_escape_string(b);
            s.push_str(&format!("    {{\"{a_esc}\", \"{b_esc}\"}},\n"));
        }
    }
    s.push_str("};\n\n");

    // Sorted vocab for binary search encode
    s.push_str("// Vocab lookup: sorted by token string for binary search\n");
    let mut sorted_vocab: Vec<(String, u32)> = vocab.iter()
        .map(|(id, tok)| (tok.clone(), *id))
        .collect();
    sorted_vocab.sort_by(|a, b| a.0.cmp(&b.0));

    s.push_str("static const struct { const char* tok; uint32_t id; } unc_vocab_sorted[] = {\n");
    for (tok, id) in &sorted_vocab {
        let escaped = c_escape_string(tok);
        s.push_str(&format!("    {{\"{escaped}\", {id}}},\n"));
    }
    s.push_str("};\n\n");

    s.push_str(BPE_C_CODE);

    Ok(s)
}

fn c_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c if c.is_ascii_graphic() || c == ' ' => out.push(c),
            c => {
                // Emit as UTF-8 octal bytes (octal avoids C's greedy hex parsing)
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf);
                for b in bytes.bytes() {
                    out.push_str(&format!("\\{:03o}", b));
                }
            }
        }
    }
    out
}

/// Minimal BPE tokenizer implementation in C.
static BPE_C_CODE: &str = r#"
// ---------------------------------------------------------------------------
// Minimal BPE tokenizer (encode + decode)
// ---------------------------------------------------------------------------

static int unc_tok_vocab_lookup(const char* tok, int len) {
    // Binary search in unc_vocab_sorted
    int lo = 0, hi = UNC_TOK_VOCAB_SIZE - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strncmp(unc_vocab_sorted[mid].tok, tok, len);
        if (cmp == 0 && unc_vocab_sorted[mid].tok[len] == '\0') {
            return (int)unc_vocab_sorted[mid].id;
        }
        if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// Simple BPE encode: split into bytes, iteratively merge
typedef struct { char* str; int len; } BPEPiece;

// Return byte length of UTF-8 character starting at *p
static int utf8_char_len(const char* p) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // fallback
}

static int unc_tok_encode(const char* text, uint32_t* out_ids, int max_ids) {
    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    // SentencePiece preprocessing: prepend ▁ and replace all spaces with ▁
    // ▁ = UTF-8 bytes \xe2\x96\x81 (3 bytes)
    // "Hello world" → "▁Hello▁world"
    int preprocessed_cap = text_len * 4 + 4;
    char* preprocessed = (char*)malloc(preprocessed_cap);
    int pp_len = 0;

    // Prepend ▁
    preprocessed[pp_len++] = '\xe2';
    preprocessed[pp_len++] = '\x96';
    preprocessed[pp_len++] = '\x81';

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            preprocessed[pp_len++] = '\xe2';
            preprocessed[pp_len++] = '\x96';
            preprocessed[pp_len++] = '\x81';
        } else {
            preprocessed[pp_len++] = text[i];
        }
    }
    preprocessed[pp_len] = '\0';

    // Split into UTF-8 characters (not bytes!)
    // Count characters first
    int n_chars = 0;
    for (int i = 0; i < pp_len; ) {
        i += utf8_char_len(preprocessed + i);
        n_chars++;
    }

    int n_pieces = n_chars;
    BPEPiece* pieces = (BPEPiece*)malloc(n_pieces * sizeof(BPEPiece));
    int ci = 0;
    for (int i = 0; i < pp_len; ) {
        int clen = utf8_char_len(preprocessed + i);
        pieces[ci].str = (char*)malloc(clen + 1);
        memcpy(pieces[ci].str, preprocessed + i, clen);
        pieces[ci].str[clen] = '\0';
        pieces[ci].len = clen;
        i += clen;
        ci++;
    }
    free(preprocessed);

    // Iterative merge: apply merges in priority order
    for (int mi = 0; mi < UNC_TOK_N_MERGES && n_pieces > 1; mi++) {
        const char* ma = unc_merges[mi][0];
        const char* mb = unc_merges[mi][1];
        int ma_len = (int)strlen(ma);
        int mb_len = (int)strlen(mb);

        for (int i = 0; i < n_pieces - 1; i++) {
            if (pieces[i].len == ma_len && pieces[i+1].len == mb_len &&
                memcmp(pieces[i].str, ma, ma_len) == 0 &&
                memcmp(pieces[i+1].str, mb, mb_len) == 0)
            {
                // Merge pieces[i] and pieces[i+1]
                int new_len = ma_len + mb_len;
                char* merged = (char*)malloc(new_len + 1);
                memcpy(merged, ma, ma_len);
                memcpy(merged + ma_len, mb, mb_len);
                merged[new_len] = '\0';

                free(pieces[i].str);
                free(pieces[i+1].str);
                pieces[i].str = merged;
                pieces[i].len = new_len;

                // Shift remaining pieces down
                for (int j = i + 1; j < n_pieces - 1; j++) {
                    pieces[j] = pieces[j+1];
                }
                n_pieces--;
                i--; // Re-check this position
            }
        }
    }

    // Convert pieces to token IDs
    int n_ids = 0;
    for (int i = 0; i < n_pieces && n_ids < max_ids; i++) {
        int id = unc_tok_vocab_lookup(pieces[i].str, pieces[i].len);
        if (id >= 0) {
            out_ids[n_ids++] = (uint32_t)id;
        } else {
            // Fallback: encode individual bytes as byte tokens <0xXX>
            for (int j = 0; j < pieces[i].len && n_ids < max_ids; j++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)pieces[i].str[j]);
                int bid = unc_tok_vocab_lookup(byte_tok, (int)strlen(byte_tok));
                if (bid >= 0) out_ids[n_ids++] = (uint32_t)bid;
            }
        }
        free(pieces[i].str);
    }
    free(pieces);
    return n_ids;
}

static const char* unc_tok_decode(uint32_t id) {
    if (id < UNC_TOK_VOCAB_SIZE) return unc_vocab[id];
    return "";
}

// Decode a token, replacing ▁ with space
static void unc_tok_print(uint32_t id) {
    const char* tok = unc_tok_decode(id);
    while (*tok) {
        // Check for ▁ (0xe2 0x96 0x81)
        if ((unsigned char)tok[0] == 0xe2 &&
            (unsigned char)tok[1] == 0x96 &&
            (unsigned char)tok[2] == 0x81) {
            putchar(' ');
            tok += 3;
        } else {
            putchar(*tok);
            tok++;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// AOT main generation
// ---------------------------------------------------------------------------

fn generate_aot_main(
    orchestrator_src: &str,
    metallib_bytes: &[u8],
    tokenizer_src: &str,
    vocab_size: u32,
    bos_id: u32,
    eos_id: u32,
) -> String {
    let mut s = String::new();

    // Embedded metallib
    s.push_str(&emit_metallib_bytes(metallib_bytes));

    // Headers
    s.push_str(r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <mach-o/dyld.h>

"#);

    // Tokenizer code
    s.push_str(tokenizer_src);
    s.push_str("\n");

    // The orchestrator
    s.push_str(orchestrator_src);
    s.push_str("\n");

    // AOT main with self-mmap for weights
    s.push_str(&format!(r#"
// ---------------------------------------------------------------------------
// AOT entry point
// ---------------------------------------------------------------------------

static uint32_t aot_argmax(const float* logits, uint32_t n) {{
    uint32_t best = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < n; i++) {{
        if (logits[i] > best_val) {{ best_val = logits[i]; best = i; }}
    }}
    return best;
}}

// Read weight data appended after the Mach-O binary.
// Layout: [Mach-O][weights][8B offset][8B size][8B magic "UNCW8TSX"]
static void* load_appended_weights(size_t* out_size) {{
    // Get path to self
    char self_path[4096];
    uint32_t path_len = sizeof(self_path);
    if (_NSGetExecutablePath(self_path, &path_len) != 0) {{
        fprintf(stderr, "Error: _NSGetExecutablePath failed\\n");
        return NULL;
    }}
    // Resolve symlinks
    char* real_path = realpath(self_path, NULL);
    if (!real_path) real_path = self_path;

    int fd = open(real_path, O_RDONLY);
    if (fd < 0) {{
        perror("open self");
        if (real_path != self_path) free(real_path);
        return NULL;
    }}

    // Read trailer (last 24 bytes): [offset u64][size u64][magic 8B]
    struct stat st;
    fstat(fd, &st);
    off_t file_size = st.st_size;

    uint8_t trailer[24];
    pread(fd, trailer, 24, file_size - 24);

    // Verify magic
    if (memcmp(trailer + 16, "UNCW8TSX", 8) != 0) {{
        fprintf(stderr, "Error: no UNC weight trailer found in binary\\n");
        close(fd);
        if (real_path != self_path) free(real_path);
        return NULL;
    }}

    uint64_t weight_offset, weight_size;
    memcpy(&weight_offset, trailer, 8);
    memcpy(&weight_size, trailer + 8, 8);

    // mmap the weight region
    void* mapped = mmap(NULL, (size_t)weight_size, PROT_READ, MAP_PRIVATE, fd, (off_t)weight_offset);
    close(fd);
    if (real_path != self_path) free(real_path);

    if (mapped == MAP_FAILED) {{
        perror("mmap weights");
        return NULL;
    }}

    *out_size = (size_t)weight_size;
    return mapped;
}}

int main(int argc, char* argv[]) {{
    const char* prompt = "Hello";
    int max_tokens = 200;

    // Simple arg parsing
    for (int i = 1; i < argc; i++) {{
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {{
            prompt = argv[++i];
        }} else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {{
            max_tokens = atoi(argv[++i]);
        }} else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {{
            fprintf(stderr, "Usage: %s [--prompt TEXT] [--max-tokens N]\\n", argv[0]);
            return 0;
        }}
    }}

    // 1. Write embedded metallib to temp file
    char metallib_tmp[] = "/tmp/unc_metallib_XXXXXX.metallib";
    int fd = mkstemps(metallib_tmp, 9);  // 9 = strlen(".metallib")
    if (fd < 0) {{ perror("mkstemps"); return 1; }}
    write(fd, unc_metallib_data, unc_metallib_size);
    close(fd);

    // 2. Load weights from appended data (mmap'd from self)
    size_t weight_size = 0;
    void* weight_ptr = load_appended_weights(&weight_size);
    if (!weight_ptr) {{
        fprintf(stderr, "Error: failed to load embedded weights\\n");
        unlink(metallib_tmp);
        return 1;
    }}

    // 3. Initialize Metal state
    unc_init(metallib_tmp, weight_ptr, weight_size, 0);
    unlink(metallib_tmp);  // Clean up temp file

    // 4. Tokenize prompt
    uint32_t token_ids[4096];
    token_ids[0] = {bos_id};  // BOS
    int n_tokens = unc_tok_encode(prompt, token_ids + 1, 4095) + 1;

    // 5. Prefill
    float* logits = (float*)malloc({vocab_size} * sizeof(float));
    unc_forward(token_ids, (uint32_t)n_tokens, (uint32_t)n_tokens, logits);

    // 6. Decode loop
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);
    int decode_tokens = 0;
    uint32_t next = aot_argmax(logits, {vocab_size});
    for (int step = 0; step < max_tokens; step++) {{
        unc_tok_print(next);
        fflush(stdout);
        decode_tokens++;

        if (next == {eos_id}) break;

        uint32_t pos = (uint32_t)(n_tokens + step + 1);
        unc_forward(&next, 1, pos, logits);
        next = aot_argmax(logits, {vocab_size});
    }}
    gettimeofday(&tv_end, NULL);
    double elapsed = (tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1e6;
    double tok_per_sec = (elapsed > 0) ? decode_tokens / elapsed : 0;

    printf("\\n");
    fprintf(stderr, "\\n--- %d tokens in %.2fs = %.1f tok/s ---\\n", decode_tokens, elapsed, tok_per_sec);
    free(logits);
    return 0;
}}
"#));

    s
}
