# UNC — Universal Neural Compiler

Compiles HuggingFace transformer models into optimised native Metal inference binaries. No runtime framework, no Python — just a compiled binary that runs your model at near-hardware-limit speed on Apple Silicon, using **25% less GPU power** and **1.7x better energy efficiency** than mlx-lm. See the full [resource & power report](unc_resource_usage.md).

## Performance

TinyLlama 1.1B on Apple M1 Pro (16GB, 200 GB/s):

```
UNC Q4_0  ████████████████████████████████████████████████████████████  152.0 tok/s
mlx-lm Q4 ████████████████████████████████████████████                 112.7 tok/s
UNC Q8_0  ███████████████████████████████                               76.6 tok/s
UNC F16   ███████████████████                                           47.9 tok/s
```

Qwen3-4B on Apple M1 Pro (Q4_0):

```
mlx-lm Q4 ████████████████████████████████████████████████████  49.2 tok/s
UNC Q4_0  ██████████████████████████████████████████            38.7 tok/s
```

### Energy Efficiency (Q4_0 TinyLlama 1.1B, measured via macmon)

| Metric | UNC Metal | mlx-lm Q4 |
|--------|----------|-----------|
| Throughput | 152 tok/s | 113 tok/s |
| GPU power (decode) | 11.3W | 14.1W |
| Energy per token | 74 mJ | 125 mJ |
| Tokens per watt-hour | 12,800 | 8,000 |
| CPU instructions (200 tok) | 5.3B | 31.4B |
| Peak memory | 4.2 GB | 0.9 GB |

UNC is **1.35x faster** while using **25% less GPU power**, resulting in **1.7x better energy efficiency**. The compiled approach eliminates Python runtime and framework dispatch overhead entirely — 8.4x fewer CPU instructions means less heat, less power, and more headroom for the GPU. See [unc_resource_usage.md](unc_resource_usage.md) for full methodology and traces.

## Architecture

```
HuggingFace model
       |
  [ Frontend ]     Parse config.json + safetensors → IR graph
       |
  [ Compiler ]     Optimization passes: fusion, quantization, memory planning
       |
  [ Metal emit ]   Objective-C orchestrator + custom Metal kernels
       |
  Native binary    Standalone Mach-O (AOT) or .unc bundle (JIT)
```

**IR**: Typed tensor graph with `BatchMatMul`, `QuantizedMatVec`, `RMSNorm`, `LayerNorm`, `QKNorm`, `RoPE`, `SDPA`, `SwiGLU`, `KVCacheAppend`, `Gather`, etc.

**Compiler passes**: Weight binding, dead code elimination, QKV fusion, Gate+Up fusion, SwiGLU fusion, Add+RMSNorm fusion, RoPE+KV fusion, PSQ pipeline, dual-path (GEMM/GEMV), kernel matching, barrier analysis, memory planning with buffer aliasing.

**Output modes**:
| Mode | Output | Use case |
|------|--------|----------|
| JIT (default) | `.unc` bundle — JIT-compiled via clang at first run, cached thereafter | Development, iteration |
| AOT (`--binary`) | Standalone Mach-O with embedded weights — zero dependencies | Deployment, distribution |

## Setup

```bash
# Prerequisites: Rust toolchain, Xcode Command Line Tools (macOS)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone <repo-url> && cd unc
cargo build --release
```

## Usage

### Compile a model

```bash
# JIT bundle (default)
unc compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant q4-0 -o ./tinyllama
unc compile --model Qwen/Qwen3-4B --quant q4-0 -o ./qwen3

# AOT standalone binary (single Mach-O, zero dependencies)
unc compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant q4-0 --binary -o ./tinyllama
```

### Run inference

```bash
unc run ./tinyllama.unc --prompt "The history of" --max-tokens 200
```

### Quantization options

| Flag | Precision | Size (1.1B) | Speed |
|------|-----------|-------------|-------|
| `f16` | 16-bit float | 2.2 GB | 47.9 tok/s |
| `q8-0` | 8-bit | 1.1 GB | 76.6 tok/s |
| `q4-0` | 4-bit | 0.6 GB | 152.0 tok/s |

### Supported architectures

```bash
unc list-architectures
```

LLaMA, Mistral, Qwen, Phi, Gemma.

## Project Structure

```
src/
  frontend/    HuggingFace config parsing, model templates
  ir/          Typed tensor IR (ops, graph, types)
  compile/     Optimization passes, memory planner
  kernel/      Kernel registry, Metal kernel definitions
  emit/        Metal orchestrator codegen, AOT binary emission
  runtime/     JIT compilation, weight loading, tokenizer
  target/      Apple Silicon target detection
  unc_format/  .unc bundle serialization
kernel_sources/
  metal/
    unc_kernels/   Custom Metal shaders (fused GEMV, SDPA, RoPE, RMSNorm, etc.)
    upstream_mlx/  MLX reference kernels (QMV, sdpa_vector headers)
```

## License

MIT — see [LICENSE](LICENSE).
