# Qwen3-4B Benchmark

**Model:** Qwen/Qwen3-4B (3.6B parameters, 36 layers, 32 heads, 8 kv heads)
**Hardware:** Apple M1 Pro, 16GB, 14 GPU cores, 200 GB/s memory bandwidth
**Date:** 2026-03-13
**Prompt:** "The history of"
**Max tokens:** 200

## Throughput

```
mlx-lm Q4 ████████████████████████████████████████████████████  49.2 tok/s
UNC Q4_0  ██████████████████████████████████████████            38.7 tok/s
```

| Engine | Quant | tok/s | Decode latency |
|--------|-------|-------|----------------|
| mlx-lm | Q4 | 49.2 | — |
| UNC Metal | Q4_0 | 38.7 | 22.88-23.0 ms/tok |

UNC at **79% of mlx-lm** on Qwen3-4B (vs 135% on TinyLlama). The gap is due to:
- Extra dispatches from QKNorm (2/layer x 36 = 72 extra vs TinyLlama)
- 36 layers vs 22 — dispatch overhead scales linearly
- MLX's concurrent encoder with lazy barriers skips ~90/160 barriers

## Model Details

| Param | Value |
|-------|-------|
| Layers | 36 |
| Heads (Q) | 32 |
| KV heads | 8 |
| Head dim | 128 |
| Hidden size | 2560 |
| FFN size | 9728 |
| Q4_0 weights | ~2.26 GB |
| Dispatches per decode | ~399 |
| Safetensor shards | 3 |

## Compilation

```
graph: 723 nodes, quant: q4_0
registry: 19 kernels for metal
QKV fusion: 36 groups
Fused RoPE+KV+SDPA: 36 decode dispatches eliminated
PSQ pipeline: 71 writer/reader pairs, eliminating 71 Add+RMSNorm dispatches
Barrier plan: 398 barriers for 399 dispatches
```

## Thermal

- Decode latency stable at 22.9-23.0 ms/tok across windows 2-4
- Less thermal variance than TinyLlama (longer per-token time = lower instantaneous power)
