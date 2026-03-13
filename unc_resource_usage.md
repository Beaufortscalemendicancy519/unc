# UNC Resource Usage & Power Report

**Hardware**: Apple M1 Pro (14 GPU cores, 16GB, 200 GB/s)
**Date**: 2026-03-13
**Tools**: macmon (sudoless Apple Silicon power monitoring, 200ms sampling), /usr/bin/time -l
**Methodology**: 200 tokens generated per run, prompt "The history of", 15-30s cooldown between A/B runs, decode phase identified by GPU utilization > 50%

---

## UNC vs mlx-lm (Q4_0 TinyLlama 1.1B)

| Metric | UNC JIT | UNC AOT | mlx-lm Q4 |
|--------|---------|---------|-----------|
| **Throughput** | 152 tok/s | 152 tok/s | 112.7 tok/s |
| **Speedup** | 1.35x | 1.35x | 1.0x |
| **GPU power** | 11.2W | 11.4W | 14.1W |
| **CPU power** | 0.9W | 2.1W | 2.1W |
| **SoC power** | 12.1W | 13.5W | 16.2W |
| **Energy/token (GPU)** | 74 mJ | 75 mJ | 125 mJ |
| **Energy/token (SoC)** | 80 mJ | 89 mJ | 144 mJ |
| **Tokens per Wh** | 12,800 | 12,600 | 8,000 |
| **Peak RSS** | 4.24 GB | 4.11 GB | 0.87 GB |
| **Instructions retired** | 5.33B | 3.75B | 31.4B |
| **GPU utilization** | 88% | 86% | 88% |
| **GPU temp** | 72°C | 75°C | 74°C |

**Key takeaways**:
- UNC is **1.35x faster** while using **25% less GPU power** (11.3W vs 14.1W)
- **1.7x more energy-efficient** per token (74 vs 125 mJ)
- **8.4x fewer CPU instructions** — no Python/framework overhead
- mlx-lm uses **4-5x less memory** — UNC loads BF16 then quantizes at init

---

## UNC Performance Detail (TinyLlama Q4_0)

| Metric | JIT (.unc bundle) | AOT (standalone binary) |
|--------|-------------------|------------------------|
| Throughput (cold) | 152 tok/s | 152 tok/s |
| Throughput (warm) | 100-110 tok/s | 100-110 tok/s |
| Decode latency (cold) | 6.36 ms/tok | 6.3 ms/tok |
| Decode latency (warm) | 8.3-10.0 ms/tok | 8.3-10.7 ms/tok |
| Init time | ~1.7s (cached) / ~2s (first) | ~1.5s |
| Binary size | 544 KB + HF weights | 2.1 GB self-contained |
| Dependencies | clang (for JIT) | None |

## UNC Performance Detail (Qwen3-4B Q4_0)

| Metric | Value |
|--------|-------|
| Throughput (cold) | 38.7 tok/s |
| Decode latency | 22.9-23.0 ms/tok |
| Dispatches per decode | ~399 |
| Q4_0 weights | ~2.26 GB |
| vs mlx-lm | 79% (38.7 vs 49.2 tok/s) |

---

## Power Traces

### UNC JIT Decode
```
GPU: ████████████████████████████████████████████  11.2W avg
CPU: ███                                            0.9W avg
SoC: █████████████████████████████████████████████  12.1W avg
```

### UNC AOT Decode
```
GPU: ████████████████████████████████████████████   11.4W avg
CPU: ████████                                        2.1W avg
SoC: █████████████████████████████████████████████████ 13.5W avg
```

### mlx-lm Q4 Decode
```
GPU: ████████████████████████████████████████████████████████ 14.1W avg
CPU: ████████                                                  2.1W avg
SoC: █████████████████████████████████████████████████████████████████ 16.2W avg
```

JIT vs AOT GPU power is identical — same Metal kernels, same dispatch pattern. AOT uses ~1W more CPU from mmap'ing the 2.1GB embedded binary.

---

## Memory

| Metric | UNC JIT | UNC AOT | mlx-lm Q4 |
|--------|---------|---------|-----------|
| Peak RSS | 4.24 GB | 4.11 GB | 0.87 GB |
| Peak footprint | 3.61 GB | 3.70 GB | 0.81 GB |
| Q4_0 weights (GPU) | 618 MB | 618 MB | ~618 MB |

## GPU Bandwidth

| Metric | Value |
|--------|-------|
| Q4_0 weights per token | 618 MB |
| Effective bandwidth (cold) | ~97 GB/s |
| Bandwidth utilization | 48% of 200 GB/s |

## Thermal

| Metric | UNC | mlx-lm |
|--------|-----|--------|
| GPU temp (idle) | 66-70°C | — |
| GPU temp (decode) | 70-75°C | 68-74°C |
| CPU temp | 71-73°C | 72-73°C |
| Throttling onset | ~100-150 tokens | — |
| Cold → warm drop | 30-40% | — |

## CPU Usage (200 tokens)

| Metric | UNC JIT | UNC AOT | mlx-lm Q4 |
|--------|---------|---------|-----------|
| User CPU time | 0.30s | 0.16s | 3.27s |
| System CPU time | 0.60s | 0.69s | 0.80s |
| Instructions | 5.33B | 3.75B | 31.4B |
| Context switches | ~4,000 | ~6,200 | ~15,700 |
