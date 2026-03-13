# TinyLlama 1.1B Benchmark

**Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
**Hardware:** Apple M1 Pro, 16GB, 14 GPU cores, 200 GB/s memory bandwidth
**Date:** 2026-03-13
**Prompt:** "The history of"
**Max tokens:** 200

## Throughput

```
UNC Q4_0  ████████████████████████████████████████████████████████████  152.0 tok/s
mlx-lm Q4 ████████████████████████████████████████████                 112.7 tok/s
UNC Q8_0  ███████████████████████████████                               76.6 tok/s
UNC F16   ███████████████████                                           47.9 tok/s
```

| Engine | Quant | tok/s | Decode latency |
|--------|-------|-------|----------------|
| UNC Metal | Q4_0 | 152.0 | 6.36-6.40 ms/tok |
| mlx-lm | Q4 | 112.7 | — |
| UNC Metal | Q8_0 | 76.6 | — |
| UNC Metal | F16 | 47.9 | — |

## Power & Energy (measured via macmon, 200ms sampling)

| Metric | UNC Metal | mlx-lm Q4 |
|--------|----------|-----------|
| GPU power (decode avg) | 11.2W | 14.1W |
| GPU power (peak) | 11.8W | 18.4W |
| CPU power (decode avg) | 0.9W | 2.1W |
| SoC power (decode avg) | 12.1W | 16.2W |
| Energy per token (GPU) | 74 mJ | 125 mJ |
| Energy per token (SoC) | 80 mJ | 144 mJ |
| Tokens per watt-hour (GPU) | 12,800 | 8,000 |
| GPU utilization | 88% | 88% |
| GPU temp | 72°C | 74°C |

UNC is **1.7x more energy-efficient** per token than mlx-lm.

## Resource Usage

| Metric | UNC JIT | UNC AOT | mlx-lm Q4 |
|--------|---------|---------|-----------|
| Peak RSS | 4.24 GB | 4.11 GB | 0.87 GB |
| Peak memory footprint | 3.61 GB | 3.70 GB | 0.81 GB |
| Q4_0 weights (GPU) | 618 MB | 618 MB | ~618 MB |
| CPU instructions (200 tok) | 5.33B | 3.75B | 31.4B |
| User CPU time | 0.30s | 0.16s | 3.27s |

## Bandwidth

| Metric | Value |
|--------|-------|
| Q4_0 weights per token | 618 MB |
| Effective bandwidth (cold) | ~97 GB/s |
| Bandwidth utilization | 48% of 200 GB/s |

## JIT vs AOT

| Metric | JIT (.unc) | AOT (binary) |
|--------|-----------|-------------|
| Throughput | 152 tok/s | 152 tok/s |
| GPU power | 11.2W | 11.4W |
| Init time | ~1.7s (cached) | ~1.5s |
| Binary size | 544 KB + HF weights | 2.1 GB self-contained |
| Dependencies | clang | None |

GPU power identical — same Metal kernels. AOT uses ~1W more CPU (mmap overhead).

## Thermal

- GPU temp idle: 66-70°C, decode: 70-75°C
- Throttling onset: ~100-150 tokens
- Cold → warm throughput drop: 30-40% (152 → 100-110 tok/s)
