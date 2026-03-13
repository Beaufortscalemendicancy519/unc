// mlx_kernels.metal — Thin instantiation of MLX kernel templates for UNC.
// Only instantiates the specific variants needed for Q4_0 inference.
// MLX kernels are MIT licensed (Copyright © 2023-2024 Apple Inc.)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ─── defines ──────────────────────────────────────────────────────────────
typedef half float16_t;

#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

static constant constexpr const int RMS_N_READS = 4;

// Limits template (needed by sdpa_vector.h)
template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

template <>
struct Limits<float> {
  static constexpr constant float max = metal::numeric_limits<float>::infinity();
  static constexpr constant float min = -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max = metal::numeric_limits<float>::max();
  static constexpr constant float finite_min = -metal::numeric_limits<float>::max();
};

template <>
struct Limits<half> {
  static constexpr constant half max = metal::numeric_limits<half>::infinity();
  static constexpr constant half min = -metal::numeric_limits<half>::infinity();
  static constexpr constant half finite_max = metal::numeric_limits<half>::max();
  static constexpr constant half finite_min = -metal::numeric_limits<half>::max();
};

// ─── QMV (quantized matrix-vector multiply) ───────────────────────────────
#include "mlx_qmv_fast.h"

// Q4_0: group_size=32, bits=4, float16
// MLX-native entry point (separate weight/scale/bias buffers)
instantiate_kernel("mlx_qmv_fast_f16_gs32_b4", mlx_qmv_fast, float16_t, 32, 4)
// UNC entry point (Q4_0_FAST buffer layout, same buffer convention as qmv_fast_q4_0)
instantiate_kernel("unc_mlx_qmv_fast_q4_0", unc_mlx_qmv_fast, float16_t, 32, 4)

// ─── SDPA (scaled dot-product attention, vectorized decode) ───────────────
#include "sdpa_vector.h"

// head_dim=64 (TinyLlama, Phi-2)
instantiate_kernel("sdpa_vector_float16_t_64_64", sdpa_vector, float16_t, 64, 64)
instantiate_kernel("sdpa_vector_2pass_1_float16_t_64_64", sdpa_vector_2pass_1, float16_t, 64, 64)
instantiate_kernel("sdpa_vector_2pass_2_float16_t_64", sdpa_vector_2pass_2, float16_t, 64)

// head_dim=128 (Llama-2/3, Mistral, Qwen, Gemma)
instantiate_kernel("sdpa_vector_float16_t_128_128", sdpa_vector, float16_t, 128, 128)
instantiate_kernel("sdpa_vector_2pass_1_float16_t_128_128", sdpa_vector_2pass_1, float16_t, 128, 128)
instantiate_kernel("sdpa_vector_2pass_2_float16_t_128", sdpa_vector_2pass_2, float16_t, 128)
