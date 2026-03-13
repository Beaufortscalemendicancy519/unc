// unc_kernels.metal — Reference Metal kernel implementations for Universal Neural Compiler
//
// Tensor layout convention (matches unc IR):
//   Activation tensors: [h, N_max]  — first dim is static hidden size, second is dynamic seq stride
//   Weight tensors:     [h_out, h_in] — fully static
//   KV cache:           [num_kv_heads, max_seq, head_dim] — per layer
//
// params[0] = seq_len (current batch size, runtime)
// params[1] = n       (last dim of output = N_max = max_position_embeddings = stride)
// params[2] = k       (first dim of in0 = hidden/K dimension)
// params[3..7]        = op-specific (see each kernel)
//
// Buffer binding convention (all kernels):
//   [[buffer(0)]] = output
//   [[buffer(1)]] = input[0]
//   [[buffer(2)]] = input[1]   (if binary)
//   [[buffer(3)]] = input[2]   (if ternary, e.g. attention V)
//   [[buffer(4)]] = constant uint32_t params[8]

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// BF16 → F16 in-place conversion (used at init time)
// dispatch: (ceil(count/256), 1, 1) x (256, 1, 1)
// buffer(0)=data (uint16_t*, in-place), buffer(1)=count (uint32_t)
// ---------------------------------------------------------------------------
kernel void bf16_to_f16(device uint16_t* data [[buffer(0)]],
                        constant uint& count  [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    uint16_t b = data[gid];
    uint16_t sign = (b >> 15) & 1;
    int exp8 = (b >> 7) & 0xFF;
    uint16_t mant7 = b & 0x7F;
    uint16_t f16;
    if (exp8 == 0) {
        f16 = sign << 15;
    } else if (exp8 == 0xFF) {
        f16 = (sign << 15) | (0x1F << 10) | (uint16_t(mant7) << 3);
    } else {
        int u = exp8 - 127;
        if (u > 15) f16 = (sign << 15) | (0x1F << 10);
        else if (u < -14) f16 = sign << 15;
        else f16 = (sign << 15) | (uint16_t(u + 15) << 10) | (uint16_t(mant7) << 3);
    }
    data[gid] = f16;
}

// ---------------------------------------------------------------------------
// Gather: embedding lookup
// dispatch: (seq_len, 1, 1) threadgroups x (256, 1, 1) threads
// params[1]=act_stride (stride), params[2]=h (hidden=k), params[3]=vocab_size
// buffer(0)=out[h,act_stride], buffer(1)=embed[vocab,h], buffer(2)=token_ids[seq]
// ---------------------------------------------------------------------------
kernel void gather_f16(
    device half*             out        [[buffer(0)]],
    device const half*       embed      [[buffer(1)]],
    device const uint*       token_ids  [[buffer(2)]],
    constant uint*           params     [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len   = params[0];
    uint act_stride = params[1];   // stride (max_seq)
    uint h          = params[2];   // hidden size

    uint si = tgid_x;
    if (si >= seq_len) return;

    uint token = token_ids[si];
    // 256 threads split hidden dimension
    uint chunk = (h + 255u) / 256u;
    uint hi_start = lid_x * chunk;
    uint hi_end   = min(hi_start + chunk, h);

    for (uint hi = hi_start; hi < hi_end; hi++) {
        // out layout: [h, act_stride] -> element [hi, si] at hi*act_stride+si
        // embed layout: [vocab, h] -> element [token, hi] at token*h+hi
        out[hi * act_stride + si] = embed[token * h + hi];
    }
}

// ---------------------------------------------------------------------------
// RMSNorm: per-token normalisation
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=h, params[3]=eps (as float bits)
// buffer(0)=out[h,act_stride], buffer(1)=x[h,act_stride], buffer(2)=weight[h]
// ---------------------------------------------------------------------------
kernel void rms_norm_f16(
    device half*             out     [[buffer(0)]],
    device const half*       x       [[buffer(1)]],
    device const half*       weight  [[buffer(2)]],
    constant uint*           params  [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]],
    threadgroup float*       shared  [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];
    float eps       = as_type<float>(params[3]);

    uint si = tgid_x;
    if (si >= seq_len) return;

    // Strided partial sum of squares — each thread handles h/256 elements
    float partial_sq = 0.0f;
    for (uint hi = lid_x; hi < h; hi += 256u) {
        float v = float(x[hi * act_stride + si]);
        partial_sq += v * v;
    }
    shared[lid_x] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid_x < stride) {
            shared[lid_x] += shared[lid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(h) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Strided output write
    for (uint hi = lid_x; hi < h; hi += 256u) {
        float v = float(x[hi * act_stride + si]);
        out[hi * act_stride + si] = half(v * scale * float(weight[hi]));
    }
}

// ---------------------------------------------------------------------------
// QKNorm: per-head RMSNorm for Q/K projections (Qwen3)
// dispatch: (seq_len * num_heads, 1, 1) x (128, 1, 1), shared_memory = 128*4 bytes
// params[0]=seq_len, params[1]=act_stride, params[2]=head_dim, params[3]=eps (as float bits)
// params[4]=num_heads
// buffer(0)=out[total_dim, act_stride], buffer(1)=x[total_dim, act_stride], buffer(2)=weight[head_dim]
// ---------------------------------------------------------------------------
kernel void qk_norm_f16(
    device half*             out     [[buffer(0)]],
    device const half*       x       [[buffer(1)]],
    device const half*       weight  [[buffer(2)]],
    constant uint*           params  [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]],
    threadgroup float*       shared  [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint head_dim   = params[2];
    float eps       = as_type<float>(params[3]);
    uint num_heads  = params[4];

    uint si = tgid_x / num_heads;  // seq position
    uint hi = tgid_x % num_heads;  // head index
    if (si >= seq_len) return;

    uint base = hi * head_dim;

    // Partial sum of squares — each thread handles head_dim/128 elements
    float partial_sq = 0.0f;
    for (uint d = lid_x; d < head_dim; d += 128u) {
        float v = float(x[(base + d) * act_stride + si]);
        partial_sq += v * v;
    }
    shared[lid_x] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction (128 threads)
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lid_x < stride) {
            shared[lid_x] += shared[lid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(head_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = lid_x; d < head_dim; d += 128u) {
        float v = float(x[(base + d) * act_stride + si]);
        out[(base + d) * act_stride + si] = half(v * scale * float(weight[d]));
    }
}

// ---------------------------------------------------------------------------
// Fused Add + RMSNorm: residual = x + delta, out = rmsnorm(residual)
// dispatch: (seq_len, 1, 1) x (256, 1, 1), shared_memory = 256*4 bytes
// params[1]=act_stride, params[2]=h, params[3]=eps (as float bits)
// buffer(0)=out_norm[h,act_stride], buffer(1)=x[h,act_stride],
// buffer(2)=delta[h,act_stride], buffer(3)=weight[h],
// buffer(5)=out_residual[h,act_stride] (the sum x+delta, for next residual)
// ---------------------------------------------------------------------------
kernel void add_rms_norm_f16(
    device half*             out_norm     [[buffer(0)]],
    device const half*       x            [[buffer(1)]],
    device const half*       delta        [[buffer(2)]],
    device const half*       weight       [[buffer(3)]],
    constant uint*           params       [[buffer(4)]],
    device half*             out_residual [[buffer(5)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]],
    threadgroup float*       shared       [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];
    float eps       = as_type<float>(params[3]);

    uint si = tgid_x;
    if (si >= seq_len) return;

    // Pass 1: compute residual = x + delta, and sum of squares
    float partial_sq = 0.0f;
    for (uint hi = lid_x; hi < h; hi += 256u) {
        uint idx = hi * act_stride + si;
        float sum = float(x[idx]) + float(delta[idx]);
        out_residual[idx] = half(sum);
        partial_sq += sum * sum;
    }
    shared[lid_x] = partial_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid_x < stride) {
            shared[lid_x] += shared[lid_x + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(shared[0] / float(h) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: read residual back and apply normalization
    for (uint hi = lid_x; hi < h; hi += 256u) {
        uint idx = hi * act_stride + si;
        float v = float(out_residual[idx]);
        out_norm[idx] = half(v * scale * float(weight[hi]));
    }
}

// ---------------------------------------------------------------------------
// RoPE: rotary position embedding (non-interleaved)
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=rope_dim, params[3]=head_dim,
// params[4]=num_heads, params[5]=base (float bits), params[6]=total_seq_len
// buffer(0)=out[rope_dim,act_stride], buffer(1)=x[rope_dim,act_stride]
// ---------------------------------------------------------------------------
kernel void rope_f16(
    device half*             out    [[buffer(0)]],
    device const half*       x      [[buffer(1)]],
    constant uint*           params [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len      = params[0];
    uint act_stride   = params[1];
    uint rope_dim     = params[2];   // total elements = num_heads * head_dim
    uint head_dim     = params[3];
    float base        = as_type<float>(params[5]);
    uint total_seq    = params[6];   // total context length (for absolute position)

    uint si = tgid_x;
    if (si >= seq_len) return;

    // Absolute position = total_seq - seq_len + si
    uint abs_pos = total_seq - seq_len + si;

    uint half_dim = head_dim / 2u;

    // 256 threads split the rope_dim/2 pairs
    uint n_pairs = rope_dim / 2u;
    uint chunk   = (n_pairs + 255u) / 256u;
    uint p_start = lid_x * chunk;
    uint p_end   = min(p_start + chunk, n_pairs);

    for (uint pi = p_start; pi < p_end; pi++) {
        // pair index: pi = head_idx * half_dim + pair_within_head
        uint head_idx = pi / half_dim;
        uint pair_pos = pi % half_dim;

        // hidden indices of the two paired elements
        uint hi0 = head_idx * head_dim + pair_pos;
        uint hi1 = head_idx * head_dim + pair_pos + half_dim;

        float theta = (float)abs_pos / pow(base, 2.0f * (float)pair_pos / (float)head_dim);
        float cos_t = cos(theta);
        float sin_t = sin(theta);

        float x0 = float(x[hi0 * act_stride + si]);
        float x1 = float(x[hi1 * act_stride + si]);

        out[hi0 * act_stride + si] = half(x0 * cos_t - x1 * sin_t);
        out[hi1 * act_stride + si] = half(x0 * sin_t + x1 * cos_t);
    }
}

// ---------------------------------------------------------------------------
// Fused RoPE Q+K: apply rotary position embedding to both Q and K in one dispatch
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// buffer(0)=out_q, buffer(1)=q, buffer(2)=out_k, buffer(3)=k,
// params: [seq_len, act_stride, rope_dim_q, head_dim, 0, base, total_seq_len, rope_dim_k]
// ---------------------------------------------------------------------------
kernel void rope_qk_f16(
    device half*             out_q  [[buffer(0)]],
    device const half*       q      [[buffer(1)]],
    device half*             out_k  [[buffer(2)]],
    device const half*       k      [[buffer(3)]],
    constant uint*           params [[buffer(4)]],
    device const half*       rope_table [[buffer(5)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len      = params[0];
    uint act_stride   = params[1];
    uint rope_dim_q   = params[2];
    uint head_dim     = params[3];
    uint total_seq    = params[6];
    uint rope_dim_k   = params[7];

    uint si = tgid_x;
    if (si >= seq_len) return;

    uint abs_pos = total_seq - seq_len + si;
    uint half_dim = head_dim / 2u;

    // Process Q pairs
    uint n_pairs_q = rope_dim_q / 2u;
    uint chunk_q = (n_pairs_q + 255u) / 256u;
    uint p_start = lid_x * chunk_q;
    uint p_end   = min(p_start + chunk_q, n_pairs_q);

    for (uint pi = p_start; pi < p_end; pi++) {
        uint head_idx = pi / half_dim;
        uint pair_pos = pi % half_dim;
        uint hi0 = head_idx * head_dim + pair_pos;
        uint hi1 = head_idx * head_dim + pair_pos + half_dim;
        uint tidx = abs_pos * half_dim + pair_pos;
        float cos_t = float(rope_table[tidx * 2u]);
        float sin_t = float(rope_table[tidx * 2u + 1u]);
        float x0 = float(q[hi0 * act_stride + si]);
        float x1 = float(q[hi1 * act_stride + si]);
        out_q[hi0 * act_stride + si] = half(x0 * cos_t - x1 * sin_t);
        out_q[hi1 * act_stride + si] = half(x0 * sin_t + x1 * cos_t);
    }

    // Process K pairs
    uint n_pairs_k = rope_dim_k / 2u;
    uint chunk_k = (n_pairs_k + 255u) / 256u;
    p_start = lid_x * chunk_k;
    p_end   = min(p_start + chunk_k, n_pairs_k);

    for (uint pi = p_start; pi < p_end; pi++) {
        uint head_idx = pi / half_dim;
        uint pair_pos = pi % half_dim;
        uint hi0 = head_idx * head_dim + pair_pos;
        uint hi1 = head_idx * head_dim + pair_pos + half_dim;
        uint tidx = abs_pos * half_dim + pair_pos;
        float cos_t = float(rope_table[tidx * 2u]);
        float sin_t = float(rope_table[tidx * 2u + 1u]);
        float x0 = float(k[hi0 * act_stride + si]);
        float x1 = float(k[hi1 * act_stride + si]);
        out_k[hi0 * act_stride + si] = half(x0 * cos_t - x1 * sin_t);
        out_k[hi1 * act_stride + si] = half(x0 * sin_t + x1 * cos_t);
    }
}

// ---------------------------------------------------------------------------
// Fused KV cache append: append both K and V to the KV cache in one dispatch
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// buffer(0)=k_cache, buffer(1)=new_k, buffer(2)=v_cache, buffer(3)=new_v,
// params: [seq_len, total_seq, act_stride, head_dim, max_seq, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void kv_cache_append_fused_f16(
    device half*             k_cache [[buffer(0)]],
    device const half*       new_k   [[buffer(1)]],
    device half*             v_cache [[buffer(2)]],
    device const half*       new_v   [[buffer(3)]],
    constant uint*           params  [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len    = params[0];
    uint total_seq  = params[1];
    uint act_stride = params[2];
    uint head_dim   = params[3];
    uint max_seq    = params[4];
    uint kv_dim     = params[5];  // num_kv_heads * head_dim (total KV elements)

    uint kv_total = (kv_dim > 0u) ? kv_dim : 256u;  // backward compat: default 256

    uint kv_pos_start = total_seq - seq_len;
    uint si = tgid_x;
    if (si >= seq_len) return;

    uint global_pos = kv_pos_start + si;

    // Loop to cover all kv_dim elements with 256 threads
    for (uint e = lid_x; e < kv_total; e += 256u) {
        uint kv_head = e / head_dim;
        uint d       = e % head_dim;

        // Append K
        half k_val = new_k[e * act_stride + si];
        k_cache[kv_head * max_seq * head_dim + global_pos * head_dim + d] = k_val;

        // Append V
        half v_val = new_v[e * act_stride + si];
        v_cache[kv_head * max_seq * head_dim + global_pos * head_dim + d] = v_val;
    }
}

// ---------------------------------------------------------------------------
// Fused RoPE Q+K + KV cache append: apply RoPE to Q and K, then write
// rotated K and raw V directly to KV cache in one dispatch.
// Saves 1 dispatch per layer vs separate RoPE QK + KV Append.
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// buffer(0)=out_q, buffer(1)=q_in, buffer(2)=k_cache, buffer(3)=k_in,
// buffer(4)=v_cache, buffer(5)=v_in
// params(6): [seq_len, act_stride, rope_dim_q, head_dim, base_bits, total_seq, rope_dim_k, max_seq]
// ---------------------------------------------------------------------------
kernel void rope_qk_kv_append_f16(
    device half*             out_q   [[buffer(0)]],
    device const half*       q_in    [[buffer(1)]],
    device half*             k_cache [[buffer(2)]],
    device const half*       k_in    [[buffer(3)]],
    device half*             v_cache [[buffer(4)]],
    device const half*       v_in    [[buffer(5)]],
    constant uint*           params  [[buffer(6)]],
    device const half*       rope_table [[buffer(7)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len      = params[0];
    uint act_stride   = params[1];
    uint rope_dim_q   = params[2];
    uint head_dim     = params[3];
    uint total_seq    = params[5];
    uint rope_dim_k   = params[6];
    uint max_seq      = params[7];

    uint si = tgid_x;
    if (si >= seq_len) return;

    uint abs_pos = total_seq - seq_len + si;
    uint half_dim = head_dim / 2u;

    // --- RoPE Q: write rotated Q to activation buffer ---
    uint n_pairs_q = rope_dim_q / 2u;
    uint chunk_q = (n_pairs_q + 255u) / 256u;
    uint p_start = lid_x * chunk_q;
    uint p_end   = min(p_start + chunk_q, n_pairs_q);

    for (uint pi = p_start; pi < p_end; pi++) {
        uint head_idx = pi / half_dim;
        uint pair_pos = pi % half_dim;
        uint hi0 = head_idx * head_dim + pair_pos;
        uint hi1 = head_idx * head_dim + pair_pos + half_dim;
        uint tidx = abs_pos * half_dim + pair_pos;
        float cos_t = float(rope_table[tidx * 2u]);
        float sin_t = float(rope_table[tidx * 2u + 1u]);
        float x0 = float(q_in[hi0 * act_stride + si]);
        float x1 = float(q_in[hi1 * act_stride + si]);
        out_q[hi0 * act_stride + si] = half(x0 * cos_t - x1 * sin_t);
        out_q[hi1 * act_stride + si] = half(x0 * sin_t + x1 * cos_t);
    }

    // --- RoPE K + write to K cache ---
    uint n_pairs_k = rope_dim_k / 2u;
    uint chunk_k = (n_pairs_k + 255u) / 256u;
    p_start = lid_x * chunk_k;
    p_end   = min(p_start + chunk_k, n_pairs_k);

    for (uint pi = p_start; pi < p_end; pi++) {
        uint head_idx = pi / half_dim;
        uint pair_pos = pi % half_dim;
        uint hi0 = head_idx * head_dim + pair_pos;
        uint hi1 = head_idx * head_dim + pair_pos + half_dim;
        uint tidx = abs_pos * half_dim + pair_pos;
        float cos_t = float(rope_table[tidx * 2u]);
        float sin_t = float(rope_table[tidx * 2u + 1u]);
        float x0 = float(k_in[hi0 * act_stride + si]);
        float x1 = float(k_in[hi1 * act_stride + si]);
        float k0_rot = x0 * cos_t - x1 * sin_t;
        float k1_rot = x0 * sin_t + x1 * cos_t;
        // Write rotated K directly to KV cache
        k_cache[head_idx * max_seq * head_dim + abs_pos * head_dim + pair_pos] = half(k0_rot);
        k_cache[head_idx * max_seq * head_dim + abs_pos * head_dim + pair_pos + half_dim] = half(k1_rot);
    }

    // --- Copy V directly to V cache ---
    uint kv_size = rope_dim_k; // num_kv_heads * head_dim
    uint chunk_v = (kv_size + 255u) / 256u;
    uint v_start = lid_x * chunk_v;
    uint v_end   = min(v_start + chunk_v, kv_size);
    for (uint vi = v_start; vi < v_end; vi++) {
        uint kv_head = vi / head_dim;
        uint d       = vi % head_dim;
        half v_val = v_in[vi * act_stride + si];
        v_cache[kv_head * max_seq * head_dim + abs_pos * head_dim + d] = v_val;
    }
}

// ---------------------------------------------------------------------------
// Fused SDPA + RoPE + KV Cache Write (decode-only, single-pass)
// Combines rope_qk_kv_append_f16 + sdpa_vector into one dispatch.
// - Loads raw Q and applies RoPE inline via simd_shuffle
// - Writes RoPE'd K and raw V to KV cache
// - Computes scaled dot-product attention (single-pass vectorized)
//
// dispatch: (num_q_heads, 1, 1) x (32, 32, 1)  [32 simdgroups × 32 threads]
// buffer(0)=out, buffer(1)=raw_q, buffer(2)=raw_k, buffer(3)=raw_v,
// buffer(4)=kv_k_cache, buffer(5)=kv_v_cache, buffer(6)=rope_table,
// buffer(7)=params: [gqa_factor, N (total_seq), head_dim, scale_bits, max_seq, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void sdpa_rope_kv_decode_f16(
    device half*             out         [[buffer(0)]],
    device const half*       raw_q       [[buffer(1)]],
    device const half*       raw_k       [[buffer(2)]],
    device const half*       raw_v       [[buffer(3)]],
    device half*             kv_k_cache  [[buffer(4)]],
    device half*             kv_v_cache  [[buffer(5)]],
    device const half*       rope_table  [[buffer(6)]],
    constant uint*           params      [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;      // simdgroups = KV positions processed in parallel
    constexpr int BD = 32;      // threads per simdgroup
    constexpr int HEAD_DIM = 64;  // compile-time head dimension
    constexpr int qk_per_thread = HEAD_DIM / BD;  // = 2
    constexpr int half_dim = HEAD_DIM / 2;  // = 32

    uint gqa_factor = params[0];
    int  N          = int(params[1]);   // total_seq_len (including current token)
    float scale     = as_type<float>(params[3]);
    uint max_seq    = params[4];

    uint q_head_idx  = tgid_x;
    uint kv_head_idx = q_head_idx / gqa_factor;
    int  abs_pos     = N - 1;  // position of current token

    // RoPE pairing: thread t pairs with thread t+16 (or t-16)
    uint pair_thread = simd_lid < 16u ? simd_lid + 16u : simd_lid - 16u;

    // ---- Phase 1: Apply RoPE to Q inline (all threads) ----
    float q[qk_per_thread];
    {
        device const half* q_ptr = raw_q + q_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            q[i] = float(q_ptr[i]);
        }
        for (int i = 0; i < qk_per_thread; i++) {
            float q_pair = simd_shuffle(q[i], pair_thread);
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint pair_pos = pos_in_head < uint(half_dim) ? pos_in_head : pos_in_head - uint(half_dim);
            uint tidx = uint(abs_pos) * uint(half_dim) + pair_pos;
            float cos_t = float(rope_table[tidx * 2u]);
            float sin_t = float(rope_table[tidx * 2u + 1u]);
            if (pos_in_head < uint(half_dim)) {
                q[i] = q[i] * cos_t - q_pair * sin_t;
            } else {
                q[i] = q_pair * sin_t + q[i] * cos_t;
            }
            q[i] *= scale;
        }
    }

    // ---- Phase 2: RoPE K + store K/V in threadgroup memory (no device write yet) ----
    // Use TG memory so all simdgroups can access current token's K/V without device fence
    threadgroup half tg_k[HEAD_DIM];
    threadgroup half tg_v[HEAD_DIM];

    if (simd_gid == 0) {
        // K: load, apply RoPE, store to TG memory
        device const half* k_ptr = raw_k + kv_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        float k_vals[qk_per_thread];
        for (int i = 0; i < qk_per_thread; i++) {
            k_vals[i] = float(k_ptr[i]);
        }
        for (int i = 0; i < qk_per_thread; i++) {
            float k_pair = simd_shuffle(k_vals[i], pair_thread);
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint pair_pos = pos_in_head < uint(half_dim) ? pos_in_head : pos_in_head - uint(half_dim);
            uint tidx = uint(abs_pos) * uint(half_dim) + pair_pos;
            float cos_t = float(rope_table[tidx * 2u]);
            float sin_t = float(rope_table[tidx * 2u + 1u]);
            float k_roped;
            if (pos_in_head < uint(half_dim)) {
                k_roped = k_vals[i] * cos_t - k_pair * sin_t;
            } else {
                k_roped = k_pair * sin_t + k_vals[i] * cos_t;
            }
            tg_k[pos_in_head] = half(k_roped);
        }

        // V: load raw, store to TG memory
        device const half* v_ptr = raw_v + kv_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            tg_v[simd_lid * qk_per_thread + i] = v_ptr[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // cheap: on-chip SRAM only

    // ---- Phase 3: SDPA over cached positions 0..N-2 ----
    uint k_head_stride = max_seq * HEAD_DIM;
    uint k_seq_stride  = HEAD_DIM;
    device const half* keys   = kv_k_cache + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    device const half* values = kv_v_cache + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    int inner_k_stride = BN * int(k_seq_stride);
    int inner_v_stride = BN * int(k_seq_stride);

    float o[qk_per_thread] = {};
    float max_score = -__FLT_MAX__;
    float sum_exp_score = 0.0f;

    // Attend to all PREVIOUS positions (already in KV cache from prior tokens)
    for (int i = simd_gid; i < N - 1; i += BN) {
        float k[qk_per_thread];
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = float(keys[j]);
        }
        float score = 0.0f;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < qk_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * float(values[j]);
        }
        keys += inner_k_stride;
        values += inner_v_stride;
    }

    // Attend to CURRENT position (N-1) using threadgroup memory K/V
    {
        int target_sg = (N - 1) % BN;
        if (int(simd_gid) == target_sg) {
            float k[qk_per_thread];
            for (int j = 0; j < qk_per_thread; j++) {
                k[j] = float(tg_k[simd_lid * qk_per_thread + j]);
            }
            float score = 0.0f;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);

            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < qk_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * float(tg_v[simd_lid * qk_per_thread + j]);
            }
        }
    }

    // ---- Phase 4: Reduce across simdgroups ----
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = tg_max[simd_lid];
    float new_max = simd_max(max_score);
    float factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(tg_sum[simd_lid] * factor);

    for (int i = 0; i < qk_per_thread; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write attention output
    device half* out_ptr = out + q_head_idx * HEAD_DIM + simd_gid * qk_per_thread;
    if (simd_lid == 0) {
        for (int i = 0; i < qk_per_thread; i++) {
            out_ptr[i] = half(o[i]);
        }
    }

    // ---- Phase 5: Write K/V to cache AFTER attention (no device fence needed) ----
    // Future tokens will see these writes via dispatch ordering (serial encoder).
    if (simd_gid == 0) {
        for (int i = 0; i < qk_per_thread; i++) {
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint cache_idx = kv_head_idx * max_seq * HEAD_DIM + uint(abs_pos) * HEAD_DIM + pos_in_head;
            kv_k_cache[cache_idx] = tg_k[pos_in_head];
            kv_v_cache[cache_idx] = tg_v[pos_in_head];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused SDPA + RoPE + KV Cache Write (decode-only, HEAD_DIM=128 variant)
// Same algorithm as sdpa_rope_kv_decode_f16 but for head_dim=128 models
// (Llama-2/3, Mistral, Qwen, Gemma).
// dispatch: (num_q_heads, 1, 1) x (32, 32, 1)  [32 simdgroups × 32 threads]
// buffer(0)=out, buffer(1)=raw_q, buffer(2)=raw_k, buffer(3)=raw_v,
// buffer(4)=kv_k_cache, buffer(5)=kv_v_cache, buffer(6)=rope_table,
// buffer(7)=params: [gqa_factor, N (total_seq), head_dim, scale_bits, max_seq, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void sdpa_rope_kv_decode_f16_128(
    device half*             out         [[buffer(0)]],
    device const half*       raw_q       [[buffer(1)]],
    device const half*       raw_k       [[buffer(2)]],
    device const half*       raw_v       [[buffer(3)]],
    device half*             kv_k_cache  [[buffer(4)]],
    device half*             kv_v_cache  [[buffer(5)]],
    device const half*       rope_table  [[buffer(6)]],
    constant uint*           params      [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;      // simdgroups = KV positions processed in parallel
    constexpr int BD = 32;      // threads per simdgroup
    constexpr int HEAD_DIM = 128;  // compile-time head dimension
    constexpr int qk_per_thread = HEAD_DIM / BD;  // = 4
    constexpr int half_dim = HEAD_DIM / 2;  // = 64

    uint gqa_factor = params[0];
    int  N          = int(params[1]);   // total_seq_len (including current token)
    float scale     = as_type<float>(params[3]);
    uint max_seq    = params[4];

    uint q_head_idx  = tgid_x;
    uint kv_head_idx = q_head_idx / gqa_factor;
    int  abs_pos     = N - 1;  // position of current token

    // RoPE pairing: thread t pairs with thread t+16 (or t-16)
    // For HEAD_DIM=128: threads 0..15 hold positions 0..63, threads 16..31 hold 64..127
    uint pair_thread = simd_lid < 16u ? simd_lid + 16u : simd_lid - 16u;

    // ---- Phase 1: Apply RoPE to Q inline (all threads) ----
    float q[qk_per_thread];
    {
        device const half* q_ptr = raw_q + q_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            q[i] = float(q_ptr[i]);
        }
        for (int i = 0; i < qk_per_thread; i++) {
            float q_pair = simd_shuffle(q[i], pair_thread);
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint pair_pos = pos_in_head < uint(half_dim) ? pos_in_head : pos_in_head - uint(half_dim);
            uint tidx = uint(abs_pos) * uint(half_dim) + pair_pos;
            float cos_t = float(rope_table[tidx * 2u]);
            float sin_t = float(rope_table[tidx * 2u + 1u]);
            if (pos_in_head < uint(half_dim)) {
                q[i] = q[i] * cos_t - q_pair * sin_t;
            } else {
                q[i] = q_pair * sin_t + q[i] * cos_t;
            }
            q[i] *= scale;
        }
    }

    // ---- Phase 2: RoPE K + store K/V in threadgroup memory ----
    threadgroup half tg_k[HEAD_DIM];
    threadgroup half tg_v[HEAD_DIM];

    if (simd_gid == 0) {
        // K: load, apply RoPE, store to TG memory
        device const half* k_ptr = raw_k + kv_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        float k_vals[qk_per_thread];
        for (int i = 0; i < qk_per_thread; i++) {
            k_vals[i] = float(k_ptr[i]);
        }
        for (int i = 0; i < qk_per_thread; i++) {
            float k_pair = simd_shuffle(k_vals[i], pair_thread);
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint pair_pos = pos_in_head < uint(half_dim) ? pos_in_head : pos_in_head - uint(half_dim);
            uint tidx = uint(abs_pos) * uint(half_dim) + pair_pos;
            float cos_t = float(rope_table[tidx * 2u]);
            float sin_t = float(rope_table[tidx * 2u + 1u]);
            float k_roped;
            if (pos_in_head < uint(half_dim)) {
                k_roped = k_vals[i] * cos_t - k_pair * sin_t;
            } else {
                k_roped = k_pair * sin_t + k_vals[i] * cos_t;
            }
            tg_k[pos_in_head] = half(k_roped);
        }

        // V: load raw, store to TG memory
        device const half* v_ptr = raw_v + kv_head_idx * HEAD_DIM + simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            tg_v[simd_lid * qk_per_thread + i] = v_ptr[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: SDPA over cached positions 0..N-2 ----
    uint k_head_stride = max_seq * HEAD_DIM;
    uint k_seq_stride  = HEAD_DIM;
    device const half* keys   = kv_k_cache + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    device const half* values = kv_v_cache + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
    int inner_k_stride = BN * int(k_seq_stride);
    int inner_v_stride = BN * int(k_seq_stride);

    float o[qk_per_thread] = {};
    float max_score = -__FLT_MAX__;
    float sum_exp_score = 0.0f;

    for (int i = simd_gid; i < N - 1; i += BN) {
        float k[qk_per_thread];
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = float(keys[j]);
        }
        float score = 0.0f;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        float new_max = max(max_score, score);
        float factor = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < qk_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * float(values[j]);
        }
        keys += inner_k_stride;
        values += inner_v_stride;
    }

    // Attend to CURRENT position (N-1) using threadgroup memory K/V
    {
        int target_sg = (N - 1) % BN;
        if (int(simd_gid) == target_sg) {
            float k[qk_per_thread];
            for (int j = 0; j < qk_per_thread; j++) {
                k[j] = float(tg_k[simd_lid * qk_per_thread + j]);
            }
            float score = 0.0f;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * k[j];
            }
            score = simd_sum(score);

            float new_max = max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < qk_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * float(tg_v[simd_lid * qk_per_thread + j]);
            }
        }
    }

    // ---- Phase 4: Reduce across simdgroups ----
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = tg_max[simd_lid];
    float new_max = simd_max(max_score);
    float factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(tg_sum[simd_lid] * factor);

    for (int i = 0; i < qk_per_thread; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write attention output
    device half* out_ptr = out + q_head_idx * HEAD_DIM + simd_gid * qk_per_thread;
    if (simd_lid == 0) {
        for (int i = 0; i < qk_per_thread; i++) {
            out_ptr[i] = half(o[i]);
        }
    }

    // ---- Phase 5: Write K/V to cache AFTER attention ----
    if (simd_gid == 0) {
        for (int i = 0; i < qk_per_thread; i++) {
            uint pos_in_head = simd_lid * qk_per_thread + i;
            uint cache_idx = kv_head_idx * max_seq * HEAD_DIM + uint(abs_pos) * HEAD_DIM + pos_in_head;
            kv_k_cache[cache_idx] = tg_k[pos_in_head];
            kv_v_cache[cache_idx] = tg_v[pos_in_head];
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax: row-wise softmax over last axis
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=row_width
// buffer(0)=out[row_width,act_stride], buffer(1)=x[row_width,act_stride]
// ---------------------------------------------------------------------------
kernel void softmax_f16(
    device half*             out    [[buffer(0)]],
    device const half*       x      [[buffer(1)]],
    constant uint*           params [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]],
    threadgroup float*       shared [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint row_width  = params[2];

    uint si = tgid_x;
    if (si >= seq_len) return;

    // Pass 1: parallel max reduction
    float local_max = -INFINITY;
    for (uint i = lid_x; i < row_width; i += 256u) {
        float v = float(x[i * act_stride + si]);
        local_max = max(local_max, v);
    }
    shared[lid_x] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid_x < stride) shared[lid_x] = max(shared[lid_x], shared[lid_x + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: parallel exp-sum reduction
    float local_sum = 0.0f;
    for (uint i = lid_x; i < row_width; i += 256u) {
        local_sum += fast::exp(float(x[i * act_stride + si]) - mx);
    }
    shared[lid_x] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid_x < stride) shared[lid_x] += shared[lid_x + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: parallel normalize
    float inv_sum = 1.0f / total;
    for (uint i = lid_x; i < row_width; i += 256u) {
        out[i * act_stride + si] = half(fast::exp(float(x[i * act_stride + si]) - mx) * inv_sum);
    }
}

// ---------------------------------------------------------------------------
// GEMM (prefill): C[M,N] = B[M,K] x A[K,N]
// dispatch: (ceil(M/32), ceil(seq_len/32), 1) x (32, 1, 1)
// params[1]=act_stride (seq stride), params[2]=K, params[3]=M (output rows)
// buffer(0)=C[M,act_stride], buffer(1)=A[K,act_stride] (activation), buffer(2)=B[M,K] (weight)
// ---------------------------------------------------------------------------
kernel void steel_gemm_f16_32x32x16(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const half*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint m = tgid.x * 32u + lid.x;
    if (m >= M) return;

    uint n_start = tgid.y * 32u;
    for (uint cn = 0u; cn < 32u; cn++) {
        uint n = n_start + cn;
        if (n >= seq_len) break;

        float sum = 0.0f;
        for (uint kk = 0u; kk < K; kk++) {
            sum += float(A[kk * act_stride + n]) * float(B[m * K + kk]);
        }
        C[m * act_stride + n] = half(sum);
    }
}

// 64x64 tile alias (same implementation)
kernel void steel_gemm_f16_64x64x32(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const half*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint m = tgid.x * 32u + lid.x;
    if (m >= M) return;

    uint n_start = tgid.y * 32u;
    for (uint cn = 0u; cn < 32u; cn++) {
        uint n = n_start + cn;
        if (n >= seq_len) break;
        float sum = 0.0f;
        for (uint kk = 0u; kk < K; kk++) {
            sum += float(A[kk * act_stride + n]) * float(B[m * K + kk]);
        }
        C[m * act_stride + n] = half(sum);
    }
}

kernel void steel_gemm_bf16_64x64x32(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const half*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint m = tgid.x * 32u + lid.x;
    if (m >= M) return;

    uint n_start = tgid.y * 32u;
    for (uint cn = 0u; cn < 32u; cn++) {
        uint n = n_start + cn;
        if (n >= seq_len) break;
        float sum = 0.0f;
        for (uint kk = 0u; kk < K; kk++) {
            sum += float(A[kk * act_stride + n]) * float(B[m * K + kk]);
        }
        C[m * act_stride + n] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// GEMV (decode): y[M] = B[M,K] x A[K] (seq_len=1)
// dispatch: (ceil(M/GEMV_BM), 1, 1) x (GEMV_TG_SIZE, 1, 1)
// params[1]=act_stride, params[2]=K, params[3]=M
// buffer(0)=C[M,act_stride], buffer(1)=A[K,act_stride] (activation), buffer(2)=B[M,K] (weight)
//
// Shared-memory-free design: activation is read directly from device memory
// into registers. With compact decode buffer (act_stride=1), reads are
// contiguous and hit L1/L2 cache across SIMD groups. Eliminates
// threadgroup_barrier overhead and shared memory reservation, improving
// GPU occupancy.
//
// Each threadgroup processes GEMV_BM output rows.
// Each SIMD group handles GEMV_BR rows, computing partial dot products across K.
// Weight reads use half4 vectorization for 4x fewer memory transactions.
// ---------------------------------------------------------------------------
#define GEMV_BR    8u                          // rows per SIMD group
#define GEMV_BK    128u                        // K-tile processed per iteration
#define GEMV_TG_SIZE 256u                      // threads per threadgroup
#define GEMV_N_SIMD  (GEMV_TG_SIZE / 32u)      // SIMD groups per TG = 8
#define GEMV_BM    (GEMV_N_SIMD * GEMV_BR)     // rows per TG = 64

kernel void gemv_f16(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const half*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];   // 1 for compact decode
    uint K          = params[2];
    uint M          = params[3];

    // Each SIMD group handles GEMV_BR output rows
    uint m_base = tgid_x * GEMV_BM + simd_gid * GEMV_BR;

    for (uint n = 0u; n < seq_len; n++) {
        // Each thread accumulates GEMV_BR partial sums
        float acc[GEMV_BR] = {};

        // Vectorized half4 reads — contiguous when act_stride=1 (decode)
        uint k_vec = K / 4u;
        for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
            uint base_k = kv * 4u;
            half4 a = *(device const half4*)(A + base_k * act_stride + n);
            float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);

            for (uint r = 0u; r < GEMV_BR; r++) {
                uint m = m_base + r;
                if (m < M) {
                    half4 w = *(device const half4*)(B + m * K + base_k);
                    acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                            + float(w[2]) * a2 + float(w[3]) * a3;
                }
            }
        }
        // Handle remainder
        for (uint kk = k_vec * 4u + simd_lid; kk < K; kk += 32u) {
            float a_val = float(A[kk * act_stride + n]);
            for (uint r = 0u; r < GEMV_BR; r++) {
                uint m = m_base + r;
                if (m < M) {
                    acc[r] += float(B[m * K + kk]) * a_val;
                }
            }
        }

        // Reduce within SIMD group and write output
        for (uint r = 0u; r < GEMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV (decode): 3 GEMVs sharing the same activation input
// dispatch: (ceil(M_q/GEMV_BM), 1, 1) x (GEMV_TG_SIZE, 1, 1)
// buffer(0)=out_q[M_q,act_stride], buffer(1)=A[K,act_stride] (activation),
// buffer(2)=W_q[M_q,K], buffer(3)=out_k[M_kv,act_stride],
// buffer(4)=W_k[M_kv,K], buffer(5)=out_v[M_kv,act_stride], buffer(6)=W_v[M_kv,K]
// params: [seq_len, act_stride, K, M_q, M_kv, 0, 0, 0] at buffer(7)
// Shared-memory-free: reads activation directly from device memory
// ---------------------------------------------------------------------------
kernel void fused_qkv_gemv_f16(
    device half*             out_q  [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const half*       W_q    [[buffer(2)]],
    device half*             out_k  [[buffer(3)]],
    device const half*       W_k    [[buffer(4)]],
    device half*             out_v  [[buffer(5)]],
    device const half*       W_v    [[buffer(6)]],
    constant uint*           params [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];   // 1 for compact decode
    uint K          = params[2];
    uint M_q        = params[3];
    uint M_kv       = params[4];

    uint m_base = tgid_x * GEMV_BM + simd_gid * GEMV_BR;

    for (uint n = 0u; n < seq_len; n++) {
        // --- Q projection ---
        {
            float acc[GEMV_BR] = {};
            uint k_vec = K / 4u;
            for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
                uint base_k = kv * 4u;
                half4 a = *(device const half4*)(A + base_k * act_stride + n);
                float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);
                for (uint r = 0u; r < GEMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M_q) {
                        half4 w = *(device const half4*)(W_q + m * K + base_k);
                        acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                                + float(w[2]) * a2 + float(w[3]) * a3;
                    }
                }
            }
            for (uint r = 0u; r < GEMV_BR; r++) {
                float total = simd_sum(acc[r]);
                uint m = m_base + r;
                if (simd_lid == 0u && m < M_q) {
                    out_q[m * act_stride + n] = half(total);
                }
            }
        }

        // --- K projection (only if this threadgroup covers M_kv range) ---
        if (m_base < M_kv) {
            float acc[GEMV_BR] = {};
            uint k_vec = K / 4u;
            for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
                uint base_k = kv * 4u;
                half4 a = *(device const half4*)(A + base_k * act_stride + n);
                float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);
                for (uint r = 0u; r < GEMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M_kv) {
                        half4 w = *(device const half4*)(W_k + m * K + base_k);
                        acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                                + float(w[2]) * a2 + float(w[3]) * a3;
                    }
                }
            }
            for (uint r = 0u; r < GEMV_BR; r++) {
                float total = simd_sum(acc[r]);
                uint m = m_base + r;
                if (simd_lid == 0u && m < M_kv) {
                    out_k[m * act_stride + n] = half(total);
                }
            }
        }

        // --- V projection (same range as K) ---
        if (m_base < M_kv) {
            float acc[GEMV_BR] = {};
            uint k_vec = K / 4u;
            for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
                uint base_k = kv * 4u;
                half4 a = *(device const half4*)(A + base_k * act_stride + n);
                float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);
                for (uint r = 0u; r < GEMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M_kv) {
                        half4 w = *(device const half4*)(W_v + m * K + base_k);
                        acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                                + float(w[2]) * a2 + float(w[3]) * a3;
                    }
                }
            }
            for (uint r = 0u; r < GEMV_BR; r++) {
                float total = simd_sum(acc[r]);
                uint m = m_base + r;
                if (simd_lid == 0u && m < M_kv) {
                    out_v[m * act_stride + n] = half(total);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up GEMV (decode): 2 GEMVs sharing the same activation input
// dispatch: (ceil(M/GEMV_BM), 1, 1) x (GEMV_TG_SIZE, 1, 1)
// buffer(0)=out_gate[M,act_stride], buffer(1)=A[K,act_stride] (activation),
// buffer(2)=W_gate[M,K], buffer(3)=out_up[M,act_stride], buffer(4)=W_up[M,K]
// params: [seq_len, act_stride, K, M, 0, 0, 0, 0] at buffer(5)
// Shared-memory-free: reads activation directly from device memory
// ---------------------------------------------------------------------------
kernel void fused_gate_up_gemv_f16(
    device half*             out_gate [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const half*       W_gate   [[buffer(2)]],
    device half*             out_up   [[buffer(3)]],
    device const half*       W_up     [[buffer(4)]],
    constant uint*           params   [[buffer(5)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];   // 1 for compact decode
    uint K          = params[2];
    uint M          = params[3];

    uint m_base = tgid_x * GEMV_BM + simd_gid * GEMV_BR;

    for (uint n = 0u; n < seq_len; n++) {
        // --- Gate projection ---
        {
            float acc[GEMV_BR] = {};
            uint k_vec = K / 4u;
            for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
                uint base_k = kv * 4u;
                half4 a = *(device const half4*)(A + base_k * act_stride + n);
                float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);
                for (uint r = 0u; r < GEMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M) {
                        half4 w = *(device const half4*)(W_gate + m * K + base_k);
                        acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                                + float(w[2]) * a2 + float(w[3]) * a3;
                    }
                }
            }
            for (uint r = 0u; r < GEMV_BR; r++) {
                float total = simd_sum(acc[r]);
                uint m = m_base + r;
                if (simd_lid == 0u && m < M) {
                    out_gate[m * act_stride + n] = half(total);
                }
            }
        }

        // --- Up projection ---
        {
            float acc[GEMV_BR] = {};
            uint k_vec = K / 4u;
            for (uint kv = simd_lid; kv < k_vec; kv += 32u) {
                uint base_k = kv * 4u;
                half4 a = *(device const half4*)(A + base_k * act_stride + n);
                float a0 = float(a[0]), a1 = float(a[1]), a2 = float(a[2]), a3 = float(a[3]);
                for (uint r = 0u; r < GEMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M) {
                        half4 w = *(device const half4*)(W_up + m * K + base_k);
                        acc[r] += float(w[0]) * a0 + float(w[1]) * a1
                                + float(w[2]) * a2 + float(w[3]) * a3;
                    }
                }
            }
            for (uint r = 0u; r < GEMV_BR; r++) {
                float total = simd_sum(acc[r]);
                uint m = m_base + r;
                if (simd_lid == 0u && m < M) {
                    out_up[m * act_stride + n] = half(total);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SiLU: out = x * sigmoid(x)  — elementwise on [h, act_stride]
// dispatch: (ceil(h/256), 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=h
// buffer(0)=out[h,act_stride], buffer(1)=x[h,act_stride]
// ---------------------------------------------------------------------------
kernel void silu_f16(
    device half*             out    [[buffer(0)]],
    device const half*       x      [[buffer(1)]],
    constant uint*           params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];

    if (gid >= h) return;

    for (uint si = 0u; si < seq_len; si++) {
        float val = float(x[gid * act_stride + si]);
        float sig = 1.0f / (1.0f + fast::exp(-val));
        out[gid * act_stride + si] = half(val * sig);
    }
}

// ---------------------------------------------------------------------------
// Add: out = a + b  — vectorized elementwise on [h, act_stride]
// For decode (seq_len=1): reads are strided so use scalar path
// For prefill: inner loop over seq positions, also scalar (strided layout)
// dispatch: (ceil(h/256), 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=h
// buffer(0)=out[h,act_stride], buffer(1)=a[h,act_stride], buffer(2)=b[h,act_stride]
// ---------------------------------------------------------------------------
kernel void add_f16(
    device half*             out    [[buffer(0)]],
    device const half*       a      [[buffer(1)]],
    device const half*       b      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];

    if (gid >= h) return;

    for (uint si = 0u; si < seq_len; si++) {
        out[gid * act_stride + si] = a[gid * act_stride + si] + b[gid * act_stride + si];
    }
}

// ---------------------------------------------------------------------------
// Mul: out = a * b  — elementwise on [h, act_stride]
// dispatch: (ceil(h/256), 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=h
// buffer(0)=out[h,act_stride], buffer(1)=a[h,act_stride], buffer(2)=b[h,act_stride]
// ---------------------------------------------------------------------------
kernel void mul_f16(
    device half*             out    [[buffer(0)]],
    device const half*       a      [[buffer(1)]],
    device const half*       b      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];

    if (gid >= h) return;

    for (uint si = 0u; si < seq_len; si++) {
        out[gid * act_stride + si] = a[gid * act_stride + si] * b[gid * act_stride + si];
    }
}

// ---------------------------------------------------------------------------
// SwiGLU: out = silu(gate) * up  (fused)
// dispatch: (ceil(h/256), 1, 1) x (256, 1, 1)
// params[1]=act_stride, params[2]=h
// buffer(0)=out[h,act_stride], buffer(1)=gate[h,act_stride], buffer(2)=up[h,act_stride]
// ---------------------------------------------------------------------------
kernel void swiglu_f16(
    device half*             out    [[buffer(0)]],
    device const half*       gate   [[buffer(1)]],
    device const half*       up     [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint h          = params[2];

    if (gid >= h) return;

    for (uint si = 0u; si < seq_len; si++) {
        float g    = float(gate[gid * act_stride + si]);
        float silu = g / (1.0f + fast::exp(-g));
        out[gid * act_stride + si] = half(silu * float(up[gid * act_stride + si]));
    }
}

// ---------------------------------------------------------------------------
// KV cache append: write new K or V into the persistent cache
// dispatch: (seq_len, 1, 1) x (256, 1, 1)
// params[0]=seq_len, params[1]=total_seq_len, params[2]=act_stride (N_max or 1),
// params[3]=head_dim, params[4]=max_seq_len (KV cache stride), params[5]=layer (offset pre-applied)
// buffer(0)=kv_cache[kv_dim, max_seq, head_dim] (layer offset already applied to buffer base)
// buffer(1)=kv_cache (same, unused in append path)
// buffer(2)=new_kv[kv_dim, act_stride]  where kv_dim = num_kv_heads * head_dim
// ---------------------------------------------------------------------------
kernel void kv_cache_append_f16(
    device half*             kv_out  [[buffer(0)]],
    device const half*       kv_in   [[buffer(1)]],
    device const half*       new_kv  [[buffer(2)]],
    constant uint*           params  [[buffer(4)]],
    uint tgid_x [[threadgroup_position_in_grid]],
    uint lid_x  [[thread_position_in_threadgroup]])
{
    uint seq_len    = params[0];
    uint total_seq  = params[1];   // total context length after this step
    uint act_stride = params[2];   // activation stride: N_max for prefill, 1 for decode
    uint head_dim   = params[3];
    uint max_seq    = params[4];   // max_position_embeddings (KV cache stride)

    uint kv_pos_start = total_seq - seq_len;
    uint si = tgid_x;
    if (si >= seq_len) return;

    uint global_pos = kv_pos_start + si;

    // new_kv layout: [kv_dim, act_stride] where kv_dim = num_kv_heads * head_dim
    // kv_out layout (layer offset pre-applied): [num_kv_heads, max_seq, head_dim]
    // Thread lid_x directly indexes one kv_dim element
    // (works for kv_dim = num_kv_heads * head_dim <= 256; TinyLlama: 4*64=256 exactly)
    uint kv_head = lid_x / head_dim;
    uint d       = lid_x % head_dim;

    half val = new_kv[lid_x * act_stride + si];
    kv_out[kv_head * max_seq * head_dim + global_pos * head_dim + d] = val;
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention (causal) — parallel across heads with simd_sum
// dispatch: (seq_len, num_heads, 1) x (32, 1, 1)   (one SIMD group per head per token)
// params[0]=seq_len, params[1]=total_seq_len, params[2]=N_max (KV cache stride),
// params[3]=head_dim, params[4]=gqa_factor, params[5]=num_kv_heads, params[6]=act_stride
// buffer(0)=out[q_dim,act_stride], buffer(1)=Q[q_dim,act_stride],
// buffer(2)=K_cache[num_kv_heads, max_seq, head_dim] (layer offset pre-applied),
// buffer(3)=V_cache[num_kv_heads, max_seq, head_dim] (layer offset pre-applied)
// ---------------------------------------------------------------------------
kernel void steel_attention_causal_f16(
    device half*             out    [[buffer(0)]],
    device const half*       Q      [[buffer(1)]],
    device const half*       K      [[buffer(2)]],
    device const half*       V      [[buffer(3)]],
    constant uint*           params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint seq_len      = params[0];
    uint total_seq    = params[1];
    uint N_max        = params[2];   // KV cache max_seq stride
    uint head_dim     = params[3];
    uint gqa_factor   = params[4];
    uint num_kv_heads = params[5];
    uint act_stride   = params[6];   // activation stride: N_max for prefill, 1 for decode
    float scale       = rsqrt((float)head_dim);

    uint local_t  = tgid.x;
    uint h        = tgid.y;      // head index from 2D grid
    if (local_t >= seq_len) return;

    uint q_pos_start = total_seq - seq_len;
    uint global_t    = q_pos_start + local_t;
    uint kv_h        = h / gqa_factor;

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    // Each thread accumulates head_dim/32 V values
    float acc[4] = {};  // max head_dim/32 = 128/32 = 4

    uint my_dims = (head_dim + 31u) / 32u;  // elements per thread

    for (uint pos = 0u; pos <= global_t; pos++) {
        // Compute Q·K dot product — 32 threads split head_dim
        float partial = 0.0f;
        for (uint d = lid.x; d < head_dim; d += 32u) {
            float q_d = float(Q[(h * head_dim + d) * act_stride + local_t]);
            float k_d = float(K[kv_h * N_max * head_dim + pos * head_dim + d]);
            partial += q_d * k_d;
        }
        float score = simd_sum(partial) * scale;

        // Online softmax update — all threads see same score via simd_sum
        float new_max = max(max_score, score);
        float decay   = fast::exp(max_score - new_max);
        float w       = fast::exp(score - new_max);
        sum_exp = sum_exp * decay + w;

        // Accumulate V weighted by attention — each thread handles its stripe
        for (uint di = 0u; di < my_dims; di++) {
            uint d = lid.x + di * 32u;
            if (d < head_dim) {
                float v_d = float(V[kv_h * N_max * head_dim + pos * head_dim + d]);
                acc[di] = acc[di] * decay + w * v_d;
            }
        }
        max_score = new_max;
    }

    // Write output — each thread writes its stripe of head_dim
    float inv_sum = 1.0f / sum_exp;
    for (uint di = 0u; di < my_dims; di++) {
        uint d = lid.x + di * 32u;
        if (d < head_dim) {
            out[(h * head_dim + d) * act_stride + local_t] = half(acc[di] * inv_sum);
        }
    }
}

// Non-causal attention — parallel across heads with simd_sum
// dispatch: (seq_len, num_heads, 1) x (32, 1, 1)
kernel void steel_attention_f16(
    device half*             out    [[buffer(0)]],
    device const half*       Q      [[buffer(1)]],
    device const half*       K      [[buffer(2)]],
    device const half*       V      [[buffer(3)]],
    constant uint*           params [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    uint seq_len      = params[0];
    uint total_seq    = params[1];
    uint N_max        = params[2];   // KV cache max_seq stride
    uint head_dim     = params[3];
    uint gqa_factor   = params[4];
    uint num_kv_heads = params[5];
    uint act_stride   = params[6];   // activation stride: N_max for prefill, 1 for decode
    float scale       = rsqrt((float)head_dim);

    uint local_t = tgid.x;
    uint h       = tgid.y;
    if (local_t >= seq_len) return;

    uint kv_h = h / gqa_factor;

    float max_score = -INFINITY;
    float sum_exp   = 0.0f;
    float acc[4]    = {};
    uint my_dims    = (head_dim + 31u) / 32u;

    for (uint pos = 0u; pos < total_seq; pos++) {
        float partial = 0.0f;
        for (uint d = lid.x; d < head_dim; d += 32u) {
            float q_d = float(Q[(h * head_dim + d) * act_stride + local_t]);
            float k_d = float(K[kv_h * N_max * head_dim + pos * head_dim + d]);
            partial += q_d * k_d;
        }
        float score = simd_sum(partial) * scale;

        float new_max = max(max_score, score);
        float decay   = fast::exp(max_score - new_max);
        float w       = fast::exp(score - new_max);
        sum_exp = sum_exp * decay + w;

        for (uint di = 0u; di < my_dims; di++) {
            uint d = lid.x + di * 32u;
            if (d < head_dim) {
                float v_d = float(V[kv_h * N_max * head_dim + pos * head_dim + d]);
                acc[di] = acc[di] * decay + w * v_d;
            }
        }
        max_score = new_max;
    }

    float inv_sum = 1.0f / sum_exp;
    for (uint di = 0u; di < my_dims; di++) {
        uint d = lid.x + di * 32u;
        if (d < head_dim) {
            out[(h * head_dim + d) * act_stride + local_t] = half(acc[di] * inv_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// Logit gather: extract last-token logits from strided [vocab, N_max] f16 → f32
// dispatch: (ceil(vocab_size/256), 1, 1) x (256, 1, 1)
// buffer(0)=out_f32[vocab], buffer(1)=act_f16[vocab, N_max]
// buffer(2)=params: [vocab_size, N_max, last_si]
// ---------------------------------------------------------------------------
kernel void logit_gather_f32(
    device float*            out    [[buffer(0)]],
    device const half*       act    [[buffer(1)]],
    constant uint*           params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint vocab_size = params[0];
    uint N_max      = params[1];
    uint last_si    = params[2];
    if (gid >= vocab_size) return;
    out[gid] = float(act[gid * N_max + last_si]);
}

// ===========================================================================
// Quantized weight support
// ===========================================================================
//
// Q8_0 block: 34 bytes per 32 elements
//   - 2 bytes: half scale (delta)
//   - 32 bytes: int8_t quants[32]
//
// Q4_0 block: 18 bytes per 32 elements
//   - 2 bytes: half scale (delta)
//   - 16 bytes: uint8_t quants[16] (each byte = 2 × 4-bit values)
// ===========================================================================

// ---------------------------------------------------------------------------
// F16 → Q8_0 in-place quantization (used at init time)
// dispatch: (n_blocks, 1, 1) x (1, 1, 1)
// Each thread quantizes one block of 32 F16 elements.
// buffer(0) = source F16 data (read)
// buffer(1) = destination Q8_0 data (write)
// buffer(2) = n_blocks (uint32_t)
// ---------------------------------------------------------------------------
kernel void quantize_f16_to_q8_0(
    device const half*   src     [[buffer(0)]],
    device char*         dst     [[buffer(1)]],
    constant uint&       n_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    // Read 32 F16 source elements
    device const half* block_src = src + gid * 32u;
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) {
        vals[i] = float(block_src[i]);
        amax = max(amax, abs(vals[i]));
    }

    // Compute scale
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

    // Write block: 2-byte scale + 32 × 1-byte quants = 34 bytes
    device char* block_dst = dst + gid * 34u;
    *(device half*)block_dst = half(d);

    for (uint i = 0; i < 32u; i++) {
        int q = (int)round(vals[i] * id);
        q = clamp(q, -128, 127);
        block_dst[2 + i] = (char)q;
    }
}

// ---------------------------------------------------------------------------
// F16 → Q4_0 in-place quantization
// dispatch: (n_blocks, 1, 1) x (1, 1, 1)
// buffer(0) = source F16, buffer(1) = destination Q4_0, buffer(2) = n_blocks
// ---------------------------------------------------------------------------
kernel void quantize_f16_to_q4_0(
    device const half*   src     [[buffer(0)]],
    device uchar*        dst     [[buffer(1)]],
    constant uint&       n_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    device const half* block_src = src + gid * 32u;
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) {
        vals[i] = float(block_src[i]);
        amax = max(amax, abs(vals[i]));
    }

    float d = amax / 7.0f;
    float id = (d != 0.0f) ? (7.0f / amax) : 0.0f;

    // Write block: 2-byte scale + 16 bytes (32 × 4-bit) = 18 bytes
    device uchar* block_dst = dst + gid * 18u;
    *(device half*)block_dst = half(d);

    for (uint i = 0; i < 16u; i++) {
        int q0 = (int)round(vals[i * 2]     * id) + 8;
        int q1 = (int)round(vals[i * 2 + 1] * id) + 8;
        q0 = clamp(q0, 0, 15);
        q1 = clamp(q1, 0, 15);
        block_dst[2 + i] = (uchar)(q0 | (q1 << 4));
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMV (Q8_0): y[M] = B_q8[M,K] x A[K]
// dispatch: (ceil(M/GEMV_BM), 1, 1) x (GEMV_TG_SIZE, 1, 1)
// params[1]=act_stride, params[2]=K, params[3]=M
// buffer(0)=C[M,act_stride] (f16 output)
// buffer(1)=A[K,act_stride] (f16 activation)
// buffer(2)=B_q8[M, K/32 * 34] (Q8_0 quantized weights)
// threadgroup(0) = half[K] activation cache
// ---------------------------------------------------------------------------
kernel void quantized_gemv_q8_0(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const char*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];   // 1 for compact decode
    uint K          = params[2];
    uint M          = params[3];

    uint m_base = tgid_x * GEMV_BM + simd_gid * GEMV_BR;
    uint n_blocks_per_row = K / 32u;
    uint row_bytes = n_blocks_per_row * 34u;  // 34 bytes per Q8_0 block

    for (uint n = 0u; n < seq_len; n++) {
        float acc[GEMV_BR] = {};

        // Process weight blocks — each SIMD lane handles different blocks
        for (uint bi = simd_lid; bi < n_blocks_per_row; bi += 32u) {
            uint base_k = bi * 32u;
            for (uint r = 0u; r < GEMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;

                device const char* block = B + m * row_bytes + bi * 34u;
                float d = float(*(device const half*)block);
                float partial = 0.0f;

                for (uint j = 0u; j < 32u; j++) {
                    float w = d * float((int)block[2 + j]);
                    partial += w * float(A[(base_k + j) * act_stride + n]);
                }
                acc[r] += partial;
            }
        }

        // Reduce and write
        for (uint r = 0u; r < GEMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantized GEMV (Q4_0): y[M] = B_q4[M,K] x A[K]
// Same dispatch as Q8_0 but with 4-bit dequantization.
// ---------------------------------------------------------------------------
kernel void quantized_gemv_q4_0(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const uchar*      B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];   // 1 for compact decode
    uint K          = params[2];
    uint M          = params[3];

    uint m_base = tgid_x * GEMV_BM + simd_gid * GEMV_BR;
    uint n_blocks_per_row = K / 32u;
    uint row_bytes = n_blocks_per_row * 18u;  // 18 bytes per Q4_0 block

    for (uint n = 0u; n < seq_len; n++) {
        float acc[GEMV_BR] = {};

        for (uint bi = simd_lid; bi < n_blocks_per_row; bi += 32u) {
            uint base_k = bi * 32u;

            for (uint r = 0u; r < GEMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;

                device const uchar* block = B + m * row_bytes + bi * 18u;
                float d = float(*(device const half*)block);
                float partial = 0.0f;

                for (uint j = 0u; j < 16u; j++) {
                    uchar packed = block[2 + j];
                    float w0 = d * float((int)(packed & 0xF) - 8);
                    float w1 = d * float((int)(packed >> 4) - 8);
                    partial += w0 * float(A[(base_k + j * 2) * act_stride + n]);
                    partial += w1 * float(A[(base_k + j * 2 + 1) * act_stride + n]);
                }
                acc[r] += partial;
            }
        }

        for (uint r = 0u; r < GEMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ===========================================================================
// Fast quantized kernels — separated scale/data layout for vectorized reads
// ===========================================================================
//
// Q4_0_FAST layout (per tensor with M rows, K cols, nb = K/32 blocks per row):
//   Scales region:  half[M][nb]      at offset 0                 (M * nb * 2 bytes)
//   Nibbles region: uchar[M][nb][16] at offset M * nb * 2 bytes  (M * nb * 16 bytes)
//   Total: M * nb * 18 bytes (same as interleaved Q4_0)
//
// Q8_0_FAST layout:
//   Scales region:  half[M][nb]      at offset 0                 (M * nb * 2 bytes)
//   Weights region: char[M][nb][32]  at offset M * nb * 2 bytes  (M * nb * 32 bytes)
//   Total: M * nb * 34 bytes (same as interleaved Q8_0)
// ===========================================================================

// ---------------------------------------------------------------------------
// F16 → Q4_0_FAST quantization (separated scales/nibbles)
// dispatch: (ceil(n_blocks/256), 1, 1) x (256, 1, 1)
// buffer(0) = source F16, buffer(1) = destination Q4_0_FAST
// buffer(2) = n_blocks, buffer(3) = nb_per_row (K/32), buffer(4) = M
// ---------------------------------------------------------------------------
kernel void quantize_f16_to_q4_0_fast(
    device const half*   src        [[buffer(0)]],
    device uchar*        dst        [[buffer(1)]],
    constant uint&       n_blocks   [[buffer(2)]],
    constant uint&       nb_per_row [[buffer(3)]],
    constant uint&       M_val      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    uint row = gid / nb_per_row;
    uint bi  = gid % nb_per_row;

    device const half* block_src = src + gid * 32u;
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) {
        vals[i] = float(block_src[i]);
        amax = max(amax, abs(vals[i]));
    }

    float d = amax / 7.0f;
    float id = (d != 0.0f) ? (7.0f / amax) : 0.0f;

    // Write scale to scales region: half[M][nb_per_row]
    device half* scale_dst = (device half*)dst + row * nb_per_row + bi;
    *scale_dst = half(d);

    // Write nibbles to nibbles region (after all scales)
    size_t nibbles_base = (size_t)M_val * (size_t)nb_per_row * 2u;
    device uchar* nib_dst = dst + nibbles_base + (row * nb_per_row + bi) * 16u;
    for (uint i = 0; i < 16u; i++) {
        int q0 = (int)round(vals[i * 2]     * id) + 8;
        int q1 = (int)round(vals[i * 2 + 1] * id) + 8;
        q0 = clamp(q0, 0, 15);
        q1 = clamp(q1, 0, 15);
        nib_dst[i] = (uchar)(q0 | (q1 << 4));
    }
}

// ---------------------------------------------------------------------------
// F16 → Q8_0_FAST quantization (separated scales/weights)
// dispatch: (ceil(n_blocks/256), 1, 1) x (256, 1, 1)
// buffer(0) = source F16, buffer(1) = destination Q8_0_FAST
// buffer(2) = n_blocks, buffer(3) = nb_per_row (K/32), buffer(4) = M
// ---------------------------------------------------------------------------
kernel void quantize_f16_to_q8_0_fast(
    device const half*   src        [[buffer(0)]],
    device char*         dst        [[buffer(1)]],
    constant uint&       n_blocks   [[buffer(2)]],
    constant uint&       nb_per_row [[buffer(3)]],
    constant uint&       M_val      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    uint row = gid / nb_per_row;
    uint bi  = gid % nb_per_row;

    device const half* block_src = src + gid * 32u;
    float vals[32];
    float amax = 0.0f;
    for (uint i = 0; i < 32u; i++) {
        vals[i] = float(block_src[i]);
        amax = max(amax, abs(vals[i]));
    }

    float d = amax / 127.0f;
    float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

    // Write scale to scales region: half[M][nb_per_row]
    device half* scale_dst = (device half*)((device uchar*)dst) + row * nb_per_row + bi;
    *scale_dst = half(d);

    // Write weights to weights region (after all scales)
    size_t weights_base = (size_t)M_val * (size_t)nb_per_row * 2u;
    device char* w_dst = dst + weights_base + (row * nb_per_row + bi) * 32u;
    for (uint i = 0; i < 32u; i++) {
        int q = (int)round(vals[i] * id);
        q = clamp(q, -128, 127);
        w_dst[i] = (char)q;
    }
}

// ---------------------------------------------------------------------------
// Fast Quantized GEMV (Q4_0): y[M] = B_q4fast[M,K] x A[K]
// Reads from separated Q4_0_FAST layout for vectorized uint32 weight reads
// and half4 activation reads.
// dispatch: (ceil(M/32), 1, 1) x (256, 1, 1)  — 8 simdgroups × 4 rows = 32 rows/TG
// NOTE: Optimized for decode (act_stride=1). half4 reads assume contiguous A.
// ---------------------------------------------------------------------------

#define QMV_BR    4u        // rows per SIMD group
#define QMV_BM    16u       // rows per threadgroup = 4 SIMD × 4 rows

kernel void qmv_fast_q4_0(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const uchar*      B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    // Separated layout: scales [M, nb] then nibbles [M, nb, 16]
    device const half*  scales  = (device const half*)B;
    device const uchar* nibbles = B + M * nb * 2u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;

            // Load 32 activation values and pre-scale for shift-free dequant
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;

                float d = float(scales[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nibbles + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc[r] += d * (partial - 8.0f * act_sum);
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Quantized GEMV (Q8_0): y[M] = B_q8fast[M,K] x A[K]
// Reads from separated Q8_0_FAST layout for vectorized char4 weight reads.
// dispatch: (ceil(M/32), 1, 1) x (256, 1, 1)  — 8 simdgroups × 4 rows = 32 rows/TG
// ---------------------------------------------------------------------------
kernel void qmv_fast_q8_0(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const char*       B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    // Separated layout: scales [M, nb] then weights [M, nb, 32]
    device const half* scales  = (device const half*)((device const uchar*)B);
    device const char* weights = (device const char*)((device const uchar*)B + M * nb * 2u);

    for (uint n = 0u; n < seq_len; n++) {
        float acc[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;

            // Load 32 activation values as 8 × half4
            half4 a_v[8];
            for (uint i = 0u; i < 8u; i++) {
                a_v[i] = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;

                float d = float(scales[m * nb + bi]);
                device const char4* w4 = (device const char4*)(weights + (m * nb + bi) * 32u);
                float partial = 0.0f;

                for (uint g = 0u; g < 8u; g++) {
                    char4 w = w4[g];
                    half4 a = a_v[g];
                    partial += float(w[0]) * float(a[0]);
                    partial += float(w[1]) * float(a[1]);
                    partial += float(w[2]) * float(a[2]);
                    partial += float(w[3]) * float(a[3]);
                }
                acc[r] += d * partial;
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV (Q4_0_FAST): 3 GEMVs sharing the same activation input
// dispatch: (ceil(M_q/QMV_BM), 1, 1) x (256, 1, 1)
// buffer(0)=out_q, buffer(1)=A, buffer(2)=B_q, buffer(3)=out_k,
// buffer(4)=B_k, buffer(5)=out_v, buffer(6)=B_v
// params at buffer(7): [seq_len, act_stride, K, M_q, M_kv, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void fused_qkv_qmv_fast_q4_0(
    device half*             out_q  [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const uchar*      B_q    [[buffer(2)]],
    device half*             out_k  [[buffer(3)]],
    device const uchar*      B_k    [[buffer(4)]],
    device half*             out_v  [[buffer(5)]],
    device const uchar*      B_v    [[buffer(6)]],
    constant uint*           params [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M_q        = params[3];
    uint M_kv       = params[4];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq = (device const half*)B_q;
    device const uchar* nq = B_q + M_q * nb * 2u;
    device const half*  sk = (device const half*)B_k;
    device const uchar* nk = B_k + M_kv * nb * 2u;
    device const half*  sv = (device const half*)B_v;
    device const uchar* nv = B_v + M_kv * nb * 2u;

    bool do_kv = (m_base < M_kv);

    for (uint n = 0u; n < seq_len; n++) {
        float acc_q[QMV_BR] = {};
        float acc_k[QMV_BR] = {};
        float acc_v[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            // Q projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M_q) continue;
                float d = float(sq[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nq + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_q[r] += d * (partial - 8.0f * act_sum);
            }

            if (do_kv) {
                // K projection
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M_kv) continue;
                    float d = float(sk[m * nb + bi]);
                    device const ushort* ws = (device const ushort*)(nk + (m * nb + bi) * 16u);
                    float partial = 0.0f;
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial += xs[j][0] * float(w & 0x000fu)
                                 + xs[j][1] * float(w & 0x00f0u)
                                 + xs[j][2] * float(w & 0x0f00u)
                                 + xs[j][3] * float(w & 0xf000u);
                    }
                    acc_k[r] += d * (partial - 8.0f * act_sum);
                }
                // V projection
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M_kv) continue;
                    float d = float(sv[m * nb + bi]);
                    device const ushort* ws = (device const ushort*)(nv + (m * nb + bi) * 16u);
                    float partial = 0.0f;
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial += xs[j][0] * float(w & 0x000fu)
                                 + xs[j][1] * float(w & 0x00f0u)
                                 + xs[j][2] * float(w & 0x0f00u)
                                 + xs[j][3] * float(w & 0xf000u);
                    }
                    acc_v[r] += d * (partial - 8.0f * act_sum);
                }
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            float tq = simd_sum(acc_q[r]);
            if (simd_lid == 0u && m < M_q)
                out_q[m * act_stride + n] = half(tq);
            if (do_kv) {
                float tk = simd_sum(acc_k[r]);
                float tv = simd_sum(acc_v[r]);
                if (simd_lid == 0u && m < M_kv) {
                    out_k[m * act_stride + n] = half(tk);
                    out_v[m * act_stride + n] = half(tv);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up GEMV (Q4_0_FAST): 2 GEMVs sharing the same activation input
// dispatch: (ceil(M/QMV_BM), 1, 1) x (256, 1, 1)
// buffer(0)=out_gate, buffer(1)=A, buffer(2)=B_gate, buffer(3)=out_up,
// buffer(4)=B_up, params at buffer(5): [seq_len, act_stride, K, M, 0, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void fused_gate_up_qmv_fast_q4_0(
    device half*             out_gate [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B_gate   [[buffer(2)]],
    device half*             out_up   [[buffer(3)]],
    device const uchar*      B_up     [[buffer(4)]],
    constant uint*           params   [[buffer(5)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const uchar* n_gate = B_gate + M * nb * 2u;
    device const half*  s_up   = (device const half*)B_up;
    device const uchar* n_up   = B_up + M * nb * 2u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {};
        float acc_up[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            // Gate projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_gate[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_gate[r] += d * (partial - 8.0f * act_sum);
            }

            // Up projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_up[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_up[r] += d * (partial - 8.0f * act_sum);
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            float total_gate = simd_sum(acc_gate[r]);
            float total_up   = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                out_gate[m * act_stride + n] = half(total_gate);
                out_up[m * act_stride + n]   = half(total_up);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up GEMV + SwiGLU (Q4_0_FAST): 2 GEMVs + SiLU(gate)*up in one dispatch
// dispatch: (ceil(M/QMV_BM), 1, 1) x (128, 1, 1)
// buffer(0)=out (SwiGLU result), buffer(1)=A, buffer(2)=B_gate, buffer(3)=B_up
// params at buffer(4): [seq_len, act_stride, K, M, 0, 0, 0, 0]
// ---------------------------------------------------------------------------
kernel void fused_gate_up_swiglu_qmv_fast_q4_0(
    device half*             out      [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B_gate   [[buffer(2)]],
    device const uchar*      B_up     [[buffer(3)]],
    constant uint*           params   [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const uchar* n_gate = B_gate + M * nb * 2u;
    device const half*  s_up   = (device const half*)B_up;
    device const uchar* n_up   = B_up + M * nb * 2u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {};
        float acc_up[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            // Gate projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_gate[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_gate[r] += d * (partial - 8.0f * act_sum);
            }

            // Up projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_up[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_up[r] += d * (partial - 8.0f * act_sum);
            }
        }

        // Fused SwiGLU write-back: silu(gate) * up
        for (uint r = 0u; r < QMV_BR; r++) {
            float g = simd_sum(acc_gate[r]);
            float u = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float silu_g = g / (1.0f + fast::exp(-g));
                out[m * act_stride + n] = half(silu_g * u);
            }
        }
    }
}

// ===========================================================================
// Fused RoPE Q+K + KV Append + SDPA (decode only, seq_len=1)
// Combines 3 operations into 1 dispatch, keeping rotated Q in registers.
// dispatch: (num_kv_heads, 1, 1) x (256, 1, 1)
// Each TG handles one KV head group (gqa_factor Q heads).
// Phase 1: RoPE K + KV Append K,V (all TGs read v_in here)
// Phase 2: Atomic cross-TG barrier (ensure all v_in reads done before attn_out writes)
// Phase 3: RoPE Q (in registers) + SDPA for gqa_factor Q heads (1 per simdgroup)
// buffer(0)=attn_out, buffer(1)=q_in, buffer(2)=k_cache, buffer(3)=k_in,
// buffer(4)=v_cache, buffer(5)=v_in, buffer(6)=sync_buf,
// buffer(7)=params [total_seq, head_dim, num_kv_heads, gqa_factor,
//                   rope_base_bits, max_seq, act_stride, layer_idx]
// ===========================================================================
kernel void fused_rope_kv_sdpa_f16(
    device half*       attn_out [[buffer(0)]],  // dedicated buffer, NOT aliased with v_in
    device const half* q_in     [[buffer(1)]],
    device half*       k_cache  [[buffer(2)]],
    device const half* k_in     [[buffer(3)]],
    device half*       v_cache  [[buffer(4)]],
    device const half* v_in     [[buffer(5)]],
    constant uint*     params   [[buffer(6)]],
    device const half* rope_table [[buffer(7)]],
    uint tgid     [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    uint total_seq   = params[0];
    uint head_dim    = params[1];
    uint num_kv_h    = params[2];
    uint gqa_factor  = params[3];
    uint max_seq     = params[5];
    uint act_stride  = params[6];

    uint kv_h = tgid;
    uint abs_pos = total_seq - 1u;
    uint half_dim = head_dim / 2u;
    uint tid = simd_gid * 32u + simd_lid;  // flat thread id 0..255

    // ---- Phase 1: RoPE K + write K to cache ----
    for (uint pi = tid; pi < half_dim; pi += 256u) {
        uint tidx = abs_pos * half_dim + pi;
        float cos_t = float(rope_table[tidx * 2u]);
        float sin_t = float(rope_table[tidx * 2u + 1u]);
        uint hi0 = kv_h * head_dim + pi;
        uint hi1 = kv_h * head_dim + pi + half_dim;
        float x0 = float(k_in[hi0 * act_stride]);
        float x1 = float(k_in[hi1 * act_stride]);
        uint cache_base = kv_h * max_seq * head_dim + abs_pos * head_dim;
        k_cache[cache_base + pi] = half(x0 * cos_t - x1 * sin_t);
        k_cache[cache_base + pi + half_dim] = half(x0 * sin_t + x1 * cos_t);
    }

    // ---- Copy V to cache ----
    for (uint di = tid; di < head_dim; di += 256u) {
        half v_val = v_in[(kv_h * head_dim + di) * act_stride];
        v_cache[kv_h * max_seq * head_dim + abs_pos * head_dim + di] = v_val;
    }

    // Ensure own K/V cache writes are visible before reading in SDPA
    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 1.5: Write rotated Q to attn_out (temporary storage) ----
    for (uint qh_local = simd_gid; qh_local < gqa_factor; qh_local += 8u) {
        uint q_h = kv_h * gqa_factor + qh_local;
        uint my_dims = (head_dim + 31u) / 32u;
        for (uint di = 0u; di < my_dims; di++) {
            uint d = simd_lid + di * 32u;
            if (d < head_dim) {
                uint pair_pos = d % half_dim;
                uint base_idx = q_h * head_dim;
                float x0 = float(q_in[(base_idx + pair_pos) * act_stride]);
                float x1 = float(q_in[(base_idx + pair_pos + half_dim) * act_stride]);
                uint tidx = abs_pos * half_dim + pair_pos;
                float cos_t = float(rope_table[tidx * 2u]);
                float sin_t = float(rope_table[tidx * 2u + 1u]);
                if (d < half_dim) {
                    attn_out[(q_h * head_dim + d) * act_stride] = half(x0 * cos_t - x1 * sin_t);
                } else {
                    attn_out[(q_h * head_dim + d) * act_stride] = half(x0 * sin_t + x1 * cos_t);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ---- Phase 2: SDPA reading Q from attn_out buffer — single simdgroup only ----
    if (simd_gid == 0u)
    for (uint qh_local = 0u; qh_local < gqa_factor; qh_local++) {
        uint q_h = kv_h * gqa_factor + qh_local;
        uint my_dims = (head_dim + 31u) / 32u;

        float scale = rsqrt((float)head_dim);
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float acc[4] = {};

        uint kv_base = kv_h * max_seq * head_dim;

        for (uint pos = 0u; pos <= abs_pos; pos++) {
            float partial = 0.0f;
            for (uint di = 0u; di < my_dims; di++) {
                uint d = simd_lid + di * 32u;
                if (d < head_dim) {
                    // Read Q from buffer (not registers)
                    float q_d = float(attn_out[(q_h * head_dim + d) * act_stride]);
                    float k_d = float(k_cache[kv_base + pos * head_dim + d]);
                    partial += q_d * k_d;
                }
            }
            float score = simd_sum(partial) * scale;

            float new_max = max(max_score, score);
            float decay = fast::exp(max_score - new_max);
            float w = fast::exp(score - new_max);
            sum_exp = sum_exp * decay + w;

            for (uint di = 0u; di < my_dims; di++) {
                uint d = simd_lid + di * 32u;
                if (d < head_dim) {
                    float v_d = float(v_cache[kv_base + pos * head_dim + d]);
                    acc[di] = acc[di] * decay + w * v_d;
                }
            }
            max_score = new_max;
        }

        float inv_sum = 1.0f / sum_exp;
        for (uint di = 0u; di < my_dims; di++) {
            uint d = simd_lid + di * 32u;
            if (d < head_dim) {
                attn_out[(q_h * head_dim + d) * act_stride] = half(acc[di] * inv_sum);
            }
        }
    }
}

// ===========================================================================
// PSQ Pipeline: partial sum-of-squares for cross-dispatch RMSNorm elimination
// ===========================================================================

// ---------------------------------------------------------------------------
// GEMV + Residual Add + Partial Sum-of-Squares (Q4_0_FAST)
// Used for O-proj and Down-proj to fuse the residual add and prepare partial
// sums for the next GEMV's inline normalization.
// dispatch: (ceil(M/QMV_BM), 1, 1) x (128, 1, 1)
// buffer(0)=out, buffer(1)=act, buffer(2)=weight, buffer(3)=residual,
// buffer(4)=psq_buf (float[num_tgs]), buffer(5)=params
// params: [seq_len, act_stride, K, M, 0, 0, 0, 0]
// threadgroup(0) = 16 bytes (float[4] for simdgroup reduction)
// ---------------------------------------------------------------------------
kernel void qmv_add_psq_q4_0(
    device half*             C        [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B        [[buffer(2)]],
    device const half*       residual [[buffer(3)]],
    device float*            psq_buf  [[buffer(4)]],
    constant uint*           params   [[buffer(5)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    threadgroup float* tg_sq [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  scales  = (device const half*)B;
    device const uchar* nibbles = B + M * nb * 2u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(scales[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nibbles + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc[r] += d * (partial - 8.0f * act_sum);
            }
        }

        // Write GEMV result + residual, accumulate sum-of-squares
        float my_sq = 0.0f;
        for (uint r = 0u; r < QMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float val = total + float(residual[m * act_stride + n]);
                C[m * act_stride + n] = half(val);
                my_sq += val * val;
            }
        }

        // Reduce sum-of-squares across simdgroups within TG
        float sg_sq = simd_sum(my_sq);  // only lid==0 contributed, this broadcasts
        if (simd_lid == 0u) tg_sq[simd_gid] = sg_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u && simd_gid == 0u) {
            psq_buf[tgid_x] = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV with inline RMSNorm from partial sums (Q4_0_FAST)
// Reads psq_buf from preceding GEMV+Add+PSQ dispatch, computes norm_scale,
// normalizes activation on-the-fly. Eliminates Add+RMSNorm dispatch.
// dispatch: (ceil(M_q/QMV_BM), 1, 1) x (128, 1, 1)
// buffer(0)=out_q, buffer(1)=A (un-normalized residual sum),
// buffer(2)=B_q, buffer(3)=out_k, buffer(4)=B_k, buffer(5)=out_v,
// buffer(6)=B_v, buffer(7)=psq_buf, buffer(8)=norm_weight,
// buffer(9)=params [seq_len, act_stride, K, M_q, M_kv, eps_bits, num_psq, 0]
// ---------------------------------------------------------------------------
kernel void fused_qkv_norm_qmv_fast_q4_0(
    device half*             out_q       [[buffer(0)]],
    device const half*       A           [[buffer(1)]],
    device const uchar*      B_q         [[buffer(2)]],
    device half*             out_k       [[buffer(3)]],
    device const uchar*      B_k         [[buffer(4)]],
    device half*             out_v       [[buffer(5)]],
    device const uchar*      B_v         [[buffer(6)]],
    device const float*      psq_buf     [[buffer(7)]],
    device const half*       norm_weight [[buffer(8)]],
    constant uint*           params      [[buffer(9)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M_q        = params[3];
    uint M_kv       = params[4];
    float eps       = as_type<float>(params[5]);
    uint num_psq    = params[6];

    // Compute RMSNorm scale from partial sums
    float sum_sq = 0.0f;
    for (uint i = simd_lid; i < num_psq; i += 32u) {
        sum_sq += psq_buf[i];
    }
    sum_sq = simd_sum(sum_sq);
    float norm_scale = rsqrt(sum_sq / float(K) + eps);

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq = (device const half*)B_q;
    device const uchar* nq = B_q + M_q * nb * 2u;
    device const half*  sk = (device const half*)B_k;
    device const uchar* nk = B_k + M_kv * nb * 2u;
    device const half*  sv = (device const half*)B_v;
    device const uchar* nv = B_v + M_kv * nb * 2u;

    bool do_kv = (m_base < M_kv);

    for (uint n = 0u; n < seq_len; n++) {
        float acc_q[QMV_BR] = {};
        float acc_k[QMV_BR] = {};
        float acc_v[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                half4 nw = *(device const half4*)(norm_weight + base_k + i * 4u);
                // Normalize on-the-fly: activation * norm_weight * norm_scale
                float x0 = float(av[0]) * float(nw[0]) * norm_scale;
                float x1 = float(av[1]) * float(nw[1]) * norm_scale;
                float x2 = float(av[2]) * float(nw[2]) * norm_scale;
                float x3 = float(av[3]) * float(nw[3]) * norm_scale;
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M_q) continue;
                float d = float(sq[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nq + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_q[r] += d * (partial - 8.0f * act_sum);
            }

            if (do_kv) {
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M_kv) continue;
                    float d = float(sk[m * nb + bi]);
                    device const ushort* ws = (device const ushort*)(nk + (m * nb + bi) * 16u);
                    float partial = 0.0f;
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial += xs[j][0] * float(w & 0x000fu)
                                 + xs[j][1] * float(w & 0x00f0u)
                                 + xs[j][2] * float(w & 0x0f00u)
                                 + xs[j][3] * float(w & 0xf000u);
                    }
                    acc_k[r] += d * (partial - 8.0f * act_sum);
                }
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M_kv) continue;
                    float d = float(sv[m * nb + bi]);
                    device const ushort* ws = (device const ushort*)(nv + (m * nb + bi) * 16u);
                    float partial = 0.0f;
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial += xs[j][0] * float(w & 0x000fu)
                                 + xs[j][1] * float(w & 0x00f0u)
                                 + xs[j][2] * float(w & 0x0f00u)
                                 + xs[j][3] * float(w & 0xf000u);
                    }
                    acc_v[r] += d * (partial - 8.0f * act_sum);
                }
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            float tq = simd_sum(acc_q[r]);
            if (simd_lid == 0u && m < M_q)
                out_q[m * act_stride + n] = half(tq);
            if (do_kv) {
                float tk = simd_sum(acc_k[r]);
                float tv = simd_sum(acc_v[r]);
                if (simd_lid == 0u && m < M_kv) {
                    out_k[m * act_stride + n] = half(tk);
                    out_v[m * act_stride + n] = half(tv);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up+SwiGLU GEMV with inline RMSNorm from partial sums (Q4_0_FAST)
// Same PSQ pipeline as above but for the FFN block.
// dispatch: (ceil(M/QMV_BM), 1, 1) x (128, 1, 1)
// buffer(0)=out, buffer(1)=A (un-normalized residual sum),
// buffer(2)=B_gate, buffer(3)=B_up, buffer(4)=psq_buf,
// buffer(5)=norm_weight, buffer(6)=params
// params: [seq_len, act_stride, K, M, eps_bits, num_psq, 0, 0]
// ---------------------------------------------------------------------------
kernel void fused_gate_up_swiglu_norm_qmv_fast_q4_0(
    device half*             out         [[buffer(0)]],
    device const half*       A           [[buffer(1)]],
    device const uchar*      B_gate      [[buffer(2)]],
    device const uchar*      B_up        [[buffer(3)]],
    device const float*      psq_buf     [[buffer(4)]],
    device const half*       norm_weight [[buffer(5)]],
    constant uint*           params      [[buffer(6)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];
    float eps       = as_type<float>(params[4]);
    uint num_psq    = params[5];

    // Compute RMSNorm scale from partial sums
    float sum_sq = 0.0f;
    for (uint i = simd_lid; i < num_psq; i += 32u) {
        sum_sq += psq_buf[i];
    }
    sum_sq = simd_sum(sum_sq);
    float norm_scale = rsqrt(sum_sq / float(K) + eps);

    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const uchar* n_gate = B_gate + M * nb * 2u;
    device const half*  s_up   = (device const half*)B_up;
    device const uchar* n_up   = B_up + M * nb * 2u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {};
        float acc_up[QMV_BR] = {};

        for (uint bi = simd_lid; bi < nb; bi += 32u) {
            uint base_k = bi * 32u;
            float4 xs[8];
            float act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                half4 av = *(device const half4*)(A + (base_k + i * 4u) * act_stride + n);
                half4 nw = *(device const half4*)(norm_weight + base_k + i * 4u);
                float x0 = float(av[0]) * float(nw[0]) * norm_scale;
                float x1 = float(av[1]) * float(nw[1]) * norm_scale;
                float x2 = float(av[2]) * float(nw[2]) * norm_scale;
                float x3 = float(av[3]) * float(nw[3]) * norm_scale;
                act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_gate[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_gate[r] += d * (partial - 8.0f * act_sum);
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(s_up[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_up[r] += d * (partial - 8.0f * act_sum);
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            float g = simd_sum(acc_gate[r]);
            float u = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float silu_g = g / (1.0f + fast::exp(-g));
                out[m * act_stride + n] = half(silu_g * u);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Add+RMSNorm + QKV GEMV (Q4_0_FAST): Decode only (seq_len=1)
// Each TG redundantly computes Add+RMSNorm in shared memory, then performs
// QKV GEMV reading the normalized activation from shared memory.
// Eliminates the dispatch boundary between Add+RMSNorm and QKV GEMV.
// dispatch: (ceil(M_q/QMV_BM), 1, 1) x (128, 1, 1)
// shared memory: 2048 floats = 8192 bytes (normalized activation)
//
// buffer(0)=out_q, buffer(1)=x (residual), buffer(2)=delta (o_proj or down),
// buffer(3)=norm_weight, buffer(4)=B_q (quant weights), buffer(5)=out_k,
// buffer(6)=B_k, buffer(7)=out_v, buffer(8)=B_v, buffer(9)=out_residual,
// buffer(10)=params [K, M_q, M_kv, eps_bits, has_add]
// ---------------------------------------------------------------------------
kernel void fused_add_rmsnorm_qkv_qmv_fast_q4_0(
    device half*             out_q         [[buffer(0)]],
    device const half*       x             [[buffer(1)]],
    device const half*       delta         [[buffer(2)]],
    device const half*       norm_weight   [[buffer(3)]],
    device const uchar*      B_q           [[buffer(4)]],
    device half*             out_k         [[buffer(5)]],
    device const uchar*      B_k           [[buffer(6)]],
    device half*             out_v         [[buffer(7)]],
    device const uchar*      B_v           [[buffer(8)]],
    device half*             out_residual  [[buffer(9)]],
    constant uint*           params        [[buffer(10)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  lid      [[thread_position_in_threadgroup]],
    threadgroup float*       reduce_f      [[threadgroup(0)]])
{
    uint K       = params[0];
    uint M_q     = params[1];
    uint M_kv    = params[2];
    float eps    = as_type<float>(params[3]);
    uint has_add = params[4];
    uint H       = K;

    // ---- Phase 1: Compute RMSNorm scale + write residual (TG 0 only) ----
    float partial_sq = 0.0f;
    for (uint hi = lid; hi < H; hi += 128u) {
        float v;
        if (has_add != 0u) {
            v = float(x[hi]) + float(delta[hi]);
            // Only TG 0 writes residual (other TGs skip to avoid write amplification)
            if (tgid_x == 0u) out_residual[hi] = half(v);
        } else {
            v = float(x[hi]);
        }
        partial_sq += v * v;
    }

    float sg_sum = simd_sum(partial_sq);
    if (simd_lid == 0u) {
        reduce_f[simd_gid] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = reduce_f[0] + reduce_f[1] + reduce_f[2] + reduce_f[3];
    float scale = rsqrt(total_sq / float(H) + eps);

    // ---- Phase 2: QKV GEMV with on-the-fly x+delta+norm (reads from device, L2 cached) ----
    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq = (device const half*)B_q;
    device const uchar* nq = B_q + M_q * nb * 2u;
    device const half*  sk = (device const half*)B_k;
    device const uchar* nk = B_k + M_kv * nb * 2u;
    device const half*  sv = (device const half*)B_v;
    device const uchar* nv = B_v + M_kv * nb * 2u;

    bool do_kv = (m_base < M_kv);

    float acc_q[QMV_BR] = {};
    float acc_k[QMV_BR] = {};
    float acc_v[QMV_BR] = {};

    for (uint bi = simd_lid; bi < nb; bi += 32u) {
        uint base_k = bi * 32u;
        float4 xs[8];
        float act_sum = 0.0f;
        // Read x+delta from device (L2 cached), normalize on-the-fly
        for (uint i = 0u; i < 8u; i++) {
            uint idx = base_k + i * 4u;
            float r0, r1, r2, r3;
            if (has_add != 0u) {
                r0 = float(x[idx+0u]) + float(delta[idx+0u]);
                r1 = float(x[idx+1u]) + float(delta[idx+1u]);
                r2 = float(x[idx+2u]) + float(delta[idx+2u]);
                r3 = float(x[idx+3u]) + float(delta[idx+3u]);
            } else {
                r0 = float(x[idx+0u]); r1 = float(x[idx+1u]);
                r2 = float(x[idx+2u]); r3 = float(x[idx+3u]);
            }
            float x0 = r0 * scale * float(norm_weight[idx + 0u]);
            float x1 = r1 * scale * float(norm_weight[idx + 1u]);
            float x2 = r2 * scale * float(norm_weight[idx + 2u]);
            float x3 = r3 * scale * float(norm_weight[idx + 3u]);
            act_sum += x0 + x1 + x2 + x3;
            xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
        }

        // Q projection
        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            if (m >= M_q) continue;
            float d = float(sq[m * nb + bi]);
            device const ushort* ws = (device const ushort*)(nq + (m * nb + bi) * 16u);
            float partial = 0.0f;
            for (uint j = 0u; j < 8u; j++) {
                ushort w = ws[j];
                partial += xs[j][0] * float(w & 0x000fu)
                         + xs[j][1] * float(w & 0x00f0u)
                         + xs[j][2] * float(w & 0x0f00u)
                         + xs[j][3] * float(w & 0xf000u);
            }
            acc_q[r] += d * (partial - 8.0f * act_sum);
        }

        if (do_kv) {
            // K projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M_kv) continue;
                float d = float(sk[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nk + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_k[r] += d * (partial - 8.0f * act_sum);
            }
            // V projection
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M_kv) continue;
                float d = float(sv[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(nv + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_v[r] += d * (partial - 8.0f * act_sum);
            }
        }
    }

    for (uint r = 0u; r < QMV_BR; r++) {
        uint m = m_base + r;
        float tq = simd_sum(acc_q[r]);
        if (simd_lid == 0u && m < M_q)
            out_q[m] = half(tq);
        if (do_kv) {
            float tk = simd_sum(acc_k[r]);
            float tv = simd_sum(acc_v[r]);
            if (simd_lid == 0u && m < M_kv) {
                out_k[m] = half(tk);
                out_v[m] = half(tv);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Add+RMSNorm + Gate+Up+SwiGLU GEMV (Q4_0_FAST): Decode only (seq_len=1)
// Same shared-memory trick as above but for the FFN block.
// dispatch: (ceil(M/QMV_BM), 1, 1) x (128, 1, 1)
// shared memory: 2048 floats = 8192 bytes (normalized activation)
//
// buffer(0)=out (SwiGLU result), buffer(1)=x (residual), buffer(2)=delta (o_proj),
// buffer(3)=norm_weight, buffer(4)=B_gate, buffer(5)=B_up,
// buffer(6)=out_residual, buffer(7)=params [K, M, eps_bits]
// ---------------------------------------------------------------------------
kernel void fused_add_rmsnorm_gate_up_swiglu_qmv_fast_q4_0(
    device half*             out           [[buffer(0)]],
    device const half*       x             [[buffer(1)]],
    device const half*       delta         [[buffer(2)]],
    device const half*       norm_weight   [[buffer(3)]],
    device const uchar*      B_gate        [[buffer(4)]],
    device const uchar*      B_up          [[buffer(5)]],
    device half*             out_residual  [[buffer(6)]],
    constant uint*           params        [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  lid      [[thread_position_in_threadgroup]])
{
    uint K       = params[0];
    uint M       = params[1];
    float eps    = as_type<float>(params[2]);
    uint H       = K;

    // Only 16 bytes of threadgroup memory for cross-simdgroup reduction
    threadgroup float reduce_f[4];

    // ---- Phase 1: Compute RMSNorm scale + write residual (TG 0 only) ----
    float partial_sq = 0.0f;
    for (uint hi = lid; hi < H; hi += 128u) {
        float v = float(x[hi]) + float(delta[hi]);
        if (tgid_x == 0u) out_residual[hi] = half(v);
        partial_sq += v * v;
    }

    float sg_sum = simd_sum(partial_sq);
    if (simd_lid == 0u) {
        reduce_f[simd_gid] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = reduce_f[0] + reduce_f[1] + reduce_f[2] + reduce_f[3];
    float scale = rsqrt(total_sq / float(H) + eps);

    // ---- Phase 2: Gate+Up+SwiGLU GEMV with on-the-fly x+delta+norm ----
    uint nb = K / 32u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const uchar* n_gate = B_gate + M * nb * 2u;
    device const half*  s_up   = (device const half*)B_up;
    device const uchar* n_up   = B_up + M * nb * 2u;

    float acc_gate[QMV_BR] = {};
    float acc_up[QMV_BR] = {};

    for (uint bi = simd_lid; bi < nb; bi += 32u) {
        uint base_k = bi * 32u;
        float4 xs[8];
        float act_sum = 0.0f;
        for (uint i = 0u; i < 8u; i++) {
            uint idx = base_k + i * 4u;
            float r0 = float(x[idx+0u]) + float(delta[idx+0u]);
            float r1 = float(x[idx+1u]) + float(delta[idx+1u]);
            float r2 = float(x[idx+2u]) + float(delta[idx+2u]);
            float r3 = float(x[idx+3u]) + float(delta[idx+3u]);
            float x0 = r0 * scale * float(norm_weight[idx + 0u]);
            float x1 = r1 * scale * float(norm_weight[idx + 1u]);
            float x2 = r2 * scale * float(norm_weight[idx + 2u]);
            float x3 = r3 * scale * float(norm_weight[idx + 3u]);
            act_sum += x0 + x1 + x2 + x3;
            xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            if (m >= M) continue;
            {
                float d = float(s_gate[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_gate[r] += d * (partial - 8.0f * act_sum);
            }
            {
                float d = float(s_up[m * nb + bi]);
                device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi) * 16u);
                float partial = 0.0f;
                for (uint j = 0u; j < 8u; j++) {
                    ushort w = ws[j];
                    partial += xs[j][0] * float(w & 0x000fu)
                             + xs[j][1] * float(w & 0x00f0u)
                             + xs[j][2] * float(w & 0x0f00u)
                             + xs[j][3] * float(w & 0xf000u);
                }
                acc_up[r] += d * (partial - 8.0f * act_sum);
            }
        }
    }

    for (uint r = 0u; r < QMV_BR; r++) {
        float g = simd_sum(acc_gate[r]);
        float u = simd_sum(acc_up[r]);
        uint m = m_base + r;
        if (simd_lid == 0u && m < M) {
            float silu_g = g / (1.0f + fast::exp(-g));
            out[m] = half(silu_g * u);
        }
    }
}

// ===========================================================================
// Q4_MLX kernels: Affine 4-bit quantization, group_size=128, scale+bias
// Layout: [scales half[M][nb]][biases half[M][nb]][nibbles uchar[M][nb][64]]
// Dequant: w_orig ≈ scale * nibble + bias (asymmetric, better accuracy)
// ===========================================================================

// ---------------------------------------------------------------------------
// F16 → Q4_MLX_FAST quantization (separated scales/biases/weights, group_size=128)
// dispatch: (ceil(n_blocks/256), 1, 1) x (256, 1, 1)
// buffer(0) = source F16, buffer(1) = destination Q4_MLX_FAST
// buffer(2) = n_blocks, buffer(3) = nb_per_row (K/128), buffer(4) = M
// ---------------------------------------------------------------------------
kernel void quantize_f16_to_q4_mlx_fast(
    device const half*   src        [[buffer(0)]],
    device uchar*        dst        [[buffer(1)]],
    constant uint&       n_blocks   [[buffer(2)]],
    constant uint&       nb_per_row [[buffer(3)]],
    constant uint&       M_val      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    uint row = gid / nb_per_row;
    uint bi  = gid % nb_per_row;

    device const half* block_src = src + gid * 128u;
    float vals[128];
    float vmin = INFINITY, vmax = -INFINITY;
    for (uint i = 0; i < 128u; i++) {
        vals[i] = float(block_src[i]);
        vmin = min(vmin, vals[i]);
        vmax = max(vmax, vals[i]);
    }

    float range = vmax - vmin;
    float d = range / 15.0f;
    float id = (d > 0.0f) ? (15.0f / range) : 0.0f;
    float bias = vmin;

    // Write scale: half[M][nb]
    device half* scale_dst = (device half*)dst + row * nb_per_row + bi;
    *scale_dst = half(d);

    // Write bias: half[M][nb], after all scales
    size_t bias_offset = (size_t)M_val * (size_t)nb_per_row * 2u;
    device half* bias_dst = (device half*)(dst + bias_offset) + row * nb_per_row + bi;
    *bias_dst = half(bias);

    // Write nibbles: uchar[M][nb][64], after scales+biases
    size_t nibbles_base = (size_t)M_val * (size_t)nb_per_row * 4u;
    device uchar* nib_dst = dst + nibbles_base + (size_t)(row * nb_per_row + bi) * 64u;
    for (uint i = 0; i < 64u; i++) {
        int q0 = (int)round((vals[i * 2]     - bias) * id);
        int q1 = (int)round((vals[i * 2 + 1] - bias) * id);
        q0 = clamp(q0, 0, 15);
        q1 = clamp(q1, 0, 15);
        nib_dst[i] = (uchar)(q0 | (q1 << 4));
    }
}

// ---------------------------------------------------------------------------
// Fast Quantized GEMV (Q4_MLX): y[M] = B_q4mlx[M,K] x A[K]
// dispatch: (ceil(M/QMV_BM), 1, 1) x (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void qmv_fast_q4_mlx(
    device half*             C      [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const uchar*      B      [[buffer(2)]],
    constant uint*           params [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  scales  = (device const half*)B;
    device const half*  biases  = (device const half*)(B + M * nb * 2u);
    device const uchar* nibbles = B + M * nb * 4u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float partial_r[QMV_BR] = {};
            float act_sum = 0.0f;

            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;

                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M) continue;
                    device const ushort* ws = (device const ushort*)(nibbles + (m * nb + bi_) * 64u + sub * 16u);
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial_r[r] += xs[j][0] * float(w & 0x000fu)
                                      + xs[j][1] * float(w & 0x00f0u)
                                      + xs[j][2] * float(w & 0x0f00u)
                                      + xs[j][3] * float(w & 0xf000u);
                    }
                }
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                float d = float(scales[m * nb + bi_]);
                float b = float(biases[m * nb + bi_]);
                acc[r] += d * partial_r[r] + b * act_sum;
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                C[m * act_stride + n] = half(total);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void fused_qkv_qmv_fast_q4_mlx(
    device half*             out_q  [[buffer(0)]],
    device const half*       A      [[buffer(1)]],
    device const uchar*      B_q    [[buffer(2)]],
    device half*             out_k  [[buffer(3)]],
    device const uchar*      B_k    [[buffer(4)]],
    device half*             out_v  [[buffer(5)]],
    device const uchar*      B_v    [[buffer(6)]],
    constant uint*           params [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M_q        = params[3];
    uint M_kv       = params[4];

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq = (device const half*)B_q;
    device const half*  bq = (device const half*)(B_q + M_q * nb * 2u);
    device const uchar* nq = B_q + M_q * nb * 4u;
    device const half*  sk = (device const half*)B_k;
    device const half*  bk = (device const half*)(B_k + M_kv * nb * 2u);
    device const uchar* nk = B_k + M_kv * nb * 4u;
    device const half*  sv = (device const half*)B_v;
    device const half*  bv = (device const half*)(B_v + M_kv * nb * 2u);
    device const uchar* nv = B_v + M_kv * nb * 4u;

    bool do_kv = (m_base < M_kv);

    for (uint n = 0u; n < seq_len; n++) {
        float acc_q[QMV_BR] = {};
        float acc_k[QMV_BR] = {};
        float acc_v[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float pq[QMV_BR] = {}, pk[QMV_BR] = {}, pv[QMV_BR] = {};
            float act_sum = 0.0f;

            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;

                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M_q) {
                        device const ushort* ws = (device const ushort*)(nq + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            pq[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                   + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                        }
                    }
                    if (do_kv && m < M_kv) {
                        {
                            device const ushort* ws = (device const ushort*)(nk + (m * nb + bi_) * 64u + sub * 16u);
                            for (uint j = 0u; j < 8u; j++) {
                                ushort w = ws[j];
                                pk[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                       + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                            }
                        }
                        {
                            device const ushort* ws = (device const ushort*)(nv + (m * nb + bi_) * 64u + sub * 16u);
                            for (uint j = 0u; j < 8u; j++) {
                                ushort w = ws[j];
                                pv[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                       + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                            }
                        }
                    }
                }
            }

            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m < M_q) { acc_q[r] += float(sq[m * nb + bi_]) * pq[r] + float(bq[m * nb + bi_]) * act_sum; }
                if (do_kv && m < M_kv) {
                    acc_k[r] += float(sk[m * nb + bi_]) * pk[r] + float(bk[m * nb + bi_]) * act_sum;
                    acc_v[r] += float(sv[m * nb + bi_]) * pv[r] + float(bv[m * nb + bi_]) * act_sum;
                }
            }
        }

        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            float tq = simd_sum(acc_q[r]);
            if (simd_lid == 0u && m < M_q) out_q[m * act_stride + n] = half(tq);
            if (do_kv) {
                float tk = simd_sum(acc_k[r]);
                float tv = simd_sum(acc_v[r]);
                if (simd_lid == 0u && m < M_kv) {
                    out_k[m * act_stride + n] = half(tk);
                    out_v[m * act_stride + n] = half(tv);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up GEMV (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void fused_gate_up_qmv_fast_q4_mlx(
    device half*             out_gate [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B_gate   [[buffer(2)]],
    device half*             out_up   [[buffer(3)]],
    device const uchar*      B_up     [[buffer(4)]],
    constant uint*           params   [[buffer(5)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const half*  b_gate = (device const half*)(B_gate + M * nb * 2u);
    device const uchar* n_gate = B_gate + M * nb * 4u;
    device const half*  s_up   = (device const half*)B_up;
    device const half*  b_up   = (device const half*)(B_up + M * nb * 2u);
    device const uchar* n_up   = B_up + M * nb * 4u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {};
        float acc_up[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float pg[QMV_BR] = {}, pu[QMV_BR] = {};
            float act_sum = 0.0f;

            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M) continue;
                    {
                        device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            pg[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                   + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                        }
                    }
                    {
                        device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            pu[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                   + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                        }
                    }
                }
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                acc_gate[r] += float(s_gate[m * nb + bi_]) * pg[r] + float(b_gate[m * nb + bi_]) * act_sum;
                acc_up[r]   += float(s_up[m * nb + bi_])   * pu[r] + float(b_up[m * nb + bi_])   * act_sum;
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            float total_gate = simd_sum(acc_gate[r]);
            float total_up   = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                out_gate[m * act_stride + n] = half(total_gate);
                out_up[m * act_stride + n]   = half(total_up);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up GEMV + SwiGLU (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void fused_gate_up_swiglu_qmv_fast_q4_mlx(
    device half*             out      [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B_gate   [[buffer(2)]],
    device const uchar*      B_up     [[buffer(3)]],
    constant uint*           params   [[buffer(4)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const half*  b_gate = (device const half*)(B_gate + M * nb * 2u);
    device const uchar* n_gate = B_gate + M * nb * 4u;
    device const half*  s_up   = (device const half*)B_up;
    device const half*  b_up   = (device const half*)(B_up + M * nb * 2u);
    device const uchar* n_up   = B_up + M * nb * 4u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {};
        float acc_up[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float pg[QMV_BR] = {}, pu[QMV_BR] = {};
            float act_sum = 0.0f;
            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M) continue;
                    {
                        device const ushort* ws = (device const ushort*)(n_gate + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            pg[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                   + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                        }
                    }
                    {
                        device const ushort* ws = (device const ushort*)(n_up + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            pu[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                   + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                        }
                    }
                }
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                acc_gate[r] += float(s_gate[m * nb + bi_]) * pg[r] + float(b_gate[m * nb + bi_]) * act_sum;
                acc_up[r]   += float(s_up[m * nb + bi_])   * pu[r] + float(b_up[m * nb + bi_])   * act_sum;
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            float g = simd_sum(acc_gate[r]);
            float u = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float silu_g = g / (1.0f + fast::exp(-g));
                out[m * act_stride + n] = half(silu_g * u);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEMV + Residual Add + Partial Sum-of-Squares (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void qmv_add_psq_q4_mlx(
    device half*             C        [[buffer(0)]],
    device const half*       A        [[buffer(1)]],
    device const uchar*      B        [[buffer(2)]],
    device const half*       residual [[buffer(3)]],
    device float*            psq_buf  [[buffer(4)]],
    constant uint*           params   [[buffer(5)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    threadgroup float* tg_sq [[threadgroup(0)]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  scales  = (device const half*)B;
    device const half*  biases  = (device const half*)(B + M * nb * 2u);
    device const uchar* nibbles = B + M * nb * 4u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float partial_r[QMV_BR] = {};
            float act_sum = 0.0f;
            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    float x0 = float(av[0]), x1 = float(av[1]), x2 = float(av[2]), x3 = float(av[3]);
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M) continue;
                    device const ushort* ws = (device const ushort*)(nibbles + (m * nb + bi_) * 64u + sub * 16u);
                    for (uint j = 0u; j < 8u; j++) {
                        ushort w = ws[j];
                        partial_r[r] += xs[j][0] * float(w & 0x000fu) + xs[j][1] * float(w & 0x00f0u)
                                      + xs[j][2] * float(w & 0x0f00u) + xs[j][3] * float(w & 0xf000u);
                    }
                }
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                acc[r] += float(scales[m * nb + bi_]) * partial_r[r] + float(biases[m * nb + bi_]) * act_sum;
            }
        }

        float my_sq = 0.0f;
        for (uint r = 0u; r < QMV_BR; r++) {
            float total = simd_sum(acc[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float val = total + float(residual[m * act_stride + n]);
                C[m * act_stride + n] = half(val);
                my_sq += val * val;
            }
        }
        float sg_sq = simd_sum(my_sq);
        if (simd_lid == 0u) tg_sq[simd_gid] = sg_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lid == 0u && simd_gid == 0u) {
            psq_buf[tgid_x] = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused QKV GEMV with inline RMSNorm from partial sums (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void fused_qkv_norm_qmv_fast_q4_mlx(
    device half*             out_q       [[buffer(0)]],
    device const half*       A           [[buffer(1)]],
    device const uchar*      B_q         [[buffer(2)]],
    device half*             out_k       [[buffer(3)]],
    device const uchar*      B_k         [[buffer(4)]],
    device half*             out_v       [[buffer(5)]],
    device const uchar*      B_v         [[buffer(6)]],
    device const float*      psq_buf     [[buffer(7)]],
    device const half*       norm_weight [[buffer(8)]],
    constant uint*           params      [[buffer(9)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M_q        = params[3];
    uint M_kv       = params[4];
    float eps       = as_type<float>(params[5]);
    uint num_psq    = params[6];

    float sum_sq = 0.0f;
    for (uint i = simd_lid; i < num_psq; i += 32u) sum_sq += psq_buf[i];
    sum_sq = simd_sum(sum_sq);
    float norm_scale = rsqrt(sum_sq / float(K) + eps);

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq_ = (device const half*)B_q;
    device const half*  bq_ = (device const half*)(B_q + M_q * nb * 2u);
    device const uchar* nq_ = B_q + M_q * nb * 4u;
    device const half*  sk_ = (device const half*)B_k;
    device const half*  bk_ = (device const half*)(B_k + M_kv * nb * 2u);
    device const uchar* nk_ = B_k + M_kv * nb * 4u;
    device const half*  sv_ = (device const half*)B_v;
    device const half*  bv_ = (device const half*)(B_v + M_kv * nb * 2u);
    device const uchar* nv_ = B_v + M_kv * nb * 4u;

    bool do_kv = (m_base < M_kv);

    for (uint n = 0u; n < seq_len; n++) {
        float acc_q[QMV_BR] = {}, acc_k[QMV_BR] = {}, acc_v[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float pq[QMV_BR] = {}, pk[QMV_BR] = {}, pv[QMV_BR] = {};
            float act_sum = 0.0f;

            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    half4 nw = *(device const half4*)(norm_weight + base_k + sub * 32u + i * 4u);
                    float x0 = float(av[0]) * float(nw[0]) * norm_scale;
                    float x1 = float(av[1]) * float(nw[1]) * norm_scale;
                    float x2 = float(av[2]) * float(nw[2]) * norm_scale;
                    float x3 = float(av[3]) * float(nw[3]) * norm_scale;
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1 * (1.0f/16.0f), x2 * (1.0f/256.0f), x3 * (1.0f/4096.0f));
                }
                act_sum += sub_act_sum;
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m < M_q) {
                        device const ushort* ws = (device const ushort*)(nq_ + (m * nb + bi_) * 64u + sub * 16u);
                        for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pq[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); }
                    }
                    if (do_kv && m < M_kv) {
                        { device const ushort* ws = (device const ushort*)(nk_ + (m * nb + bi_) * 64u + sub * 16u);
                          for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pk[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                        { device const ushort* ws = (device const ushort*)(nv_ + (m * nb + bi_) * 64u + sub * 16u);
                          for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pv[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                    }
                }
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m < M_q)  acc_q[r] += float(sq_[m*nb+bi_])*pq[r] + float(bq_[m*nb+bi_])*act_sum;
                if (do_kv && m < M_kv) {
                    acc_k[r] += float(sk_[m*nb+bi_])*pk[r] + float(bk_[m*nb+bi_])*act_sum;
                    acc_v[r] += float(sv_[m*nb+bi_])*pv[r] + float(bv_[m*nb+bi_])*act_sum;
                }
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            float tq = simd_sum(acc_q[r]);
            if (simd_lid == 0u && m < M_q) out_q[m * act_stride + n] = half(tq);
            if (do_kv) {
                float tk = simd_sum(acc_k[r]), tv = simd_sum(acc_v[r]);
                if (simd_lid == 0u && m < M_kv) { out_k[m*act_stride+n] = half(tk); out_v[m*act_stride+n] = half(tv); }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Gate+Up+SwiGLU GEMV with inline RMSNorm from partial sums (Q4_MLX_FAST)
// ---------------------------------------------------------------------------
kernel void fused_gate_up_swiglu_norm_qmv_fast_q4_mlx(
    device half*             out         [[buffer(0)]],
    device const half*       A           [[buffer(1)]],
    device const uchar*      B_gate      [[buffer(2)]],
    device const uchar*      B_up        [[buffer(3)]],
    device const float*      psq_buf     [[buffer(4)]],
    device const half*       norm_weight [[buffer(5)]],
    constant uint*           params      [[buffer(6)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]])
{
    uint seq_len    = params[0];
    uint act_stride = params[1];
    uint K          = params[2];
    uint M          = params[3];
    float eps       = as_type<float>(params[4]);
    uint num_psq    = params[5];

    float sum_sq = 0.0f;
    for (uint i = simd_lid; i < num_psq; i += 32u) sum_sq += psq_buf[i];
    sum_sq = simd_sum(sum_sq);
    float norm_scale = rsqrt(sum_sq / float(K) + eps);

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const half*  b_gate = (device const half*)(B_gate + M * nb * 2u);
    device const uchar* n_gate = B_gate + M * nb * 4u;
    device const half*  s_up   = (device const half*)B_up;
    device const half*  b_up   = (device const half*)(B_up + M * nb * 2u);
    device const uchar* n_up   = B_up + M * nb * 4u;

    for (uint n = 0u; n < seq_len; n++) {
        float acc_gate[QMV_BR] = {}, acc_up[QMV_BR] = {};

        for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
            uint base_k = bi_ * 128u;
            float pg[QMV_BR] = {}, pu[QMV_BR] = {};
            float act_sum = 0.0f;
            for (uint sub = 0u; sub < 4u; sub++) {
                float4 xs[8];
                float sub_act_sum = 0.0f;
                for (uint i = 0u; i < 8u; i++) {
                    half4 av = *(device const half4*)(A + (base_k + sub * 32u + i * 4u) * act_stride + n);
                    half4 nw = *(device const half4*)(norm_weight + base_k + sub * 32u + i * 4u);
                    float x0 = float(av[0])*float(nw[0])*norm_scale, x1 = float(av[1])*float(nw[1])*norm_scale;
                    float x2 = float(av[2])*float(nw[2])*norm_scale, x3 = float(av[3])*float(nw[3])*norm_scale;
                    sub_act_sum += x0 + x1 + x2 + x3;
                    xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                }
                act_sum += sub_act_sum;
                for (uint r = 0u; r < QMV_BR; r++) {
                    uint m = m_base + r;
                    if (m >= M) continue;
                    { device const ushort* ws = (device const ushort*)(n_gate + (m*nb+bi_)*64u + sub*16u);
                      for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pg[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                    { device const ushort* ws = (device const ushort*)(n_up + (m*nb+bi_)*64u + sub*16u);
                      for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pu[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                }
            }
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                acc_gate[r] += float(s_gate[m*nb+bi_])*pg[r] + float(b_gate[m*nb+bi_])*act_sum;
                acc_up[r]   += float(s_up[m*nb+bi_])  *pu[r] + float(b_up[m*nb+bi_])  *act_sum;
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            float g = simd_sum(acc_gate[r]), u = simd_sum(acc_up[r]);
            uint m = m_base + r;
            if (simd_lid == 0u && m < M) {
                float silu_g = g / (1.0f + fast::exp(-g));
                out[m * act_stride + n] = half(silu_g * u);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Add+RMSNorm + QKV GEMV (Q4_MLX_FAST): Decode only (seq_len=1)
// ---------------------------------------------------------------------------
kernel void fused_add_rmsnorm_qkv_qmv_fast_q4_mlx(
    device half*             out_q         [[buffer(0)]],
    device const half*       x             [[buffer(1)]],
    device const half*       delta         [[buffer(2)]],
    device const half*       norm_weight   [[buffer(3)]],
    device const uchar*      B_q           [[buffer(4)]],
    device half*             out_k         [[buffer(5)]],
    device const uchar*      B_k           [[buffer(6)]],
    device half*             out_v         [[buffer(7)]],
    device const uchar*      B_v           [[buffer(8)]],
    device half*             out_residual  [[buffer(9)]],
    constant uint*           params        [[buffer(10)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  lid      [[thread_position_in_threadgroup]],
    threadgroup float*       reduce_f      [[threadgroup(0)]])
{
    uint K       = params[0];
    uint M_q     = params[1];
    uint M_kv    = params[2];
    float eps    = as_type<float>(params[3]);
    uint has_add = params[4];
    uint H       = K;

    float partial_sq = 0.0f;
    for (uint hi = lid; hi < H; hi += 128u) {
        float v;
        if (has_add != 0u) {
            v = float(x[hi]) + float(delta[hi]);
            if (tgid_x == 0u) out_residual[hi] = half(v);
        } else {
            v = float(x[hi]);
        }
        partial_sq += v * v;
    }
    float sg_sum = simd_sum(partial_sq);
    if (simd_lid == 0u) reduce_f[simd_gid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = reduce_f[0] + reduce_f[1] + reduce_f[2] + reduce_f[3];
    float scale = rsqrt(total_sq / float(H) + eps);

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  sq_ = (device const half*)B_q;
    device const half*  bq_ = (device const half*)(B_q + M_q * nb * 2u);
    device const uchar* nq_ = B_q + M_q * nb * 4u;
    device const half*  sk_ = (device const half*)B_k;
    device const half*  bk_ = (device const half*)(B_k + M_kv * nb * 2u);
    device const uchar* nk_ = B_k + M_kv * nb * 4u;
    device const half*  sv_ = (device const half*)B_v;
    device const half*  bv_ = (device const half*)(B_v + M_kv * nb * 2u);
    device const uchar* nv_ = B_v + M_kv * nb * 4u;

    bool do_kv = (m_base < M_kv);
    float acc_q[QMV_BR] = {}, acc_k[QMV_BR] = {}, acc_v[QMV_BR] = {};

    for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
        uint base_k = bi_ * 128u;
        float pq[QMV_BR] = {}, pk[QMV_BR] = {}, pv[QMV_BR] = {};
        float act_sum = 0.0f;

        for (uint sub = 0u; sub < 4u; sub++) {
            float4 xs[8];
            float sub_act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                uint idx = base_k + sub * 32u + i * 4u;
                float r0, r1, r2, r3;
                if (has_add != 0u) {
                    r0 = float(x[idx+0u]) + float(delta[idx+0u]);
                    r1 = float(x[idx+1u]) + float(delta[idx+1u]);
                    r2 = float(x[idx+2u]) + float(delta[idx+2u]);
                    r3 = float(x[idx+3u]) + float(delta[idx+3u]);
                } else {
                    r0 = float(x[idx+0u]); r1 = float(x[idx+1u]);
                    r2 = float(x[idx+2u]); r3 = float(x[idx+3u]);
                }
                float x0 = r0 * scale * float(norm_weight[idx+0u]);
                float x1 = r1 * scale * float(norm_weight[idx+1u]);
                float x2 = r2 * scale * float(norm_weight[idx+2u]);
                float x3 = r3 * scale * float(norm_weight[idx+3u]);
                sub_act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
            }
            act_sum += sub_act_sum;
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m < M_q) {
                    device const ushort* ws = (device const ushort*)(nq_ + (m*nb+bi_)*64u + sub*16u);
                    for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pq[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); }
                }
                if (do_kv && m < M_kv) {
                    { device const ushort* ws = (device const ushort*)(nk_ + (m*nb+bi_)*64u + sub*16u);
                      for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pk[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                    { device const ushort* ws = (device const ushort*)(nv_ + (m*nb+bi_)*64u + sub*16u);
                      for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pv[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                }
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            if (m < M_q)  acc_q[r] += float(sq_[m*nb+bi_])*pq[r] + float(bq_[m*nb+bi_])*act_sum;
            if (do_kv && m < M_kv) {
                acc_k[r] += float(sk_[m*nb+bi_])*pk[r] + float(bk_[m*nb+bi_])*act_sum;
                acc_v[r] += float(sv_[m*nb+bi_])*pv[r] + float(bv_[m*nb+bi_])*act_sum;
            }
        }
    }

    for (uint r = 0u; r < QMV_BR; r++) {
        uint m = m_base + r;
        float tq = simd_sum(acc_q[r]);
        if (simd_lid == 0u && m < M_q) out_q[m] = half(tq);
        if (do_kv) {
            float tk = simd_sum(acc_k[r]), tv = simd_sum(acc_v[r]);
            if (simd_lid == 0u && m < M_kv) { out_k[m] = half(tk); out_v[m] = half(tv); }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Add+RMSNorm + Gate+Up+SwiGLU GEMV (Q4_MLX_FAST): Decode only (seq_len=1)
// ---------------------------------------------------------------------------
kernel void fused_add_rmsnorm_gate_up_swiglu_qmv_fast_q4_mlx(
    device half*             out           [[buffer(0)]],
    device const half*       x             [[buffer(1)]],
    device const half*       delta         [[buffer(2)]],
    device const half*       norm_weight   [[buffer(3)]],
    device const uchar*      B_gate        [[buffer(4)]],
    device const uchar*      B_up          [[buffer(5)]],
    device half*             out_residual  [[buffer(6)]],
    constant uint*           params        [[buffer(7)]],
    uint  tgid_x  [[threadgroup_position_in_grid]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  lid      [[thread_position_in_threadgroup]])
{
    uint K       = params[0];
    uint M       = params[1];
    float eps    = as_type<float>(params[2]);
    uint H       = K;

    threadgroup float reduce_f[4];

    float partial_sq = 0.0f;
    for (uint hi = lid; hi < H; hi += 128u) {
        float v = float(x[hi]) + float(delta[hi]);
        if (tgid_x == 0u) out_residual[hi] = half(v);
        partial_sq += v * v;
    }
    float sg_sum = simd_sum(partial_sq);
    if (simd_lid == 0u) reduce_f[simd_gid] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = reduce_f[0] + reduce_f[1] + reduce_f[2] + reduce_f[3];
    float scale = rsqrt(total_sq / float(H) + eps);

    uint nb = K / 128u;
    uint m_base = tgid_x * QMV_BM + simd_gid * QMV_BR;

    device const half*  s_gate = (device const half*)B_gate;
    device const half*  b_gate = (device const half*)(B_gate + M * nb * 2u);
    device const uchar* n_gate = B_gate + M * nb * 4u;
    device const half*  s_up   = (device const half*)B_up;
    device const half*  b_up   = (device const half*)(B_up + M * nb * 2u);
    device const uchar* n_up   = B_up + M * nb * 4u;

    float acc_gate[QMV_BR] = {}, acc_up[QMV_BR] = {};

    for (uint bi_ = simd_lid; bi_ < nb; bi_ += 32u) {
        uint base_k = bi_ * 128u;
        float pg[QMV_BR] = {}, pu[QMV_BR] = {};
        float act_sum = 0.0f;
        for (uint sub = 0u; sub < 4u; sub++) {
            float4 xs[8];
            float sub_act_sum = 0.0f;
            for (uint i = 0u; i < 8u; i++) {
                uint idx = base_k + sub * 32u + i * 4u;
                float r0 = float(x[idx+0u]) + float(delta[idx+0u]);
                float r1 = float(x[idx+1u]) + float(delta[idx+1u]);
                float r2 = float(x[idx+2u]) + float(delta[idx+2u]);
                float r3 = float(x[idx+3u]) + float(delta[idx+3u]);
                float x0 = r0 * scale * float(norm_weight[idx+0u]);
                float x1 = r1 * scale * float(norm_weight[idx+1u]);
                float x2 = r2 * scale * float(norm_weight[idx+2u]);
                float x3 = r3 * scale * float(norm_weight[idx+3u]);
                sub_act_sum += x0 + x1 + x2 + x3;
                xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
            }
            act_sum += sub_act_sum;
            for (uint r = 0u; r < QMV_BR; r++) {
                uint m = m_base + r;
                if (m >= M) continue;
                { device const ushort* ws = (device const ushort*)(n_gate + (m*nb+bi_)*64u + sub*16u);
                  for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pg[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
                { device const ushort* ws = (device const ushort*)(n_up + (m*nb+bi_)*64u + sub*16u);
                  for (uint j = 0u; j < 8u; j++) { ushort w = ws[j]; pu[r] += xs[j][0]*float(w&0x000fu)+xs[j][1]*float(w&0x00f0u)+xs[j][2]*float(w&0x0f00u)+xs[j][3]*float(w&0xf000u); } }
            }
        }
        for (uint r = 0u; r < QMV_BR; r++) {
            uint m = m_base + r;
            if (m >= M) continue;
            acc_gate[r] += float(s_gate[m*nb+bi_])*pg[r] + float(b_gate[m*nb+bi_])*act_sum;
            acc_up[r]   += float(s_up[m*nb+bi_])  *pu[r] + float(b_up[m*nb+bi_])  *act_sum;
        }
    }

    for (uint r = 0u; r < QMV_BR; r++) {
        float g = simd_sum(acc_gate[r]), u = simd_sum(acc_up[r]);
        uint m = m_base + r;
        if (simd_lid == 0u && m < M) {
            float silu_g = g / (1.0f + fast::exp(-g));
            out[m] = half(silu_g * u);
        }
    }
}

// ===========================================================================
// Per-layer megakernel: one transformer layer per dispatch
// ===========================================================================
//
// Reduces ~137 dispatches to ~26 by fusing all ops within a layer into
// one kernel. Each dispatch processes one layer with 128 threads (4 SGs).
// Phase synchronization via threadgroup_barrier(mem_flags::mem_device).
//
// Launch: (1, 1, 1) x (128, 1, 1)   — called 22 times (once per layer)
// Also used for final Add+RMSNorm with is_final_norm=1.
//
// Buffer layout:
//   buffer(0)  = decode_act_buf    (compact stride=1 activations)
//   buffer(1)  = weight_buf        (F16 weights, for norm weights)
//   buffer(2)  = quant_weight_buf  (Q4_0 fast layout quantized weights)
//   buffer(3)  = kv_buf            (KV cache)
//   buffer(4)  = layer_params      (MegaLayerParams — single layer)
//   buffer(5)  = runtime_params    (MegaRuntimeParams)
// ===========================================================================

#define MEGA_TG_SIZE 128u
#define MEGA_NUM_SGS 4u
#define MEGA_QMV_BR 4u
#define MEGA_QMV_BM (MEGA_QMV_BR * MEGA_NUM_SGS)  // 16

struct MegaLayerParams {
    // Quant weight offsets in quant_weight_buf
    uint wq_off;
    uint wk_off;
    uint wv_off;
    uint wo_off;
    uint wgate_off;
    uint wup_off;
    uint wdown_off;
    // Norm weight offsets in weight_buf (F16, not quantized)
    uint norm_attn_off;
    uint norm_ffn_off;
    // KV cache offsets in kv_buf
    uint kv_k_off;
    uint kv_v_off;
    // Activation offsets in decode_act_buf (compact stride=1)
    uint act_input_off;     // residual stream input to this layer
    uint act_norm1_off;     // attention norm output
    uint act_q_off;
    uint act_k_off;
    uint act_v_off;
    uint act_attn_off;      // attention output
    uint act_o_proj_off;    // output projection output
    uint act_residual2_off; // residual after attention add (x + o_proj)
    uint act_norm2_off;     // FFN norm output
    uint act_swiglu_off;    // Gate+Up+SwiGLU output
    uint act_down_off;      // Down projection output
};

struct MegaRuntimeParams {
    uint total_seq_len;
    uint num_layers;
    uint hidden_size;
    uint head_dim;
    uint num_heads;
    uint num_kv_heads;
    uint max_seq;
    uint gqa_factor;
    uint rope_dim_q;    // num_heads * head_dim
    uint rope_dim_k;    // num_kv_heads * head_dim
    uint ffn_size;      // intermediate_size
    uint M_q;           // num_heads * head_dim (Q output dim)
    uint M_kv;          // num_kv_heads * head_dim (K/V output dim)
    uint K_attn;        // hidden_size (attention projection input dim)
    uint K_ffn;         // hidden_size (FFN projection input dim)
    uint K_down;        // intermediate_size (down projection input dim)
    uint eps_bits;      // as_type<float>() for RMSNorm epsilon
    uint rope_base_bits;// as_type<float>() for RoPE base
    uint num_tgs;       // (unused, kept for struct compat)
    // Final output
    uint final_norm_weight_off; // weight_buf offset for final norm
    uint final_output_off;      // activation offset for final norm output
};

kernel void mega_decode_q4_0(
    device half*                    act         [[buffer(0)]],
    device const half*              weights     [[buffer(1)]],
    device const uchar*             qweights    [[buffer(2)]],
    device half*                    kv_buf      [[buffer(3)]],
    constant MegaLayerParams*       layers      [[buffer(4)]],
    constant MegaRuntimeParams&     rt          [[buffer(5)]],
    constant uint&                  layer_idx   [[buffer(6)]],
    uint  simd_gid [[simdgroup_index_in_threadgroup]],
    uint  simd_lid [[thread_index_in_simdgroup]],
    uint  lid      [[thread_position_in_threadgroup]],
    threadgroup float*              shared      [[threadgroup(0)]])
{
    const uint H       = rt.hidden_size;
    const float eps    = as_type<float>(rt.eps_bits);
    const float rope_base = as_type<float>(rt.rope_base_bits);
    const uint abs_pos = rt.total_seq_len - 1u;  // decode: position of the new token

    // Final norm dispatch: layer_idx == num_layers means just do Add+RMSNorm
    if (layer_idx == rt.num_layers) {
        constant MegaLayerParams& Ll = layers[rt.num_layers - 1u];
        device const half* res = act + Ll.act_residual2_off;
        device const half* delta = act + Ll.act_down_off;
        device const half* norm_w = (device const half*)(((device const uchar*)weights) + rt.final_norm_weight_off);

        for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
            act[rt.final_output_off + hi] = half(float(res[hi]) + float(delta[hi]));
        }
        threadgroup_barrier(mem_flags::mem_device);

        float partial_sq = 0.0f;
        for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
            float v = float(act[rt.final_output_off + hi]);
            partial_sq += v * v;
        }
        shared[lid] = partial_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = MEGA_TG_SIZE / 2u; s > 0u; s >>= 1u) {
            if (lid < s) shared[lid] += shared[lid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float scale = rsqrt(shared[0] / float(H) + eps);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
            float v = float(act[rt.final_output_off + hi]);
            act[rt.final_output_off + hi] = half(v * scale * float(norm_w[hi]));
        }
        return;
    }

    constant MegaLayerParams& L = layers[layer_idx];

        // =====================================================================
        // Phase 0: Add+RMSNorm (or just RMSNorm for layer 0)
        // =====================================================================
        {
            device const half* norm_w = (device const half*)(((device const uchar*)weights) + L.norm_attn_off);

            if (layer_idx > 0u) {
                constant MegaLayerParams& Lp = layers[layer_idx - 1u];
                device const half* res = act + Lp.act_residual2_off;
                device const half* delta = act + Lp.act_down_off;
                for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                    act[L.act_input_off + hi] = half(float(res[hi]) + float(delta[hi]));
                }
                threadgroup_barrier(mem_flags::mem_device);
            }

            device const half* input = act + L.act_input_off;

            // RMSNorm: 1024-thread tree reduction
            float partial_sq = 0.0f;
            for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                float v = float(input[hi]);
                partial_sq += v * v;
            }
            shared[lid] = partial_sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = MEGA_TG_SIZE / 2u; s > 0u; s >>= 1u) {
                if (lid < s) shared[lid] += shared[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float scale = rsqrt(shared[0] / float(H) + eps);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                float v = float(input[hi]);
                act[L.act_norm1_off + hi] = half(v * scale * float(norm_w[hi]));
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 1: Fused QKV GEMV (Q4_0) — 32 simdgroups, tile-looped
        // =====================================================================
        {
            device const half* a = act + L.act_norm1_off;
            uint nb = rt.K_attn / 32u;

            // Q projection
            device const half*  sq = (device const half*)(qweights + L.wq_off);
            device const uchar* nq = (device const uchar*)(qweights + L.wq_off) + rt.M_q * nb * 2u;
            uint total_tiles_q = (rt.M_q + MEGA_QMV_BM - 1u) / MEGA_QMV_BM;
            for (uint tile = 0u; tile < total_tiles_q; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= rt.M_q) continue;
                        float d = float(sq[m*nb+bi]);
                        device const ushort* ws = (device const ushort*)(nq + (m*nb+bi)*16u);
                        float p = 0.0f;
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                               + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                        }
                        acc[r] += d * (p - 8.0f * act_sum);
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float t = simd_sum(acc[r]); uint m = m_base + r;
                    if (simd_lid == 0u && m < rt.M_q) act[L.act_q_off + m] = half(t);
                }
            }

            // K projection
            device const half*  sk = (device const half*)(qweights + L.wk_off);
            device const uchar* nk = (device const uchar*)(qweights + L.wk_off) + rt.M_kv * nb * 2u;
            uint total_tiles_kv = (rt.M_kv + MEGA_QMV_BM - 1u) / MEGA_QMV_BM;
            for (uint tile = 0u; tile < total_tiles_kv; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= rt.M_kv) continue;
                        float d = float(sk[m*nb+bi]);
                        device const ushort* ws = (device const ushort*)(nk + (m*nb+bi)*16u);
                        float p = 0.0f;
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                               + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                        }
                        acc[r] += d * (p - 8.0f * act_sum);
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float t = simd_sum(acc[r]); uint m = m_base + r;
                    if (simd_lid == 0u && m < rt.M_kv) act[L.act_k_off + m] = half(t);
                }
            }

            // V projection
            device const half*  sv = (device const half*)(qweights + L.wv_off);
            device const uchar* nv = (device const uchar*)(qweights + L.wv_off) + rt.M_kv * nb * 2u;
            for (uint tile = 0u; tile < total_tiles_kv; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= rt.M_kv) continue;
                        float d = float(sv[m*nb+bi]);
                        device const ushort* ws = (device const ushort*)(nv + (m*nb+bi)*16u);
                        float p = 0.0f;
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                               + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                        }
                        acc[r] += d * (p - 8.0f * act_sum);
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float t = simd_sum(acc[r]); uint m = m_base + r;
                    if (simd_lid == 0u && m < rt.M_kv) act[L.act_v_off + m] = half(t);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 2: RoPE Q+K — all 1024 threads
        // =====================================================================
        {
            uint half_dim = rt.head_dim / 2u;

            // RoPE Q
            uint n_pairs_q = rt.rope_dim_q / 2u;
            for (uint pi = lid; pi < n_pairs_q; pi += MEGA_TG_SIZE) {
                uint head_idx = pi / half_dim;
                uint pair_pos = pi % half_dim;
                uint hi0 = head_idx * rt.head_dim + pair_pos;
                uint hi1 = hi0 + half_dim;
                float theta = (float)abs_pos / pow(rope_base, 2.0f * (float)pair_pos / (float)rt.head_dim);
                float cos_t = cos(theta), sin_t = sin(theta);
                float x0 = float(act[L.act_q_off + hi0]);
                float x1 = float(act[L.act_q_off + hi1]);
                act[L.act_q_off + hi0] = half(x0 * cos_t - x1 * sin_t);
                act[L.act_q_off + hi1] = half(x0 * sin_t + x1 * cos_t);
            }

            // RoPE K
            uint n_pairs_k = rt.rope_dim_k / 2u;
            for (uint pi = lid; pi < n_pairs_k; pi += MEGA_TG_SIZE) {
                uint head_idx = pi / half_dim;
                uint pair_pos = pi % half_dim;
                uint hi0 = head_idx * rt.head_dim + pair_pos;
                uint hi1 = hi0 + half_dim;
                float theta = (float)abs_pos / pow(rope_base, 2.0f * (float)pair_pos / (float)rt.head_dim);
                float cos_t = cos(theta), sin_t = sin(theta);
                float x0 = float(act[L.act_k_off + hi0]);
                float x1 = float(act[L.act_k_off + hi1]);
                act[L.act_k_off + hi0] = half(x0 * cos_t - x1 * sin_t);
                act[L.act_k_off + hi1] = half(x0 * sin_t + x1 * cos_t);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 3: KV Cache Append — all 1024 threads
        // =====================================================================
        {
            uint kv_dim = rt.M_kv;  // num_kv_heads * head_dim
            device half* k_cache = kv_buf + L.kv_k_off / 2u;
            device half* v_cache = kv_buf + L.kv_v_off / 2u;
            for (uint i = lid; i < kv_dim; i += MEGA_TG_SIZE) {
                uint kv_head = i / rt.head_dim;
                uint d = i % rt.head_dim;
                k_cache[kv_head * rt.max_seq * rt.head_dim + abs_pos * rt.head_dim + d] = act[L.act_k_off + i];
                v_cache[kv_head * rt.max_seq * rt.head_dim + abs_pos * rt.head_dim + d] = act[L.act_v_off + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 4: SDPA (causal, decode) — 1 simdgroup per head, online softmax
        // 4 SGs process 4 heads in parallel, looping over all heads
        // =====================================================================
        {
            float attn_scale = rsqrt((float)rt.head_dim);
            uint qk_per_thread = rt.head_dim / 32u;  // e.g. 2 for head_dim=64
            device const half* k_cache = kv_buf + L.kv_k_off / 2u;
            device const half* v_cache = kv_buf + L.kv_v_off / 2u;

            for (uint h = simd_gid; h < rt.num_heads; h += MEGA_NUM_SGS) {
                uint kv_h = h / rt.gqa_factor;

                // Load Q slice for this head
                float q[4];
                for (uint i = 0u; i < qk_per_thread; i++) {
                    q[i] = attn_scale * float(act[L.act_q_off + h * rt.head_dim + simd_lid * qk_per_thread + i]);
                }

                // Online softmax over all KV positions
                float o[4] = {};
                float max_score = -INFINITY;
                float sum_exp = 0.0f;

                for (uint pos = 0u; pos < rt.total_seq_len; pos++) {
                    float score = 0.0f;
                    for (uint i = 0u; i < qk_per_thread; i++) {
                        uint d = simd_lid * qk_per_thread + i;
                        score += q[i] * float(k_cache[kv_h * rt.max_seq * rt.head_dim + pos * rt.head_dim + d]);
                    }
                    score = simd_sum(score);

                    float new_max = max(max_score, score);
                    float factor = fast::exp(max_score - new_max);
                    float exp_score = fast::exp(score - new_max);
                    max_score = new_max;
                    sum_exp = sum_exp * factor + exp_score;

                    for (uint i = 0u; i < qk_per_thread; i++) {
                        uint d = simd_lid * qk_per_thread + i;
                        o[i] = o[i] * factor + exp_score * float(v_cache[kv_h * rt.max_seq * rt.head_dim + pos * rt.head_dim + d]);
                    }
                }

                // Normalize and write output
                for (uint i = 0u; i < qk_per_thread; i++) {
                    uint d = simd_lid * qk_per_thread + i;
                    act[L.act_attn_off + h * rt.head_dim + d] = half(sum_exp > 0.0f ? (o[i] / sum_exp) : 0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 5: Output projection GEMV (Q4_0)
        // =====================================================================
        {
            device const half* a = act + L.act_attn_off;
            uint K = rt.K_attn;
            uint M = rt.hidden_size;
            uint nb = K / 32u;
            device const half*  sc = (device const half*)(qweights + L.wo_off);
            device const uchar* ni = (device const uchar*)(qweights + L.wo_off) + M * nb * 2u;
            uint total_tiles = (M + MEGA_QMV_BM - 1u) / MEGA_QMV_BM;
            for (uint tile = 0u; tile < total_tiles; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= M) continue;
                        float d = float(sc[m*nb+bi]);
                        device const ushort* ws = (device const ushort*)(ni + (m*nb+bi)*16u);
                        float p = 0.0f;
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                               + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                        }
                        acc[r] += d * (p - 8.0f * act_sum);
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float t = simd_sum(acc[r]); uint m = m_base + r;
                    if (simd_lid == 0u && m < M) act[L.act_o_proj_off + m] = half(t);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 6: Add+RMSNorm (residual + o_proj → residual2, norm2)
        // =====================================================================
        {
            device const half* res = act + L.act_input_off;
            device const half* delta = act + L.act_o_proj_off;
            device const half* norm_w = (device const half*)(((device const uchar*)weights) + L.norm_ffn_off);

            // Write residual2 = res + delta
            for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                act[L.act_residual2_off + hi] = half(float(res[hi]) + float(delta[hi]));
            }
            threadgroup_barrier(mem_flags::mem_device);

            // RMSNorm
            float partial_sq = 0.0f;
            for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                float v = float(act[L.act_residual2_off + hi]);
                partial_sq += v * v;
            }
            shared[lid] = partial_sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = MEGA_TG_SIZE / 2u; s > 0u; s >>= 1u) {
                if (lid < s) shared[lid] += shared[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float scale = rsqrt(shared[0] / float(H) + eps);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint hi = lid; hi < H; hi += MEGA_TG_SIZE) {
                float v = float(act[L.act_residual2_off + hi]);
                act[L.act_norm2_off + hi] = half(v * scale * float(norm_w[hi]));
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 7: Gate+Up+SwiGLU GEMV (Q4_0)
        // =====================================================================
        {
            device const half* a = act + L.act_norm2_off;
            uint M = rt.ffn_size;
            uint K = rt.K_ffn;
            uint nb = K / 32u;
            device const half*  sg = (device const half*)(qweights + L.wgate_off);
            device const uchar* ng = (device const uchar*)(qweights + L.wgate_off) + M * nb * 2u;
            device const half*  su = (device const half*)(qweights + L.wup_off);
            device const uchar* nu = (device const uchar*)(qweights + L.wup_off) + M * nb * 2u;

            uint total_tiles = (M + MEGA_QMV_BM - 1u) / MEGA_QMV_BM;
            for (uint tile = 0u; tile < total_tiles; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc_g[MEGA_QMV_BR] = {};
                float acc_u[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= M) continue;
                        {
                            float d = float(sg[m*nb+bi]);
                            device const ushort* ws = (device const ushort*)(ng + (m*nb+bi)*16u);
                            float p = 0.0f;
                            for (uint j = 0u; j < 8u; j++) {
                                ushort w = ws[j];
                                p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                                   + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                            }
                            acc_g[r] += d * (p - 8.0f * act_sum);
                        }
                        {
                            float d = float(su[m*nb+bi]);
                            device const ushort* ws = (device const ushort*)(nu + (m*nb+bi)*16u);
                            float p = 0.0f;
                            for (uint j = 0u; j < 8u; j++) {
                                ushort w = ws[j];
                                p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                                   + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                            }
                            acc_u[r] += d * (p - 8.0f * act_sum);
                        }
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float g = simd_sum(acc_g[r]);
                    float u = simd_sum(acc_u[r]);
                    uint m = m_base + r;
                    if (simd_lid == 0u && m < M) {
                        float silu_g = g / (1.0f + fast::exp(-g));
                        act[L.act_swiglu_off + m] = half(silu_g * u);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // Phase 8: Down projection GEMV (Q4_0)
        // =====================================================================
        {
            device const half* a = act + L.act_swiglu_off;
            uint M = rt.hidden_size;
            uint K = rt.K_down;
            uint nb = K / 32u;
            device const half*  sc = (device const half*)(qweights + L.wdown_off);
            device const uchar* ni = (device const uchar*)(qweights + L.wdown_off) + M * nb * 2u;
            uint total_tiles = (M + MEGA_QMV_BM - 1u) / MEGA_QMV_BM;
            for (uint tile = 0u; tile < total_tiles; tile++) {
                uint m_base = tile * MEGA_QMV_BM + simd_gid * MEGA_QMV_BR;
                float acc[MEGA_QMV_BR] = {};
                for (uint bi = simd_lid; bi < nb; bi += 32u) {
                    uint base_k = bi * 32u;
                    float4 xs[8]; float act_sum = 0.0f;
                    for (uint i = 0u; i < 8u; i++) {
                        half4 av = *(device const half4*)(a + base_k + i * 4u);
                        float x0=float(av[0]),x1=float(av[1]),x2=float(av[2]),x3=float(av[3]);
                        act_sum += x0+x1+x2+x3;
                        xs[i] = float4(x0, x1*(1.0f/16.0f), x2*(1.0f/256.0f), x3*(1.0f/4096.0f));
                    }
                    for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                        uint m = m_base + r; if (m >= M) continue;
                        float d = float(sc[m*nb+bi]);
                        device const ushort* ws = (device const ushort*)(ni + (m*nb+bi)*16u);
                        float p = 0.0f;
                        for (uint j = 0u; j < 8u; j++) {
                            ushort w = ws[j];
                            p += xs[j][0]*float(w&0x000fu) + xs[j][1]*float(w&0x00f0u)
                               + xs[j][2]*float(w&0x0f00u) + xs[j][3]*float(w&0xf000u);
                        }
                        acc[r] += d * (p - 8.0f * act_sum);
                    }
                }
                for (uint r = 0u; r < MEGA_QMV_BR; r++) {
                    float t = simd_sum(acc[r]); uint m = m_base + r;
                    if (simd_lid == 0u && m < M) act[L.act_down_off + m] = half(t);
                }
            }
        }

    // No end-of-layer barrier needed — dispatch boundary provides it
}

