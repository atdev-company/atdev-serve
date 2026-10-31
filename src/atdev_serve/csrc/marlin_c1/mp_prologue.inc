// ---------------------------------------------------------------------------
// marlin_plus: fused per-token int8 activation quantisation + grid barrier.
// Replaces vLLM's separate `_per_token_quant_int8` Triton launch and the
// Inductor `a_scales * input_global_scale` multiply (2 launches per linear,
// ~5 us at c=32). Same formula as vllm int8_utils._per_token_quant_int8:
//   absmax = max(max|x|, 1e-10); q = round(x * (127 / absmax)); s = absmax/127
// Requires all CTAs of the grid to be co-resident (Marlin's persistent grid
// of sms * blocks_per_sm CTAs; Marlin's own lock-based reduction already
// relies on that).
// ---------------------------------------------------------------------------
// grid barrier (bar[0] = count, bar[1] = generation); self-resetting,
// CUDA-graph safe; all CTAs of the grid must be co-resident.
__device__ __forceinline__ void mp_grid_barrier(int* bar) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    int gen;
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                 : "=r"(gen)
                 : "l"(bar + 1)
                 : "memory");
    int arrived = atomicAdd(bar, 1);
    if (arrived == (int)gridDim.x - 1) {
      atomicExch(bar, 0);
      __threadfence();
      atomicAdd(bar + 1, 1);
    } else {
      int g;
      do {
        asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                     : "=r"(g)
                     : "l"(bar + 1)
                     : "memory");
      } while (g == gen);
    }
  }
  __syncthreads();
}

template <int threads>
__device__ __forceinline__ void mp_fused_quant_prologue(
    const int4* __restrict__ src, int lda_src, int prob_m, int prob_k,
    int4* dst, int lda_dst, float* a_s, const float* __restrict__ gscale,
    int* bar, int* sa = nullptr) {
  __shared__ float mp_red[threads / 32];
  const int nvec = prob_k / 8;  // 8 bf16 per int4
  const float gs = gscale[0];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int2* dst8 = reinterpret_cast<int2*>(dst);
  for (int row = blockIdx.x; row < prob_m; row += gridDim.x) {
    const int4* x = src + (size_t)row * (lda_src / 8);
    float amax = 0.f;
    for (int v = threadIdx.x; v < nvec; v += threads) {
      int4 d = x[v];
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&d);
  #pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __bfloat1622float2(h[i]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) mp_red[warp] = amax;
    __syncthreads();
    amax = 0.f;
  #pragma unroll
    for (int w = 0; w < threads / 32; w++) amax = fmaxf(amax, mp_red[w]);
    __syncthreads();
    amax = fmaxf(amax, 1e-10f);
    const float inv = 127.f / amax;
    if (sa == nullptr) {
      for (int v = threadIdx.x; v < nvec; v += threads) {
        int4 d = x[v];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&d);
        int2 o;
        int8_t* ob = reinterpret_cast<int8_t*>(&o);
    #pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 f = __bfloat1622float2(h[i]);
          ob[2 * i] = (int8_t)(int)roundf(f.x * inv);
          ob[2 * i + 1] = (int8_t)(int)roundf(f.y * inv);
        }
        dst8[(size_t)row * (lda_dst / 8) + v] = o;
      }
    } else {
      // identical int8 values, plus the exact per-(row, 128-group) sum of the codes
      // (16 consecutive 8-element vectors = one group = 16 consecutive lanes)
      for (int base = 0; base < nvec; base += threads) {
        const int v = base + threadIdx.x;
        int gsum = 0;
        if (v < nvec) {
          int4 d = x[v];
          const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&d);
          int2 o;
          int8_t* ob = reinterpret_cast<int8_t*>(&o);
    #pragma unroll
          for (int i = 0; i < 4; i++) {
            float2 f = __bfloat1622float2(h[i]);
            ob[2 * i] = (int8_t)(int)roundf(f.x * inv);
            ob[2 * i + 1] = (int8_t)(int)roundf(f.y * inv);
            gsum += (int)ob[2 * i] + (int)ob[2 * i + 1];
          }
          dst8[(size_t)row * (lda_dst / 8) + v] = o;
        }
    #pragma unroll
        for (int off = 8; off > 0; off >>= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, off);
        if (v < nvec && (v & 15) == 0) sa[(size_t)(v / 16) * 256 + row] = gsum;  // [K/128][256]
      }
    }
    if (threadIdx.x == 0) a_s[row] = (amax / 127.f) * gs;
  }
  mp_grid_barrier(bar);
}
