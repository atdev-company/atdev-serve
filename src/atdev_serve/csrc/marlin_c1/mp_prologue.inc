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
      // P3 U8: identical int8 values, plus the exact per-(row, 128-group) sum of the codes
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

// ---------------------------------------------------------------------------
// P1 stage 2 chain phases (run by the whole grid between GEMM jobs)
// ---------------------------------------------------------------------------
// residual add + GemmaRMSNorm, one CTA per row (CTAs >= m idle). fp32 math as
// vllm.ir.ops.fused_add_rms_norm: x = h + res; res_out = bf16(x);
// out = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)).
template <int threads>
__device__ __forceinline__ void mp_norm_phase(const int4* __restrict__ h, int ldh,
                                              const int4* __restrict__ res, int4* out,
                                              int4* res_out, const __nv_bfloat16* __restrict__ w,
                                              int m, int k, float eps) {
  __shared__ float red[threads / 32];
  const int nvec = k / 8;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int row = blockIdx.x; row < m; row += gridDim.x) {
    const int4* hp = h + (size_t)row * (ldh / 8);
    const int4* rp = res ? res + (size_t)row * nvec : nullptr;
    // pass 1: sum of squares (values kept in registers: k/8/threads int4 per thread, <= 3 for k<=5120... loop twice instead)
    float ss = 0.f;
    for (int v = threadIdx.x; v < nvec; v += threads) {
      int4 d = hp[v];
      const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&d);
      float2 f[4];
#pragma unroll
      for (int i = 0; i < 4; i++) f[i] = __bfloat1622float2(hh[i]);
      if (rp) {
        int4 r = rp[v];
        const __nv_bfloat162* rr = reinterpret_cast<const __nv_bfloat162*>(&r);
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 g = __bfloat1622float2(rr[i]);
          f[i].x += g.x; f[i].y += g.y;
        }
        int4 o;
        __nv_bfloat162* oo = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
        for (int i = 0; i < 4; i++) oo[i] = __float22bfloat162_rn(f[i]);
        res_out[(size_t)row * nvec + v] = o;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) ss += f[i].x * f[i].x + f[i].y * f[i].y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    ss = 0.f;
#pragma unroll
    for (int i = 0; i < threads / 32; i++) ss += red[i];
    __syncthreads();
    const float rstd = rsqrtf(ss / (float)k + eps);
    // pass 2 (re-read h/res from L2: the row is 10 KB, cheaper than holding it)
    for (int v = threadIdx.x; v < nvec; v += threads) {
      int4 d = hp[v];
      const __nv_bfloat162* hh = reinterpret_cast<const __nv_bfloat162*>(&d);
      int4 wv = reinterpret_cast<const int4*>(w)[v];
      const __nv_bfloat162* ww = reinterpret_cast<const __nv_bfloat162*>(&wv);
      int4 o;
      __nv_bfloat162* oo = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 f = __bfloat1622float2(hh[i]);
        if (rp) {
          int4 r = rp[v];
          float2 g = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&r)[i]);
          f.x += g.x; f.y += g.y;
        }
        float2 wf = __bfloat1622float2(ww[i]);
        f.x = f.x * rstd * (wf.x + 1.f);
        f.y = f.y * rstd * (wf.y + 1.f);
        oo[i] = __float22bfloat162_rn(f);
      }
      out[(size_t)row * nvec + v] = o;
    }
  }
}

// out[m, n] = bf16(silu(g) * u), g = A[:, :n], u = A[:, n:], grid-stride over int4s
template <int threads>
__device__ __forceinline__ void mp_silu_phase(const int4* __restrict__ gu, int ldgu, int4* out,
                                              int m, int n) {
  const int nvec = n / 8;
  const size_t total = (size_t)m * nvec;
  for (size_t idx = (size_t)blockIdx.x * threads + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * threads) {
    const int row = idx / nvec, v = idx % nvec;
    const int4* rowp = gu + (size_t)row * (ldgu / 8);
    int4 g4 = rowp[v], u4 = rowp[nvec + v];
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g4);
    const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&u4);
    int4 o;
    __nv_bfloat162* oo = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 g = __bfloat1622float2(g2[i]), u = __bfloat1622float2(u2[i]);
      float2 y;
      y.x = g.x / (1.f + __expf(-g.x)) * u.x;
      y.y = g.y / (1.f + __expf(-g.y)) * u.y;
      oo[i] = __float22bfloat162_rn(y);
    }
    out[idx] = o;
  }
}
