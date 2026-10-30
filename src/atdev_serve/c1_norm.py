"""E6 (docs/research/E6-c1-norms.md): low-latency residual-add + Gemma RMSNorm / SiLU*mul kernels for the
c=1 verify batch (csrc/c1_norm.cu). Engine hook: ATDEV_OPT=c1_norm (register())."""
from __future__ import annotations

import logging
import os

import torch

OPT_NAME = "c1_norm"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")
_EXT = None


def build(defines: str = ""):
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "c1_norm.cu")
    defs = [f"-D{d}" for d in defines.split(",") if d]
    name = "atdev_c1_norm" + ("_" + "_".join(d.replace("=", "") for d in defines.split(",") if d) if defines else "")
    return load(name=name, sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"] + defs)


def ext():
    global _EXT
    if _EXT is None:
        _EXT = build(os.environ.get("ATDEV_C1_NORM_DEFS", ""))
    return _EXT


_NS = (5120, 4096, 2048)


def _al16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and (t.stride(0) * t.element_size()) % 16 == 0


def add_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """(norm(x + residual) * (1 + w), bf16(x + residual)) -- Gemma RMSNorm with fused residual add."""
    if not (_al16(x) and _al16(residual) and weight.is_contiguous() and weight.data_ptr() % 16 == 0):
        xf = x.float() + residual.float()
        var = (xf * xf).mean(-1, keepdim=True)
        return ((xf * torch.rsqrt(var + eps)) * (weight.float() + 1.0)).to(x.dtype), xf.to(x.dtype)
    out = torch.empty_like(x)
    res = torch.empty_like(x)
    ext().add_norm(x, residual, None, weight, out, res, eps)
    return out, res


def _add_norm_fake(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(x)


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1] // 2
    if not _al16(x):
        return (torch.nn.functional.silu(x[:, :d].float()) * x[:, d:].float()).to(x.dtype)
    y = torch.empty((x.shape[0], d), dtype=x.dtype, device=x.device)
    ext().silu_mul(x, y)
    return y


def _silu_mul_fake(x):
    return torch.empty((x.shape[0], x.shape[1] // 2), dtype=x.dtype, device=x.device)


def _norm_ok(x, residual, weight) -> bool:
    return (residual is not None and x.dim() == 2 and residual.dim() == 2 and x.is_cuda
            and x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16 and x.shape[-1] in _NS
            and x.shape == residual.shape and x.stride(-1) == 1 and residual.stride(-1) == 1)


_REGISTERED = False


def register() -> None:
    """Gemma RMSNorm with residual (Qwen3-Next decoder input / post-attention norms, the final norm) and
    SiluAndMul (MLP) -> csrc/c1_norm.cu kernels as opaque custom ops (Inductor no longer generates them).
    ATDEV_C1_SILU=0 keeps Inductor's SiLU*mul."""
    global _REGISTERED
    import vllm
    from vllm.model_executor.layers import activation, layernorm
    from vllm.utils.torch_utils import direct_register_custom_op

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written for vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    if _REGISTERED:
        return
    _REGISTERED = True
    ext()  # build / load before any graph capture
    direct_register_custom_op("atdev_c1_add_norm", add_norm, fake_impl=_add_norm_fake)
    direct_register_custom_op("atdev_c1_silu_mul", silu_mul, fake_impl=_silu_mul_fake)

    g_fwd = layernorm.GemmaRMSNorm.forward

    def gemma_forward(self, x, residual=None):
        if _norm_ok(x, residual, self.weight):
            return torch.ops.vllm.atdev_c1_add_norm(x, residual, self.weight, self.variance_epsilon)
        return g_fwd(self, x, residual)

    layernorm.GemmaRMSNorm.forward = gemma_forward
    msg = "GemmaRMSNorm(x, residual) -> atdev_c1_add_norm"
    if os.environ.get("ATDEV_C1_SILU", "1") != "0":
        s_fwd = activation.SiluAndMul.forward

        def silu_forward(self, x):
            if (x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16 and x.stride(-1) == 1
                    and x.shape[-1] % 16 == 0):
                return torch.ops.vllm.atdev_c1_silu_mul(x)
            return s_fwd(self, x)

        activation.SiluAndMul.forward = silu_forward
        msg += ", SiluAndMul -> atdev_c1_silu_mul"
    log.warning("atdev %s: %s (csrc/c1_norm.cu)", OPT_NAME, msg)


# ---------------------------------------------------------------------------------------------------------
# E6 F3/F4: norm folded into the GEMM epilogues (csrc/marlin_c1, marlin_c1_build) -- see kernel.h MpEpi
# ---------------------------------------------------------------------------------------------------------
def mc1(mode: int):
    from atdev_serve import marlin_c1_build
    return marlin_c1_build.load(mode)


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _ba_gemv_r_kernel(x_ptr, stride_x, w_ptr, stride_w, o_ptr, stride_o, ws_ptr, tick_ptr, r_ptr,
                          M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          SPLIT: tl.constexpr):
        # ba_gemv._ba_gemv_kernel with the output rows scaled by rstd r[m] (fp32) before rounding
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < M
        kk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        k0 = pid_k * (K // SPLIT)
        for ks in range(0, K // SPLIT, BK):
            kidx = k0 + ks + kk
            xk = tl.load(x_ptr + rows[:, None] * stride_x + kidx[None, :], mask=rmask[:, None], other=0.0)
            wk = tl.load(w_ptr + cols[None, :] * stride_w + kidx[:, None])
            acc = tl.dot(xk, wk, acc)
        ws = ws_ptr + (pid_k * BM + rows)[:, None] * N + cols[None, :]
        tl.store(ws, acc)
        tl.debug_barrier()
        t = tl.atomic_add(tick_ptr + pid_n, 1, sem="acq_rel")
        if t == SPLIT - 1:
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(ws_ptr + (s * BM + rows)[:, None] * N + cols[None, :])
            r = tl.load(r_ptr + rows, mask=rmask, other=0.0)
            tot = tot * r[:, None]
            tl.store(o_ptr + rows[:, None] * stride_o + cols[None, :], tot.to(o_ptr.dtype.element_ty),
                     mask=rmask[:, None])
            tl.atomic_xchg(tick_ptr + pid_n, 0)


_FB: dict = {}


def _fbufs(dev):
    b = _FB.get(dev.index)
    if b is None:
        from atdev_serve import marlin_plus
        mb = marlin_plus.bufs(dev)
        b = _FB[dev.index] = dict(
            locks=torch.zeros(mb["locks"].numel(), dtype=torch.int32, device=dev),
            c_tmp=mb["c_tmp"],  # fp32 split-K scratch: stream-ordered reuse is safe
            ws=torch.empty(32 * 64 * 256, dtype=torch.float32, device=dev),
            tick=torch.zeros(64, dtype=torch.int32, device=dev),
        )
    return b


def mp_cfg(n: int, k: int, m: int):
    from atdev_serve import marlin_plus
    ent = marlin_plus.lookup(n, k, m, False)
    return None if ent is None else list(ent["cfg"])


def ba_gemv_r(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, cfg=(8, 128, 4), out=None) -> torch.Tensor:
    split, bk, warps = cfg
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    b = _fbufs(x.device)
    _ba_gemv_r_kernel[(n // 16, split)](x, x.stride(0), w, w.stride(0), out, out.stride(0), b["ws"], b["tick"], r,
                                        m, n, k, BM=16, BN=16, BK=bk, SPLIT=split, num_warps=warps)
    return out


# ---- ATDEV_C1_BA_STREAM (c1_fuse): ba_gemv_r split in two so the GEMV can run on a side stream before the
# qkvz consumer has produced rstd.  ba_tot = the _ba_gemv_r_kernel accumulation (same tiles, same split-K
# partials, same in-order 0..SPLIT-1 sum) stored as fp32 WITHOUT the rstd row scale;  ba_scale = bf16(tot * r).
# fp32 mul + RN round of the same fp32 operands -> bit-identical to ba_gemv_r.
if triton is not None:

    @triton.jit
    def _ba_tot_kernel(x_ptr, stride_x, w_ptr, stride_w, o_ptr, ws_ptr, tick_ptr,
                       M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        rmask = rows < M
        kk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        k0 = pid_k * (K // SPLIT)
        for ks in range(0, K // SPLIT, BK):
            kidx = k0 + ks + kk
            xk = tl.load(x_ptr + rows[:, None] * stride_x + kidx[None, :], mask=rmask[:, None], other=0.0)
            wk = tl.load(w_ptr + cols[None, :] * stride_w + kidx[:, None])
            acc = tl.dot(xk, wk, acc)
        ws = ws_ptr + (pid_k * BM + rows)[:, None] * N + cols[None, :]
        tl.store(ws, acc)
        tl.debug_barrier()
        t = tl.atomic_add(tick_ptr + pid_n, 1, sem="acq_rel")
        if t == SPLIT - 1:
            tot = tl.zeros([BM, BN], dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                tot += tl.load(ws_ptr + (s * BM + rows)[:, None] * N + cols[None, :])
            tl.store(o_ptr + rows[:, None] * N + cols[None, :], tot, mask=rmask[:, None])  # [M, N] fp32
            tl.atomic_xchg(tick_ptr + pid_n, 0)

    @triton.jit
    def _ba_scale_kernel(t_ptr, r_ptr, o_ptr, stride_o, M, N, BM: tl.constexpr, BN: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.arange(0, BN)
        msk = (rows < M)[:, None] & (cols < N)[None, :]
        tot = tl.load(t_ptr + rows[:, None] * N + cols[None, :], mask=msk, other=0.0)
        r = tl.load(r_ptr + rows, mask=rows < M, other=0.0)
        tot = tot * r[:, None]
        tl.store(o_ptr + rows[:, None] * stride_o + cols[None, :], tot.to(o_ptr.dtype.element_ty), mask=msk)


_SB: dict = {}


def _sbufs(dev):
    """side-stream ba_tot scratch (separate from _fbufs: may run concurrently with a main-stream ba_gemv_r)."""
    b = _SB.get(dev.index)
    if b is None:
        b = _SB[dev.index] = dict(ws=torch.empty(32 * 64 * 256, dtype=torch.float32, device=dev),
                                  tick=torch.zeros(64, dtype=torch.int32, device=dev))
    return b


def ba_tot_ok(x: torch.Tensor, w: torch.Tensor, cfg=(8, 128, 4)) -> bool:
    """shapes on which ba_gemv_r's kernel (and so ba_tot) is the production path (c1_fuse.e6_ba condition)."""
    return (x.shape[0] <= 16 and x.stride(-1) == 1 and w.stride(-1) == 1 and w.shape[0] % 16 == 0
            and x.shape[1] % 1024 == 0 and w.shape[0] <= 256)


_BA_SIDE_EXT = None


def ba_side_ext():
    """csrc/ba_side.cu (smem-free one-warp-per-CTA ba_tot, co-resides with the Marlin consumer)."""
    global _BA_SIDE_EXT
    if _BA_SIDE_EXT is None:
        from torch.utils.cpp_extension import load
        _BA_SIDE_EXT = load(name="atdev_ba_side", sources=[os.path.join(os.path.dirname(__file__), "csrc",
                                                                         "ba_side.cu")],
                            extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"],
                            verbose=False)
    return _BA_SIDE_EXT


# ATDEV_C1_BA_TOT: cuda (default: csrc/ba_side.cu) | triton (_ba_tot_kernel; smem-staged, does not co-reside)
BA_TOT_IMPL = os.environ.get("ATDEV_C1_BA_TOT", "cuda")


def ba_tot(x: torch.Tensor, w: torch.Tensor, cfg=(8, 128, 4), out=None, impl=None) -> torch.Tensor:
    """fp32 [M, N] unscaled ba accumulator of ba_gemv_r(x, w, r) (same cfg); out must be contiguous."""
    split, bk, warps = cfg
    m, k = x.shape
    n = w.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    assert out.is_contiguous() and out.shape == (m, n)
    b = _sbufs(x.device)
    if (impl or BA_TOT_IMPL) == "cuda" and cfg == (8, 128, 4):
        ba_side_ext().ba_tot(x, w, out, b["ws"], b["tick"])
        return out
    _ba_tot_kernel[(n // 16, split)](x, x.stride(0), w, w.stride(0), out, b["ws"], b["tick"],
                                     m, n, k, BM=16, BN=16, BK=bk, SPLIT=split, num_warps=warps)
    return out


def ba_scale(tot: torch.Tensor, r: torch.Tensor, m: int, out=None) -> torch.Tensor:
    """bf16(tot[:m] * r[:, None]) == ba_gemv_r's epilogue."""
    n = tot.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=tot.device)
    assert tot.is_contiguous() and tot.shape[0] >= m and r.shape[0] >= m
    _ba_scale_kernel[(triton.cdiv(m, 16),)](tot, r, out, out.stride(0), m, n, BM=16, BN=triton.next_power_of_2(n), num_warps=4)
    return out


def lin_prod(x, wq, ws, n, k, res, nw):
    """producer: h = x @ W (n = norm width) -> (a = bf16((h + res) * (1 + nw)), res_out = bf16(h + res), P)."""
    m = x.shape[0]
    cfg = mp_cfg(n, k, m)
    t = n // cfg[1]
    a = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ro = torch.empty_like(a)
    p = torch.empty((m, t), dtype=torch.float32, device=x.device)
    b = _fbufs(x.device)
    mc1(1).gemm_epi(x, wq, ws, a, b["locks"], b["c_tmp"], n, k, cfg, 1, R=res, W=nw, C2=ro, P=p)
    return a, ro, p


def lin_cons(a, wq, ws, n, k, p, n_norm, eps, silu=False, want_r=False):
    """consumer: y = diag(rstd) (a @ W) [silu pair -> [M, n/2]]; rstd from the producer partials p [M, T]."""
    m = a.shape[0]
    cfg = mp_cfg(n, k, m)
    out = torch.empty((m, n // 2 if silu else n), dtype=torch.bfloat16, device=a.device)
    r = torch.empty((m,), dtype=torch.float32, device=a.device) if want_r else None
    b = _fbufs(a.device)
    mc1(3 if silu else 2).gemm_epi(a, wq, ws, out, b["locks"], b["c_tmp"], n, k, cfg, 3 if silu else 2, Pin=p,
                   T_in=p.shape[1], n_norm=n_norm, eps=eps, Rout=r)
    return out, r


# ---- W3 (3-bit planar weights, w3_mixed) variants: csrc/marlin_c1 + the W3 B stream (marlin_c1_w3_build)
_W3C1_TABLE: dict | None = None


def mc1w3(mode: int):
    from atdev_serve import marlin_c1_w3_build
    return marlin_c1_w3_build.load(mode)


def w3c1_table() -> dict:
    """{"N,K": {"16": [tk, tn, threads, stages, bps, max_tmb]}} -- ATDEV_E6W3_CONFIG or w3_c1_cfg.json"""
    global _W3C1_TABLE
    if _W3C1_TABLE is None:
        import json
        p = os.environ.get("ATDEV_E6W3_CONFIG") or os.path.join(os.path.dirname(__file__), "w3_c1_cfg.json")
        _W3C1_TABLE = json.load(open(p)) if os.path.exists(p) else {}
    return _W3C1_TABLE


def w3c1_cfg(n: int, k: int, m: int):
    if not 8 < m <= 16:
        return None
    ent = w3c1_table().get(f"{n},{k}", {}).get("16")
    if ent is None:
        return None
    return list(ent["cfg"] if isinstance(ent, dict) else ent)


def _w3_set_next(ext, nxt, m):
    """L2PF (ATDEV_C1W3_X=l2pf build): announce the next W3 linear (q3, ws, n, k[, cfg]) to the next launch of ext;
    its kernel prefetches that linear's first k-stages into L2 from its tail. No-op on builds without set_next."""
    if nxt is None or not hasattr(ext, "set_next"):
        return
    q3, ws, n, k = nxt[:4]
    cfg = (nxt[4] if len(nxt) > 4 else None) or w3c1_cfg(n, k, m)
    if cfg is not None:
        ext.set_next(q3, ws, n, k, list(cfg), int(os.environ.get("ATDEV_W3_L2PF_STAGES", "3")))


def _w3_sk(n, k, mode, m) -> bool:
    if os.environ.get("ATDEV_C1_W3_SK", "0") != "1":
        return False
    from atdev_serve import w3_sk
    return w3_sk.supported(n, k, mode, m)


def lin_prod_w3(x, q3, ws, n, k, res, nw, cfg=None, nxt=None):
    """lin_prod on W3 planar weights (q3 = w3_mixed.pack_w3 layout). nxt: see _w3_set_next.
    ATDEV_C1_W3_SK=1 (and no explicit cfg): the stream-K kernel (w3_sk.py) on its shapes, P = [M, N / 128]."""
    m = x.shape[0]
    if cfg is None and _w3_sk(n, k, 1, m):
        from atdev_serve import w3_sk
        return w3_sk.lin_prod(x.contiguous(), q3, ws, n, k, res.contiguous(), nw)
    cfg = cfg or w3c1_cfg(n, k, m)
    t = n // cfg[1]
    a = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    ro = torch.empty_like(a)
    p = torch.empty((m, t), dtype=torch.float32, device=x.device)
    b = _fbufs(x.device)
    _w3_set_next(mc1w3(1), nxt, m)
    mc1w3(1).gemm_epi(x, q3, ws, a, b["locks"], b["c_tmp"], n, k, cfg, 1, R=res, W=nw, C2=ro, P=p)
    return a, ro, p


def lin_cons_w3(a, q3, ws, n, k, p, n_norm, eps, silu=False, want_r=False, cfg=None, nxt=None, qkc=None):
    """qkc (C1gdn2, ATDEV_C1W3_X=qkc builds, mode 2): dict of mc1w3(2).set_qkc kwargs -> the GDN q / k / v conv runs in
    this launch's epilogue (scratch qkc["out"] + tags; the GEMM output itself is unchanged)."""
    m = a.shape[0]
    if qkc is not None and not silu:
        ext = mc1w3(2)
        if hasattr(ext, "set_qkc") and not (cfg is None and _w3_sk(n, k, 2, m) and p.shape[1] <= 80):
            ext.set_qkc(**qkc)
    if cfg is None and _w3_sk(n, k, 3 if silu else 2, m) and p.shape[1] <= 80:
        from atdev_serve import w3_sk
        return w3_sk.lin_cons(a.contiguous(), q3, ws, n, k, p.contiguous(), n_norm, eps, silu=silu, want_r=want_r)
    cfg = cfg or w3c1_cfg(n, k, m)
    out = torch.empty((m, n // 2 if silu else n), dtype=torch.bfloat16, device=a.device)
    r = torch.empty((m,), dtype=torch.float32, device=a.device) if want_r else None
    b = _fbufs(a.device)
    _w3_set_next(mc1w3(3 if silu else 2), nxt, m)
    mc1w3(3 if silu else 2).gemm_epi(a, q3, ws, out, b["locks"], b["c_tmp"], n, k, cfg, 3 if silu else 2, Pin=p,
                                     T_in=p.shape[1], n_norm=n_norm, eps=eps, Rout=r)
    return out, r


if triton is not None:

    @triton.jit
    def _w3fb_prod_kernel(h_ptr, r_ptr, w_ptr, a_ptr, ro_ptr, p_ptr, N, T, BLOCK: tl.constexpr,
                          TB: tl.constexpr):
        # x = fp32(h) + fp32(res); ro = bf16(x); a = bf16(x * (1 + w)); p[row] = [sum x^2, 0, ...]
        row = tl.program_id(0)
        s = tl.zeros([BLOCK], dtype=tl.float32)
        for c0 in range(0, N, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            msk = cols < N
            x = tl.load(h_ptr + row * N + cols, mask=msk, other=0.0).to(tl.float32) + \
                tl.load(r_ptr + row * N + cols, mask=msk, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=msk, other=0.0).to(tl.float32)
            tl.store(ro_ptr + row * N + cols, x.to(tl.bfloat16), mask=msk)
            tl.store(a_ptr + row * N + cols, (x * (w + 1.0)).to(tl.bfloat16), mask=msk)
            s += x * x
        tot = tl.sum(s, 0)
        t = tl.arange(0, TB)
        tl.store(p_ptr + row * T + t, tl.where(t == 0, tot, 0.0), mask=t < T)

    @triton.jit
    def _w3fb_cons_kernel(a_ptr, p_ptr, xs_ptr, r_ptr, N, T, n_norm, eps, BLOCK: tl.constexpr, TB: tl.constexpr):
        # r = rsqrt(sum_t p[row, t] / n_norm + eps); xs = bf16(fp32(a) * r)
        row = tl.program_id(0)
        t = tl.arange(0, TB)
        ps = tl.sum(tl.load(p_ptr + row * T + t, mask=t < T, other=0.0), 0)
        r = 1.0 / tl.sqrt(ps / n_norm + eps)
        tl.store(r_ptr + row, r)
        for c0 in range(0, N, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            msk = cols < N
            x = tl.load(a_ptr + row * N + cols, mask=msk, other=0.0).to(tl.float32)
            tl.store(xs_ptr + row * N + cols, (x * r).to(tl.bfloat16), mask=msk)

    @triton.jit
    def _w3fb_silu_kernel(gu_ptr, y_ptr, NH, BLOCK: tl.constexpr):
        # gate_up interleaved per 64 columns: y[j] = silu(g) * u, g = gu[(j // 64) * 128 + j % 64], u = g + 64
        row = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        msk = j < NH
        src = (j // 64) * 128 + j % 64
        g = tl.load(gu_ptr + row * 2 * NH + src, mask=msk, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * 2 * NH + src + 64, mask=msk, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * NH + j, (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16), mask=msk)


def w3fb_prod_epi(h, res, nw, t):
    """unfused-W3 producer epilogue in one kernel (c1_fuse fallback at 8 < M <= 16)."""
    m, n = h.shape
    a = torch.empty_like(h)
    ro = torch.empty_like(h)
    p = torch.empty((m, t), dtype=torch.float32, device=h.device)
    _w3fb_prod_kernel[(m,)](h.contiguous(), res.contiguous(), nw, a, ro, p, n, t, BLOCK=1024,
                            TB=triton.next_power_of_2(t), num_warps=4)
    return a, ro, p


def w3fb_cons_pro(a, p, n_norm, eps):
    m, n = a.shape
    xs = torch.empty_like(a)
    r = torch.empty((m,), dtype=torch.float32, device=a.device)
    t = p.shape[1]
    _w3fb_cons_kernel[(m,)](a.contiguous(), p.contiguous(), xs, r, n, t, float(n_norm), float(eps), BLOCK=1024,
                            TB=triton.next_power_of_2(t), num_warps=4)
    return xs, r


def w3fb_silu(gu):
    m, n = gu.shape
    y = torch.empty((m, n // 2), dtype=gu.dtype, device=gu.device)
    _w3fb_silu_kernel[(m, triton.cdiv(n // 2, 1024))](gu.contiguous(), y, n // 2, BLOCK=1024, num_warps=4)
    return y


def permute_gu_w3(q3: torch.Tensor, ws: torch.Tensor, n: int):
    """W3 planar gate_up [K/16, 3, N/2] int32 (+ scales) -> n64-tile interleaved copies. Plane word j of a k16
    row <-> int4 chunk j of the Marlin layout, an n64 tile = 32 chunks, so the tiles move as 32-word blocks."""
    idx = gu_perm_index(n // 2, q3.device)
    kt = q3.shape[0]
    q2 = q3.view(kt, 3, n // 64, 32)[:, :, idx].reshape(q3.shape).contiguous()
    g = ws.shape[0]
    ws2 = ws.view(g, n // 64, 64)[:, idx].reshape(ws.shape).contiguous()
    return q2, ws2


def gu_perm_index(n_half: int, dev) -> torch.Tensor:
    """n64-tile order of the interleaved gate_up weight: tile 2t = gate tile t, 2t + 1 = up tile t."""
    t = n_half // 64
    return torch.stack([torch.arange(t, device=dev), torch.arange(t, device=dev) + t], 1).reshape(-1)


def permute_gu_marlin(wq: torch.Tensor, ws: torch.Tensor, n: int):
    """Marlin-layout gate_up weight [K/16, 2N] int32 and scales [G, N] -> n64-tile interleaved copies."""
    idx = gu_perm_index(n // 2, wq.device)
    kt = wq.shape[0]
    wq2 = wq.view(kt, n // 64, -1)[:, idx].reshape(wq.shape).contiguous()
    g = ws.shape[0]
    ws2 = ws.view(g, n // 64, 64)[:, idx].reshape(ws.shape).contiguous()
    return wq2, ws2
