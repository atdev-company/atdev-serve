"""LL1: lower-level, bit-identical fusions of the c=1 DFlash2 drafter graph glue (ATDEV_OPT=ll1_fuse).

Each fusion is opt-in by its own env flag (default off = stock behaviour, nothing patched):

  ATDEV_LL1_CTX=1   drafter_ctxfold context-K/V chain of a layer (k copy + vLLM rms_norm K-norm + vLLM rotary +
                    reshape_and_cache_flash, 4 launches) as ONE CUDA kernel (csrc/ll1_ctx.cu) that reproduces each
                    step's arithmetic exactly (tests/test_ll1_fuse.py: bitwise vs the stock chain).
  ATDEV_LL1_CBUF=1  the attention_conv.prepare grouped conv writes its output straight into drafter_ctxfold's
                    block rows buf[rows:] (the fold's qkv op then copies onto itself = torch's same-data no-op), so
                    the per-layer memcpy into the Marlin m32 tile disappears. Same Triton conv kernel -> identical.
  ATDEV_LL1_KPC=1   DFlashGroupedConv.prepare at the c=1 block (16 rows): kernel_projection GEMM + split-K reduce +
                    side-0 grouped conv as ONE kernel (csrc/ll1_kpc.cu) emulating cuBLAS' split-K (S = 5) mma order
                    and the Triton conv arithmetic exactly (with CBUF the attention conv writes into the fold buffer).
  ATDEV_LL1_SEL=1   DFlash2 CandidateSelector at c=1 (hidden_projection split-K GEMM + reduce + codebook gathers + bf16
                    mul + einsum + unary add, 6 launches) as ONE kernel (csrc/ll1_sel.cu, cuBLAS / CUTLASS mma order
                    emulated; batch != 1 -> stock ops inside the op).
  ATDEV_LL1_QKR=1   drafter attention: the Inductor q/k RMSNorm + RoPE kernel (copied verbatim from the dump,
                    ll1_qkr.py) also writes the block rows' K / V into the paged KV cache, replacing the separate
                    reshape_and_cache_flash launch (unified_kv_cache_update) of each drafter layer.
  ATDEV_LL1_TAIL=1  draft-vocab top-k tail (TOPKX=1, k = screen = 16): merge of the per-chunk candidates + full-head row
                    gather + exact bf16 rescoring (cuBLAS gemvx order emulated) + sorted top-16 + ids gather + float
                    as ONE kernel per row (csrc/ll1_tail.cu) after the stock _topkx_chunk_kernel.
  ATDEV_LL1_QKG=1   target full-attention layers: vLLM's fused q/k RMSNorm + RoPE + gate kernel with its independent
                    loads (positions, cos/sin, rotary halves, weights, gate copy) issued before the norm (ll1_qkg.py,
                    arithmetic verbatim -> bit-identical); 16 launches per step in the T graph.
  ATDEV_LL1_GTIME=n diagnostic (no profiler): CUDA events around every torch.cuda.CUDAGraph.replay; every n replays
                    of a graph, log the median GPU time of the graphs replayed >= 100 times (D / T graph timing).
  ATDEV_LL1_DUMP=d  diagnostic: after the CUDA graph capture, write every Inductor CachingAutotuner kernel (name,
                    source, launcher configs, generated-module path) of this process to d/kernels.json and copy the
                    generated modules to d/ (to emulate Inductor kernels inside fused ones).
Requires drafter_ctxfold.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import shutil

import torch

OPT_NAME = "ll1_fuse"
log = logging.getLogger("atdev_serve")

CTX = os.environ.get("ATDEV_LL1_CTX", "0") == "1"
CBUF = os.environ.get("ATDEV_LL1_CBUF", "0") == "1"
KPC = os.environ.get("ATDEV_LL1_KPC", "0") == "1"
SEL = os.environ.get("ATDEV_LL1_SEL", "0") == "1"
QKR = os.environ.get("ATDEV_LL1_QKR", "0") == "1"
TAIL = os.environ.get("ATDEV_LL1_TAIL", "0") == "1"
TAIL_CHUNK = os.environ.get("ATDEV_LL1_TAIL_CHUNK", "0") == "1"  # CUDA warp-per-chunk stage 1 (else Triton)
TAIL_V = int(os.environ.get("ATDEV_LL1_TAIL_V", "2"))  # 2: warp per (row, candidate); 1: CTA per row
_TAIL_BUF: dict = {}
QKG = os.environ.get("ATDEV_LL1_QKG", "0") == "1"
DUMP = os.environ.get("ATDEV_LL1_DUMP", "")
GTIME = int(os.environ.get("ATDEV_LL1_GTIME", "0"))

_EXT = None
_KEXT = None
_N = {"ctx": 0, "cbuf": 0, "kpc": 0}


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="atdev_ll1_ctx", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll1_ctx.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _EXT


def kext():
    global _KEXT
    if _KEXT is None:
        from torch.utils.cpp_extension import load
        _KEXT = load(name="atdev_ll1_kpc", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll1_kpc.cu")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _KEXT


_KPC_BUF: dict = {}


def kpc_run(x: torch.Tensor, w: torch.Tensor, base: torch.Tensor, block_size: int,
            out: torch.Tensor | None) -> torch.Tensor:
    """bf16 kernel_projection(x) [16, 1280] (returned) + side-0 grouped conv into ``out`` (bitwise == stock)."""
    dev = x.device
    b = _KPC_BUF.get(dev)
    if b is None:
        b = (torch.empty(5 * 16 * 1280, dtype=torch.float32, device=dev), torch.zeros(64, dtype=torch.int32, device=dev))
        _KPC_BUF[dev] = b
    kp = torch.empty(16, 1280, dtype=x.dtype, device=dev)
    kext().kpc(x, w, base, b[0], b[1], kp, out, block_size)
    return kp


def kpc_ok(x: torch.Tensor, w: torch.Tensor, base: torch.Tensor, taps: int, group_size: int) -> bool:
    return (x.is_cuda and x.dim() == 2 and x.shape[0] == 16 and x.shape[1] == 5120 and x.stride(1) == 1
            and x.stride(0) % 8 == 0 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and tuple(w.shape) == (1280, 5120) and w.is_contiguous() and taps == 2 and group_size == 16
            and base.dtype == torch.bfloat16 and tuple(base.shape) == (2, 5120) and base.stride(1) == 1)


# ---------------------------------------------------------------------------------------------------------------
# LL1-ctx: fused context K/V chain
# ---------------------------------------------------------------------------------------------------------------


def ctx_kv_fused(qkv_ctx: torch.Tensor, q_size: int, kv_size: int, nkv: int, k_norm_w: torch.Tensor, eps: float,
                 pos: torch.Tensor, cos_sin: torch.Tensor, slots: torch.Tensor, kv_cache: torch.Tensor,
                 head_size: int) -> None:
    """== k-norm / rope / reshape_and_cache_flash of drafter_ctxfold._ctx_kv_store (bitwise)."""
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    ext().ctx_kv(qkv_ctx, q_size, q_size + kv_size, nkv, k_norm_w, eps, pos, cos_sin, slots, key_cache, value_cache)


def _ctx_ok(layer, m) -> bool:
    cs = m._rope_cos_sin_cache
    return (layer.head_dim == 128 and layer.v_scale is None and m._rope_is_neox and m._rope_head_size == 128
            and cs.dtype == torch.bfloat16 and cs.shape[1] == 128 and cs.is_contiguous()
            and layer.k_norm.weight.dtype == torch.bfloat16 and layer.attn.kv_cache_dtype in ("auto", "bfloat16")
            and 16 * layer.num_kv_heads <= 1024)


def _install_ctx() -> None:
    from atdev_serve import drafter_ctxfold as cf

    orig = cf._ctx_kv_store
    if getattr(orig, "_atdev_ll1", False):
        return

    def _ctx_kv_store(layer, qkv_ctx):
        m = cf.S.model
        if not _ctx_ok(layer, m):
            return orig(layer, qkv_ctx)
        kv = layer.attn.kv_cache
        if isinstance(kv, (list, tuple)):
            return orig(layer, qkv_ctx)
        _N["ctx"] += 1
        if _N["ctx"] == 1:
            log.warning("atdev %s: LL1-ctx fused context K/V kernel ENGAGED", OPT_NAME)
        ctx_kv_fused(qkv_ctx, layer.q_size, layer.kv_size, layer.num_kv_heads, layer.k_norm.weight,
                     m._rms_norm_eps, cf.S.ctx_pos, m._rope_cos_sin_cache, cf.S.ctx_slots[layer.layer_name], kv,
                     layer.head_dim)

    _ctx_kv_store._atdev_ll1 = True
    cf._ctx_kv_store = _ctx_kv_store


# ---------------------------------------------------------------------------------------------------------------
# LL1-cbuf: attention_conv.prepare output written into the fold buffer
# ---------------------------------------------------------------------------------------------------------------


def _conv_buf(x: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, block_size: int,
              group_size: int) -> torch.Tensor:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    from atdev_serve import drafter_ctxfold as cf

    S = cf.S
    rows = S.rows
    if not (S.fold and S.buf is not None and x.shape[0] == rows and x.shape[1] == S.buf.shape[1]
            and x.dtype == S.buf.dtype):
        return q2.dflash2_grouped_conv_impl(x, delta, base, block_size, group_size)
    out = S.buf[rows:]
    num_rows, num_channels = x.shape
    element_block = 1024 if num_rows >= 128 and num_channels % 1024 == 0 else 512
    grid = (num_rows * q2.triton.cdiv(num_channels, element_block),)
    q2._dflash2_grouped_conv_kernel[grid](
        x, delta, base, out, x.stride(0), delta.stride(0), delta.stride(1), base.stride(0), out.stride(0),
        NUM_CHANNELS=num_channels, BLOCK_SIZE=block_size, GROUP_SIZE=group_size, TAPS=base.shape[0],
        ELEMENT_BLOCK=element_block, num_warps=4)
    _N["cbuf"] += 1
    if _N["cbuf"] == 1:
        log.warning("atdev %s: LL1-cbuf conv -> fold buffer ENGAGED", OPT_NAME)
    return out


def _conv_buf_fake(x, delta, base, block_size, group_size):
    return torch.empty_like(x)


def _fold_out(x: torch.Tensor, use_buf: bool) -> torch.Tensor | None:
    from atdev_serve import drafter_ctxfold as cf

    S = cf.S
    if (use_buf and S.fold and S.buf is not None and x.shape[0] == S.rows and x.shape[1] == S.buf.shape[1]
            and x.dtype == S.buf.dtype):
        return S.buf[S.rows:]
    return None


def _kpc_op(x: torch.Tensor, w: torch.Tensor, base: torch.Tensor, block_size: int,
            use_buf: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.models import qwen3_dflash2 as q2

    out = _fold_out(x, use_buf)
    if _N.get("kpc_dbg", 0) < 4:
        _N["kpc_dbg"] = _N.get("kpc_dbg", 0) + 1
        log.warning("atdev %s: kpc op x %s %s %s w %s %s base %s %s %s fold_out %s ok %s", OPT_NAME, tuple(x.shape),
                    x.stride(), x.dtype, tuple(w.shape), w.dtype, tuple(base.shape), base.stride(), base.dtype,
                    out is not None, kpc_ok(x, w, base, 2, 16))
    if not kpc_ok(x, w, base, 2, 16):  # not the c=1 block (eager prefill, nr > 1): stock kernels
        kp = torch.nn.functional.linear(x, w)
        coefficients = kp.reshape(x.shape[0], 2, 2, w.shape[0] // 4)
        if out is None:
            return q2.dflash2_grouped_conv_impl(x, coefficients[:, 0], base, block_size, 16), kp
        return _conv_buf(x, coefficients[:, 0], base, block_size, 16), kp
    if out is None:
        out = torch.empty_like(x)
    elif _N["cbuf"] == 0:
        _N["cbuf"] = 1
        log.warning("atdev %s: LL1-cbuf conv -> fold buffer ENGAGED (kpc)", OPT_NAME)
    kp = kpc_run(x, w, base, block_size, out)
    _N["kpc"] += 1
    if _N["kpc"] == 1:
        log.warning("atdev %s: LL1-kpc fused kernel_projection + split-K + conv ENGAGED", OPT_NAME)
    return out, kp


def _kpc_op_fake(x, w, base, block_size, use_buf):
    return torch.empty_like(x), x.new_empty(x.shape[0], w.shape[0])


def _install_cbuf() -> None:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    from vllm.utils.torch_utils import direct_register_custom_op

    if getattr(q2.DFlash2Qwen3DecoderLayer.__init__, "_atdev_ll1", False):
        return
    if CBUF and not hasattr(torch.ops.vllm, "atdev_ll1_conv_buf"):
        direct_register_custom_op(op_name="atdev_ll1_conv_buf", op_func=_conv_buf, fake_impl=_conv_buf_fake,
                                  tags=(torch.Tag.needs_fixed_stride_order,))
    if KPC and not hasattr(torch.ops.vllm, "atdev_ll1_kpc"):
        direct_register_custom_op(op_name="atdev_ll1_kpc", op_func=_kpc_op, fake_impl=_kpc_op_fake,
                                  tags=(torch.Tag.needs_fixed_stride_order,))
    orig_init = q2.DFlash2Qwen3DecoderLayer.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.attention_conv._atdev_cbuf = CBUF
        for cv in (self.attention_conv, self.mlp_conv):
            w = cv.kernel_projection.weight
            cv._atdev_kpc = bool(KPC and w.dtype == torch.bfloat16 and tuple(w.shape) == (1280, 5120)
                                 and cv.taps == 2 and cv.group_size == 16 and cv.base_kernel.dtype == torch.bfloat16)
            if KPC and not _N.get("kpc_init"):
                _N["kpc_init"] = 1
                log.warning("atdev %s: kpc layer check: %s (w %s %s, taps %d, group %d, base %s)", OPT_NAME,
                            cv._atdev_kpc, getattr(w, "dtype", None), tuple(getattr(w, "shape", ())), cv.taps,
                            cv.group_size, cv.base_kernel.dtype)

    __init__._atdev_ll1 = True
    q2.DFlash2Qwen3DecoderLayer.__init__ = __init__

    orig_prep = q2.DFlashGroupedConv.prepare

    def prepare(self, hidden_states):
        w = self.kernel_projection.weight
        if getattr(self, "_atdev_kpc", False) and hidden_states.is_cuda:
            # static checks only (the drafter is compiled for a symbolic row count): rows != 16 -> stock kernels
            # inside the op
            out, kp = torch.ops.vllm.atdev_ll1_kpc(hidden_states, w, self.base_kernel[0], self.block_size,
                                                   bool(getattr(self, "_atdev_cbuf", False)))
            coefficients = kp.reshape(hidden_states.shape[0], 2, self.taps, self.num_groups)
            return out, coefficients[:, 1]
        if not getattr(self, "_atdev_cbuf", False) or not hidden_states.is_cuda:
            return orig_prep(self, hidden_states)
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups)
        out = torch.ops.vllm.atdev_ll1_conv_buf(hidden_states, coefficients[:, 0], self.base_kernel[0],
                                                self.block_size, self.group_size)
        return out, coefficients[:, 1]

    q2.DFlashGroupedConv.prepare = prepare


# ---------------------------------------------------------------------------------------------------------------
# LL1-sel: fused candidate selector
# ---------------------------------------------------------------------------------------------------------------
_SEXT = None
_SEL_BUF: dict = {}


def sext():
    global _SEXT
    if _SEXT is None:
        from torch.utils.cpp_extension import load
        _SEXT = load(name="atdev_ll1_sel", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll1_sel.cu")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _SEXT


def sel_run(cand, unary, hidden, anchor, w, pcb, scb) -> torch.Tensor:
    """scores f32 [1, 15, 16, 16] == stock CandidateSelector (bitwise)."""
    dev = hidden.device
    b = _SEL_BUF.get(dev)
    if b is None:
        b = (torch.empty(16 * 15 * 256, dtype=torch.float32, device=dev),
             torch.empty(15 * 256, dtype=torch.bfloat16, device=dev), torch.zeros(16, dtype=torch.int32, device=dev))
        _SEL_BUF[dev] = b
    out = torch.empty(1, 15, 16, 16, dtype=torch.float32, device=dev)
    sext().sel(hidden.reshape(15, -1), w, pcb, scb, cand.reshape(-1), unary.reshape(-1), anchor, b[0], b[1], b[2],
               out)
    return out


def _sel_eligible(cand, unary, hidden, anchor, w, pcb, scb) -> bool:
    return (cand.dim() == 3 and tuple(cand.shape) == (1, 15, 16) and cand.dtype == torch.int64 and cand.is_contiguous()
            and unary.dtype == torch.float32 and unary.is_contiguous() and tuple(unary.shape) == (1, 15, 16)
            and hidden.dtype == torch.bfloat16 and tuple(hidden.shape) == (1, 15, 5120) and hidden.stride(-1) == 1
            and hidden.stride(1) % 8 == 0 and anchor.dtype == torch.int32 and anchor.numel() == 1
            and tuple(w.shape) == (256, 5120) and w.is_contiguous() and pcb.shape[1] == 256 and pcb.is_contiguous()
            and scb.is_contiguous() and sext().sel_ok())


def _sel_op(cand: torch.Tensor, unary: torch.Tensor, hidden: torch.Tensor, anchor: torch.Tensor, w: torch.Tensor,
            pcb: torch.Tensor, scb: torch.Tensor) -> torch.Tensor:
    if not _sel_eligible(cand, unary, hidden, anchor, w, pcb, scb):
        from vllm.model_executor.models import qwen3_dflash2 as q2
        hid = torch.nn.functional.linear(hidden, w)
        return q2._score_edges(pcb, scb, cand, unary, hid, anchor, cand.shape[-1]).float()
    _N.setdefault("sel", 0)
    _N["sel"] += 1
    if _N["sel"] == 1:
        log.warning("atdev %s: LL1-sel fused candidate selector ENGAGED", OPT_NAME)
    return sel_run(cand, unary, hidden, anchor, w, pcb, scb)


def _sel_fake(cand, unary, hidden, anchor, w, pcb, scb):
    return unary.new_empty(*cand.shape, cand.shape[-1])


def _install_sel() -> None:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    from vllm.utils.torch_utils import direct_register_custom_op

    cls = q2.CandidateSelector
    if getattr(cls.forward, "_atdev_ll1", False):
        return
    if not hasattr(torch.ops.vllm, "atdev_ll1_sel"):
        direct_register_custom_op(op_name="atdev_ll1_sel", op_func=_sel_op, fake_impl=_sel_fake)
    orig = cls.forward

    def forward(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        if self.top_k != 16 or self.predecessor_codebook.dtype != torch.bfloat16:
            return orig(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids)
        return torch.ops.vllm.atdev_ll1_sel(candidate_ids, unary_logits, hidden_states, anchor_token_ids,
                                            self.hidden_projection.weight, self.predecessor_codebook,
                                            self.successor_codebook)

    forward._atdev_ll1 = True
    cls.forward = forward


# ---------------------------------------------------------------------------------------------------------------
# LL1-qkr: q/k norm + rope + block KV insert
# ---------------------------------------------------------------------------------------------------------------


def qkr_run(qkv, positions, qw, kw, cs, slots, kv_cache, head_size):
    """(q [T, 4096], k [T, 1024]) == the Inductor q/k norm + rope kernel; block K / V written to the cache."""
    from atdev_serve.ll1_qkr import _qkr_kernel

    t = qkv.shape[0]
    q = torch.empty(t, 4096, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(t, 1024, dtype=qkv.dtype, device=qkv.device)
    xn0, xn1 = t * 32, t * 8
    grid = (triton_cdiv(xn0, 8) + triton_cdiv(xn1, 8),)
    if slots is None:
        kc = vc = q  # never stored: n_slots = 0
        ns, bs_, ps, hs, bsz = 0, 0, 0, 0, 1
    else:
        kc, vc = kv_cache.transpose(1, 2).split(head_size, dim=-1)
        ns, bs_, ps, hs, bsz = slots.shape[0], kc.stride(0), kc.stride(1), kc.stride(2), kc.shape[1]
    _qkr_kernel[grid](qkv, qw, positions, cs, kw, q, k, xn0, xn1, slots if slots is not None else positions, ns,
                      kc, vc, bs_, ps, hs, bsz, XBLOCK=8, R0_BLOCK=64, num_warps=8, num_stages=1)
    return q, k


def triton_cdiv(a, b):
    return (a + b - 1) // b


def _qkr_ok(qkv) -> bool:
    return (qkv.dim() == 2 and qkv.shape[1] == 6144 and qkv.stride(0) == 6144 and qkv.stride(1) == 1
            and qkv.dtype == torch.bfloat16)


def _qkr_op(qkv: torch.Tensor, positions: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, cs: torch.Tensor,
            layer_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, attn_layer, kv_cache, slots = get_attention_context(layer_name)
    if not _qkr_ok(qkv):
        raise RuntimeError(f"atdev {OPT_NAME}: qkr layout {tuple(qkv.shape)} {qkv.stride()}")
    q, k = qkr_run(qkv, positions, qw, kw, cs, slots, kv_cache, attn_layer.impl.head_size)
    if slots is not None:
        _N.setdefault("qkr", 0)
        _N["qkr"] += 1
        if _N["qkr"] == 1:
            log.warning("atdev %s: LL1-qkr q/k norm + rope + block KV insert ENGAGED", OPT_NAME)
    return q, k, qkv.new_empty(0)


def _qkr_fake(qkv, positions, qw, kw, cs, layer_name):
    t = qkv.shape[0]
    return qkv.new_empty(t, 4096), qkv.new_empty(t, 1024), qkv.new_empty(0)


def _install_qkr() -> None:
    from vllm.model_executor.layers.attention.attention import _encode_layer_name
    from vllm.model_executor.models import qwen3_dflash as qd
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "atdev_ll1_qkr"):
        direct_register_custom_op(op_name="atdev_ll1_qkr", op_func=_qkr_op, fake_impl=_qkr_fake)
    base_fwd = qd.DFlashQwen3Attention.forward
    if not getattr(base_fwd, "_atdev_ctxfold", False):
        raise RuntimeError(f"atdev {OPT_NAME}: LL1-qkr expects drafter_ctxfold's attention forward")

    def forward(self, positions, hidden_states):
        attn = self.attn
        ok = (self.head_dim == 128 and self.q_size == 4096 and self.kv_size == 1024 and self.v_scale is None
              and attn.kv_sharing_target_layer_name is None and attn.query_quant is None
              and not attn.attn_backend.forward_includes_kv_cache_update
              and attn.kv_cache_dtype in ("auto", "bfloat16") and self.rotary_emb.is_neox_style
              and self.rotary_emb.cos_sin_cache.dtype == torch.bfloat16
              and tuple(self.rotary_emb.cos_sin_cache.shape[1:]) == (128,))
        if not ok:
            return base_fwd(self, positions, hidden_states)
        qkv = torch.ops.vllm.atdev_ctxfold_qkv(hidden_states, self.layer_name)
        v = qkv[:, self.q_size + self.kv_size:]
        q, k, dep = torch.ops.vllm.atdev_ll1_qkr(qkv, positions, self.q_norm.weight, self.k_norm.weight,
                                                 self.rotary_emb.cos_sin_cache, attn.layer_name)
        # == Attention.forward minus the KV cache update (done by the qkr kernel above)
        num_tokens = q.shape[0]
        output = torch.empty((num_tokens, attn.num_heads * attn.head_size_v), dtype=q.dtype, device=q.device)
        hidden_size = output.shape[-1]
        query = q.view(-1, attn.num_heads, attn.head_size)
        output = output.view(-1, attn.num_heads, attn.head_size_v)
        key = k.view(-1, attn.num_kv_heads, attn.head_size)
        value = v.view(-1, attn.num_kv_heads, attn.head_size_v)
        torch.ops.vllm.unified_attention_with_output(query, key, value, output, _encode_layer_name(attn.layer_name),
                                                     kv_cache_dummy_dep=dep)
        attn_output = output.view(-1, hidden_size)
        out, _ = self.o_proj(attn_output)
        return out

    forward._atdev_ctxfold = True
    forward._atdev_ll1 = True
    qd.DFlashQwen3Attention.forward = forward


# ---------------------------------------------------------------------------------------------------------------
# LL1-tail: draft-vocab top-k merge + rescoring + sort
# ---------------------------------------------------------------------------------------------------------------
_TEXT = None


def text():
    global _TEXT
    if _TEXT is None:
        from torch.utils.cpp_extension import load
        _TEXT = load(name="atdev_ll1_tail", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll1_tail.cu")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _TEXT


def tail_run(approx: torch.Tensor, ids: torch.Tensor, weight: torch.Tensor, hidden: torch.Tensor):
    """(full-vocab ids int64 [M, 16], float32 values [M, 16]) == draft_vocab TOPKX=1 path + .float() (bitwise)."""
    from atdev_serve import draft_vocab as dv

    m, n = approx.shape
    nc = dv.triton.cdiv(n, dv._XCH)
    cc = torch.empty((m, nc * 16), dtype=torch.int32, device=approx.device)
    oid = torch.empty((m, 16), dtype=torch.int64, device=approx.device)
    oval = torch.empty((m, 16), dtype=torch.float32, device=approx.device)
    if m:
        if TAIL_CHUNK:
            text().chunk(approx, cc)
        else:
            dv._topkx_chunk_kernel[(m, nc)](approx, approx.stride(0), cc, n, nc, S=16, CH=dv._XCH, num_warps=4)
        if TAIL_V == 2:
            b = _TAIL_BUF.get(approx.device)
            if b is None or b[1].numel() < m:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(f"atdev {OPT_NAME}: tail scratch must exist before capture")
                rows = max(m, 8192)
                b = (torch.empty(rows * 16, dtype=torch.bfloat16, device=approx.device),
                     torch.zeros(rows, dtype=torch.int32, device=approx.device))
                _TAIL_BUF[approx.device] = b
            text().tail2(cc, ids, weight, hidden, oid, oval, b[0], b[1])
        else:
            text().tail(cc, ids, weight, hidden, oid, oval)
    return oid, oval


def _install_tail() -> None:
    from atdev_serve import draft_vocab as dv

    cls = dv.DraftVocabHead
    if getattr(cls.top_k, "_atdev_ll1", False):
        return
    orig = cls.top_k

    def top_k(self, hidden, k):
        if not (self.shadow is not None and k == 16 and dv._TOPKX == 1 and self.cfg.screen == 16
                and self.weight.dtype == torch.bfloat16 and self.weight.stride(-1) == 1
                and hidden.dtype == torch.bfloat16 and hidden.dim() == 2 and hidden.stride(-1) == 1
                and self.ids.numel() <= 65536):
            return orig(self, hidden, k)
        approx = self.shadow.approx_logits(hidden)
        if not dv.topkx_ok(approx, 16) or dv.triton.cdiv(approx.shape[1], dv._XCH) * 16 > 2048:
            raise RuntimeError(f"atdev {OPT_NAME}: LL1-tail approx layout {tuple(approx.shape)}")
        _N.setdefault("tail", 0)
        _N["tail"] += 1
        if _N["tail"] == 1:
            log.warning("atdev %s: LL1-tail fused top-k merge + rescore + sort ENGAGED", OPT_NAME)
        return tail_run(approx, self.ids, self.weight, hidden)

    top_k._atdev_ll1 = True
    cls.top_k = top_k


# ---------------------------------------------------------------------------------------------------------------
# diagnostic dump of the Inductor kernels
# ---------------------------------------------------------------------------------------------------------------


def dump_inductor(d: str) -> None:
    import gc
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    os.makedirs(d, exist_ok=True)
    out, mods = [], {}
    for o in gc.get_objects():
        try:
            if not isinstance(o, CachingAutotuner):
                continue
        except Exception:
            continue
        try:
            fn = o.fn
            src = getattr(fn, "src", None)
            pyfn = getattr(fn, "fn", None)
            mod = getattr(getattr(pyfn, "__code__", None), "co_filename", None)
            cfgs = [str(l.config) for l in (o.launchers or [])]
            out.append({"name": getattr(fn, "__name__", "?"), "src": src, "launchers": cfgs,
                        "configs": [str(c) for c in (o.configs or [])], "module": mod,
                        "heuristic": str(getattr(o, "heuristic_type", "")),
                        "triton_meta": str(getattr(o, "triton_meta", ""))[:2000],
                        "inductor_meta": str(getattr(o, "inductor_meta", ""))[:2000]})
            if mod and os.path.exists(mod) and mod not in mods:
                mods[mod] = os.path.join(d, f"mod{len(mods):03d}_" + os.path.basename(mod))
                shutil.copy(mod, mods[mod])
        except Exception as e:  # pragma: no cover - diagnostic
            out.append({"error": repr(e)})
    with open(os.path.join(d, "kernels.json"), "w") as f:
        json.dump({"kernels": out, "modules": mods}, f, indent=1)
    log.warning("atdev %s: dumped %d Inductor kernels (%d modules) to %s", OPT_NAME, len(out), len(mods), d)


def _install_dump() -> None:
    from vllm.v1.worker.gpu import model_runner as mr

    cls = mr.GPUModelRunner
    orig = cls.capture_model
    if getattr(orig, "_atdev_ll1", False):
        return

    @functools.wraps(orig)
    def capture_model(self, *a, **kw):
        r = orig(self, *a, **kw)
        try:
            dump_inductor(DUMP)
        except Exception as e:  # pragma: no cover
            log.warning("atdev %s: dump failed: %r", OPT_NAME, e)
        return r

    capture_model._atdev_ll1 = True
    cls.capture_model = capture_model


def _install_gtime() -> None:
    import statistics

    G = torch.cuda.CUDAGraph
    if getattr(G.replay, "_atdev_ll1", False):
        return
    orig = G.replay
    pend: list = []
    acc: dict = {}
    cnt: dict = {}

    def replay(self):
        if torch.cuda.is_current_stream_capturing():
            return orig(self)
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        r = orig(self)
        b.record()
        pend.append((id(self), a, b))
        while pend and pend[0][2].query():
            k, x, y = pend.pop(0)
            acc.setdefault(k, []).append(x.elapsed_time(y) * 1000.0)
            cnt[k] = cnt.get(k, 0) + 1
            if cnt[k] % GTIME == 0 and len(acc[k]) >= 100:
                v = acc[k]
                log.warning("atdev %s GTIME: graph %x replays %d: median %.1f us mean %.1f us (last %d)", OPT_NAME,
                            k & 0xffffff, cnt[k], statistics.median(v), statistics.mean(v), len(v))
                acc[k] = []
        return r

    replay._atdev_ll1 = True
    G.replay = replay


def register() -> None:
    from atdev_serve.register import enabled

    if "drafter_ctxfold" not in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: requires ATDEV_OPT drafter_ctxfold")
    if os.environ.get("ATDEV_LL1_RECOMPILE"):  # diagnostic: force a fresh torch.compile (e.g. TORCH_LOGS=output_code)
        from atdev_serve import compile_cache
        compile_cache._SALTS["ll1_recompile"] = os.environ["ATDEV_LL1_RECOMPILE"]
    if CTX:
        ext()  # build / load before any CUDA graph capture
        _install_ctx()
    if KPC:
        kext()
    if CBUF or KPC:
        _install_cbuf()
    if SEL:
        sext()
        _install_sel()
    if QKR:
        _install_qkr()
    if TAIL:
        text()
        _install_tail()
    if QKG:
        from vllm.model_executor.models import qwen3_next as q3n
        from atdev_serve import ll1_qkg
        q3n.fused_qk_rmsnorm_rope_gate = ll1_qkg.fused_qk_rmsnorm_rope_gate_ll1
    if DUMP:
        _install_dump()
    if GTIME:
        _install_gtime()
    log.warning("atdev %s: registered (ctx=%d cbuf=%d kpc=%d sel=%d qkr=%d tail=%d qkg=%d dump=%s)", OPT_NAME, CTX,
                CBUF, KPC, SEL, QKR, TAIL, QKG,
                DUMP or "-")
