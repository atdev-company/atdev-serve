"""ll1_fuse: bit-identical fusions of the c=1 DFlash2 drafter graph glue (ATDEV_OPT=ll1_fuse, requires
drafter_ctxfold):

  ctx   drafter_ctxfold context-K/V chain of a layer (k copy + vLLM rms_norm K-norm + vLLM rotary +
        reshape_and_cache_flash, 4 launches) as ONE CUDA kernel (csrc/ll1_ctx.cu) that reproduces each step's
        arithmetic exactly.
  cbuf  the attention_conv.prepare grouped conv writes its output straight into drafter_ctxfold's block rows
        buf[rows:] (the fold's qkv op then copies onto itself = torch's same-data no-op), so the per-layer memcpy
        into the Marlin m32 tile disappears. Same Triton conv kernel -> identical.
  kpc   DFlashGroupedConv.prepare at the c=1 block (16 rows): kernel_projection GEMM + split-K reduce + side-0
        grouped conv as ONE kernel (csrc/ll1_kpc.cu) emulating cuBLAS' split-K (S = 5) mma order and the Triton
        conv arithmetic exactly (the attention conv writes into the fold buffer).
  sel   DFlash2 CandidateSelector at c=1 (hidden_projection split-K GEMM + reduce + codebook gathers + bf16 mul +
        einsum + unary add, 6 launches) as ONE kernel (csrc/ll1_sel.cu, cuBLAS / CUTLASS mma order emulated;
        batch != 1 -> stock ops inside the op).
  tail  draft-vocab top-k tail (k = screen = 16): merge of the per-chunk candidates + full-head row gather + exact
        bf16 rescoring (cuBLAS gemvx order emulated) + sorted top-16 (torch.topk(sorted) emulated in registers) +
        ids gather + float, one warp per (row, candidate) (csrc/ll5_tail.cu) after the stock _topkx_chunk_kernel.
  qkg   target full-attention layers: vLLM's fused q/k RMSNorm + RoPE + gate kernel with its independent loads
        (positions, cos/sin, rotary halves, weights, gate copy) issued before the norm (ll1_qkg.py, arithmetic
        verbatim -> bit-identical).
"""
from __future__ import annotations

import functools
import logging
import os

import torch

OPT_NAME = "ll1_fuse"
log = logging.getLogger("atdev_serve")

_TAIL_BUF: dict = {}
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
# ctx: fused context K/V chain
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
            log.warning("atdev %s: fused context K/V kernel ENGAGED", OPT_NAME)
        ctx_kv_fused(qkv_ctx, layer.q_size, layer.kv_size, layer.num_kv_heads, layer.k_norm.weight,
                     m._rms_norm_eps, cf.S.ctx_pos, m._rope_cos_sin_cache, cf.S.ctx_slots[layer.layer_name], kv,
                     layer.head_dim)

    _ctx_kv_store._atdev_ll1 = True
    cf._ctx_kv_store = _ctx_kv_store


# ---------------------------------------------------------------------------------------------------------------
# cbuf / kpc: attention_conv.prepare output written into the fold buffer
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
        log.warning("atdev %s: conv -> fold buffer ENGAGED", OPT_NAME)
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
        log.warning("atdev %s: conv -> fold buffer ENGAGED (kpc)", OPT_NAME)
    kp = kpc_run(x, w, base, block_size, out)
    _N["kpc"] += 1
    if _N["kpc"] == 1:
        log.warning("atdev %s: fused kernel_projection + split-K + conv ENGAGED", OPT_NAME)
    return out, kp


def _kpc_op_fake(x, w, base, block_size, use_buf):
    return torch.empty_like(x), x.new_empty(x.shape[0], w.shape[0])


def _install_cbuf() -> None:
    from vllm.model_executor.models import qwen3_dflash2 as q2
    from vllm.utils.torch_utils import direct_register_custom_op

    if getattr(q2.DFlash2Qwen3DecoderLayer.__init__, "_atdev_ll1", False):
        return
    if not hasattr(torch.ops.vllm, "atdev_ll1_conv_buf"):
        direct_register_custom_op(op_name="atdev_ll1_conv_buf", op_func=_conv_buf, fake_impl=_conv_buf_fake,
                                  tags=(torch.Tag.needs_fixed_stride_order,))
    if not hasattr(torch.ops.vllm, "atdev_ll1_kpc"):
        direct_register_custom_op(op_name="atdev_ll1_kpc", op_func=_kpc_op, fake_impl=_kpc_op_fake,
                                  tags=(torch.Tag.needs_fixed_stride_order,))
    orig_init = q2.DFlash2Qwen3DecoderLayer.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.attention_conv._atdev_cbuf = True
        for cv in (self.attention_conv, self.mlp_conv):
            w = cv.kernel_projection.weight
            cv._atdev_kpc = bool(w.dtype == torch.bfloat16 and tuple(w.shape) == (1280, 5120)
                                 and cv.taps == 2 and cv.group_size == 16 and cv.base_kernel.dtype == torch.bfloat16)

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
# sel: fused candidate selector
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
        log.warning("atdev %s: fused candidate selector ENGAGED", OPT_NAME)
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
# tail: draft-vocab top-k merge + rescoring + sort
# ---------------------------------------------------------------------------------------------------------------
_TEXT = None


def text():
    global _TEXT
    if _TEXT is None:
        from torch.utils.cpp_extension import load
        _TEXT = load(name="atdev_ll5_tail", sources=[os.path.join(os.path.dirname(__file__), "csrc", "ll5_tail.cu")],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"], verbose=False)
    return _TEXT


def tail_run(approx: torch.Tensor, ids: torch.Tensor, weight: torch.Tensor, hidden: torch.Tensor):
    """(full-vocab ids int64 [M, 16], float32 values [M, 16]) == draft_vocab's top-k path + .float() (bitwise)."""
    from atdev_serve import draft_vocab as dv

    m, n = approx.shape
    nc = dv.triton.cdiv(n, dv._XCH)
    cc = torch.empty((m, nc * 16), dtype=torch.int32, device=approx.device)
    oid = torch.empty((m, 16), dtype=torch.int64, device=approx.device)
    oval = torch.empty((m, 16), dtype=torch.float32, device=approx.device)
    if m:
        dv._topkx_chunk_kernel[(m, nc)](approx, approx.stride(0), cc, n, nc, S=16, CH=dv._XCH, num_warps=4)
        b = _TAIL_BUF.get(approx.device)
        if b is None or b[1].numel() < m:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"atdev {OPT_NAME}: tail scratch must exist before capture")
            rows = max(m, 8192)
            b = (torch.empty(rows * 16, dtype=torch.bfloat16, device=approx.device),
                 torch.zeros(rows, dtype=torch.int32, device=approx.device))
            _TAIL_BUF[approx.device] = b
        text().tail2(cc, ids, weight, hidden, oid, oval, b[0], b[1])
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
            raise RuntimeError(f"atdev {OPT_NAME}: tail approx layout {tuple(approx.shape)}")
        _N.setdefault("tail", 0)
        _N["tail"] += 1
        if _N["tail"] == 1:
            log.warning("atdev %s: fused top-k merge + rescore + sort ENGAGED", OPT_NAME)
        return tail_run(approx, self.ids, self.weight, hidden)

    top_k._atdev_ll1 = True
    cls.top_k = top_k


def register() -> None:
    from vllm.model_executor.models import qwen3_next as q3n
    from atdev_serve import ll1_qkg
    from atdev_serve.register import enabled

    if "drafter_ctxfold" not in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: requires ATDEV_OPT drafter_ctxfold")
    ext()  # build / load before any CUDA graph capture
    _install_ctx()
    kext()
    _install_cbuf()
    sext()
    _install_sel()
    text()
    _install_tail()
    q3n.fused_qk_rmsnorm_rope_gate = ll1_qkg.fused_qk_rmsnorm_rope_gate_ll1
    log.warning("atdev %s: registered (ctx, cbuf, kpc, sel, tail, qkg)", OPT_NAME)
