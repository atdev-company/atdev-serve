"""Context-KV fold for the DFlash2 drafter (ATDEV_OPT=drafter_ctxfold), P7-e.

Stock (vLLM 0.31.0, MRV2 ``DFlashSpeculator``): the drafter CUDA graph for one request first
runs ``precompute_and_store_context_kv`` on the step's target rows -- with ``dflash_tree`` that is
N = 16 rows (one tree verify), of which at most D + 1 = 8 (accepted path + bonus) are valid, the
rest carry PAD slots -- as 5 extra Marlin ``qkv_proj`` passes (M = 16, 32 us each in situ: the q
columns are computed and thrown away), a stack/permute copy, a grouped K-norm, RoPE and 5 cache
inserts; then the 8 query rows run the 5 layers (``qkv_proj`` at M = 8).

Here the nr = 1 graph instead carries the <= 8 valid context rows *inside* each layer's block
``qkv_proj`` call: rows [0, 8) = hidden_norm(context), rows [8, 16) = the block -> one Marlin m16
tile (``results/p7-drafter-micro.txt``: M = 16 costs +3 us over M = 8 and rows are bit-identical
to the separate calls, since Marlin's per-row math does not depend on the other rows). The context
K/V of layer i come out of layer i's call, get the layer's K-norm (``ops.rms_norm``, the kernel
the stock grouped K-norm runs per row), RoPE (``ops.rotary_embedding``, same kernel) and the
FlashAttention cache insert (``do_kv_cache_update`` -> ``reshape_and_cache_flash``, same kernel)
before the layer's attention reads them -- so the KV cache content is bit-identical to stock.
The graph's context rows shrink from 16 to 8 (``_num_graph_context_tokens``); rows of a prefill
beyond 8 still take the stock eager path, and rows 8..15 of a 16-row verify step (always rejected
or junk) are skipped. nr > 1 graphs and every eager path keep the stock precompute.

Mechanism: ``DFlashQwen3Attention.forward`` calls the opaque custom op ``atdev_ctxfold_qkv`` in
place of ``self.qkv_proj`` (vLLM compiles the drafter once and bypasses Dynamo guards, so the
stock/fold decision cannot be a Python branch inside the traced forward); the op reads a runtime
flag that the speculator sets while capturing the nr = 1 graph. Requires ``dflash_tree``
(accepted tokens per step <= D + 1 and N <= 2 (D + 1) hold for both its tree and chain steps).

Env:
  ATDEV_CTXFOLD_FORK    1: inside the drafter graph capture the context chain of each layer runs on a forked
                        stream (joined by that layer's FlashAttention forward), overlapping the block rows'
                        q/k norm + RoPE + cache insert. Same kernels / inputs -> bit-identical (C1-agf)
  ATDEV_CTXFOLD_VERIFY  N > 0: every N proposals read the folded context K/V back from the cache
                        and compare bitwise with a stock eager recompute (logs; slow, for tests)
"""
from __future__ import annotations

import functools
import logging
import os
import types

import torch

OPT_NAME = "drafter_ctxfold"
SUPPORTED_VLLM_VERSION = "0.31.0"
log = logging.getLogger("atdev_serve")


class _State:
    def __init__(self) -> None:
        self.fold = False  # runtime flag read by the custom op (set during nr = 1 graph capture)
        self.in_capture = False
        self.rows = 0  # context rows carried by the fold (= D + 1 = query rows per request)
        self.buf: torch.Tensor | None = None  # [2 * rows, H] bf16: [:rows] normed ctx, [rows:] block
        self.layers: dict[str, torch.nn.Module] = {}  # attention prefix -> DFlashQwen3Attention
        self.out_size: dict[str, int] = {}
        self.ctx_slots: dict[str, torch.Tensor] = {}  # attention prefix -> [rows] int64 slot view
        self.ctx_pos: torch.Tensor | None = None  # [rows] int64 view of context_positions
        self.model = None  # DFlashQwen3Model (hidden_norm, rope cache, eps)
        self.spec = None
        self.cur_nr = 0
        self.cur_has_prefill = False
        self.n_fold_capture = 0
        self.n_skip = 0
        self.n_prop = 0
        self.verify_every = int(os.environ.get("ATDEV_CTXFOLD_VERIFY", "0"))
        # ATDEV_CTXFOLD_FORK: context K-norm / RoPE / cache insert on a forked capture stream
        self.fork = os.environ.get("ATDEV_CTXFOLD_FORK", "0") == "1"
        self.side = None
        self.join_ev = None  # event the next drafter FlashAttention forward waits on
        self.keep: dict[str, torch.Tensor] = {}  # qkv outputs read by the side branch (no reuse in capture)
        self.n_fork = 0


S = _State()


# ---------------------------------------------------------------------------
# custom op: block qkv_proj with the context rows folded into the same Marlin tile
# ---------------------------------------------------------------------------


def _ctx_kv_store(layer, qkv_ctx: torch.Tensor) -> None:
    """K-norm + RoPE + cache insert of the folded context rows of one layer (stock kernels)."""
    from vllm import _custom_ops as ops

    m = S.model
    rows = S.rows
    nkv, hd = layer.num_kv_heads, layer.head_dim
    k = qkv_ctx[:, layer.q_size: layer.q_size + layer.kv_size].contiguous()  # [rows, kv]
    v = qkv_ctx[:, layer.q_size + layer.kv_size:]  # strided view; reshape_and_cache uses stride(0)
    kn = torch.empty_like(k)
    ops.rms_norm(kn.view(-1, hd), k.view(-1, hd), layer.k_norm.weight, m._rms_norm_eps)
    cs = m._rope_cos_sin_cache
    if cs.dtype != kn.dtype:
        cs = cs.to(dtype=kn.dtype)
    ops.rotary_embedding(S.ctx_pos, kn, None, m._rope_head_size, cs, m._rope_is_neox)
    if layer.v_scale is not None:
        v = v * layer.v_scale
    attn = layer.attn
    attn.impl.do_kv_cache_update(attn, kn.view(rows, nkv, hd), v.view(rows, nkv, hd), attn.kv_cache,
                                 S.ctx_slots[layer.layer_name])


def _fold_qkv(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = S.layers[layer_name]
    if not S.fold:
        out, _ = layer.qkv_proj(x)
        return out
    rows = S.rows
    buf = S.buf
    assert x.shape[0] == rows and buf is not None and buf.shape[0] == 2 * rows, (x.shape, rows)
    buf[rows:].copy_(x)
    qkv, _ = layer.qkv_proj(buf)  # [2 * rows, q + 2 kv]: one Marlin m16 tile
    if S.fork and torch.cuda.is_current_stream_capturing():
        # ATDEV_CTXFOLD_FORK: the context chain (copy, K-norm, RoPE, cache insert of the ctx slots) runs
        # on a forked branch of the drafter graph, concurrent with the block rows' q/k norm + RoPE + cache
        # insert; the layer's FlashAttention forward joins it. Same kernels / inputs -> bit-identical.
        cur = torch.cuda.current_stream(qkv.device)
        if S.side is None:
            S.side = torch.cuda.Stream(device=qkv.device)
        S.side.wait_stream(cur)
        with torch.cuda.stream(S.side):
            _ctx_kv_store(layer, qkv[:rows])
        ev = torch.cuda.Event()
        ev.record(S.side)
        assert S.join_ev is None, "ctxfold fork: previous branch not joined"
        S.join_ev = ev
        S.keep[layer_name] = qkv  # stays allocated for the whole capture (graph pool: no reuse)
        S.n_fork += 1
        if S.n_fork == 1:
            log.warning("atdev %s: context K/V chain on a forked drafter-graph branch ENGAGED", OPT_NAME)
    else:
        _ctx_kv_store(layer, qkv[:rows])
    return qkv[rows:]


def _install_join() -> None:
    from vllm.v1.attention.backends import flash_attn as fa

    orig = fa.FlashAttentionImpl.forward
    if getattr(orig, "_atdev_ctxjoin", False):
        return

    @functools.wraps(orig)
    def forward(self, *a, **k):
        ev, S.join_ev = S.join_ev, None
        if ev is not None:
            torch.cuda.current_stream().wait_event(ev)
        return orig(self, *a, **k)

    forward._atdev_ctxjoin = True
    fa.FlashAttentionImpl.forward = forward


def _fold_qkv_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    return x.new_empty(x.shape[0], S.out_size[layer_name])


# ---------------------------------------------------------------------------
# drafter attention forward: stock body with the qkv call routed through the op
# ---------------------------------------------------------------------------


def _attn_forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    qkv = torch.ops.vllm.atdev_ctxfold_qkv(hidden_states, self.layer_name)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = self.q_norm(q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)).view(q_shape)
    k = self.k_norm(k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)).view(k_shape)
    q, k = self.rotary_emb(positions, q, k)
    if self.v_scale is not None:
        v = v * self.v_scale
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


# ---------------------------------------------------------------------------
# speculator hooks (instance-level, installed on the speculator init_speculator returns)
# ---------------------------------------------------------------------------


def _install_speculator(spec) -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if not isinstance(spec, DFlashSpeculator):
        raise RuntimeError(f"atdev {OPT_NAME}: expected a DFlash speculator, got {type(spec).__name__}")
    from atdev_serve import dflash_tree

    S.spec = spec
    S.rows = spec.num_query_per_req  # 1 + D = block_size (8)
    orig_ngc = spec._num_graph_context_tokens
    orig_pre = spec._precompute_context_kv
    orig_cap = spec.capture
    orig_prop = spec.propose

    def _num_graph_context_tokens(self, num_reqs: int) -> int:
        return S.rows if num_reqs == 1 else orig_ngc(num_reqs)

    def _precompute_context_kv(self, start: int, end: int, dummy_run: bool = False) -> None:
        from vllm import _custom_ops as ops

        if S.in_capture and start == 0:
            if end == S.rows:  # the nr = 1 graph: fold the context rows into the block qkv tiles
                _bind_views(self)
                S.fold = True
                S.n_fold_capture += 1
                ops.rms_norm(S.buf[: S.rows], self.hidden_states[: S.rows],
                             S.model.hidden_norm.weight, S.model._rms_norm_eps)
                return
            S.fold = False
            return orig_pre(start, end, dummy_run)
        n = dflash_tree.S.N
        if (not dummy_run and S.cur_nr == 1 and not S.cur_has_prefill and start == S.rows
                and n is not None and end <= n):
            # a 16-row verify step: rows >= D + 1 are rejected drafts / junk (PAD slots)
            S.n_skip += 1
            return
        if not dummy_run and S.cur_nr == 1 and start == S.rows and end > n:
            # Prefill: reproduce stock's Marlin M partition exactly. Stock ran rows [0, N) in the graph
            # (M = 16 tile) and rows [N, end) in one eager call; Marlin's per-row results depend on the
            # k-split chosen for M, so rows [8, N) must stay in the <= 16-row regime (an M = 8 call is
            # row-identical to the M = 16 tile, results/p7-drafter-micro.txt) and rows [N, end) must keep
            # M = end - N. One eager (8, end) call changed both -> drafter K/V of every prompt differed by
            # ulps -> 94/96 outputs reshuffled (P7-ctxfold-96).
            if n > start:  # block 8 (rows 8 < N 16); block 16 has rows == N: nothing in between
                orig_pre(start, n, dummy_run)
            return orig_pre(n, end, dummy_run)
        return orig_pre(start, end, dummy_run)

    def capture(self) -> None:
        # The runner captures twice (memory-profiling pass, then the real one after initialize_kv_cache ->
        # set_attn reallocates _context_slot_mappings): rebind the fixed-address views on every capture.
        S.ctx_pos = None
        S.ctx_slots.clear()
        S.in_capture = True
        try:
            orig_cap()
        finally:
            S.in_capture = False
            S.fold = False
        log.warning("atdev %s: context-KV fold captured in %d nr=1 drafter graph(s) (%d ctx rows "
                    "+ %d block rows per Marlin tile)", OPT_NAME, S.n_fold_capture, S.rows, S.rows)

    def propose(self, input_batch, *a, **kw):
        S.cur_nr = input_batch.num_reqs
        S.cur_has_prefill = bool(getattr(input_batch, "has_prefill", False))
        out = orig_prop(input_batch, *a, **kw)
        if S.verify_every and not kw.get("dummy_run", False):
            S.n_prop += 1
            if S.n_prop % S.verify_every == 0 and S.cur_nr == 1 and not S.cur_has_prefill:
                _verify(self)
        return out

    spec._num_graph_context_tokens = types.MethodType(_num_graph_context_tokens, spec)
    spec._precompute_context_kv = types.MethodType(_precompute_context_kv, spec)
    spec.capture = types.MethodType(capture, spec)
    spec.propose = types.MethodType(propose, spec)


def _bind_views(spec) -> None:
    """Fixed-address views the captured graph reads: context positions / per-layer slots."""
    if S.ctx_pos is not None:
        return
    rows = S.rows
    S.ctx_pos = spec.context_positions[:rows]
    gidx = spec._layer_group_idx
    for i, layer in enumerate(S.model.layers):
        g = gidx[i] if gidx is not None else 0
        S.ctx_slots[layer.self_attn.layer_name] = spec._context_slot_mappings[g][:rows]


def _install_model(model) -> None:
    """Register the drafter's attention layers for the op; allocate the fold buffer."""
    m = model.model
    S.model = m
    if not hasattr(m, "_num_attn_layers"):
        m._build_fused_kv_buffers()
    dev = m.hidden_norm.weight.device
    dt = m.hidden_norm.weight.dtype
    rows = S.rows or (1 + int(m.config.dflash_config["block_size"]) - 1)
    S.rows = rows
    S.buf = torch.zeros(2 * rows, m.config.hidden_size, dtype=dt, device=dev)
    for layer in m.layers:
        a = layer.self_attn
        S.layers[a.layer_name] = a
        S.out_size[a.layer_name] = a.q_size + 2 * a.kv_size
    log.warning("atdev %s: %d drafter attention layers registered (rows %d, buf %s)", OPT_NAME,
                len(S.layers), rows, tuple(S.buf.shape))


# ---------------------------------------------------------------------------
# verification: folded cache content vs a stock eager recompute (bitwise)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _verify(spec) -> None:
    from vllm import _custom_ops as ops

    m = S.model
    rows = S.rows
    L = m._num_attn_layers
    nkv, hd = m._num_kv_heads, m._head_dim
    torch.cuda.synchronize()
    ctx = spec.hidden_states[:rows]
    all_k, all_v = m._project_context_kv(ctx, rows, L, nkv, hd)
    all_k = m._normalize_context_k(all_k)
    k_flat = all_k.view(L * rows, m._kv_size)
    cs = m._rope_cos_sin_cache
    if cs.dtype != k_flat.dtype:
        cs = cs.to(dtype=k_flat.dtype)
    ops.rotary_embedding(S.ctx_pos.repeat(L), k_flat, None, m._rope_head_size, cs, m._rope_is_neox)
    k_ref = k_flat.view(L, rows, nkv, hd)
    v_ref = all_v
    bad = 0
    checked = 0
    for i, layer in enumerate(m.layers):
        a = layer.self_attn
        slots = S.ctx_slots[a.layer_name]
        kv = a.attn.kv_cache
        kv = kv[0] if isinstance(kv, (list, tuple)) else kv
        valid = slots >= 0
        if not bool(valid.any()):
            continue
        # FlashAttention layout [blocks, nkv, block_size, 2 * hd]; slot = block * block_size + offset
        bs = kv.shape[2]
        sv = slots[valid]
        got = kv[sv // bs, :, sv % bs, :]  # [n, nkv, 2 * hd]
        kr = k_ref[i][valid]
        vr = v_ref[i][valid]
        if a.v_scale is not None:
            vr = vr * a.v_scale
        bad += int((got[..., :hd] != kr).sum()) + int((got[..., hd:] != vr).sum())
        checked += kr.numel() + vr.numel()
    n_valid = int((torch.stack([S.ctx_slots[l.self_attn.layer_name] for l in m.layers]) >= 0).sum())
    log.warning("atdev %s: VERIFY proposal %d: %d / %d cached context K/V elements differ from the "
                "stock recompute, %d valid ctx slots over %d layers (%s)", OPT_NAME, S.n_prop, bad, checked,
                n_valid, L, "bit-identical" if bad == 0 and checked > 0 else
                ("NO-VALID-SLOTS" if checked == 0 else "MISMATCH"))


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def register() -> None:
    import vllm
    from vllm.model_executor.models import qwen3_dflash as qd
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    from atdev_serve.register import enabled

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, "
                           f"found {vllm.__version__}")
    if "dflash_tree" not in enabled():
        raise RuntimeError(f"atdev {OPT_NAME}: requires ATDEV_OPT dflash_tree (accepted <= D+1 per step)")
    if getattr(qd.DFlashQwen3Attention.forward, "_atdev_ctxfold", False):
        return
    if not hasattr(torch.ops.vllm, "atdev_ctxfold_qkv"):
        direct_register_custom_op("atdev_ctxfold_qkv", _fold_qkv, fake_impl=_fold_qkv_fake)
    if S.fork:
        _install_join()
    _attn_forward._atdev_ctxfold = True
    qd.DFlashQwen3Attention.forward = _attn_forward

    orig_load = DFlashSpeculator.load_draft_model

    @functools.wraps(orig_load)
    def load_draft_model(self, *a, **kw):
        # Instance-level hooks: dflash_tree replaces init_speculator with its own subclass, so the
        # speculator is patched here (load precedes set_attn and capture), independent of
        # registration order.
        model = orig_load(self, *a, **kw)
        S.rows = self.num_query_per_req
        _install_model(model)
        _install_speculator(self)
        return model

    load_draft_model._atdev_ctxfold = True
    DFlashSpeculator.load_draft_model = load_draft_model
    log.info("atdev %s: registered", OPT_NAME)
