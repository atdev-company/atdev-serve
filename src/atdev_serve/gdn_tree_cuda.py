"""CUDA tree-verify GDN kernel (csrc/gdn_tree.cu): drop-in for gdn_tree.gdn_tree_verify (+ conv_tree_update + the
fused gated norm) at N <= 16, BV = 32, bf16 state, K = V = 128 (used by dflash_tree.TreeGDNAttention).

Two builds: the plain one (b / a as bf16) and the GT_BAP one, where b / a are the unscaled fp32 in_proj_ba
accumulator (c1_norm.ba_tot, side stream) + rstd (bap=dict(b=, a=, r=)) and the kernel rounds bf16(tot * rstd)
itself (bit-identical to materialising b / a first)."""

from __future__ import annotations

import os

import torch

from atdev_serve.gdn_tree import tree_log_layout

_EXT = None
_BUF: dict = {}
_EXT_BAP = None
# c1_fuse -> dflash_tree: the deferred in_proj_ba of the current GDN layer
# (kind="tot": fp32 accumulator tot, rstd r, bf16 placeholder ba, done)
BA_CTX: dict = {}


def build(defines: tuple[str, ...] = ()):
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "csrc", "gdn_tree.cu")
    name = "atdev_gdn_tree" + "".join("_" + d.replace("=", "") for d in defines)
    return load(name=name, sources=[src], verbose=False,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80",
                                   "-std=c++17", "--expt-relaxed-constexpr"] + [f"-D{d}" for d in defines])


def cuda_ext():
    """Plain build; also builds the GT_BAP one (both must exist before any graph capture)."""
    global _EXT
    if _EXT is None:
        _EXT = build()
        cuda_ext_bap()
    return _EXT


def cuda_ext_bap():
    """GT_BAP build (fp32 unscaled b / a + rstd); used only for calls that pass bap."""
    global _EXT_BAP
    if _EXT_BAP is None:
        _EXT_BAP = build(("GT_BAP=1",))
    return _EXT_BAP


def _bufs(dev: torch.device, n_req: int, h: int, hv: int):
    """Exchange buffers (zero-initialised once; the kernel's generation tags make reuse safe):
    cnt [R*H] u64 arrival counters, part [prof area + R*HV*4*16 (value, gen) pairs],
    qkx [R*H*16*256] bf16 conv output exchange, qkf [R*H*16] u32 slice flags."""
    key = (dev, n_req, h, hv)
    b = _BUF.get(key)
    if b is None:
        cnt = torch.zeros(n_req * h, dtype=torch.int64, device=dev)
        part = torch.zeros((1 << 16) + max(n_req * hv * 4 * 16 * 2, 4096 * 64 * 2), dtype=torch.float32,
                           device=dev)
        qkx = torch.zeros(3 * n_req * h * 16 * 256, dtype=torch.bfloat16, device=dev)
        qkf = torch.zeros(n_req * h * 16, dtype=torch.int32, device=dev)
        b = _BUF[key] = (cnt, part, qkx, qkf)
    return b


def prealloc(dev: torch.device, n_req: int, h: int, hv: int) -> None:
    """Allocate the exchange buffers outside any CUDA-graph capture (call at layer init)."""
    _bufs(dev, n_req, h, hv)


def _sm_count(dev) -> int:
    return torch.cuda.get_device_properties(dev).multi_processor_count


def _al16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and (t.stride(0) * t.element_size()) % 16 == 0


def supported(*, state, state_indices, q_dtype=torch.bfloat16, block_v=32, num_k_heads=None,
              num_requests=1) -> bool:
    _, hv, vd, kd = state.shape
    return (state.dtype == torch.bfloat16 and vd == 128 and kd == 128 and block_v == 32
            and state_indices.size(1) <= 16 and q_dtype == torch.bfloat16
            and (num_k_heads is None or hv % num_k_heads == 0)
            # every CTA of the grid must be resident (the q/k and norm exchanges spin-wait)
            and num_requests * hv * 4 <= 2 * _sm_count(state.device))


def gdn_tree_verify_cuda(*, q=None, k=None, v=None, a, b, A_log, dt_bias, state, state_indices,
                         cu_seqlens, num_accepted, anc_bits, out, num_k_heads, scale,
                         softplus_beta=1.0, softplus_threshold=20.0, num_requests=None,
                         output_gate=None, norm_weight=None, norm_eps=1e-6, sigmoid_gate=False,
                         conv=None, bap=None) -> None:
    """Same contract as gdn_tree.gdn_tree_verify (BV fixed at 32). conv = dict(x, weight, bias,
    conv_state, conv_src, activation): fuse conv_tree_update (q/k/v are then ignored; x is the
    raw mixed_qkv [L, 2*H*K + HV*V]). bap = dict(b, a, r) (GT_BAP build): fp32 unscaled b / a + rstd."""
    slots, hv, vdim, kdim = state.shape
    S = state_indices.size(1)
    n = state_indices.size(0) if num_requests is None else num_requests
    if n == 0:
        return
    assert S <= 16 and state.dtype == torch.bfloat16
    assert state.stride()[1:] == (vdim * kdim, kdim, 1)
    lay = tree_log_layout(state, num_k_heads, S, 32)
    assert lay["buf"] % 4 == 0 and lay["v_off"] % 4 == 0 and lay["log_base"] % 4 == 0
    assert (state.stride(0) * 2) % 16 == 0 and state.data_ptr() % 16 == 0
    assert out.stride(-1) == 1 and out.stride(-2) == vdim
    if bap is not None:
        a, b = bap["a"], bap["b"]
        assert a.dtype == torch.float32 and b.dtype == torch.float32 and bap["r"].dtype == torch.float32
        assert a.stride(-1) == 1 and b.stride(-1) == 1 and bap["r"].is_contiguous()
    else:
        assert a.stride(-1) == 1 and b.stride(-1) == 1 and a.dtype == torch.bfloat16
    assert anc_bits.dtype == torch.int64 and anc_bits.stride(-1) == 1
    dev = out.device
    assert n * hv * 4 <= 2 * _sm_count(dev), "gdn_tree_cuda: grid must be co-resident (R == 1)"
    if (dev, n, num_k_heads, hv) not in _BUF and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("gdn_tree_cuda: exchange buffers must be preallocated before graph capture")
    cnt, part, qkx, qkf = _bufs(dev, n, num_k_heads, hv)
    e = cuda_ext_bap() if bap is not None else cuda_ext()
    extra = (bap["r"],) if bap is not None else ()
    norm = output_gate is not None
    empty = out.new_empty(0)
    if norm:
        assert output_gate.stride(-1) == 1 and output_gate.stride(-2) == vdim
        assert _al16(output_gate) and norm_weight is not None
        z, nw = output_gate, norm_weight
    else:
        z, nw = empty, empty
    if conv is not None:
        x = conv["x"]
        assert x.stride(1) == 1 and _al16(x) and conv["weight"].is_contiguous()
        assert x.dtype == torch.bfloat16 and conv["weight"].dtype == torch.bfloat16
        cs = conv["conv_state"]
        assert cs.dtype == torch.bfloat16
        csrc = conv["conv_src"]
        assert csrc.dtype == torch.int32 and csrc.stride(2) == 1 and csrc.stride(1) == 3
        bias = conv["bias"] if conv["bias"] is not None else empty
        qq = kk = vv = empty
        act = bool(conv.get("activation", True))
    else:
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
        qq = q.reshape(q.shape[0], -1) if q.dim() > 2 else q
        kk = k.reshape(k.shape[0], -1) if k.dim() > 2 else k
        vv = v.reshape(v.shape[0], -1) if v.dim() > 2 else v
        assert _al16(qq) and _al16(kk) and _al16(vv)
        x = cs = csrc = bias = empty
        act = False
        x_w = empty
    e.gdn_tree_fwd(
        qq, kk, vv, x, conv["weight"] if conv is not None else x_w, bias, cs, csrc,
        conv is not None, act, a, b, A_log, dt_bias, state, state_indices, cu_seqlens,
        num_accepted, anc_bits, out, z, nw, norm, bool(sigmoid_gate), cnt, part, qkx, qkf,
        int(num_k_heads), int(n), float(scale), float(norm_eps), float(softplus_beta),
        float(softplus_threshold), S - 1, lay["log_base"], lay["buf"], lay["v_off"],
        lay["d_off"], lay["b_off"], *extra)
