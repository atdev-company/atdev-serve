"""ATDEV_OPT=w3_mixed: 3-bit weights for selected linears of the RedHatAI INT4 target.

The linears matched by W3_LAYERS are replaced at load time by 3-bit g128 symmetric re-quantisations from a
*sidecar* safetensors overlay, stored in the checkpoint's own compressed-tensors int4 layout (nibbles q + 8 with
q in [-4, 3]). After vLLM's Marlin repack the int4-layout weight is re-packed to the 3-bit planar layout
(pack_w3, 0.758x the bytes) and the layer runs the marlin_plus W3 kernel (csrc/marlin_plus_w3, tile configs
w3_fastdq_cfg.json; shapes / M buckets without an entry use the stock Marlin heuristic).

Env:
  ATDEV_W3_SIDECAR   comma-separated sidecar files (later files win per tensor); each entry may also be
                     an HF spec ``org/name[@rev]`` (all *.safetensors at the repo root, sorted),
                     ``org/name[@rev]:folder/`` (all *.safetensors in that subfolder, sorted, e.g.
                     ``atdev-company/Qwen3.8-27B-atdev-c1:w3/``) or ``org/name[@rev]:path/file.safetensors``
                     (atdev_serve.hub)
"""
from __future__ import annotations

import json
import logging
import os
import re

import torch

OPT_NAME = "w3_mixed"
log = logging.getLogger("atdev_serve")
P_HF = "model.language_model."
# checkpoint tensor suffix -> vLLM fused layer suffix (all shards of a fused layer must be present)
FUSED = {"linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz", "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
         "linear_attn.out_proj": "linear_attn.out_proj",
         "self_attn.q_proj": "self_attn.qkv_proj", "self_attn.k_proj": "self_attn.qkv_proj",
         "self_attn.v_proj": "self_attn.qkv_proj", "self_attn.o_proj": "self_attn.o_proj",
         "mlp.gate_proj": "mlp.gate_up_proj", "mlp.up_proj": "mlp.gate_up_proj", "mlp.down_proj": "mlp.down_proj"}
SHARDS = {"linear_attn.in_proj_qkvz": 2, "linear_attn.out_proj": 1, "self_attn.qkv_proj": 3,
          "self_attn.o_proj": 1, "mlp.gate_up_proj": 2, "mlp.down_proj": 1}

# checkpoint tensors taken from the sidecar: MLP, GDN in_proj_qkv / in_proj_z / out_proj, attention q / k / v / o
W3_LAYERS = r"mlp\.|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$|self_attn\.(q|k|v|o)_proj$"
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w3_fastdq_cfg.json")
_STATE: dict = {"sub": {}, "fused_count": {}, "fused_ok": set(), "n_sub": 0, "drafter_loading": False}


def norm_prefix(prefix: str) -> str:
    """'language_model.model.layers.3.mlp.down_proj' -> 'model.layers.3.mlp.down_proj'"""
    i = prefix.find("model.layers.")
    return prefix[i:] if i >= 0 else prefix
_BUF: dict = {}
_MOD = None
_TABLE: dict | None = None
_SEEN: set = set()


def sidecar_files() -> list[str]:
    from atdev_serve.hub import resolve_files

    out: list[str] = []
    for p in os.environ.get("ATDEV_W3_SIDECAR", "").split(","):
        if p.strip():
            out.extend(resolve_files(p.strip()))
    return out


def load_sidecar_index():
    """{checkpoint tensor name (without .weight_packed): (file, meta)}"""
    from safetensors import safe_open
    sub = {}
    for path in sidecar_files():
        with safe_open(path, "pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                if not k.endswith(".weight_packed"):
                    continue
                base = k[: -len(".weight_packed")]
                if not re.search(W3_LAYERS, base):
                    continue
                sub[base] = (path, meta.get("bits", "?"))
    return sub


def vllm_prefix(ckpt_base: str) -> str | None:
    """model.language_model.layers.3.mlp.gate_proj -> model.layers.3.mlp.gate_up_proj"""
    if not ckpt_base.startswith(P_HF + "layers."):
        return None
    _, li, suffix = ckpt_base[len(P_HF):].split(".", 2)
    fused = FUSED.get(suffix)
    return f"model.layers.{li}.{fused}" if fused else None


def pack_w3(mq: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Marlin int4 repack (int32 [K/16, 2N], nibbles q+8 with q in [-4, 3]) -> 3-bit planar
    int32 [K/16, 3, N/2]: plane p = u(w_p) | (bit p of u(w3)) << 3 per nibble, u = q + 4."""
    assert mq.dtype == torch.int32 and mq.shape == (k // 16, 2 * n), (mq.shape, n, k)
    w = mq.to(torch.int64) & 0xFFFFFFFF
    w = w.view(k // 16, n // 2, 4)
    nib_min = min(((w >> (4 * i)) & 0xF).min().item() for i in range(8))
    nib_max = max(((w >> (4 * i)) & 0xF).max().item() for i in range(8))
    if nib_min < 4 or nib_max > 11:
        raise ValueError(f"pack_w3: nibbles {nib_min}..{nib_max} are not 3-bit values (expected 4..11)")
    u = w - 0x44444444
    u3 = u[:, :, 3]
    p0 = u[:, :, 0] | ((u3 & 0x11111111) << 3)
    p1 = u[:, :, 1] | ((u3 & 0x22222222) << 2)
    p2 = u[:, :, 2] | ((u3 & 0x44444444) << 1)
    out = torch.stack([p0, p1, p2], dim=1)  # [K/16, 3, N/2]
    out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
    return out.contiguous()


def mod():
    global _MOD
    if _MOD is None:
        from atdev_serve import marlin_plus_w3_build
        _MOD = marlin_plus_w3_build.load()
    return _MOD


def bufs(dev: torch.device) -> dict:
    b = _BUF.get(dev)
    if b is None:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        b = _BUF[dev] = dict(locks=torch.zeros(sms * 4, dtype=torch.int32, device=dev),
                             c_tmp=torch.empty(sms * 4 * 64 * 256, dtype=torch.float32, device=dev))
    return b


def m_bucket(m: int) -> str:
    if m <= 8:
        return "8"
    if m > 128:
        return "none"
    return str(min((m + 15) // 16, 4) * 16) if m <= 64 else "big"


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.load(open(CONFIG))
    return _TABLE


def w3_linear(x: torch.Tensor, q3: torch.Tensor, ms: torch.Tensor, n: int, k: int) -> torch.Tensor:
    m = x.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    if x.stride(-1) != 1 or x.stride(0) % 8:
        x = x.contiguous()
    ent = table().get(f"{n},{k}", {}).get(m_bucket(m))
    cfg = list(ent["cfg"]) if ent else [-1, -1, -1, 4, 1, 4]
    key = (n, k, m_bucket(m))
    if key not in _SEEN:
        _SEEN.add(key)
        log.warning("atdev w3: N=%d K=%d M=%d (bucket %s) -> cfg %s", n, k, m, key[2], cfg)
    b = bufs(x.device)
    return mod().gemm(x, q3, ms, out, b["locks"], b["c_tmp"], n, k, cfg)


def _fake(x, q3, ms, n, k):
    return torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)


def _wrap_loader():
    from vllm.model_executor.model_loader import default_loader as dl
    orig = dl.DefaultModelLoader._get_weights_iterator
    if getattr(orig, "_atdev_w3", False):
        return

    def _get_weights_iterator(self, source):
        from safetensors import safe_open
        sub = _STATE["sub"]
        files = {}
        try:
            for name, t in orig(self, source):
                base = None
                for suf in (".weight_packed", ".weight_scale"):
                    if name.endswith(suf):
                        base = name[: -len(suf)]
                if base is not None and base in sub and not _STATE["drafter_loading"]:
                    path = sub[base][0]
                    if path not in files:
                        files[path] = safe_open(path, "pt", device="cpu")
                    t2 = files[path].get_tensor(name)
                    assert t2.shape == t.shape and t2.dtype == t.dtype, (name, t.shape, t2.shape, t.dtype, t2.dtype)
                    _STATE["n_sub"] += 1
                    yield name, t2
                else:
                    yield name, t
        finally:
            files.clear()

    _get_weights_iterator._atdev_w3 = True
    dl.DefaultModelLoader._get_weights_iterator = _get_weights_iterator


def _wrap_kernel():
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
    from vllm.scalar_type import scalar_types
    orig_pw = MarlinLinearKernel.process_weights_after_loading
    orig_apply = MarlinLinearKernel.apply_weights
    if getattr(orig_pw, "_atdev_w3", False):
        return

    def process_weights_after_loading(self, layer):
        orig_pw(self, layer)
        prefix = norm_prefix(getattr(layer, "prefix", "") or "")
        if _STATE["drafter_loading"] or prefix not in _STATE["fused_ok"]:
            return
        c = self.config
        if c.weight_type != scalar_types.uint4b8 or c.group_size != 128 or c.act_type != torch.bfloat16:
            log.warning("atdev w3: %s not eligible for the W3 kernel (type %s, group %s, act %s)", prefix,
                        c.weight_type, c.group_size, c.act_type)
            return
        k, n = c.partition_weight_shape
        w_q = getattr(layer, self.w_q_name)
        if tuple(w_q.shape) != (k // 16, 2 * n):
            log.warning("atdev w3: %s unexpected Marlin shape %s (padded?) -> int4 path", prefix, tuple(w_q.shape))
            return
        try:
            q3 = pack_w3(w_q.data, n, k)
        except ValueError as e:  # not 3-bit valued (e.g. a same-named drafter layer) -> keep int4
            log.warning("atdev w3: %s stays int4: %s", prefix, e)
            return
        w_q.data = q3
        layer._atdev_w3 = True
        _STATE.setdefault("packed", []).append(prefix)

    def apply_weights(self, layer, x, bias=None):
        if not getattr(layer, "_atdev_w3", False) or bias is not None or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        k, n = self.config.partition_weight_shape
        w_q, w_s, _ = self._get_weight_params(layer)
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.atdev_w3_linear(x2, w_q, w_s, n, k)
        return out.reshape(x.shape[:-1] + (n,))

    process_weights_after_loading._atdev_w3 = True
    MarlinLinearKernel.process_weights_after_loading = process_weights_after_loading
    MarlinLinearKernel.apply_weights = apply_weights
    try:  # marlin_plus overrides apply_weights in a subclass; route its layers too
        from atdev_serve import marlin_plus as mpl
        cls = mpl.kernel_class()
        mp_apply = cls.apply_weights

        def mp_apply_weights(self, layer, x, bias=None):
            if getattr(layer, "_atdev_w3", False):
                return apply_weights(self, layer, x, bias)
            return mp_apply(self, layer, x, bias)
        cls.apply_weights = mp_apply_weights
    except Exception:  # noqa: BLE001
        pass


def register() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op
    from atdev_serve.compile_cache import salt_aot_cache
    salt_aot_cache(OPT_NAME, __file__)
    from atdev_serve import compile_cache
    compile_cache._SALTS[OPT_NAME] += "+" + ",".join(os.path.basename(p) for p in sidecar_files())
    sub = load_sidecar_index()
    _STATE["sub"] = sub
    cnt: dict = {}
    for base in sub:
        vp = vllm_prefix(base)
        if vp:
            cnt[vp] = cnt.get(vp, 0) + 1
    ok = {vp for vp, c in cnt.items() if c == SHARDS[vp.split(".", 3)[3]]}
    partial = sorted(set(cnt) - ok)
    _STATE["fused_count"] = cnt
    _STATE["fused_ok"] = ok
    if not hasattr(torch.ops.vllm, "atdev_w3_linear"):
        direct_register_custom_op("atdev_w3_linear", w3_linear, fake_impl=_fake)
    _wrap_loader()
    _wrap_kernel()
    try:  # the DFlash drafter's linears share 'model.layers.N.*' names: never touch them
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
        orig_load = DFlashSpeculator.load_draft_model
        if not getattr(orig_load, "_atdev_w3", False):
            def load_draft_model(self, *args, **kwargs):
                _STATE["drafter_loading"] = True
                try:
                    return orig_load(self, *args, **kwargs)
                finally:
                    _STATE["drafter_loading"] = False
            load_draft_model._atdev_w3 = True
            DFlashSpeculator.load_draft_model = load_draft_model
    except ImportError:
        pass
    bits = sorted({b for _, b in sub.values()})
    log.warning("atdev %s: %d sidecar tensors from %s (bits %s) -> %d fused layers on the W3 kernel%s",
                OPT_NAME, len(sub), sidecar_files(), bits, len(ok),
                f"; {len(partial)} partially covered fused layers stay int4: {partial[:4]}" if partial else "")
