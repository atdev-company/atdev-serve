"""Low-bit DFlash2 drafter (ATDEV_OPT=drafter_w4).

The DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) reads ~3.6 GB of bf16 linear
weights per decode step (5 layers x 666 MB + fc 262 MB), ~2.5 ms of the
~20 ms k=7 step. This module quantizes those linears to W4A16 (int4, group
128, symmetric, compressed-tensors ``pack-quantized``), the same format and
Marlin kernel path the RedHatAI INT4 target uses.

Delivery is an *offline checkpoint* (path (a)):

* ``scripts/quantize_drafter.py`` writes a compressed-tensors checkpoint
  (config.json with ``quantization_config`` + model.safetensors).
* vLLM 0.31.0 loads it unchanged: for ``method=dflash`` with an explicit
  model path the draft ``ModelConfig`` auto-detects ``quantization_config``
  (``config/speculative.py`` passes ``quantization=None`` for dflash),
  ``DFlashQwen3Model`` builds every linear with ``get_draft_quant_config``
  (``models/qwen3_dflash.py:428``), and ``_build_context_kv_buffers``
  (``:505-530``) already falls back to per-layer ``qkv_proj`` calls when the
  projections are quantized.

``register()`` (this optimization) only makes the swap honest and loud:

1. Wraps ``EngineArgs.create_speculative_config`` so that the *same CLI
   flags* as the best-vLLM baseline (``"model":"z-lab/Qwen3.8-27B-DFlash2"``)
   load the quantized checkpoint (``ATDEV_DRAFTER_QUANT_PATH``). A
   speculative model that is not the expected drafter raises.
2. Wraps ``DFlashSpeculator.load_draft_model`` (worker side) to assert the
   loaded drafter really runs quantized linears and logs the weight bytes.

Quantizer: symmetric RTN per group of 128 input channels with a per-group
clipping search (scale = alpha * amax / qhalf, alpha on a grid, bf16-rounded
scale, minimise the importance-weighted squared error). Importance is a
data-free activation proxy: for linears fed by an RMSNorm + DFlash2 grouped
conv (qkv, gate/up, kernel_projection) it is gamma_j^2 * (b0_j^2 + b1_j^2)
(norm weight times the conv's static taps); elsewhere uniform. Calibration
(GPTQ/AWQ) needs target hidden states, i.e. a GPU; see
docs/research/drafter-w4.md.

Env knobs (serving):
  ATDEV_DRAFTER_QUANT_PATH  quantized checkpoint dir or HF repo id org/name[@rev]
                            (default DEFAULT_OUT, the released W4A16 checkpoint)
  ATDEV_DRAFTER_QUANT_SRC   comma-separated drafter ids/paths to replace
                            (default z-lab/Qwen3.8-27B-DFlash2)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import torch

OPT_NAME = "drafter_w4"
SUPPORTED_VLLM_VERSION = "0.31.0"
SRC_REPO = "z-lab/Qwen3.8-27B-DFlash2"
# Default quantized drafter: the release checkpoint on the HF Hub (local dir or repo id accepted).
DEFAULT_OUT = "atdev-company/Qwen3.8-27B-atdev-DFlash2-d16"

log = logging.getLogger("atdev_serve.drafter_w4")

# ---------------------------------------------------------------------------
# Which checkpoint tensors are quantized
# ---------------------------------------------------------------------------

# Main decoder-layer linears (checkpoint names, before vLLM's q/k/v stacking).
_LAYER_LINEAR = re.compile(
    r"^layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)
_KPROJ = re.compile(r"^layers\.(\d+)\.(attention|mlp)_conv\.kernel_projection\.weight$")
_FC = "fc.weight"


@dataclass(frozen=True)
class QuantSpec:
    bits: int = 4
    group_size: int = 128
    fc_bits: int = 4  # 4, 8 or 16 (16 = keep bf16)
    kernel_proj_bits: int = 16  # 16 = keep bf16 (default), else 4/8
    clip_search: bool = True
    importance: str = "norm"  # "norm" or "none"

    def bits_for(self, name: str) -> int:
        """Bits for a checkpoint tensor name; 16 means not quantized."""
        if _LAYER_LINEAR.match(name):
            return self.bits
        if name == _FC:
            return self.fc_bits
        if _KPROJ.match(name):
            return self.kernel_proj_bits
        return 16


# ---------------------------------------------------------------------------
# Quantization math (CPU, fp32)
# ---------------------------------------------------------------------------


def qrange(bits: int) -> tuple[int, int, float]:
    """(qmin, qmax, half-range) of a symmetric signed int grid.

    compressed-tensors' minmax observer uses scale = amax / half with
    half = (qmax - qmin) / 2 (7.5 for int4); the clip grid below brackets it.
    """
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return qmin, qmax, (qmax - qmin) / 2


def clip_grid(bits: int, clip_search: bool) -> torch.Tensor:
    """Scale multipliers alpha (scale = alpha * amax / half).

    alpha = half/qmax keeps +amax representable (no clipping); smaller alpha
    clips the extremes but refines the grid.
    """
    _, qmax, half = qrange(bits)
    top = half / qmax
    if not clip_search:
        return torch.tensor([1.0])
    lo = 0.70 if bits == 4 else 0.90
    return torch.linspace(lo, top, 24)


def quantize_weight(
    w: torch.Tensor,
    bits: int,
    group_size: int,
    importance: torch.Tensor | None = None,
    clip_search: bool = True,
    row_chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric group RTN with optional weighted clip search.

    Args:
        w: [out, in] weight (any float dtype).
        importance: [in] non-negative per-input-channel weights, or None.
    Returns:
        q: int8 [out, in] in [qmin, qmax];
        scale: bf16 [out, in // group_size]. Dequant = q * scale (bf16 scale
        as stored, which is what Marlin uses).
    """
    out_f, in_f = w.shape
    if in_f % group_size:
        raise ValueError(f"in_features {in_f} not divisible by group {group_size}")
    g = in_f // group_size
    qmin, qmax, half = qrange(bits)
    alphas = clip_grid(bits, clip_search)
    imp = None
    if importance is not None:
        imp = importance.float().reshape(1, g, group_size)
        imp = imp / imp.mean().clamp_min(1e-30)
    q_out = torch.empty(out_f, in_f, dtype=torch.int8)
    s_out = torch.empty(out_f, g, dtype=torch.bfloat16)
    for r0 in range(0, out_f, row_chunk):
        wf = w[r0 : r0 + row_chunk].float().reshape(-1, g, group_size)
        amax = wf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_err = None
        best_s = None
        for a in alphas.tolist():
            s = (amax * (a / half)).to(torch.bfloat16).float().clamp_min(1e-12)
            q = (wf / s).round_().clamp_(qmin, qmax)
            d = (q.mul_(s)).sub_(wf).square_()
            if imp is not None:
                d.mul_(imp)
            err = d.sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_s = err, s
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_s = torch.where(better, s, best_s)
        q = (wf / best_s).round_().clamp_(qmin, qmax)
        rows = wf.shape[0]
        q_out[r0 : r0 + rows] = q.reshape(rows, in_f).to(torch.int8)
        s_out[r0 : r0 + rows] = best_s.reshape(rows, g).to(torch.bfloat16)
    return q_out, s_out


def dequantize(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """fp32 [out, in] from int q [out, in] and scale [out, in/group]."""
    out_f, in_f = q.shape
    s = scale.float().repeat_interleave(group_size, dim=1)
    return q.float() * s


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """compressed-tensors pack-quantized (int32 along the input dim)."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    return pack_to_int32(q, bits, packed_dim=1)


def unpack(packed: torch.Tensor, bits: int, shape) -> torch.Tensor:
    from compressed_tensors.compressors.pack_quantized.helpers import (
        unpack_from_int32,
    )

    return unpack_from_int32(packed, bits, torch.Size(shape), packed_dim=1)


def importance_for(name: str, sd: dict[str, torch.Tensor], mode: str):
    """Data-free per-input-channel importance proxy (see module docstring).

    For the DFlash2 layer, qkv/gate/up/kernel_projection inputs are
    conv(RMSNorm_gamma(x)) where conv's static part multiplies channel j by
    base_kernel[side 0, tap t, j]. Their magnitude per channel scales like
    gamma_j * ||base_kernel[0, :, j]||. o_proj/down_proj/fc have no static
    per-channel scale on the input, so uniform.
    """
    if mode == "none":
        return None
    m = _LAYER_LINEAR.match(name) or _KPROJ.match(name)
    if m is None:
        return None
    i = int(m.group(1))
    if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        norm, conv = "input_layernorm", "attention_conv"
    elif name.endswith(("gate_proj.weight", "up_proj.weight")):
        norm, conv = "post_attention_layernorm", "mlp_conv"
    elif "attention_conv.kernel_projection" in name:
        # kernel_projection sees the normed input before the conv.
        return sd[f"layers.{i}.input_layernorm.weight"].float().square() + 1e-12
    elif "mlp_conv.kernel_projection" in name:
        return sd[f"layers.{i}.post_attention_layernorm.weight"].float().square() + 1e-12
    else:
        return None
    gamma = sd[f"layers.{i}.{norm}.weight"].float()
    base = sd[f"layers.{i}.{conv}.base_kernel"].float()[0]  # [taps, hidden]
    return gamma.square() * base.square().sum(0) + 1e-12


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def _weights_args(bits: int, group_size: int) -> dict:
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": group_size,
        "num_bits": bits,
        "observer": "minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }


def _group(targets: list[str], bits: int, group_size: int) -> dict:
    return {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": targets,
        "weights": _weights_args(bits, group_size),
    }


FC_TARGET = r"re:(.*\.)?fc$"
KPROJ_TARGET = r"re:.*kernel_projection$"


def quantization_config(spec: QuantSpec) -> dict:
    """compressed-tensors quantization_config mirroring the RedHatAI target's.

    vLLM resolves a layer's group by *name* regex first, over all groups,
    then by class name ("Linear" substring), so the fc / kernel_projection
    groups take precedence over the catch-all (compressed_tensors/utils.py
    find_matched_target). The selector's hidden_projection is built with
    quant_config=None by vLLM; it is ignored here for other consumers.
    """
    ignore = ["lm_head", "re:.*candidate_selector.*"]
    groups: dict[str, dict] = {"group_0": _group(["Linear"], spec.bits, spec.group_size)}
    for target, b in ((FC_TARGET, spec.fc_bits), (KPROJ_TARGET, spec.kernel_proj_bits)):
        if b == 16:
            ignore.append(target)
        elif b != spec.bits:
            groups[f"group_{len(groups)}"] = _group([target], b, spec.group_size)
    return {
        "config_groups": groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


# ---------------------------------------------------------------------------
# vLLM integration (ATDEV_OPT=drafter_w4)
# ---------------------------------------------------------------------------


def quant_path() -> str:
    """Quantized drafter dir; ATDEV_DRAFTER_QUANT_PATH may be a local dir or an HF repo id."""
    from atdev_serve.hub import resolve_dir

    return resolve_dir(os.environ.get("ATDEV_DRAFTER_QUANT_PATH", DEFAULT_OUT))


def _sources() -> list[str]:
    raw = os.environ.get("ATDEV_DRAFTER_QUANT_SRC", SRC_REPO)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _is_source(model: str | None) -> bool:
    if not model:
        return False
    for src in _sources():
        if model == src:
            return True
        # A local HF snapshot of the repo: .../models--z-lab--X/snapshots/<rev>
        if "/" in src and f"models--{src.replace('/', '--')}" in model:
            return True
    return False


def check_checkpoint(path: str) -> dict:
    """Fail loudly unless `path` is a quantized drafter checkpoint."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path) or not os.path.isfile(
        os.path.join(path, "model.safetensors")
    ):
        raise FileNotFoundError(
            f"atdev {OPT_NAME}: no quantized drafter at {path} "
            "(run scripts/quantize_drafter.py)"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        raise ValueError(f"atdev {OPT_NAME}: {cfg_path} has no compressed-tensors config")
    if cfg.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError(f"atdev {OPT_NAME}: {cfg_path} is not a DFlash2 drafter")
    return cfg


def rewrite_spec_config(spec: dict | None) -> dict | None:
    """Point the dflash drafter at the quantized checkpoint (in place)."""
    if spec is None:
        raise ValueError(f"atdev {OPT_NAME}: enabled but no --speculative-config")
    if not isinstance(spec, dict):
        raise TypeError(f"atdev {OPT_NAME}: unexpected speculative_config {type(spec)}")
    model = spec.get("model")
    target = quant_path()
    if model == target:
        check_checkpoint(target)
        return spec
    if spec.get("method") != "dflash" or not _is_source(model):
        raise ValueError(
            f"atdev {OPT_NAME}: expected method=dflash with model in {_sources()}, "
            f"got method={spec.get('method')} model={model}"
        )
    check_checkpoint(target)
    if spec.get("quantization"):
        raise ValueError(
            f"atdev {OPT_NAME}: speculative quantization={spec['quantization']} "
            "conflicts with the pre-quantized drafter"
        )
    spec["model"] = target
    log.warning("atdev %s: drafter %s -> %s", OPT_NAME, model, target)
    return spec


def _linear_report(model: torch.nn.Module) -> tuple[int, int, int]:
    """(#quantized linears, #unquantized linears, weight bytes of linears)."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    nq = nu = nbytes = 0
    for mod in model.modules():
        if not isinstance(mod, LinearBase):
            continue
        if isinstance(mod.quant_method, UnquantizedLinearMethod):
            nu += 1
        else:
            nq += 1
        for p in mod.parameters(recurse=False):
            nbytes += p.numel() * p.element_size()
    return nq, nu, nbytes


def register() -> None:
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if vllm.__version__ != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"atdev {OPT_NAME}: written against vLLM {SUPPORTED_VLLM_VERSION}, "
            f"found {vllm.__version__}"
        )

    orig_create = EngineArgs.create_speculative_config
    if not getattr(orig_create, "_atdev_drafter_w4", False):

        def create_speculative_config(self, *args, **kwargs):
            rewrite_spec_config(self.speculative_config)
            return orig_create(self, *args, **kwargs)

        create_speculative_config._atdev_drafter_w4 = True
        EngineArgs.create_speculative_config = create_speculative_config

    orig_load = DFlashSpeculator.load_draft_model
    if not getattr(orig_load, "_atdev_drafter_w4", False):

        def load_draft_model(self, *args, **kwargs):
            model = orig_load(self, *args, **kwargs)
            nq, nu, nbytes = _linear_report(model)
            if nq == 0:
                raise RuntimeError(
                    f"atdev {OPT_NAME}: drafter loaded with no quantized linears"
                )
            log.warning(
                "atdev %s: drafter linears quantized=%d unquantized=%d, "
                "linear weight bytes %.3f GB",
                OPT_NAME,
                nq,
                nu,
                nbytes / 1e9,
            )
            return model

        load_draft_model._atdev_drafter_w4 = True
        DFlashSpeculator.load_draft_model = load_draft_model
    log.info("atdev %s: registered (path=%s)", OPT_NAME, quant_path())
