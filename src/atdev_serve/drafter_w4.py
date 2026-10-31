"""W4A16 DFlash2 drafter (ATDEV_OPT=drafter_w4).

The DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2) reads ~3.6 GB of bf16 linear weights per decode step. The
released drafter checkpoint has those linears in W4A16 (int4, group 128, symmetric, compressed-tensors
``pack-quantized``), the same format and Marlin kernel path the RedHatAI INT4 target uses. vLLM 0.31.0 loads it
unchanged: for ``method=dflash`` with an explicit model path the draft ``ModelConfig`` auto-detects
``quantization_config``, ``DFlashQwen3Model`` builds every linear with ``get_draft_quant_config``, and
``_build_context_kv_buffers`` falls back to per-layer ``qkv_proj`` calls when the projections are quantized.

``register()`` makes the swap explicit:

1. Wraps ``EngineArgs.create_speculative_config`` so that the *same CLI flags* as the best-vLLM baseline
   (``"model":"z-lab/Qwen3.8-27B-DFlash2"``) load the quantized checkpoint (``ATDEV_DRAFTER_QUANT_PATH``).
   A speculative model that is not the expected drafter raises.
2. Wraps ``DFlashSpeculator.load_draft_model`` (worker side) to assert the loaded drafter really runs
   quantized linears and logs the weight bytes.

Env:
  ATDEV_DRAFTER_QUANT_PATH  quantized checkpoint dir or HF spec org/name[@rev][:folder/]
                            (default DEFAULT_OUT, the released W4A16 checkpoint)
"""

from __future__ import annotations

import json
import logging
import os

import torch

OPT_NAME = "drafter_w4"
SUPPORTED_VLLM_VERSION = "0.31.0"
SRC_REPO = "z-lab/Qwen3.8-27B-DFlash2"
# Default quantized drafter: the release checkpoint on the HF Hub, subfolder drafter/ of the c1 repo
# (a local dir, ``org/name`` or ``org/name:subfolder/`` is accepted, see atdev_serve.hub).
DEFAULT_OUT = "atdev-company/Qwen3.8-27B-atdev-c1:drafter/"

log = logging.getLogger("atdev_serve.drafter_w4")


def quant_path() -> str:
    """Quantized drafter dir; ATDEV_DRAFTER_QUANT_PATH may be a local dir or an HF spec (org/name[:folder/])."""
    from atdev_serve.hub import resolve_dir

    return resolve_dir(os.environ.get("ATDEV_DRAFTER_QUANT_PATH", DEFAULT_OUT))


def _is_source(model: str | None) -> bool:
    if not model:
        return False
    # the repo id, or a local HF snapshot of it: .../models--z-lab--X/snapshots/<rev>
    return model == SRC_REPO or f"models--{SRC_REPO.replace('/', '--')}" in model


def check_checkpoint(path: str) -> dict:
    """Fail loudly unless `path` is a quantized drafter checkpoint."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path) or not os.path.isfile(
        os.path.join(path, "model.safetensors")
    ):
        raise FileNotFoundError(
            f"atdev {OPT_NAME}: no quantized drafter at {path}"
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
            f"atdev {OPT_NAME}: expected method=dflash with model {SRC_REPO}, "
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
