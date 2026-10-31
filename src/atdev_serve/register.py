"""vLLM general plugin entry point.

The features to install are listed in ATDEV_OPT (comma-separated names); the launch scripts
serve_c1.sh / serve_c32.sh set it.
"""
import importlib
import logging
import os

log = logging.getLogger("atdev_serve")
_DONE = False

# name -> "module:function" that installs the feature. Registration runs in every vLLM process
# (API server, engine core, workers) before any model is constructed, which is what the
# PluggableLayer / CustomOp out-of-tree swaps require.
OPTIMIZATIONS: dict[str, str] = {
    # int4 copy of lm_head used only to choose speculative candidates
    "shadow_head": "atdev_serve.shadow_head:register",
    # drafter top-k over a 64k frequent-token vocabulary (candidate selection only)
    "draft_vocab": "atdev_serve.draft_vocab:register",
    # int4 weights for the built-in MTP head
    "mtp_w4": "atdev_serve.mtp_w4:register",
    # MTP k>=2: the first draft pass runs o_proj / MLP / final norm only on the last row of each request
    "mtp_prune": "atdev_serve.mtp_prune:register",
    # confidence-scheduled variable-length MTP verification (k drafts, fixed row budget)
    "di1_varlen": "atdev_serve.di1_varlen:register",
    # verify lm_head split into an exact frequent-vocabulary part and a low-bit tail
    "verify_split_head": "atdev_serve.verify_split_head:register",
    # MTP verify: argmax-only greedy verification with the split head
    "vsplit_mtp": "atdev_serve.vsplit_mtp:register",
    # GDN verification that writes the recurrent state once and replays accepted tokens
    "gdn_single_state": "atdev_serve.gdn_single_state:register",
    # W4A4: GDN out_proj activation quantization folded into the GDN verify kernel epilogue
    "gdn_a4_fold": "atdev_serve.gdn_a4_fold:register",
    # loads the int4-quantized DFlash2 drafter
    "drafter_w4": "atdev_serve.drafter_w4:register",
    # token-tree verification for the DFlash2 drafter
    "dflash_tree": "atdev_serve.dflash_tree:register",
    # split-KV TRITON_ATTN for small verify batches
    "attn_split_verify": "atdev_serve.attn_split_verify:register",
    # drafter context rows computed inside the block qkv GEMM of each drafter layer (bit-identical KV)
    "drafter_ctxfold": "atdev_serve.drafter_ctxfold:register",
    # GDN metadata rebuild without the per-group CPU mask gather on pure spec-decode batches
    "gdn_meta_fast": "atdev_serve.gdn_meta_fast:register",
    # vLLM Marlin rebuilt with tuned launch configs (+ W4A4 kernels)
    "marlin_plus": "atdev_serve.marlin_plus:register",
    # CUDA split-KV tree-verify attention for the c=1 tree
    "tree_attn": "atdev_serve.tree_attn:register",
    # CUDA varlen decode / MTP-verify attention for c=32 (bit-identical to TRITON_ATTN)
    "c32_attn": "atdev_serve.c32_attn:register",
    # CUDA GDN causal_conv1d_update for spec decode (bit-identical to the Triton kernel)
    "c32_conv": "atdev_serve.c32_conv:register",
    # 3-bit overlay weights for selected linears + W3 Marlin kernels
    "w3_mixed": "atdev_serve.w3_mixed:register",
    # in_proj_ba as one deterministic split-K Triton GEMV
    "ba_gemv": "atdev_serve.ba_gemv:register",
    # W4A4: norm / activation kernels emit quantized activations for the following GEMMs
    "fused_quant": "atdev_serve.fused_quant:register",
    # c=1: residual-add + RMSNorm / SiLU*mul folded into the W3 Marlin GEMM epilogues
    "c1_fuse": "atdev_serve.c1_fuse:register",
    # bit-identical fusions of the c=1 drafter graph glue
    "ll1_fuse": "atdev_serve.ll1_fuse:register",
    # bit-identical fusions of the remaining c=1 drafter / tree glue
    "ll5_fuse": "atdev_serve.ll5_fuse:register",
    # bit-identical fusions of the c=32 MTP-pass / verify-screen glue
    "c32k_fuse": "atdev_serve.c32k_fuse:register",
}


def enabled() -> set[str]:
    return {s.strip() for s in os.environ.get("ATDEV_OPT", "").split(",") if s.strip()}


def register_all():
    global _DONE
    if _DONE:
        return
    _DONE = True
    opts = enabled()
    if not opts:
        return
    unknown = opts - OPTIMIZATIONS.keys()
    if unknown:
        # Fail loudly: a typo must not silently serve stock vLLM.
        raise ValueError(
            f"atdev-serve: unknown ATDEV_OPT entries {sorted(unknown)}; "
            f"known: {sorted(OPTIMIZATIONS)}"
        )
    log.warning("atdev-serve: enabled optimizations: %s", sorted(opts))
    # Determinism: fixed FLA prefill / Inductor reduction configs instead of timing-based autotuning.
    from atdev_serve import compile_cache, pin_autotune
    pin_autotune.register()
    pin_autotune.register_inductor()
    compile_cache._SALTS["pin_autotune"] = "2"  # artifacts compiled with timing-picked configs are not reused
    from atdev_serve.compile_cache import salt_aot_cache

    for name in sorted(opts):
        module_name, func_name = OPTIMIZATIONS[name].split(":")
        module = importlib.import_module(module_name)
        # Any enabled feature may change what Dynamo traces; never let an
        # ATDEV run share torch.compile AOT artifacts with stock vLLM.
        salt_aot_cache(name, module.__file__)
        getattr(module, func_name)()
