"""vLLM general plugin entry point.

Each optimization is opt-in via ATDEV_OPT (comma-separated names) so every
change can be A/B-measured against stock vLLM in the same environment.
"""
import importlib
import logging
import os

log = logging.getLogger("atdev_serve")
_DONE = False

# name -> "module:function" that installs the optimization. Registration runs
# in every vLLM process (API server, engine core, workers) before any model is
# constructed, which is what PluggableLayer / CustomOp OOT swaps require.
OPTIMIZATIONS: dict[str, str] = {
    "shadow_head": "atdev_serve.shadow_head:register",
    # reframe C: draft top-k over a frequent-token subset (docs/research/reframe-C-bytes.md)
    "draft_vocab": "atdev_serve.draft_vocab:register",
    "mtp_w4": "atdev_serve.mtp_w4:register",
    # c=32 MTP k>=2: first draft pass runs o_proj/MLP/final norm only on the last row of each request
    "mtp_prune": "atdev_serve.mtp_prune:register",
    # DI1: confidence-scheduled varlen MTP verify (MTP k=kmax drafts, fixed row budget; V2 adaptive verification)
    "di1_varlen": "atdev_serve.di1_varlen:register",
    # accuracy-gated: exact frequent-vocab + low-bit tail verify head
    "verify_split_head": "atdev_serve.verify_split_head:register",
    "verify_head_screen": "atdev_serve.verify_head_screen:register",
    # c=32 MTP: verify_split_head argmax-only greedy verify + one-pass M-split A8 tail (bit-identical)
    "vsplit_mtp": "atdev_serve.vsplit_mtp:register",
    "gdn_single_state": "atdev_serve.gdn_single_state:register",
    # c=32 W4A4: GDN out_proj s4 quant folded into the gdn_ss4 epilogue (gdn_ss4a4.cu, bit-identical)
    "gdn_a4_fold": "atdev_serve.gdn_a4_fold:register",
    "drafter_w4": "atdev_serve.drafter_w4:register",
    # A4 tree verification for DFlash2 (docs/research/tree-verify.md)
    "dflash_tree": "atdev_serve.dflash_tree:register",
    # reframe B: split-KV (3D) TRITON_ATTN for small verify batches (docs/research/reframe-B-overlap.md)
    "attn_split_verify": "atdev_serve.attn_split_verify:register",
    # P7-e (docs/research/P7-drafter.md §6.1): the <= D+1 context rows ride in the block qkv
    # Marlin tile of each drafter layer (bit-identical KV; removes 5 Marlin passes/step)
    "drafter_ctxfold": "atdev_serve.drafter_ctxfold:register",
    # P7-d (docs/research/P7-drafter.md §7): GDN metadata rebind without the per-group CPU-mask
    # gather (9 x HtoD + index + copy -> 1 copy) on pure spec-decode batches; metadata only
    "gdn_meta_fast": "atdev_serve.gdn_meta_fast:register",
    # vLLM Marlin rebuilt with tuned launch configs + fused int8 act quant (docs/research/kernel-reframe-A.md)
    "marlin_plus": "atdev_serve.marlin_plus:register",
    # KS-c1: CUDA split-KV tree-verify attention for the c=1 DFlash2 tree (+ ATDEV_TREE_ATTN_GATE / _KV folds)
    "tree_attn": "atdev_serve.tree_attn:register",
    # C32-attn: CUDA varlen decode / MTP-verify attention for c=32 (TRITON_ATTN 2D replacement; exact by default)
    "c32_attn": "atdev_serve.c32_attn:register",
    # C32b: CUDA GDN causal_conv1d_update (spec-decode form, bit-identical to the Triton kernel)
    "c32_conv": "atdev_serve.c32_conv:register",
    # bet #2 (docs/research/bet2-w3.md): 3-bit sidecar weights for selected linears + W3 Marlin kernel
    "w3_mixed": "atdev_serve.w3_mixed:register",
    # P1 fusion (docs/research/P1-fusion.md): in_proj_ba as one deterministic
    # split-K Triton GEMV (both c=1 and c=32)
    "ba_gemv": "atdev_serve.ba_gemv:register",
    # P1 fusion, c=32 W4A8: norm/act kernels emit int8 + scale, GEMMs take
    # pre-quantised A (needs marlin_plus + VLLM_MARLIN_INPUT_DTYPE=int8)
    "fused_quant": "atdev_serve.fused_quant:register",
    # E6 (docs/research/E6-c1-norms.md): c=1 residual-add + Gemma RMSNorm / SiLU*mul as one-CTA-per-row
    # CUDA kernels (custom ops) instead of the Inductor two-pass reductions
    "c1_norm": "atdev_serve.c1_norm:register",
    # E6 F3/F4: norms + SiLU*mul folded into marlin_c1 GEMM epilogues at c=1 (replaces Qwen3NextModel.forward)
    "c1_fuse": "atdev_serve.c1_fuse:register",
    # LL1 (LEDGER LL1-*): bit-identical fusions of the c=1 drafter graph glue (ATDEV_LL1_* flags)
    "ll1_fuse": "atdev_serve.ll1_fuse:register",
    # LL5 (LEDGER LL5-*): remaining c=1 glue (drafter conv.finish + add-RMSNorm fusion, ...), ATDEV_LL5_* flags
    "ll5_fuse": "atdev_serve.ll5_fuse:register",
    # C32k (LEDGER C32k-*): c=32 MTP-pass / verify-screen glue cuts, bit-identical (ATDEV_C32K_* flags)
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
        # Fail loudly: a typo must not silently benchmark stock vLLM.
        raise ValueError(
            f"atdev-serve: unknown ATDEV_OPT entries {sorted(unknown)}; "
            f"known: {sorted(OPTIMIZATIONS)}"
        )
    log.warning("atdev-serve: enabled optimizations: %s", sorted(opts))
    if os.environ.get("ATDEV_PIN_AUTOTUNE", "1") != "0":  # engine determinism: fixed FLA prefill configs
        from atdev_serve import compile_cache, pin_autotune
        pin_autotune.register()
        pin_autotune.register_inductor()
        compile_cache._SALTS["pin_autotune"] = "2"  # artifacts compiled with timing-picked reductions are not reused
    if os.environ.get("ATDEV_INDUCTOR_DET", "0") == "1":  # engine determinism: no numerics-changing Inductor tuning
        import torch._inductor.config as icfg
        icfg.deterministic = True
        from atdev_serve import compile_cache
        compile_cache._SALTS["inductor_det"] = "1"
        log.warning("atdev-serve: torch._inductor.config.deterministic = True")
    if os.environ.get("ATDEV_FILL_UNINIT") == "1":  # diagnostic: torch.empty -> filled (NaN for floats)
        import torch
        import torch.utils.deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        log.warning("atdev-serve: DIAGNOSTIC fill_uninitialized_memory=True (deterministic algorithms, warn_only)")
    from atdev_serve.compile_cache import salt_aot_cache

    for name in sorted(opts):
        module_name, func_name = OPTIMIZATIONS[name].split(":")
        module = importlib.import_module(module_name)
        # Any enabled optimization may change what Dynamo traces; never let an
        # ATDEV run share torch.compile AOT artifacts with stock vLLM.
        salt_aot_cache(name, module.__file__)
        getattr(module, func_name)()
