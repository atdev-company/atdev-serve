"""Pin the Triton ``@triton.autotune`` choices of the FLA chunked GDN prefill kernels (engine determinism).

With an empty / different Triton cache every server start re-benchmarks FLA's chunked prefill kernels
(chunk_fwd_kernel_o, chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_scaled_dot_kkt_fwd_kernel,
chunk_local_cumsum_scalar_kernel, merge_16x16_to_64x64_inverse_kernel, recompute_w_u_fwd_kernel). The winner
depends on timing noise, and some knobs change the reduction blocking, so the prefill state and every later token
would depend on which configs a given start happened to pick.

Fix: restrict each of these kernels to ONE config, so Triton never benchmarks them (``len(configs) == 1`` -> no
autotune) and every start computes the same prefill numerics. register_inductor() does the same for Inductor's
reduction autotuning. Enabled for every run with the plugin active (register_all); stock vLLM is untouched.
"""
from __future__ import annotations

import logging

log = logging.getLogger("atdev_serve")

# kernel name -> (kwargs, num_warps, num_stages)
PINS = {
    "chunk_local_cumsum_scalar_kernel": ({}, 4, 3),
    "chunk_scaled_dot_kkt_fwd_kernel": ({"BK": 32}, 8, 4),
    "merge_16x16_to_64x64_inverse_kernel": ({}, 4, 2),
    "recompute_w_u_fwd_kernel": ({}, 4, 4),
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ({"BV": 64}, 4, 2),
    "chunk_fwd_kernel_o": ({"BK": 64, "BV": 64}, 4, 2),
}
_S = {"pinned": set(), "missing": set()}


def _match(cfg, kw, nw, ns):
    return dict(cfg.kwargs) == kw and cfg.num_warps == nw and cfg.num_stages == ns


def register() -> None:
    from triton.runtime import autotuner as at
    if getattr(at.Autotuner.run, "_atdev_pin", False):
        return
    orig = at.Autotuner.run

    def run(self, *args, **kwargs):
        name = getattr(self.base_fn, "__name__", "")
        pin = PINS.get(name)
        if pin is not None and name not in _S["pinned"] and len(self.configs) > 1:
            keep = [c for c in self.configs if _match(c, *pin)]
            if keep:
                self.configs = keep[:1]
                self.cache = {}
                _S["pinned"].add(name)
                log.warning("atdev pin_autotune: %s -> %s (no benchmarking)", name, keep[0])
            elif name not in _S["missing"]:
                _S["missing"].add(name)
                log.warning("atdev pin_autotune: %s pin %s not among its configs -> stock autotune", name, pin)
        return orig(self, *args, **kwargs)

    run._atdev_pin = True
    at.Autotuner.run = run


def register_inductor() -> None:
    """Inductor's generated reduction kernels (e.g. the fused add+RMSNorm) are benchmarked over several
    (R0_BLOCK, num_warps) configs at their first run; the timing-noise winner changes the reduction blocking and is
    persisted in the compiled artifact (.best_config) -> a fresh compile lands in one of several numerical modes
    (probe MP-pin*/pd*: R0_BLOCK 2048 vs 4096 on the first start of a window). Replace the benchmark for reduction
    kernels by Inductor's own deterministic choice (filter_reduction_configs_for_determinism, forced); pointwise
    kernels (numerics-neutral) keep timing-based selection."""
    from torch._inductor.runtime import triton_heuristics as th
    CA = th.CachingAutotuner
    if getattr(CA.benchmark_all_configs, "_atdev_pin", False):
        return
    orig = CA.benchmark_all_configs
    RED = {th.HeuristicType.REDUCTION, th.HeuristicType.PERSISTENT_REDUCTION}

    def benchmark_all_configs(self, *args, **kwargs):
        if getattr(self, "heuristic_type", None) in RED and len(self.launchers) > 1:
            cfgs = [l.config for l in self.launchers]
            meta = dict(self.inductor_meta or {}, force_filter_reduction_configs=True)
            pick = th.filter_reduction_configs_for_determinism(meta, cfgs)[0]
            win = next((l for l in self.launchers if l.config is pick or str(l.config) == str(pick)), self.launchers[0])
            if not _S.get("ind_logged"):
                _S["ind_logged"] = True
                log.warning("atdev pin_autotune: Inductor reduction autotune -> deterministic pick (e.g. %s: %s)",
                            self.fn.__name__, win.config)
            return {l: (0.0 if l is win else 1.0) for l in self.launchers}
        return orig(self, *args, **kwargs)

    benchmark_all_configs._atdev_pin = True
    CA.benchmark_all_configs = benchmark_all_configs
