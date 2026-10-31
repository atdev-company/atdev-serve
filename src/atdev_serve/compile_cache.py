"""Keep vLLM's AOT torch.compile cache honest for graph-changing optimizations.

vLLM 0.31 (``VLLM_USE_AOT_COMPILE``) stores the compiled model under ``torch_aot_compile/<hash>``, where the
hash covers vLLM env factors, the VllmConfig and the forward's qualname, but *not* the plugin code that Dynamo
inlined. On load it only re-checks the source files recorded inside the artifact. So an optimization that
changes what Dynamo traces would silently reuse a stock artifact, and a later stock run could load a plugin
artifact. ``salt_aot_cache(name, module_file)`` appends the optimization name and a hash of its source file
to the AOT hash factors, in this process only; modules add extra entries to ``_SALTS`` for settings that
change the traced graph.
"""
from __future__ import annotations

import functools
import hashlib

_SALTS: dict[str, str] = {}


def salt_aot_cache(name: str, module_file: str) -> None:
    from vllm.compilation import caching

    with open(module_file, "rb") as f:
        _SALTS[name] = hashlib.sha256(f.read()).hexdigest()[:16]
    fn = caching.aot_compile_hash_factors
    if getattr(fn, "_atdev_salted", False):
        return

    @functools.wraps(fn)
    def aot_compile_hash_factors(vllm_config):
        factors = list(fn(vllm_config))
        factors.append("atdev:" + ",".join(f"{k}={v}" for k, v in sorted(_SALTS.items())))
        return factors

    aot_compile_hash_factors._atdev_salted = True
    caching.aot_compile_hash_factors = aot_compile_hash_factors

