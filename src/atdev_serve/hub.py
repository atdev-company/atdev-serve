"""Resolve model-side files given either a local path or a Hugging Face Hub repo id.

Used by the release build for the two optional model-side artifacts:

  ATDEV_W3_SIDECAR          (w3_mixed)   comma-separated list; each entry is
                                         * a local .safetensors file, or
                                         * an HF repo id (``org/name`` or ``org/name@revision``):
                                           every ``*.safetensors`` file of the repo, sorted by name, or
                                         * ``org/name[@revision]:path/in/repo.safetensors`` (one file)
  ATDEV_DRAFTER_QUANT_PATH  (drafter_w4) a local checkpoint directory or an HF repo id
                                         (``org/name[@revision]``) -> its snapshot directory

Downloads go through ``huggingface_hub`` (honours HF_HOME, HF_TOKEN, HF_HUB_OFFLINE). Each vLLM
process resolves independently; after the first download every call is a local cache hit.
"""
from __future__ import annotations

import functools
import os
import re

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _split(spec: str) -> tuple[str, str | None, str | None]:
    """'org/name@rev:sub/file' -> ('org/name', 'rev', 'sub/file')"""
    file = None
    if ":" in spec:
        spec, file = spec.split(":", 1)
    rev = None
    if "@" in spec:
        spec, rev = spec.split("@", 1)
    return spec, rev or None, file or None


def is_repo_id(spec: str) -> bool:
    if not spec or os.path.exists(spec) or spec.startswith(("/", ".", "~")):
        return False
    repo, _, _ = _split(spec)
    return bool(_REPO_RE.match(repo))


@functools.lru_cache(maxsize=None)
def snapshot_dir(spec: str, allow_patterns: tuple[str, ...] | None = None) -> str:
    from huggingface_hub import snapshot_download

    repo, rev, _ = _split(spec)
    return snapshot_download(repo_id=repo, revision=rev,
                             allow_patterns=list(allow_patterns) if allow_patterns else None)


@functools.lru_cache(maxsize=None)
def resolve_files(spec: str, suffix: str = ".safetensors") -> tuple[str, ...]:
    """One sidecar entry -> local file path(s)."""
    spec = spec.strip()
    if not is_repo_id(spec):
        return (os.path.expanduser(spec),)
    repo, rev, file = _split(spec)
    if file:
        from huggingface_hub import hf_hub_download

        return (hf_hub_download(repo_id=repo, filename=file, revision=rev),)
    d = snapshot_dir(spec, (f"*{suffix}",))
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(suffix))
    if not files:
        raise FileNotFoundError(f"atdev hub: no *{suffix} files in HF repo {spec}")
    return tuple(files)


@functools.lru_cache(maxsize=None)
def resolve_dir(spec: str) -> str:
    """Checkpoint directory: local path unchanged, HF repo id -> snapshot directory."""
    if not is_repo_id(spec):
        return os.path.expanduser(spec)
    return snapshot_dir(spec, ("*.json", "*.safetensors"))
