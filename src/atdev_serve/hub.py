"""Resolve model-side files given either a local path or a Hugging Face Hub spec.

Used by the release build for the two optional model-side artifacts. A Hub spec is

    org/name[@revision][:path/in/repo]

where ``path/in/repo`` is either one file (``w3/w3_gptqact_d30_mlpo.safetensors``) or a
subfolder (``w3/`` or ``w3``). The released files live in one repo, one subfolder each:

  ATDEV_W3_SIDECAR          (w3_mixed)   comma-separated list; each entry is
                                         * a local .safetensors file, or
                                         * ``org/name[@rev]``: every ``*.safetensors`` at the repo root,
                                         * ``org/name[@rev]:folder/``: every ``*.safetensors`` directly in
                                           that subfolder (sorted by name), e.g.
                                           ``atdev-company/Qwen3.8-27B-atdev-c1:w3/``, or
                                         * ``org/name[@rev]:path/file.safetensors`` (one file)
  ATDEV_DRAFTER_QUANT_PATH  (drafter_w4) a local checkpoint directory, ``org/name[@rev]`` (the repo
                                         root) or ``org/name[@rev]:folder/`` (that subfolder), e.g.
                                         ``atdev-company/Qwen3.8-27B-atdev-c1:drafter/``

Only the files that are needed are downloaded (``allow_patterns``). Downloads go through
``huggingface_hub`` (honours HF_HOME, HF_TOKEN, HF_HUB_OFFLINE). A complete local cache copy is used without any
network request; otherwise short Hub timeouts apply. Each vLLM process resolves independently.
"""
from __future__ import annotations

import functools
import os
import re

# A slow or unreachable Hub must not stall start-up: short metadata timeouts, and files that are already in the local cache
# are used without any network request (the Hub is only asked when the cache has no complete copy).
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _split(spec: str) -> tuple[str, str | None, str | None]:
    """'org/name@rev:sub/file' -> ('org/name', 'rev', 'sub/file')"""
    path = None
    if ":" in spec:
        spec, path = spec.split(":", 1)
    rev = None
    if "@" in spec:
        spec, rev = spec.split("@", 1)
    return spec, rev or None, path or None


def _folder(path: str | None) -> str:
    """'' for the repo root, else 'sub/dir/' (normalized, no leading slash)."""
    p = (path or "").strip().strip("/")
    if ".." in p.split("/"):
        raise ValueError(f"atdev hub: invalid subfolder {path!r}")
    return p + "/" if p else ""


def is_repo_id(spec: str) -> bool:
    if not spec or os.path.exists(spec) or spec.startswith(("/", ".", "~")):
        return False
    repo, _, _ = _split(spec)
    return bool(_REPO_RE.match(repo))


@functools.lru_cache(maxsize=None)
def snapshot_dir(repo: str, rev: str | None, allow_patterns: tuple[str, ...] | None = None) -> str:
    from huggingface_hub import snapshot_download

    kw = dict(repo_id=repo, revision=rev, allow_patterns=list(allow_patterns) if allow_patterns else None)
    try:
        return snapshot_download(local_files_only=True, **kw)  # complete cached copy: no network
    except Exception:
        return snapshot_download(**kw)


@functools.lru_cache(maxsize=None)
def resolve_files(spec: str, suffix: str = ".safetensors") -> tuple[str, ...]:
    """One sidecar entry -> local file path(s)."""
    spec = spec.strip()
    if not is_repo_id(spec):
        return (os.path.expanduser(spec),)
    repo, rev, path = _split(spec)
    if path and path.endswith(suffix):
        from huggingface_hub import hf_hub_download

        try:
            return (hf_hub_download(repo_id=repo, filename=path, revision=rev, local_files_only=True),)  # cached: no network
        except Exception:
            return (hf_hub_download(repo_id=repo, filename=path, revision=rev),)
    sub = _folder(path)
    d = os.path.join(snapshot_dir(repo, rev, (f"{sub}*{suffix}",)), sub)
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(suffix)) if os.path.isdir(d) else []
    if not files:
        raise FileNotFoundError(f"atdev hub: no *{suffix} files in HF repo {spec}")
    return tuple(files)


@functools.lru_cache(maxsize=None)
def resolve_dir(spec: str) -> str:
    """Checkpoint directory: local path unchanged, HF spec -> (subfolder of the) snapshot directory."""
    if not is_repo_id(spec):
        return os.path.expanduser(spec)
    repo, rev, path = _split(spec)
    sub = _folder(path)
    d = os.path.join(snapshot_dir(repo, rev, (f"{sub}*.json", f"{sub}*.safetensors")), sub)
    if not os.path.isfile(os.path.join(d, "config.json")):
        raise FileNotFoundError(f"atdev hub: no config.json in HF repo {spec}")
    return d.rstrip("/")
