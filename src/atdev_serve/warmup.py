"""Start-up JIT warmup for the multi-turn c1 profile (``python -m atdev_serve.warmup``).

vLLM reports "ready" before the Triton kernels of the tree / prefix-cache path have been compiled
(kernel_unified_attention, the DFlash input and tree kernels, the verify and compaction kernels:
12 kernels in all). Without a warmup the first real request pays for their compilation (about
4.7 s with an empty Triton cache, 0.1-0.4 s with a populated one). serve_c1.sh starts this module in
the background; it waits for the server, sends a small set of synthetic multi-turn conversations
(several prompt lengths, growing history so that prefix-cache hits occur, 64 decoded tokens each),
then a few sampled and concurrent requests
and exits. It uses only the standard library, needs no GPU, and leaves the answer text unused.

The synthetic prompts are random English words, so they never match real user prompts; they only
occupy a few prefix-cache blocks that are evicted first (LRU).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

_WORDS = (
    "system river window engine paper garden signal market planet bridge forest window memory orange "
    "silver thunder pattern harbor village window mirror camera ladder pocket castle desert island "
    "winter summer coffee guitar rocket button ticket lemon violin anchor candle meadow canvas copper "
    "dragon feather glacier harvest journal kitchen lantern monarch needle oyster pebble quartz ribbon "
    "saddle timber umbrella velvet walnut yellow zephyr airport balcony cabinet diamond elephant"
).split()

# (system prompt tokens, number of turns); the lengths straddle 480-token attention-block and
# 2048-token prefill-chunk boundaries so that the aligned and unaligned launch variants compile
LENGTHS = ((90, 2), (700, 3), (1900, 3), (3900, 3), (7300, 2))


def log(msg: str) -> None:
    print(f"atdev warmup: {msg}", file=sys.stderr, flush=True)


def _text(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(max(1, int(tokens / 1.35))))


def _post(base: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie (exited, not yet reaped by its parent) is gone as far as we are concerned
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


# c32 profile: shorter sequential part, then waves of concurrent multi-turn conversations (the batch shapes of 32 sessions)
LENGTHS_C32 = ((90, 2), (1900, 3), (3900, 3))


def _concurrent_sessions(base: str, model: str, max_tokens: int, seed: int, width: int) -> int:
    from concurrent.futures import ThreadPoolExecutor

    def session(i: int) -> int:
        rng = random.Random(seed * 1000 + i)
        msgs = [{"role": "system", "content": "You are a careful assistant. Notes: " + _text(rng, 1200 + 350 * (i % 9))}]
        for turn in range(3):
            msgs.append({"role": "user", "content": f"Turn {turn + 1}. " + _text(rng, 60 + 40 * turn) + " Continue the list of words above."})
            r = _post(base, "/v1/chat/completions", dict(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=0, ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False)))
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
        return 3

    with ThreadPoolExecutor(width) as ex:
        return sum(ex.map(session, range(width)))


def run(base: str, model: str, max_tokens: int, seed: int, concurrency: int = 0) -> tuple[int, float]:
    rng = random.Random(seed)
    n = 0
    t0 = time.time()
    for sys_tokens, turns in (LENGTHS_C32 if concurrency else LENGTHS):
        msgs = [{"role": "system", "content": "You are a careful assistant. Notes: " + _text(rng, sys_tokens)}]
        for turn in range(turns):
            msgs.append({"role": "user", "content": f"Turn {turn + 1}. " + _text(rng, 40 + 25 * turn)
                         + " Continue the list of words above."})
            r = _post(base, "/v1/chat/completions", dict(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=0, ignore_eos=True,
                chat_template_kwargs=dict(enable_thinking=False)))
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
            n += 1
    # sampled decoding and a small concurrent burst compile the rejection / conv-update kernels
    # that a client with temperature > 0 or a few parallel requests would otherwise hit first
    from concurrent.futures import ThreadPoolExecutor

    def one(i: int) -> None:
        _post(base, "/v1/chat/completions", dict(
            model=model, max_tokens=24, ignore_eos=True, temperature=0.7 if i % 2 else 0.0, top_p=0.9,
            messages=[{"role": "user", "content": _text(random.Random(seed + i), 60 + 30 * i)}],
            chat_template_kwargs=dict(enable_thinking=False)))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(8)))
    n += 8
    if concurrency:  # two waves: the first compiles the batch-shape kernels, the second the mixed prefill / decode steps
        for wave in range(2):
            n += _concurrent_sessions(base, model, max_tokens, seed + wave, concurrency)
    return n, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="m", help="served model name")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="c32 profile: also send waves of this many concurrent multi-turn sessions (0 = c1 profile)")
    ap.add_argument("--watch-pid", type=int, default=0, help="exit when this process (the server) is gone")
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="seconds to wait for the server")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen(base + "/v1/models", timeout=5).read()
            break
        except Exception:
            if not _alive(a.watch_pid) or time.time() - t0 > a.timeout:
                return 1
            time.sleep(2)
    log(f"server ready, sending synthetic conversations to {base}")
    try:
        n, dt = run(base, a.model, a.max_tokens, a.seed, a.concurrency)
    except Exception as e:  # warmup is best effort: the server stays usable
        log(f"stopped early ({type(e).__name__}: {e}); the server is still usable")
        return 0
    log(f"done: {n} requests in {dt:.1f} s, first-request JIT latency removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
