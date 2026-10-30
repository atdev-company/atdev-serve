"""C32h-mk in-situ bit-identity check (ATDEV_C32_MKCHK=<file>; diagnostic, not for timing runs).

After every W4A4 s4 producer call (act_a4 SiLU / gate, csum_norm_a4, Triton _norm_a4_chunk) the other quotient
variant (ATDEV_C32_MK=1 deployed -> the div.rn reference, and vice versa) runs on the same inputs into scratch
buffers and the written region (s4 words, group scales; csum also ro / xb) is compared on device. Everything is
plain torch / kernel launches, so it is captured into the CUDA graphs and runs on the real activations of every
step. A daemon thread writes {kind: [calls, word_mismatch, other_mismatch]} to <file> every 10 s.
"""
from __future__ import annotations

import json
import os
import threading
import time

import torch

MK = os.environ.get("ATDEV_C32_MK", "0") == "1"
KINDS = ("silu", "gate", "csum", "tchunk")
_S: dict = {}


def _state(dev: torch.device) -> dict:
    st = _S.get(dev.index)
    if st is None:
        from atdev_serve import marlin_plus as mp
        b = mp.bufs(dev)
        st = _S[dev.index] = dict(pk=torch.empty_like(b["a4_pk"]), s=torch.empty_like(b["a4_s"]),
                                  cnt=torch.zeros(len(KINDS), 3, dtype=torch.int64, device=dev))
        threading.Thread(target=_report, args=(dev, st["cnt"]), daemon=True).start()
    return st


def _report(dev: torch.device, cnt: torch.Tensor) -> None:
    path = os.environ["ATDEV_C32_MKCHK"]
    torch.cuda.set_device(dev)
    s = torch.cuda.Stream(dev)
    while True:
        time.sleep(10)
        with torch.cuda.stream(s):
            h = cnt.to("cpu").tolist()
        with open(path, "w") as f:
            json.dump({"mk_deployed": MK, "t": time.time(), **{k: h[i] for i, k in enumerate(KINDS)}}, f)


def _cmp(cnt, i, a_pk, b_pk, a_s, b_s, n, m_pad, tmb) -> None:
    nw = n // 64 * tmb * 128
    ns = n // 128 * m_pad
    cnt[i, 0].add_(1)
    cnt[i, 1].add_((a_pk[:nw] != b_pk[:nw]).sum())
    cnt[i, 2].add_((a_s[:ns].view(torch.int32) != b_s[:ns].view(torch.int32)).sum())


def check(kind: str, run, apk, as_, n, m_pad, tmb) -> None:
    """run(pk2, s2) launches the other variant into the scratch buffers"""
    st = _state(apk.device)
    run(st["pk"], st["s"])
    _cmp(st["cnt"], KINDS.index(kind), apk, st["pk"], as_, st["s"], n, m_pad, tmb)


def check_csum(run, apk, as_, xb, ro, n, m_pad, tmb) -> None:
    """run(pk2, s2, xb2, ro2): the other csum_norm_a4 variant (idempotent on the P slots); also compares ro / xb"""
    st = _state(apk.device)
    xb2 = None if xb is None else torch.empty_like(xb)
    ro2 = torch.empty_like(ro)
    run(st["pk"], st["s"], xb2, ro2)
    i = KINDS.index("csum")
    _cmp(st["cnt"], i, apk, st["pk"], as_, st["s"], n, m_pad, tmb)
    st["cnt"][i, 2].add_((ro.view(torch.int16) != ro2.view(torch.int16)).sum())
    if xb is not None:
        st["cnt"][i, 2].add_((xb.view(torch.int16) != xb2.view(torch.int16)).sum())
