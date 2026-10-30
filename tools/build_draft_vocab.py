"""Rebuild the draft-vocabulary frequency order (atdev_serve/data/token_order.npy).

The draft_vocab / verify_split_head features take the first ATDEV_DRAFT_VOCAB_SIZE / ATDEV_VSPLIT_SIZE
ids of this order as the "frequent" sub-vocabulary. It only selects speculative candidates: the final
greedy token is always checked against the full target lm_head, so a different order changes speed
(acceptance), not the output distribution of the exact parts.

Input: one or more JSONL files of *target-generated* responses, one object per line with
``output_token_ids`` (list of int), e.g. produced by running the target through vLLM with
``return_token_ids``. Output: all vocab ids by descending frequency, ties / unseen ids by ascending id.

usage: python tools/build_draft_vocab.py gen1.jsonl [gen2.jsonl ...] --out token_order.npy [--vocab 248320]
"""
import argparse
import collections
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--out", default="token_order.npy")
    a = ap.parse_args()
    cnt = collections.Counter()
    for path in a.inputs:
        with open(path) as f:
            for line in f:
                cnt.update(json.loads(line)["output_token_ids"])
    freq = np.zeros(a.vocab, dtype=np.int64)
    for t, c in cnt.items():
        if 0 <= t < a.vocab:
            freq[t] = c
    order = np.lexsort((np.arange(a.vocab), -freq)).astype(np.int32)  # last key is primary
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, order)
    print(f"{sum(cnt.values())} tokens, {len(cnt)} distinct ids -> {a.out}")


if __name__ == "__main__":
    main()
