#!/usr/bin/env python3
"""P6 paired per-request comparison of two c=1 bench/run.sh result dirs (same prompt set).

  python bench/paired_c1.py results/REF results/RUN [--n N] [--boot 10000] [--seed 0] [--json out.json]

Pairs prompts by index (prompts.jsonl == first 24 of prompts96.jsonl, so a 24-run can be
paired with a 96-run on the shared prefix; --n limits to the first N prompts). Reps are
averaged per prompt first (per-prompt decode tok/s = mean over reps), then:
  * per-prompt ratio r_i = run_i / ref_i  -> mean, median, geometric mean
  * identical-output count (text equal in every paired rep) and the mean ratio on that subset
    (pure kernel-speed effect: no acceptance reshuffle) and on the rest (reshuffle included)
  * bootstrap (resample prompts with replacement) CI of the mean ratio and of the
    pooled-wall ratio (sum decoded tokens / sum decode seconds, i.e. time-weighted)
  * summary statistics of both runs: median / mean / trimmed mean (10% each side) / pooled wall
Verdict line: the proposed P6 KEEP rule is paired mean ratio CI (95%) excluding 1.0 and the
point estimate >= +1.5%.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys


def load(d):
    reps = []
    for f in sorted(glob.glob(os.path.join(d, "c1_r*.json"))):
        j = json.load(open(f))
        reps.append(j["requests"])
    if not reps:
        sys.exit(f"no c1_r*.json in {d}")
    return reps


def per_prompt(reps, n):
    """-> list of dicts per prompt: tps (mean over reps), tokens, decode_s, texts"""
    out = []
    for i in range(n):
        rs = [r[i] for r in reps]
        tps = [x["decode_tps"] for x in rs if x.get("decode_tps")]
        dec_s = [x.get("decode_s", ((x["tokens"] - 1) / x["decode_tps"]) if x.get("decode_tps") else 0.0) for x in rs]
        out.append({
            "tps": statistics.mean(tps) if tps else float("nan"),
            "tokens": statistics.mean(x["tokens"] or 0 for x in rs),
            "dec_tokens": sum((x["tokens"] or 1) - 1 for x in rs),
            "dec_s": sum(dec_s),
            "texts": [x["text"] for x in rs],
        })
    return out


def stats(vals):
    v = sorted(vals)
    k = int(len(v) * 0.1)
    tr = v[k:len(v) - k] if k else v
    return {"median": statistics.median(v), "mean": statistics.mean(v), "trimmed_mean": statistics.mean(tr)}


def bootstrap(pairs, fn, B, seed):
    import random
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn([pairs[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def server_tau(d, num_spec=15):
    """Mean acceptance length over the server's SpecDecoding metric lines (10 s windows):
    1 + sum(accepted) / sum(drafted / num_spec). Lines are restricted to the c=1 bench windows when the
    rep summaries carry t_start/t_end (bench/client.py >= 2026-10-08); otherwise the whole log is used
    (then GSM8K/longctx runs in the same log pollute it). Returns (tau, filtered?)."""
    import re, time, datetime
    f = os.path.join(d, "server.log")
    if not os.path.exists(f):
        return None, False
    wins = []
    for g in sorted(glob.glob(os.path.join(d, "c1_r*.json"))):
        sm = json.load(open(g))["summary"]
        if "t_start" in sm:
            wins.append((sm["t_start"] - 1, sm["t_end"] + 12))
    year = datetime.datetime.now().year
    acc = drafted = 0
    for line in open(f, errors="replace"):
        m = re.search(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*Accepted: (\d+) tokens, Drafted: (\d+) tokens", line)
        if not m:
            continue
        if wins:
            mo, da, hh, mi, ss = (int(x) for x in m.groups()[:5])
            ts = time.mktime((year, mo, da, hh, mi, ss, 0, 0, -1))
            if not any(a <= ts <= b for a, b in wins):
                continue
        acc += int(m.group(6)); drafted += int(m.group(7))
    return ((1 + acc / (drafted / num_spec)) if drafted else None), bool(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=0, help="pair only the first N prompts")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    ap.add_argument("--gate", type=float, default=1.5, help="KEEP gate on the mean ratio, in percent")
    ap.add_argument("--num-spec", type=int, default=15, help="drafted tokens per step for the server-tau line")
    a = ap.parse_args()
    R, X = load(a.ref), load(a.run)
    n = min(len(R[0]), len(X[0]))
    if a.n:
        n = min(n, a.n)
    nrep = min(len(R), len(X))
    pr, px = per_prompt(R, n), per_prompt(X, n)
    pairs = []
    for i, (r, x) in enumerate(zip(pr, px)):
        ident = all(r["texts"][k] == x["texts"][k] for k in range(nrep))
        pairs.append({"i": i, "ref": r["tps"], "run": x["tps"], "ratio": x["tps"] / r["tps"],
                      "ident": ident, "ref_tok": r["tokens"], "run_tok": x["tokens"],
                      "ref_dec_tok": r["dec_tokens"], "ref_dec_s": r["dec_s"],
                      "run_dec_tok": x["dec_tokens"], "run_dec_s": x["dec_s"]})
    ratios = [p["ratio"] for p in pairs]
    mean_ratio = statistics.mean(ratios)
    med_ratio = statistics.median(ratios)
    gmean = math.exp(statistics.mean(math.log(r) for r in ratios))

    def wall_ratio(ps):
        rw = sum(p["ref_dec_tok"] for p in ps) / sum(p["ref_dec_s"] for p in ps)
        xw = sum(p["run_dec_tok"] for p in ps) / sum(p["run_dec_s"] for p in ps)
        return xw / rw

    ci_mean = bootstrap(pairs, lambda ps: statistics.mean(p["ratio"] for p in ps), a.boot, a.seed)
    ci_wall = bootstrap(pairs, wall_ratio, a.boot, a.seed + 1)
    ident = [p for p in pairs if p["ident"]]
    diff = [p for p in pairs if not p["ident"]]
    sr, sx = stats([p["ref"] for p in pairs]), stats([p["run"] for p in pairs])
    sr["wall"] = sum(p["ref_dec_tok"] for p in pairs) / sum(p["ref_dec_s"] for p in pairs)
    sx["wall"] = sum(p["run_dec_tok"] for p in pairs) / sum(p["run_dec_s"] for p in pairs)

    print(f"paired c=1: ref={a.ref} run={a.run}  prompts={n} reps={nrep} (per-prompt tok/s = mean over reps)")
    print(f"{'':14s} {'median':>9s} {'mean':>9s} {'trim10':>9s} {'wall':>9s}   (tok/s)")
    print(f"{'ref':14s} {sr['median']:9.1f} {sr['mean']:9.1f} {sr['trimmed_mean']:9.1f} {sr['wall']:9.1f}")
    print(f"{'run':14s} {sx['median']:9.1f} {sx['mean']:9.1f} {sx['trimmed_mean']:9.1f} {sx['wall']:9.1f}")
    print(f"{'run/ref':14s} {sx['median']/sr['median']:9.4f} {sx['mean']/sr['mean']:9.4f} "
          f"{sx['trimmed_mean']/sr['trimmed_mean']:9.4f} {sx['wall']/sr['wall']:9.4f}")
    print(f"per-prompt ratio: mean {mean_ratio:.4f}  [95% CI {ci_mean[0]:.4f}, {ci_mean[1]:.4f}]  "
          f"median {med_ratio:.4f}  gmean {gmean:.4f}  min {min(ratios):.3f} max {max(ratios):.3f}")
    print(f"pooled wall ratio: {wall_ratio(pairs):.4f}  [95% CI {ci_wall[0]:.4f}, {ci_wall[1]:.4f}]")
    if ident:
        mi = statistics.mean(p["ratio"] for p in ident)
        ci_i = bootstrap(ident, lambda ps: statistics.mean(p["ratio"] for p in ps), a.boot, a.seed + 2) if len(ident) > 1 else (mi, mi)
        print(f"identical outputs: {len(ident)}/{n}  mean ratio on identical {mi:.4f} [{ci_i[0]:.4f}, {ci_i[1]:.4f}]"
              + (f"  on differing {statistics.mean(p['ratio'] for p in diff):.4f} (n={len(diff)})" if diff else ""))
    else:
        print(f"identical outputs: 0/{n} (all outputs differ -> acceptance reshuffle present)")
    tok_r = sum(p["ref_tok"] for p in pairs); tok_x = sum(p["run_tok"] for p in pairs)
    print(f"output tokens: ref {tok_r:.0f} run {tok_x:.0f} ({tok_x/tok_r:.4f})")
    (tr, fr), (tx, fx) = server_tau(a.ref, a.num_spec), server_tau(a.run, a.num_spec)
    if tr and tx:
        scope = "bench windows" if (fr and fx) else "whole log (unfiltered: includes warmup/acc runs)"
        print(f"server tau ({scope}): ref {tr:.3f} run {tx:.3f} ({tx/tr:.4f})")
    gate = 1 + a.gate / 100
    if ci_mean[0] > 1.0 and mean_ratio >= gate:
        verdict = "KEEP (CI excludes 1.0 and mean ratio >= gate)"
    elif ci_mean[1] < 1.0:
        verdict = "WORSE (CI below 1.0)"
    elif ci_mean[0] > 1.0:
        verdict = f"BETTER but below the +{a.gate:g}% gate"
    else:
        verdict = "NEUTRAL (CI includes 1.0)"
    print(f"verdict: {verdict}")
    if a.json:
        json.dump({"ref": a.ref, "run": a.run, "n": n, "reps": nrep, "ref_stats": sr, "run_stats": sx,
                   "mean_ratio": mean_ratio, "ci_mean": ci_mean, "median_ratio": med_ratio, "gmean": gmean,
                   "wall_ratio": wall_ratio(pairs), "ci_wall": ci_wall, "identical": len(ident),
                   "verdict": verdict, "pairs": pairs}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
