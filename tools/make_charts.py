#!/usr/bin/env python3
"""Render the README / docs charts from docs/results.json.

usage:  python tools/make_charts.py [--results docs/results.json] [--out docs/img]
needs:  matplotlib (any recent version; not needed for serving)

Writes <name>.svg / <name>.png (light) and <name>-dark.svg / <name>-dark.png for:
  ladder_c1, ladder_c32     baseline -> best: stock vLLM defaults, best vLLM settings, atdev engine only, full stack
  progress_c1, progress_c32 development milestones (accepted same-session A/B steps)
  step_breakdown            where the time of one decode step goes (profiler)
Values that are null in results.json are drawn as dashed "pending" bars.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Colors: one hue per kind of result, gray for stock vLLM. Checked with a CVD validator
# (blue / orange adjacent pair: CVD dE 24.7 light, 26.8 dark).
THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df",
        "stock": "#b9b8b2", "best_vllm": "#6f6e69", "engine": "#2a78d6", "full": "#eb6834",
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e88", "grid": "#33332f",
        "stock": "#5e5d58", "best_vllm": "#9a9992", "engine": "#3987e5", "full": "#d95926",
        "cat": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}
KIND_LABEL = {
    "stock": "stock vLLM 0.31.0, default flags",
    "best_vllm": "stock vLLM 0.31.0, best settings found",
    "engine": "atdev-serve, engine only (same model files)",
    "full": "atdev-serve, full stack (engine + model-side changes)",
}


def _style(t):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "svg.fonttype": "path",
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["text2"], "xtick.color": t["text2"], "ytick.color": t["text"],
        "axes.edgecolor": t["grid"],
    })


def _axes_clean(ax, t):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out, name, mode):
    suffix = "" if mode == "light" else "-dark"
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(out, f"{name}{suffix}.{ext}"), dpi=160, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _ink(hex_color):
    """Dark or white label text, whichever reads better on the fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.42 else "#ffffff"


def _pending_bar(ax, y, width, h, t):
    ax.barh(y, width, height=h, color="none", edgecolor=t["muted"], linestyle=(0, (4, 3)), linewidth=1.2,
            hatch="//", zorder=2)


def ladder(res, conc, out, mode):
    t = THEMES[mode]
    _style(t)
    rows = res["ladder"][conc]
    unit = "decode tok/s per request (mean, 96 prompts)" if conc == "c1" else "aggregate output tok/s (96 prompts, 32 in flight)"
    title = ("Single user (c=1)" if conc == "c1" else "32 concurrent requests (c=32)") + \
        " - Qwen3.8-27B on one A100 80GB"
    base = rows[0]["tps"]
    best = next(r["tps"] for r in rows if r["key"] == "best_vllm")
    known = [r["tps"] for r in rows if r["tps"] is not None]
    xmax = max(known) * 1.42
    fig, ax = plt.subplots(figsize=(9.2, 0.62 * len(rows) + 1.25))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        v = r["tps"]
        if v is None:
            _pending_bar(ax, y, max(known) * 1.05, 0.62, t)
            ax.text(max(known) * 1.05 + xmax * 0.012, y, "pending", va="center", ha="left", color=t["muted"],
                    fontsize=10)
            continue
        ax.barh(y, v, height=0.62, color=t[r["kind"]], zorder=2)
        lab = f"{v:,.1f}"
        mult = f"   {v / base:.1f}x vs bf16 default" if r is not rows[0] else ""
        if r["kind"] in ("engine", "full"):
            mult = f"   {v / best:.2f}x vs best vLLM  |  {v / base:.1f}x vs bf16 default"
        ax.text(v + xmax * 0.012, y, lab, va="center", ha="left", color=t["text"], fontsize=10.5, fontweight="bold")
        ax.text(v + xmax * 0.012 + xmax * 0.0125 * len(lab), y, mult, va="center", ha="left", color=t["text2"], fontsize=9.2)
    for y, r in zip(ys, rows):  # mark our rows: tinted band, "OURS" tag inside the bar, bold tick label
        if r["kind"] in ("engine", "full") and r["tps"] is not None:
            ax.axhspan(y - 0.46, y + 0.46, color=t[r["kind"]], alpha=0.13, zorder=0, linewidth=0)
            ax.text(xmax * 0.012, y, "OURS", va="center", ha="left", color=_ink(t[r["kind"]]),
                    fontsize=10, fontweight="heavy", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([("OURS · " if r["kind"] in ("engine", "full") else "") + r["label"] for r in rows], fontsize=9.6)
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r["kind"] in ("engine", "full"):
            lab.set_fontweight("bold"); lab.set_color(t[r["kind"]])
    ax.set_xlim(0, xmax)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=t["text"], pad=10)
    _axes_clean(ax, t)
    kinds = [k for k in KIND_LABEL if any(r["kind"] == k for r in rows)]
    ax.legend(handles=[Patch(color=t[k], label=KIND_LABEL[k]) for k in kinds], loc="upper left",
              bbox_to_anchor=(-0.02, -0.16), ncol=2, frameon=False, fontsize=8.6, labelcolor=t["text2"])
    _save(fig, out, f"ladder_{conc}", mode)


def progress(res, conc, out, mode):
    t = THEMES[mode]
    _style(t)
    rows = res["progress"][conc]
    key = "tps" if conc == "c1" else "ratio"
    known = [r[key] for r in rows if r[key] is not None]
    lo = 0.0
    hi = max(known) * (1.08 if conc == "c1" else 1.06)
    fig, ax = plt.subplots(figsize=(9.6, 0.36 * len(rows) + 1.3))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        v = r[key]
        if v is None:
            ax.barh(y, known[-1] - lo, left=lo, height=0.66, color="none", edgecolor=t["muted"],
                    linestyle=(0, (4, 3)), linewidth=1.1, hatch="//", zorder=2)
            ax.text(known[-1] + (hi - lo) * 0.01, y, "pending", va="center", fontsize=8.8, color=t["muted"])
            continue
        ax.barh(y, v - lo, left=lo, height=0.66, color=t[r["kind"]], zorder=2)
        s = f"{v:.1f}" if conc == "c1" else f"{v:.3f}x"
        ax.text(v + (hi - lo) * 0.01, y, s, va="center", fontsize=8.8, color=t["text"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("decode tok/s per request (mean, 96 prompts)" if conc == "c1"
                  else "aggregate output tok/s relative to best vLLM (1757.1 tok/s)")
    ttl = "How c=1 got there" if conc == "c1" else "How c=32 got there"
    ax.set_title(ttl + " - accepted milestones in development order (not additive)", loc="left", fontsize=11.5,
                 fontweight="bold", pad=10)
    _axes_clean(ax, t)
    kinds = [k for k in ("best_vllm", "engine", "full") if any(r["kind"] == k for r in rows)]
    ax.legend(handles=[Patch(color=t[k], label=KIND_LABEL[k]) for k in kinds], loc="upper left",
              bbox_to_anchor=(-0.02, -0.09), ncol=3, frameon=False, fontsize=8.2, labelcolor=t["text2"])
    _save(fig, out, f"progress_{conc}", mode)


def step_breakdown(res, out, mode):
    t = THEMES[mode]
    _style(t)
    sb = res["step_breakdown"]
    comps, rows = sb["components"], sb["rows"]
    fig, ax = plt.subplots(figsize=(9.6, 0.85 * len(rows) + 1.6))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        left = 0.0
        for i, ms in enumerate(r["ms"]):
            # 2px-ish surface gap between segments
            ax.barh(y, ms, left=left, height=0.58, color=t["cat"][i], edgecolor=t["surface"], linewidth=1.5, zorder=2)
            if ms >= 1.0:
                ax.text(left + ms / 2, y, f"{ms:.1f}", ha="center", va="center", fontsize=8.6, color=_ink(t["cat"][i]),
                        fontweight="bold", zorder=3)
            left += ms
        ax.text(left + 0.25, y, f"{left:.1f} ms", va="center", fontsize=9.5, color=t["text"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.2)
    ax.set_xlabel("ms per decode step (torch profiler)")
    ax.set_xlim(0, max(sum(r["ms"]) for r in rows) * 1.12)
    ax.set_title("Where one decode step goes", loc="left", fontsize=12, fontweight="bold", pad=10)
    _axes_clean(ax, t)
    ax.legend(handles=[Patch(color=t["cat"][i], label=c) for i, c in enumerate(comps)], loc="upper center",
              bbox_to_anchor=(0.45, -0.22), ncol=4, frameon=False, fontsize=8.6, labelcolor=t["text2"])
    _save(fig, out, "step_breakdown", mode)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "docs", "results.json"))
    ap.add_argument("--out", default=os.path.join(here, "..", "docs", "img"))
    a = ap.parse_args()
    with open(a.results) as f:
        res = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    for mode in ("light", "dark"):
        for conc in ("c1", "c32"):
            ladder(res, conc, a.out, mode)
            progress(res, conc, a.out, mode)
        step_breakdown(res, a.out, mode)
    print("charts written to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
