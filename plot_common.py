#!/usr/bin/env python3
"""
plot_common.py -- shared loading and styling for the figure scripts.

Colour: the validated reference palette. Categorical slots are assigned in fixed
order and never cycled or re-assigned by rank. Several slots sit below 3:1
contrast on the light surface, so the relief rule applies throughout: every
series is directly labelled at its right-hand end, and identity never rests on
colour alone.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-8: worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

FNAME = re.compile(r"alpha([0-9.eE+-]+)_seed(\d+)_b([^_]+)_e(\d+)\.jsonl$")


def load_runs(results_dir: str) -> dict[tuple, dict[int, list[dict]]]:
    """{(alpha, batch, epochs): {seed: [round records]}} for every readable run."""
    runs: dict[tuple, dict[int, list[dict]]] = defaultdict(dict)
    for name in sorted(os.listdir(results_dir)):
        m = FNAME.search(name)
        if not m:
            continue
        alpha, seed, batch, ep = (float(m.group(1)), int(m.group(2)),
                                  m.group(3), int(m.group(4)))
        rounds = []
        with open(os.path.join(results_dir, name)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    break
                if r.get("record") == "round":
                    rounds.append(r)
        if rounds:
            runs[(alpha, batch, ep)][seed] = rounds
    return dict(runs)


def band(seed_runs: dict[int, list[dict]], key: str, flag: str | None = None):
    """Per-round median / min / max across seeds; flagged rows drop out as NaN."""
    n = min(len(v) for v in seed_runs.values())
    mat = np.full((len(seed_runs), n), np.nan)
    for i, s in enumerate(sorted(seed_runs)):
        for j, r in enumerate(seed_runs[s][:n]):
            v = r.get(key)
            bad = flag is not None and r.get(flag)
            mat[i, j] = np.nan if (bad or v is None) else v
    with np.errstate(all="ignore"):
        return (np.arange(n), np.nanmedian(mat, axis=0),
                np.nanmin(mat, axis=0), np.nanmax(mat, axis=0), mat.shape[0])


def style_axes(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURFACE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def log_ticks(ax, lo, hi):
    """Label integer-ish steps; bare decades are useless when the range is 1..10."""
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100)
             if lo <= t <= hi]
    if len(ticks) >= 2:
        ax.set_yticks(ticks, minor=False)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([f"{t:g}" for t in ticks])


def end_label(ax, x, y, color, text, dx=4):
    if np.isfinite(y):
        ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=color, clip_on=False)


def reference_lines(ax, K_c, x_frac=0.012):
    """y=1 (total decorrelation) and y=K_c (perfect agreement)."""
    for y, txt in ((1.0, "1  (mutually orthogonal)"),
                   (float(K_c), f"$K_c = {K_c}$  (all clients agree)")):
        ax.axhline(y, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
        ax.annotate(txt, xy=(x_frac, y), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8.5, color=INK_2)


def finish(fig, out, legend_ax=None, ncol=5):
    if legend_ax is not None:
        leg = legend_ax.legend(frameon=False, fontsize=8.5, ncol=ncol,
                               loc="upper center", bbox_to_anchor=(0.5, -0.16),
                               columnspacing=1.5, handlelength=1.5,
                               labelcolor=INK_2)
        for t in leg.get_texts():
            t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
