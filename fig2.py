#!/usr/bin/env python3
"""
fig2.py -- recovery factor against final accuracy, one point per (batch, tau) cell.

The figure has to carry two facts on its own, without the caption: the full-batch
cells sit alone at high recovery and low accuracy, and nothing lands between
roughly 1.4 and 4 on the recovery axis. Hence the shaded empty band with a direct
label, and marker shapes -- not colour -- keyed to batch size, so a black-and-white
print keeps the distinction.

The Pareto front is computed and printed but not plotted; see the note by the
scatter for why a connecting line would work against both facts above.
"""

from __future__ import annotations

import argparse
import math
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from plot_common import (SURFACE, INK, INK_2, INK_MUTED, GRID, AXIS, SLOTS,
                         load_runs, log_ticks)

ALPHA = 100.0
SEEDS = (0, 1, 2)
BATCHES = ("full", "128", "32")
TAUS = (1, 2, 5)
WINDOW = 50          # last 50 of 200 rounds, i.e. rounds 150-199
K_C = 10
FACE_ALPHA = 0.65   # translucent marker faces so coincident cells stay countable
# Relative ink area of each shape at equal ms; drives draw order, see below.
MARKER_AREA = {"full": 3, "32": 2, "128": 1}   # square > circle > triangle

# Shape carries batch size; colour is redundant reinforcement only.
STYLE = {"full": dict(marker="s", color=SLOTS[0], label="$b$ = full"),
         "128":  dict(marker="^", color=SLOTS[1], label="$b$ = 128"),
         "32":   dict(marker="o", color=SLOTS[2], label="$b$ = 32")}

# Nudges for the tau tags, in points. The mini-batch cells sit almost on top of
# one another in x, so neighbours are pushed to opposite sides by hand.
TAU_OFFSET = {("full", 1): (6, 2), ("full", 2): (6, 2), ("full", 5): (6, 2),
              ("128", 1): (-3, 7), ("128", 2): (-3, 7), ("128", 5): (-4, -12),
              ("32", 1): (1, -12), ("32", 2): (-5, 7), ("32", 5): (5, 1)}


def cell_median(seed_runs: dict[int, list[dict]], key: str) -> float:
    """Median across seeds of each run's median over the last WINDOW rounds.

    test_acc is only written every eval_every rounds, so Nones are dropped.
    """
    per_run = []
    for seed in sorted(seed_runs):
        rounds = sorted(seed_runs[seed], key=lambda r: r["round"])
        assert len(rounds) == 200, (key, seed, len(rounds))
        vals = [r[key] for r in rounds[-WINDOW:] if r.get(key) is not None]
        assert vals, (key, seed)
        per_run.append(st.median(vals))
    return st.median(per_run)


def pareto(points: list[dict]) -> list[dict]:
    """Cells with no other cell both higher in accuracy and higher in recovery."""
    front = [p for p in points
             if not any(q["acc"] > p["acc"] and q["inv"] > p["inv"] for q in points)]
    return sorted(front, key=lambda p: p["acc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--out", default="fig2.pdf")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    points = []
    for b in BATCHES:
        for tau in TAUS:
            seed_runs = runs[(ALPHA, b, tau)]
            assert set(seed_runs) == set(SEEDS), (b, tau, sorted(seed_runs))
            points.append(dict(b=b, tau=tau,
                               acc=cell_median(seed_runs, "test_acc"),
                               inv=cell_median(seed_runs, "inv_delta_s_unweighted")))

    front = pareto(points)
    front_ids = {(p["b"], p["tau"]) for p in front}

    print(f"cell medians (across {len(SEEDS)} seeds of each run's median over the "
          f"last {WINDOW} of 200 rounds)\n")
    hdr = f"{'batch':>5} {'tau':>4} {'acc':>7} {'1/Delta_S':>10}  pareto"
    print(hdr, "\n" + "-" * len(hdr), sep="")
    for p in sorted(points, key=lambda p: p["acc"]):
        mark = "yes" if (p["b"], p["tau"]) in front_ids else ""
        print(f"{p['b']:>5} {p['tau']:>4} {p['acc']:>7.3f} {p['inv']:>10.2f}  {mark}")
    print("\npareto set (ascending accuracy): "
          + ", ".join(f"(b={p['b']}, tau={p['tau']})" for p in front))

    lo_gap = max(p["inv"] for p in points if p["inv"] < 2)
    hi_gap = min(p["inv"] for p in points if p["inv"] > 2)
    print(f"empty recovery band: {lo_gap:.2f} to {hi_gap:.2f}")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # The empty band, called out directly so it reads without the caption.
    ax.axhspan(lo_gap, hi_gap, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.annotate("no cell lands here", xy=(0.03, math.sqrt(lo_gap * hi_gap)),
                xycoords=("axes fraction", "data"), ha="left", va="center",
                fontsize=7, style="italic", color=INK_MUTED, zorder=2)

    ax.grid(True, which="major", color=GRID, lw=0.6, zorder=0)
    # plot_common.reference_lines uses longer captions than fit beside the
    # mini-batch cluster here, so the same two lines are drawn with short ones.
    for y, txt in ((1.0, "1  (orthogonal)"), (float(K_C), f"$K_c = {K_C}$  (all agree)")):
        ax.axhline(y, color=INK_MUTED, lw=1.0, ls=(0, (5, 4)), zorder=1)
        ax.annotate(txt, xy=(0.02, y), xycoords=("axes fraction", "data"),
                    va="bottom", fontsize=8, color=INK_2, zorder=3)

    # The front is reported to stdout but deliberately not drawn: 8 of the 9
    # cells are on it, so a connecting line separates nothing, and the segment
    # from (full, 5) to (128, 2) would cross the empty band the figure exists
    # to show.

    # (32, tau=2) and (128, tau=5) land almost on top of each other. Translucent
    # faces alone are not enough -- whichever marker is drawn last hides the
    # other -- so the shapes are drawn largest-area first, letting the smallest
    # sit on top with the larger one's outline still visible around it.
    for p in sorted(points, key=lambda q: -MARKER_AREA[q["b"]]):
        s = STYLE[p["b"]]
        ax.plot(p["acc"], p["inv"], marker=s["marker"],
                mfc=to_rgba(s["color"], FACE_ALPHA), mec=INK, mew=0.6,
                ms=6, ls="none", zorder=4 + 0.1 * (3 - MARKER_AREA[p["b"]]))
        ax.annotate(rf"$\tau${p['tau']}", xy=(p["acc"], p["inv"]),
                    xytext=TAU_OFFSET[(p["b"], p["tau"])], textcoords="offset points",
                    fontsize=7, color=INK_2, zorder=5)

    log_ticks(ax, 0.9, 13)
    ax.set_xlim(0.27, 0.89)
    ax.set_xlabel("final test accuracy", fontsize=9, color=INK_2)
    ax.set_ylabel(r"recovery factor $1/\Delta_S$", fontsize=9, color=INK_2)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)

    handles = [Line2D([], [], marker=STYLE[b]["marker"],
                      mfc=to_rgba(STYLE[b]["color"], FACE_ALPHA), mec=INK,
                      mew=0.6, ms=6, ls="none", label=STYLE[b]["label"])
               for b in BATCHES]
    # Upper right is the one region no cell occupies, and it keeps the legend
    # clear of the y=1 reference label in the lower left.
    leg = ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right",
                    handlelength=1.0, borderpad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # No bbox_inches="tight": it expands the canvas to swallow the outermost
    # labels, and the 3.4 in single-column width is the spec here.
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
