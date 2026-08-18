#!/usr/bin/env python3
"""
plot_alpha.py -- fig_alpha.png, the secondary (robustness) figure.

1/Delta_S vs round, one line per alpha, median over seeds with an inter-seed
min-max band, references at y=1 and y=K_c, log y. Same references as the primary
figure so the two can be read against each other.

    python plot_alpha.py --results-dir results
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_common import (SLOTS, SURFACE, band, end_label, finish, load_runs,
                         log_ticks, reference_lines, style_axes)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--results-dir", default=os.path.join(here, "results"))
    p.add_argument("--out", default=os.path.join(here, "fig_alpha.png"))
    p.add_argument("--batch", default="32")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--variant", choices=["unweighted", "weighted"],
                   default="unweighted")
    a = p.parse_args(argv)

    runs = load_runs(a.results_dir)
    keys = [k for k in runs if k[1] == a.batch and k[2] == a.epochs]
    if not keys:
        print(f"no runs with batch={a.batch} E={a.epochs} under {a.results_dir}")
        return 1
    keys.sort(key=lambda k: -k[0])                  # most IID first, fixed order
    K_c = int(np.median([r["K_c"] for k in keys for rr in runs[k].values()
                         for r in rr]))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(SURFACE)
    reference_lines(ax, K_c)

    lo, seeds = 1.0, set()
    for i, k in enumerate(keys):
        c = SLOTS[i % len(SLOTS)]
        x, med, mn, mx, ns = band(runs[k], f"inv_delta_s_{a.variant}",
                                  f"denominator_underflow_{a.variant}")
        seeds.add(ns)
        ax.fill_between(x, mn, mx, color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, med, color=c, lw=2.0, zorder=3, label=rf"$\alpha={k[0]:g}$")
        end_label(ax, x[-1], med[-1], c, rf"$\alpha={k[0]:g}$")
        lo = min(lo, np.nanmin(mn))

    ns_txt = f"{seeds.pop()} seeds" if len(seeds) == 1 else "varying seed counts"
    style_axes(ax, "communication round", r"$1/\Delta_S$",
               "Diversity vs data heterogeneity\n"
               f"batch {a.batch}, E={a.epochs}, {a.variant}; median over {ns_txt}, "
               "band = min-max across seeds")
    log_ticks(ax, max(min(lo * 0.92, 0.98), 1e-3), K_c * 1.25)
    ax.set_xlim(0, x[-1] * 1.10)
    finish(fig, a.out, legend_ax=ax, ncol=min(len(keys), 6))
    print(f"  {len(keys)} alpha level(s), K_c={K_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
