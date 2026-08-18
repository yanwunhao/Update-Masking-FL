#!/usr/bin/env python3
"""
plot_primary.py -- fig_local_steps.png, the primary figure.

Two panels sharing a round axis:

  (a) 1/Delta_S vs round, one line per (local batch, E) configuration, median
      over seeds with an inter-seed min-max band. References at y=1 (clients
      mutually orthogonal) and y=K_c (clients identical). Log y.

  (b) ||mu_hat||^2 and sigma2_hat vs round for one representative configuration.
      THIS PANEL CARRIES THE EXPLANATION. The ratio in (a) is a function of
      r = mu2/sigma2 alone, so it cannot show WHICH term moves; the observed fall
      toward the floor is sigma2 growing, not mu2 collapsing, and only panel (b)
      distinguishes those.

    python plot_primary.py --results-dir results
    python plot_primary.py --representative 100:32:2
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_common import (SLOTS, INK_2, INK_MUTED, SURFACE, band, end_label,
                         finish, load_runs, log_ticks, reference_lines,
                         style_axes)


def cfg_label(batch: str, ep: int) -> str:
    b = "full" if str(batch).lower() == "full" else f"b{batch}"
    return f"{b}, E={ep}"


def local_steps(batch: str, ep: int, n: float = 500.0) -> float:
    per_epoch = 1.0 if str(batch).lower() == "full" else n / float(batch)
    return per_epoch * ep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--results-dir", default=os.path.join(here, "results"))
    p.add_argument("--out", default=os.path.join(here, "fig_local_steps.png"))
    p.add_argument("--alpha", type=float, default=100.0,
                   help="which alpha the primary grid was run at")
    p.add_argument("--representative", default="",
                   help='panel (b) config as "alpha:batch:E"; default is the '
                        'most-local-steps config present')
    a = p.parse_args(argv)

    runs = load_runs(a.results_dir)
    keys = [k for k in runs if k[0] == a.alpha]
    if not keys:
        print(f"no runs with alpha={a.alpha:g} under {a.results_dir}")
        return 1
    # Fixed, meaningful order: fewest local steps (least noise) first, so colour
    # tracks the axis under study rather than discovery order.
    keys.sort(key=lambda k: (local_steps(k[1], k[2]), k[1], k[2]))
    K_c = int(np.median([r["K_c"] for k in keys for sr in [runs[k]]
                         for rr in sr.values() for r in rr]))

    if a.representative:
        al, b, e = a.representative.split(":")
        rep = (float(al), b, int(e))
        if rep not in runs:
            print(f"representative {rep} not found; using the default")
            rep = keys[-1]
    else:
        rep = keys[-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    # --- (a) the ratio ----------------------------------------------------
    reference_lines(ax1, K_c)
    lo = 1.0
    for i, k in enumerate(keys):
        c = SLOTS[i % len(SLOTS)]
        x, med, mn, mx, ns = band(runs[k], "inv_delta_s_unweighted",
                                  "denominator_underflow_unweighted")
        ax1.fill_between(x, mn, mx, color=c, alpha=0.16, lw=0, zorder=2)
        ax1.plot(x, med, color=c, lw=2.0, zorder=3,
                 label=f"{cfg_label(k[1], k[2])} ({ns}s)")
        end_label(ax1, x[-1], med[-1], c, cfg_label(k[1], k[2]))
        lo = min(lo, np.nanmin(mn))
    style_axes(ax1, None, r"$1/\Delta_S$",
               f"(a)  Cohort pseudo-gradient diversity, "
               rf"$\alpha={a.alpha:g}$, median over seeds, min-max band")
    log_ticks(ax1, max(min(lo * 0.92, 0.98), 1e-3), K_c * 1.25)

    # --- (b) the decomposition -------------------------------------------
    x, mu_med, mu_lo, mu_hi, ns = band(runs[rep], "mu2_hat", "sigma2_clamped")
    _, s_med, s_lo, s_hi, _ = band(runs[rep], "sigma2_hat", "sigma2_clamped")
    for med, lo_, hi_, c, lab in (
            (mu_med, mu_lo, mu_hi, SLOTS[0], r"$\|\hat{\mu}\|^2$  (shared direction)"),
            (s_med, s_lo, s_hi, SLOTS[1], r"$\hat{\sigma}^2$  (per-client noise)")):
        ax2.fill_between(x, lo_, hi_, color=c, alpha=0.16, lw=0, zorder=2)
        ax2.plot(x, med, color=c, lw=2.0, zorder=3, label=lab)
        end_label(ax2, x[-1], med[-1], c,
                  r"$\|\hat{\mu}\|^2$" if c == SLOTS[0] else r"$\hat{\sigma}^2$")
    ax2.set_yscale("log")
    style_axes(ax2, "communication round", "energy",
               f"(b)  Decomposition for {cfg_label(rep[1], rep[2])} "
               rf"($\alpha={rep[0]:g}$, {ns} seed(s)): which term moves")
    ax2.legend(frameon=False, fontsize=8.5, loc="best", labelcolor=INK_2)

    ax1.set_xlim(0, x[-1] * 1.12)
    finish(fig, a.out, legend_ax=ax1, ncol=min(len(keys), 5))

    print(f"  panel (a): {len(keys)} configuration(s), K_c={K_c}")
    print(f"  panel (b): representative {cfg_label(rep[1], rep[2])}, alpha={rep[0]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
