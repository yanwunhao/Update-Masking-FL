#!/usr/bin/env python3
"""
merge_results.py -- pool run directories from several machines into one for plotting.

The two target machines share no filesystem, so results arrive as separate
directories that have to be combined by hand. Two things can go wrong when doing
that, and this script exists to make both loud rather than silent:

  COLLISIONS. The same config (alpha, seed, local batch, local epochs) produced
  on two hosts. Silently keeping one would hide a disagreement -- and if the two
  disagree beyond float noise, that IS the finding, not a nuisance. Collisions
  are reported with the measured difference and, by default, nothing is merged
  until you choose (--on-collision).

  ENVIRONMENT DRIFT. Runs produced under different torch/CUDA versions, dtypes,
  or compile modes are not directly comparable; --dtype tf32/amp and --compile
  perturb Delta_S at ~7e-4 relative. Every distinct environment found is printed,
  and mixing non-default numerics is called out.

    python merge_results.py results_4060 results_a6000 --out merged
    python merge_results.py a b --out merged --on-collision first
    python merge_results.py a b --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

SUMMARY_WINDOW = 50


def read_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return recs


def load_run(path: str) -> dict | None:
    recs = read_jsonl(path)
    if not recs or recs[0].get("record") != "header":
        return None
    header = recs[0]
    rounds = [r for r in recs if r.get("record") == "round"]
    complete = bool(recs[-1].get("record") == "footer" and recs[-1].get("done"))
    cfg, env = header.get("config", {}), header.get("env", {})
    if not rounds:
        return None
    tail = rounds[-SUMMARY_WINDOW:]
    return {
        "path": path,
        "key": (cfg.get("alpha"), cfg.get("seed"),
                str(cfg.get("local_batch_size")), cfg.get("local_epochs")),
        "complete": complete,
        "rounds": len(rounds),
        "env": env,
        "env_key": (env.get("torch_version"), env.get("cuda_version"),
                    env.get("dtype"), env.get("compile")),
        "host": env.get("hostname"),
        "device": env.get("device_name"),
        "median_inv": float(np.median([r["inv_delta_s_unweighted"] for r in tail])),
    }


def env_line(k) -> str:
    tv, cv, dt, comp = k
    tag = "" if (dt == "fp32" and not comp) else "   <-- NON-DEFAULT NUMERICS"
    return (f"torch {tv} / CUDA {cv} / dtype {dt} / compile {bool(comp)}{tag}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="input result directories")
    p.add_argument("--out", default="merged", help="output directory")
    p.add_argument("--on-collision", choices=["report", "first", "longest"],
                   default="report",
                   help="report (default: refuse to merge), first (input order), "
                        "or longest (most rounds)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-incomplete", action="store_true")
    a = p.parse_args(argv)

    runs: dict[tuple, list[dict]] = defaultdict(list)
    skipped_incomplete, unreadable = [], []
    for d in a.dirs:
        if not os.path.isdir(d):
            print(f"error: {d!r} is not a directory", file=sys.stderr)
            return 2
        for name in sorted(os.listdir(d)):
            if not name.endswith(".jsonl"):
                continue
            r = load_run(os.path.join(d, name))
            if r is None:
                unreadable.append(os.path.join(d, name))
                continue
            if not r["complete"] and not a.include_incomplete:
                skipped_incomplete.append((os.path.join(d, name), r["rounds"]))
                continue
            runs[r["key"]].append(r)

    if unreadable:
        print(f"{len(unreadable)} unreadable/headerless file(s) ignored:")
        for u in unreadable[:5]:
            print(f"    {u}")
    if skipped_incomplete:
        print(f"\n{len(skipped_incomplete)} incomplete run(s) skipped "
              f"(--include-incomplete to keep):")
        for path, n in skipped_incomplete[:10]:
            print(f"    {path}  ({n} rounds, no completion record)")

    # --- environments -----------------------------------------------------
    envs: dict[tuple, list[str]] = defaultdict(list)
    for rs in runs.values():
        for r in rs:
            envs[r["env_key"]].append(f"{r['host']}/{r['device']}")
    print(f"\nENVIRONMENTS ({len(envs)} distinct):")
    for k, who in sorted(envs.items(), key=lambda kv: str(kv[0])):
        hosts = sorted(set(who))
        print(f"  {env_line(k)}")
        print(f"      {len(who)} run(s) on {', '.join(hosts)}")
    nondefault = [k for k in envs if not (k[2] == "fp32" and not k[3])]
    if len(envs) > 1:
        print("  NOTE: results from more than one environment are being pooled.")
    if nondefault:
        print("  WARNING: some runs used tf32/amp/compile. Those perturb Delta_S "
              "at ~7e-4 relative\n           and must not be pooled with "
              "default-mode results for a published number.")

    # --- collisions -------------------------------------------------------
    collisions = {k: v for k, v in runs.items() if len(v) > 1}
    if collisions:
        print(f"\nCOLLISIONS ({len(collisions)} config(s) produced more than once):")
        for k, rs in sorted(collisions.items(), key=lambda kv: str(kv[0])):
            alpha, seed, batch, ep = k
            vals = [r["median_inv"] for r in rs]
            spread = max(vals) - min(vals)
            rel = spread / max(abs(np.mean(vals)), 1e-30)
            print(f"  alpha={alpha:g} seed={seed} b={batch} E={ep}: "
                  f"median 1/Delta_S spread {spread:.3e} ({rel:.2%} relative)")
            for r in rs:
                print(f"      {r['median_inv']:.6f}  {r['rounds']} rounds  "
                      f"{r['host']}/{r['device']}  {r['path']}")
            if rel > 1e-2:
                print("      ^ these disagree well beyond float noise; that is a "
                      "result to explain, not a file to pick")
        if a.on_collision == "report":
            print("\nRefusing to merge while collisions are unresolved. Re-run with "
                  "--on-collision first|longest\nonce you have decided, or remove "
                  "the duplicate files.")
            return 1

    # --- selection --------------------------------------------------------
    chosen = {}
    for k, rs in runs.items():
        if len(rs) == 1:
            chosen[k] = rs[0]
        elif a.on_collision == "longest":
            chosen[k] = max(rs, key=lambda r: r["rounds"])
        else:
            chosen[k] = rs[0]

    print(f"\nMERGE: {len(chosen)} unique config(s) from {len(a.dirs)} directory(ies)")
    if a.dry_run:
        print("(dry run: nothing written)")
        return 0

    os.makedirs(a.out, exist_ok=True)
    for k, r in sorted(chosen.items(), key=lambda kv: str(kv[0])):
        shutil.copy2(r["path"], os.path.join(a.out, os.path.basename(r["path"])))
    provenance = {
        "sources": [os.path.abspath(d) for d in a.dirs],
        "on_collision": a.on_collision,
        "collisions": {str(k): [r["path"] for r in v] for k, v in collisions.items()},
        "environments": {str(k): sorted(set(v)) for k, v in envs.items()},
        "configs": {str(k): r["path"] for k, r in chosen.items()},
    }
    with open(os.path.join(a.out, "merge_provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
    print(f"wrote {len(chosen)} file(s) to {a.out}/ "
          f"(+ merge_provenance.json recording every choice made)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
