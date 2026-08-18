#!/usr/bin/env python3
"""
run_sweep.py -- schedule a preset grid across devices as independent processes.

Parallelism is process-level on purpose: no DDP, no DataParallel. Each
fed_diversity.py run is small enough that several fit on one card, so the
scheduler keeps at most --per-gpu alive per device and refills as they exit.

    python prepare_data.py                                  # once, first
    python run_sweep.py --preset local-steps --dry-run      # what would run
    python run_sweep.py --preset local-steps --devices cuda:0 --per-gpu 3
    python run_sweep.py --preset local-steps --status       # progress
    python run_sweep.py --preset alpha-robustness
    python run_sweep.py --smoke

PRESETS -- see the module constants for the exact grids.

    local-steps        PRIMARY. Local-step count is the axis that actually moves
                       the measured ratio; alpha is not.
    alpha-robustness   SECONDARY. Shows the floor is not an artifact of near-IID.

Only machine-shaped settings are forwarded to the workers: device, dtype,
compile, data placement, thread count and paths. The experimental axes come from
the preset, and the remaining experimental constants are the worker's own
defaults -- none of them is settable per host, because a host that could change
them would silently produce incomparable runs.

RESUME is config-level only: a config whose JSONL ends in a completion record is
skipped. A killed process loses at most that one config. There is deliberately no
mid-run checkpointing -- RNG-state serialisation is bug surface with no payoff
when a single run is under an hour.

NO LOCK FILES. The two target machines share no filesystem, so they cannot
contend for a config; splitting work across them is a matter of running different
presets or --seeds on each and merging afterwards with merge_results.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import torch

import prepare_data
from fed_diversity import config_stem

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "fed_diversity.py")

# ---------------------------------------------------------------------------
# Presets. A config is (alpha, seed, local_batch_size, local_epochs).
#
# Empirical basis for making local steps the primary axis: at alpha=100 the
# measured 1/Delta_S settles at ~1.19 against a ceiling of K_c=10, and the
# decomposition shows ||mu||^2 flat from round ~25 while sigma^2 grows
# monotonically over 200 rounds. Local update noise, not data heterogeneity, is
# what pins the ratio to its floor.
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "local-steps": {
        "alphas": [100.0],
        "seeds": [0, 1, 2],
        "batches": ["32", "128", "full"],
        "epochs": [1, 2, 5],
        "rounds": 200,
        "doc": "PRIMARY: vary local-step count at alpha=100. Fewer steps and "
               "larger batches mean less accumulated local noise, hence larger r "
               "and higher 1/Delta_S. The (full, E=1) corner is minimum noise and "
               "should sit highest; if it does not rise materially, the "
               "decomposition model is wrong.",
    },
    "alpha-robustness": {
        "alphas": [100.0, 1.0, 0.1],
        "seeds": [0, 1, 2],
        "batches": ["32"],
        "epochs": [2],
        "rounds": 200,
        "doc": "SECONDARY: the floor is not an artifact of the near-IID setting. "
               "Expect all three near 1.1-1.3.",
    },
    "smoke": {
        "alphas": [100.0, 0.1],
        "seeds": [0],
        "batches": ["32"],
        "epochs": [2],
        "rounds": 5,
        "doc": "Exercise the whole path in seconds. Not a measurement.",
    },
}

CONTROL_ALPHA = 100.0

# --dry-run estimates only. Both anchors MEASURED on one RTX 4060, fp32, eager,
# 200 rounds, ~500 samples per client:
#
#     batch 32,  E=2  ->  10.5 s/round     (34.9 min total)
#     batch full, E=1 ->  37.0 s/round     (measured; 7.5 GiB peak on an 8 GB card)
#
# Step count alone does NOT predict cost, and getting this wrong by 30x once is
# how a "1 minute" config turns out to take two hours. The full-batch corner takes
# FEWER steps and processes FEWER samples than the reference, yet runs 3.5x
# SLOWER: a ~500-image ResNet-18 step is memory-bound, and near the card's limit
# it thrashes. So large batches carry an explicit measured penalty rather than
# being extrapolated from the small-batch anchor.
SECONDS_PER_ROUND_REF = 10.5
REF_STEPS = (500 / 32) * 2          # batches per epoch * epochs, at the reference
FULL_BATCH_SECONDS_PER_EPOCH = 37.0  # measured at E=1; scales with epochs
# Above this batch the step stops being launch-bound and starts being memory-bound.
LARGE_BATCH_THRESHOLD = 256


def est_seconds_per_round(batch: str, epochs: int, n_per_client: float = 500.0
                          ) -> float:
    """
    Crude, and honest about it. Small batches scale with step count; full-batch
    uses its own measured anchor. Anything in between is interpolated and may be
    off by a factor of two -- treat --dry-run totals as order-of-magnitude.
    """
    if str(batch).lower() == "full":
        return FULL_BATCH_SECONDS_PER_EPOCH * epochs
    b = float(batch)
    steps = (n_per_client / b) * epochs
    est = SECONDS_PER_ROUND_REF * steps / REF_STEPS
    if b > LARGE_BATCH_THRESHOLD:
        # blend toward the measured full-batch cost as the batch approaches n
        w = min((b - LARGE_BATCH_THRESHOLD) / max(n_per_client - LARGE_BATCH_THRESHOLD, 1.0), 1.0)
        est = (1 - w) * est + w * FULL_BATCH_SECONDS_PER_EPOCH * epochs
    return est


# ---------------------------------------------------------------------------
# Config records
# ---------------------------------------------------------------------------

class Cfg:
    __slots__ = ("alpha", "seed", "batch", "epochs", "rounds")

    def __init__(self, alpha, seed, batch, epochs, rounds):
        self.alpha, self.seed = float(alpha), int(seed)
        self.batch, self.epochs, self.rounds = str(batch), int(epochs), int(rounds)

    @property
    def stem(self) -> str:
        return config_stem(self.alpha, self.seed, self.batch, self.epochs)

    @property
    def key(self) -> str:
        return self.stem

    def path(self, results_dir: str) -> str:
        return os.path.join(results_dir, self.stem + ".jsonl")

    def __repr__(self) -> str:
        return (f"alpha={self.alpha:g} seed={self.seed} "
                f"b={self.batch} E={self.epochs}")


def build_grid(preset: str, alphas=None, seeds=None, batches=None, epochs=None
               ) -> list[Cfg]:
    if preset not in PRESETS:
        raise SystemExit(f"unknown --preset {preset!r}; "
                         f"choose from {sorted(PRESETS)}")
    p = PRESETS[preset]
    A = alphas if alphas else p["alphas"]
    S = seeds if seeds else p["seeds"]
    B = batches if batches else p["batches"]
    E = epochs if epochs else p["epochs"]
    return [Cfg(a, s, b, e, p["rounds"])
            for b in B for e in E for a in A for s in S]


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------

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
                break                        # torn final line from a killed run
    return recs


def is_complete(path: str) -> bool:
    if not os.path.exists(path):
        return False
    recs = read_jsonl(path)
    return bool(recs) and recs[-1].get("record") == "footer" and recs[-1].get("done")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def manifest_path(results_dir: str) -> str:
    return os.path.join(results_dir, "manifest.json")


def load_manifest(results_dir: str) -> dict:
    p = manifest_path(results_dir)
    if not os.path.exists(p):
        return {"host": socket.gethostname(), "configs": {}}
    try:
        with open(p) as fh:
            m = json.load(fh)
        m.setdefault("configs", {})
        return m
    except (json.JSONDecodeError, OSError):
        return {"host": socket.gethostname(), "configs": {}}


def save_manifest(results_dir: str, m: dict) -> None:
    m["host"] = socket.gethostname()
    m["updated_at"] = time.time()
    tmp = manifest_path(results_dir) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path(results_dir))      # atomic, survives a kill


def mark(results_dir: str, m: dict, cfg: Cfg, **fields) -> None:
    e = m["configs"].setdefault(cfg.key, {
        "alpha": cfg.alpha, "seed": cfg.seed,
        "local_batch_size": cfg.batch, "local_epochs": cfg.epochs,
        "rounds": cfg.rounds,
    })
    e.update(fields)
    save_manifest(results_dir, m)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def launch(cfg: Cfg, device: str, args) -> subprocess.Popen:
    path = cfg.path(args.results_dir)
    cmd = [sys.executable, WORKER,
           # experiment axes, from the preset
           "--alpha", repr(cfg.alpha), "--seed", str(cfg.seed),
           "--local-batch-size", cfg.batch, "--local-epochs", str(cfg.epochs),
           "--rounds", str(cfg.rounds), "--out", path,
           # machine-shaped only
           "--device", device, "--dtype", args.dtype,
           "--data-device", args.data_device,
           "--num-threads", str(args.num_threads),
           "--data-root", args.data_root,
           "--per-gpu", str(args.per_gpu)]
    if args.compile:
        cmd.append("--compile")
    if args.preset == "smoke":
        cmd.append("--smoke")
    log = os.path.join(args.results_dir, "logs", cfg.stem + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    fh = open(log, "w")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    p._cfg, p._log, p._device = cfg, log, device        # type: ignore[attr-defined]
    return p


def schedule(grid: list[Cfg], args) -> int:
    m = load_manifest(args.results_dir)
    queue, running = list(grid), []
    in_flight = {d: 0 for d in args.devices}
    failures, done = [], []
    started = time.time()

    while queue or running:
        for d in args.devices:
            while in_flight[d] < args.per_gpu and queue:
                cfg = queue.pop(0)
                p = launch(cfg, d, args)
                in_flight[d] += 1
                running.append(p)
                mark(args.results_dir, m, cfg, status="running", host=m["host"],
                     device=d, started_at=time.time(), ended_at=None)
                print(f"[sweep] start  {cfg} on {d}  ({len(queue)} queued)",
                      flush=True)

        time.sleep(2.0)

        for p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            running.remove(p)
            in_flight[p._device] -= 1
            cfg = p._cfg
            footer = {}
            if is_complete(cfg.path(args.results_dir)):
                footer = read_jsonl(cfg.path(args.results_dir))[-1]
            if rc == 0 and footer:
                done.append((cfg, footer))
                mark(args.results_dir, m, cfg, status="complete",
                     ended_at=time.time(),
                     seconds=footer.get("total_seconds"),
                     seconds_per_round=footer.get("seconds_per_round"),
                     peak_memory_mb=footer.get("peak_memory_mb"),
                     final_test_acc=footer.get("final_test_acc"))
                print(f"[sweep] done   {cfg}  "
                      f"{footer.get('total_seconds', 0) / 60:.1f} min, "
                      f"peak {footer.get('peak_memory_mb') or 0:.0f} MiB",
                      flush=True)
            else:
                failures.append((cfg, rc, p._log))
                mark(args.results_dir, m, cfg, status="failed", rc=rc,
                     ended_at=time.time(), log=p._log)
                print(f"[sweep] FAILED {cfg}  rc={rc}  log: {p._log}", flush=True)

    print(f"\n[sweep] {len(grid)} run(s) in {(time.time() - started) / 60:.1f} min, "
          f"{len(failures)} failure(s)")
    if done:
        peaks = [f.get("peak_memory_mb") or 0 for _, f in done]
        spr = [f.get("seconds_per_round") or 0 for _, f in done]
        print(f"[sweep] throughput: {min(spr):.1f}-{max(spr):.1f} s/round, "
              f"peak GPU {min(peaks):.0f}-{max(peaks):.0f} MiB per process "
              f"-- raise --per-gpu from this, do not guess")
    for cfg, rc, log in failures:
        print(f"        FAILED {cfg} rc={rc} -> {log}")
    return len(failures)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def dry_run(grid: list[Cfg], args) -> None:
    print(f"[dry-run] preset {args.preset!r}: {len(grid)} config(s)")
    print(f"          {PRESETS[args.preset]['doc']}\n")
    print(f"{'config':<40} {'rounds':>7} {'est/run':>9} {'status':>9}")
    total = 0.0
    for c in grid:
        est = est_seconds_per_round(c.batch, c.epochs) * c.rounds
        total += est
        st = "complete" if is_complete(c.path(args.results_dir)) else "pending"
        print(f"{c.stem:<40} {c.rounds:>7} {est / 60:>8.1f}m {st:>9}")
    todo = sum(est_seconds_per_round(c.batch, c.epochs) * c.rounds
               for c in grid if not is_complete(c.path(args.results_dir)))
    slots = max(len(args.devices) * args.per_gpu, 1)
    print(f"\n          serial total {total / 3600:.1f} h; "
          f"remaining {todo / 3600:.1f} h")
    print(f"          across {len(args.devices)} device(s) x {args.per_gpu} "
          f"= {slots} slots: roughly {todo / 3600 / slots:.1f} h wall-clock")
    print("          (order-of-magnitude only: two measured anchors, 10.5 s/round "
          "at batch 32/E=2 and\n           37 s/round at full/E=1; intermediate "
          "batches are interpolated and can be 2x off)")


def status(args) -> None:
    m = load_manifest(args.results_dir)
    grid = build_grid(args.preset, args.alphas, args.seeds, args.batches,
                      args.epochs)
    counts: dict[str, int] = {}
    print(f"{'config':<40} {'status':>9} {'host':>14} {'device':>8} {'min':>7}")
    for c in grid:
        e = m["configs"].get(c.key, {})
        st = e.get("status", "pending")
        if is_complete(c.path(args.results_dir)):
            st = "complete"
        elif st == "running" and e.get("ended_at") is None:
            st = "running?"           # manifest says running; may be a dead process
        counts[st] = counts.get(st, 0) + 1
        secs = e.get("seconds")
        print(f"{c.stem:<40} {st:>9} {str(e.get('host', '-')):>14} "
              f"{str(e.get('device', '-')):>8} "
              f"{(secs / 60 if secs else 0):>7.1f}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  manifest: {manifest_path(args.results_dir)}")


def summarise(args) -> None:
    """Median 1/Delta_S, ||mu||^2 and sigma^2 over the last N rounds, per config."""
    grid = build_grid(args.preset, args.alphas, args.seeds, args.batches,
                      args.epochs)
    window = args.summary_window
    by_cfg: dict[tuple, list[str]] = {}
    for c in grid:
        by_cfg.setdefault((c.alpha, c.batch, c.epochs), []).append(
            c.path(args.results_dir))

    print("\n" + "=" * 96)
    print(f"median over the last {window} rounds  (K_c is the ceiling for 1/Delta_S)")
    print("=" * 96)
    print(f"{'alpha':>7} {'batch':>6} {'E':>3} {'runs':>5} {'K_c':>4} "
          f"{'1/ds unw':>10} {'1/ds wtd':>10} {'mu2_hat':>11} {'sigma2_hat':>11} "
          f"{'r_hat':>8} {'flag':>5}")

    control = None
    for (alpha, batch, ep), paths in sorted(by_cfg.items()):
        u, w, mu2, s2, kcs, flagged, runs = [], [], [], [], [], 0, 0
        for p in paths:
            if not os.path.exists(p):
                continue
            R = [r for r in read_jsonl(p) if r.get("record") == "round"]
            if not R:
                continue
            runs += 1
            tail = R[-window:]
            u += [r["inv_delta_s_unweighted"] for r in tail]
            w += [r["inv_delta_s_weighted"] for r in tail]
            mu2 += [r["mu2_hat"] for r in tail if r.get("mu2_hat") is not None]
            s2 += [r["sigma2_hat"] for r in tail if r.get("sigma2_hat") is not None]
            kcs += [r["K_c"] for r in tail]
            flagged += sum(1 for r in R
                           if r.get("denominator_underflow_unweighted")
                           or r.get("sigma2_clamped"))
        if not runs:
            print(f"{alpha:>7g} {batch:>6} {ep:>3} {0:>5} {'-':>4} {'no data':>10}")
            continue
        med_u = float(np.median(u))
        kc = int(np.median(kcs))
        print(f"{alpha:>7g} {batch:>6} {ep:>3} {runs:>5} {kc:>4} "
              f"{med_u:>10.4f} {float(np.median(w)):>10.4f} "
              f"{float(np.median(mu2)):>11.3e} {float(np.median(s2)):>11.3e} "
              f"{float(np.median(mu2)) / max(float(np.median(s2)), 1e-300):>8.4f} "
              f"{flagged:>5}")
        if alpha == CONTROL_ALPHA and batch == "32" and ep == 2:
            control = (med_u, kc)

    if control:
        val, kc = control
        print("-" * 96)
        print(f"NOISE FLOOR  alpha={CONTROL_ALPHA:g}, batch 32, E=2:  "
              f"1/Delta_S = {val:.4f}  vs  K_c = {kc}  ({val / kc:.1%} of K_c)")
        print("  This is a noise-floor measurement, not a correctness check. The "
              "correctness control is test 2\n  (identical clients -> exactly K_c).")
    print("=" * 96 + "\n")


# ---------------------------------------------------------------------------

def default_devices() -> str:
    """Every visible GPU, or CPU. A default, not a hardcode -- --devices overrides."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return ",".join(f"cuda:{i}" for i in range(n)) if n else "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="local-steps", choices=sorted(PRESETS))
    p.add_argument("--devices", default=default_devices(),
                   help='comma-separated, e.g. "cuda:0,cuda:1,cuda:2" or "cpu"')
    p.add_argument("--per-gpu", type=int, default=3,
                   help="concurrent processes per device (default sized for 8 GB)")
    p.add_argument("--dtype", choices=["fp32", "tf32", "amp"], default="fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data-device", choices=["device", "cpu"], default="device")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--data-root", default=os.path.join(HERE, "data"))
    p.add_argument("--alphas", default="", help="override the preset alphas")
    p.add_argument("--seeds", default="", help="override the preset seeds")
    p.add_argument("--batches", default="", help="override the preset batch sizes")
    p.add_argument("--epochs", default="", help="override the preset local epochs")
    p.add_argument("--summary-window", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="alias for --preset smoke")
    p.add_argument("--dry-run", action="store_true",
                   help="list configs and estimated wall-clock, launch nothing")
    p.add_argument("--status", action="store_true", help="print manifest progress")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rerun configs whose output is already complete")
    args = p.parse_args(argv)

    if args.smoke:
        args.preset = "smoke"
    args.devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    args.alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else None
    args.seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    args.batches = [x.strip() for x in args.batches.split(",")] if args.batches else None
    args.epochs = [int(x) for x in args.epochs.split(",")] if args.epochs else None
    if args.preset == "smoke":
        args.summary_window = 5
    os.makedirs(args.results_dir, exist_ok=True)

    if args.status:
        status(args)
        return 0
    if args.summary_only:
        summarise(args)
        return 0

    grid = build_grid(args.preset, args.alphas, args.seeds, args.batches,
                      args.epochs)
    if args.dry_run:
        dry_run(grid, args)
        return 0

    # Pre-download guard: never let N processes race on a first-time download.
    if not prepare_data.present(args.data_root):
        print(prepare_data.missing_message(args.data_root), file=sys.stderr)
        return 1

    print(f"[sweep] host={socket.gethostname()} python={platform.python_version()} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    todo = [c for c in grid
            if args.force or not is_complete(c.path(args.results_dir))]
    print(f"[sweep] preset={args.preset} {len(todo)} run(s) to do, "
          f"{len(grid) - len(todo)} already complete, devices={args.devices} "
          f"per-gpu={args.per_gpu} dtype={args.dtype} compile={args.compile}")

    rc = schedule(todo, args) if todo else 0
    summarise(args)
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
