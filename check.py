#!/usr/bin/env python3
"""
check.py -- the `make check` gate: every test plus the smoke path, one summary line.

Run this first on any new machine. Non-zero exit on any failure, so it can be
chained ahead of a sweep with && and will stop rather than proceed on a broken
checkout.

    python check.py --device cpu
    python check.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default=os.path.join(HERE, "data"))
    a = p.parse_args(argv)

    import torch
    print("=" * 72)
    print(f"fed_diversity check | device={a.device} | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda} | python {sys.version.split()[0]}")
    print(f"host={os.uname().nodename} data-root={a.data_root}")
    print("=" * 72)

    env = dict(os.environ,
               FED_DIVERSITY_TEST_DEVICE=a.device,
               FED_DIVERSITY_DATA=a.data_root)
    stages: list[tuple[str, list[str]]] = [
        ("data present", [sys.executable, os.path.join(HERE, "prepare_data.py"),
                          "--check", "--data-root", a.data_root]),
        ("test suite", [sys.executable, os.path.join(HERE, "test_fed_diversity.py")]),
    ]

    results = []
    t0 = time.time()
    for name, cmd in stages:
        print(f"\n--- {name} " + "-" * (68 - len(name)))
        rc = subprocess.run(cmd, cwd=HERE, env=env).returncode
        results.append((name, rc == 0))
        if name == "data present" and rc != 0:
            print("\nCannot continue without CIFAR-10.")
            break
    else:
        # Smoke path end to end, into a throwaway directory.
        print("\n--- smoke path " + "-" * 57)
        with tempfile.TemporaryDirectory() as tmp:
            rc = subprocess.run(
                [sys.executable, os.path.join(HERE, "run_sweep.py"), "--smoke",
                 "--devices", a.device, "--per-gpu", "1",
                 "--data-root", a.data_root, "--results-dir", tmp],
                cwd=HERE, env=env).returncode
            results.append(("smoke path", rc == 0))

    ok = all(v for _, v in results)
    print("\n" + "=" * 72)
    for name, good in results:
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
    print(f"{'PASS' if ok else 'FAIL'} -- {sum(v for _, v in results)}/"
          f"{len(results)} stage(s) in {time.time() - t0:.0f}s on {a.device}")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
