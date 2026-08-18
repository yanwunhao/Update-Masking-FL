#!/usr/bin/env python3
"""
prepare_data.py -- download and verify CIFAR-10 once, before any sweep runs.

WHY THIS IS A SEPARATE STEP. torchvision downloads into a shared directory
without locking. Launching a sweep on a machine that has no copy yet means N
processes racing on the same archive: they interleave writes, the tar ends up
truncated, and every run then fails somewhere inside the dataset loader. The
symptom looks like a data bug and costs an afternoon. Doing the download exactly
once, in one process, up front, removes the race entirely.

    python prepare_data.py                       # download + verify
    python prepare_data.py --check               # verify only, exit 1 if missing
    python prepare_data.py --data-root /path

On a second machine with no shared filesystem, run this there too, or copy the
files it prints.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

# Upstream (Krizhevsky) archive and the extracted batch files torchvision reads.
ARCHIVE = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
EXTRACT_DIR = "cifar-10-batches-py"
BATCH_FILES = ([f"data_batch_{i}" for i in range(1, 6)]
               + ["test_batch", "batches.meta"])

EXPECT_TRAIN, EXPECT_TEST, EXPECT_CLASSES = 50000, 10000, 10


def md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def present(data_root: str) -> bool:
    """True when every extracted batch file the loader needs is on disk."""
    d = os.path.join(data_root, EXTRACT_DIR)
    return all(os.path.exists(os.path.join(d, f)) for f in BATCH_FILES)


def missing_message(data_root: str) -> str:
    return (f"CIFAR-10 not found under {data_root!r}.\n"
            f"    Run:  python prepare_data.py --data-root {data_root}\n"
            "    (do this once, before launching any sweep -- concurrent "
            "first-time downloads corrupt the archive)")


def verify(data_root: str) -> bool:
    """Load through torchvision and check the shapes actually match CIFAR-10."""
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(root=data_root, train=True, download=False)
    te = CIFAR10(root=data_root, train=False, download=False)
    ok = True
    for name, ds, expect in (("train", tr, EXPECT_TRAIN), ("test", te, EXPECT_TEST)):
        n, shape = len(ds.data), ds.data.shape[1:]
        classes = len(set(ds.targets))
        good = (n == expect and tuple(shape) == (32, 32, 3)
                and classes == EXPECT_CLASSES)
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name}: {n} images {tuple(shape)}, "
              f"{classes} classes (expected {expect}, (32, 32, 3), "
              f"{EXPECT_CLASSES})")

    arc = os.path.join(data_root, ARCHIVE)
    if os.path.exists(arc):
        got = md5(arc)
        good = got == ARCHIVE_MD5
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {ARCHIVE} md5 {got}"
              + ("" if good else f" (expected {ARCHIVE_MD5})"))
    else:
        print(f"  --  {ARCHIVE} absent (fine: the extracted files are what is read)")
    return bool(ok)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--data-root", default=os.path.join(here, "data"))
    p.add_argument("--check", action="store_true",
                   help="verify only; never download. Exit 1 if unusable.")
    a = p.parse_args(argv)
    root = os.path.abspath(a.data_root)

    if a.check:
        if not present(root):
            print(missing_message(root), file=sys.stderr)
            return 1
        print(f"CIFAR-10 under {root}:")
        return 0 if verify(root) else 1

    os.makedirs(root, exist_ok=True)
    if present(root):
        print(f"CIFAR-10 already present under {root}; verifying.")
    else:
        print(f"Downloading CIFAR-10 into {root} (~170 MB, single process).")
        from torchvision.datasets import CIFAR10
        CIFAR10(root=root, train=True, download=True)
        CIFAR10(root=root, train=False, download=True)

    if not verify(root):
        print("\nVerification FAILED. Delete the directory and re-run:\n"
              f"    rm -rf {os.path.join(root, EXTRACT_DIR)} "
              f"{os.path.join(root, ARCHIVE)}", file=sys.stderr)
        return 1

    print("\nCIFAR-10 ready. To seed another machine (no shared filesystem), copy:")
    print(f"    {os.path.join(root, EXTRACT_DIR)}/")
    print(f"and pass --data-root <destination> to run_sweep.py there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
