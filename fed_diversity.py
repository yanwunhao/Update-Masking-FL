#!/usr/bin/env python3
"""
fed_diversity.py -- per-round gradient diversity of FedAvg pseudo-gradients.

WHAT IS MEASURED
    Each round the server broadcasts w_b, a cohort C of K_c clients trains
    locally, and client k returns w_k. The round-level pseudo-gradient is

        D_k = w_b - w_k                        (descent direction, not a gradient)

    and the Yin et al. gradient-diversity ratio over the cohort is

        Delta_S = sum_{k in C} ||D_k||^2  /  || sum_{k in C} D_k ||^2 .

    We report 1/Delta_S, which by Cauchy-Schwarz is at most K_c and equals K_c
    exactly when every D_k is identical. Small 1/Delta_S means the clients pull
    in mutually cancelling directions; 1/Delta_S = K_c means they agree.

    Two variants are logged as separate columns every round:

        unweighted  -- the formula above verbatim; this is what the theory uses.
        weighted    -- each D_k first scaled by n_k / mean_j(n_j), which is what
                       FedAvg actually aggregates, so the two diverge whenever
                       the shard sizes are uneven (i.e. whenever alpha is small).

    Both the ratio (`delta_s_*`) and its reciprocal (`inv_delta_s_*`) are written
    out, so no reader has to guess which orientation a column is in.

CORRECTNESS NOTES (the whole point of the script)

  * Parameters only, never buffers. D_k is assembled from named_parameters() in a
    single fixed order that is re-asserted on every flatten/unflatten. Normal-
    isation statistics are not updates; including them corrupts the norms. The
    model is additionally asserted to expose zero buffers, which is a direct
    guard against a BatchNorm layer creeping back in.

  * GroupNorm, not BatchNorm. BN's running statistics are a known confound under
    non-IID federated training (Hsieh et al. 2020), so the ResNet-18 here uses
    GroupNorm(min(32, C), C) throughout and therefore has no buffers at all.

  * Cohort only. Every sum runs over the K_c sampled clients, never the full
    population.

  * float64 accumulation. Per-client squared norms and the cohort sum vector are
    accumulated in float64 from float32 deltas. Over d ~ 1.1e7 coordinates the
    relative error of the float64 reduction is ~1e-13, far below the 1e-9
    underflow threshold below.

  * Denominator guard. If ||sum D_k||^2 falls below UNDERFLOW_REL * sum ||D_k||^2
    the ratio is not trustworthy: the row is flagged, Delta_S is written as
    infinity and 1/Delta_S as 0.0, and the bounds assertion is skipped rather
    than a plausible-looking finite number being emitted.

  * Bounds assertion every round: 1/Delta_S in (0, K_c] up to tolerance, else
    raise. See BOUNDS_NOTE -- the upper bound is a theorem (Cauchy-Schwarz) and
    a violation is a bug, while below 1 is a real net-cancelling cohort, flagged
    with `cancellation_regime` and counted in the end-of-run summary.

PORTABILITY
    The same seed and the same experimental arguments must produce the same
    Delta_S sequence on CPU, on one GPU, or on another GPU, up to float
    tolerance. Two design rules enforce that:

      1. Every random draw that affects the experiment is made on a CPU
         generator and then moved to the compute device. A CUDA generator seeded
         with s does NOT produce the same stream as a CPU generator seeded with
         s, so drawing batch orders or augmentation offsets device-side would
         silently make the runs machine-specific.

      2. Nothing consumes global RNG state during the round loop. The cohort for
         round r comes from a generator derived from (seed, r) alone, and each
         client's local stream from (seed, r, k) alone. Global-state consumption
         order can shift with device, thread count, or library version; keyed
         generators cannot.

    Note that --dtype tf32/amp and --compile all perturb the numerics slightly
    (TF32 mantissas, bf16 autocast, and fusion-changed summation order
    respectively). fp32 without compile is the default precisely so that results
    from different machines are directly comparable; the faster modes are opt-in
    for exploratory runs. The header records which was used.

    Machine-shaped knobs (device, dtype, compile, data placement, thread count,
    paths) are arguments. Experiment-shaped constants (batch size, local epochs,
    lr, client count, cohort size, rounds, alpha, seed) are NOT derived from
    hardware in any way -- if a batch of 32 does not fit, the run fails with an
    explicit error rather than quietly shrinking and producing an incomparable
    number.

Single (alpha, seed) config, single device, no DDP and no DataParallel --
parallelism comes from running many of these processes at once (see run_sweep.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# ||sum D_k||^2 below this multiple of sum ||D_k||^2 is treated as numerical
# cancellation rather than a real measurement. float64 reduction error over ~1e7
# terms is ~1e-13 relative, so 1e-9 is ~4 orders of margin.
UNDERFLOW_REL = 1e-9

# Slack on the upper-bound (K_c) assertion, relative to K_c.
BOUNDS_TOL = 1e-6

DTYPES = ("fp32", "tf32", "amp")

# Floor below which sigma2_hat is treated as unresolvable rather than negative.
SIGMA2_EPS = 1e-30
# The predicted ratio is an identity, so this only has to absorb float64 noise.
DECOMP_TOL = 1e-9

# BOUNDS_NOTE
# -----------
# The measured range is 1/Delta_S in (0, K_c].
#
# The upper bound is a theorem: Cauchy-Schwarz gives ||sum_k D_k||^2 <= K_c *
# sum_k ||D_k||^2 for any vectors whatsoever, so 1/Delta_S <= K_c always, and a
# violation is unambiguously a bug in this file. It stays a hard raise.
#
# The lower bound is NOT 1. 1/Delta_S >= 1 holds iff the cross terms are
# non-negative on average, sum_{j != k} <D_j, D_k> >= 0. That is a
# positive-correlation condition on the cohort -- the empirically usual case,
# since clients mostly agree on the descent direction -- but it is not an
# identity, and a net-cancelling cohort genuinely produces 1/Delta_S < 1. This
# was observed in production at alpha=100, seed 1, batch 32, tau=1, which gave
# 1/Delta_S = 0.9906; the old [1, K_c] assertion crashed that run on a correct
# measurement.
#
# All that survives on the low side is positivity: ||sum_k D_k||^2 > 0 whenever
# the row is not underflow-flagged, so 1/Delta_S > 0. Rounds below 1 are real
# measurements: they are neither clamped nor warned about per round, only marked
# with the boolean column `cancellation_regime` and counted in the end-of-run
# summary.
CANCELLATION_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- experiment: never derived from hardware -------------------------
    alpha: float
    seed: int
    rounds: int = 200
    num_clients: int = 100
    cohort_size: int = 10
    local_epochs: int = 2
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    local_batch_size: str = "32"      # int as a string, or the literal "full"
    server_lr: float = 1.0
    eval_every: int = 10
    min_client_samples: int = 10
    max_partition_retries: int = 100
    max_samples_per_client: int = 0        # 0 = no cap (smoke mode sets 200)
    # --- machine: all of these may vary per host -------------------------
    device: str = "cpu"
    dtype: str = "fp32"
    compile: bool = False
    data_device: str = "device"            # "device" | "cpu"
    num_threads: int = 0                   # 0 = leave torch's default alone
    data_root: str = "./data"
    out_dir: str = "./results"
    out: str = ""                          # explicit path overrides out_dir
    per_gpu: int = 0                       # provenance only, recorded in header
    # --- test-only -------------------------------------------------------
    # every client gets the identical shard and the identical local RNG stream,
    # so every D_k must come out bit-identical (see test 2).
    debug_identical_clients: bool = False
    extra: dict = field(default_factory=dict)


def resolve_batch_size(spec: str, n: int) -> int:
    """
    "full" means one batch per epoch over the client's whole shard.

    This is an EXPERIMENT AXIS (see the presets), never a memory knob: it is
    resolved from the shard size alone and is never reduced to make a step fit.
    """
    if str(spec).strip().lower() == "full":
        return int(n)
    b = int(spec)
    if b < 1:
        raise ValueError(f"--local-batch-size must be >= 1 or 'full', got {spec!r}")
    return b


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(spec: str) -> torch.device:
    """
    "cpu" | "cuda" | "cuda:N" -> a torch.device with an explicit index.

    A bare torch.device("cuda") carries no index, which several torch entry
    points reject; resolving it once here means the rest of the file can use one
    device object uniformly and never re-inspect the string.
    """
    d = torch.device(spec)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {spec!r} requested but torch reports no CUDA device "
                "(use --device cpu)")
        if d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
    return d


# ---------------------------------------------------------------------------
# Model: ResNet-18 for 32x32, GroupNorm throughout
# ---------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.n1 = _gn(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.n2 = _gn(planes)
        self.short = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.short = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), _gn(planes)
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNet18GN(nn.Module):
    """3x3 stride-1 stem, no initial max-pool -- the standard 32x32 adaptation."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.n1 = _gn(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# Flat parameter vectors -- fixed, asserted order, parameters only
# ---------------------------------------------------------------------------

def param_names(model: nn.Module) -> list[str]:
    """Canonical parameter order. Buffers are deliberately absent."""
    names = [n for n, _ in model.named_parameters()]
    n_buffers = len(list(model.buffers()))
    if n_buffers != 0:
        raise RuntimeError(
            f"model exposes {n_buffers} buffer(s); this experiment requires a "
            "buffer-free (GroupNorm) model so that no running statistic can "
            "leak into the pseudo-gradient"
        )
    return names


def get_flat(model: nn.Module, names: list[str]) -> torch.Tensor:
    parts, i = [], 0
    for n, p in model.named_parameters():
        if n != names[i]:
            raise RuntimeError(f"parameter order drift at {i}: {n!r} != {names[i]!r}")
        parts.append(p.detach().reshape(-1))
        i += 1
    if i != len(names):
        raise RuntimeError(f"expected {len(names)} parameters, saw {i}")
    return torch.cat(parts)


def set_flat(model: nn.Module, names: list[str], vec: torch.Tensor) -> None:
    with torch.no_grad():
        off, i = 0, 0
        for n, p in model.named_parameters():
            if n != names[i]:
                raise RuntimeError(f"parameter order drift at {i}: {n!r} != {names[i]!r}")
            k = p.numel()
            p.copy_(vec[off:off + k].view_as(p))
            off += k
            i += 1
        if off != vec.numel() or i != len(names):
            raise RuntimeError("flat vector length does not match parameter set")


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def _ratio(deltas: list[torch.Tensor], scale: np.ndarray | None) -> dict[str, Any]:
    """
    Delta_S and 1/Delta_S for {scale_k * D_k}, accumulated in float64.

    Returns raw sums too, so the JSONL keeps enough to recompute the ratio.
    """
    acc = torch.zeros(deltas[0].numel(), dtype=torch.float64, device=deltas[0].device)
    sum_sq = 0.0
    norms = []
    for k, d in enumerate(deltas):
        v = d.to(torch.float64)
        if scale is not None:
            v = v * float(scale[k])
        acc += v
        sq = float(torch.dot(v, v))
        sum_sq += sq
        norms.append(math.sqrt(max(sq, 0.0)))
        del v
    sum_norm_sq = float(torch.dot(acc, acc))
    del acc

    underflow = sum_norm_sq <= UNDERFLOW_REL * sum_sq
    if underflow:
        delta_s = float("inf")
        inv = 0.0
    else:
        delta_s = sum_sq / sum_norm_sq
        inv = sum_norm_sq / sum_sq
    return {
        "delta_s": delta_s,
        "inv_delta_s": inv,
        "sum_of_sq_norms": sum_sq,          # sum_k ||D_k||^2
        "norm_sq_of_sum": sum_norm_sq,      # ||sum_k D_k||^2
        "delta_norm_mean": float(np.mean(norms)),
        "delta_norm_std": float(np.std(norms)),
        "denominator_underflow": bool(underflow),
    }


def decompose(sum_of_sq_norms: float, norm_sq_of_sum: float, K_c: int
              ) -> dict[str, Any]:
    """
    Split the cohort spread into a shared direction and per-client noise.

    MODEL AND ITS ASSUMPTIONS. We write D_k = mu + eps_k and assume the eps_k are
    INDEPENDENT ACROSS CLIENTS and have EQUAL VARIANCE, E||eps_k||^2 = sigma^2 for
    every k, with E[eps_k] = 0. Neither assumption is verified here and neither is
    exactly true in practice: clients that share classes have correlated noise,
    and clients with different n_k have different variances. Read mu2/sigma2 as a
    two-parameter summary of the cohort spread, not as a measurement of a latent
    truth.

    Given the assumptions, with
        A = (1/K) sum_k ||D_k||^2   and   B = ||mean_k D_k||^2 ,
        E[A] = ||mu||^2 + sigma^2           (each client carries the full noise)
        E[B] = ||mu||^2 + sigma^2 / K       (averaging K clients shrinks it by K)
    so subtracting gives sigma^2 and back-substituting gives ||mu||^2.

    No extra compute: A and B are rescalings of the two sums the ratio already
    needed.

    The predicted ratio is an ALGEBRAIC IDENTITY, not an independent check of the
    model -- substituting r = mu2/sigma2 into (K*r+1)/(r+1) collapses to K*B/A,
    which is 1/Delta_S. It is asserted because a mismatch means the arithmetic
    here drifted from the arithmetic in _ratio(), which would be a bug.
    """
    if K_c < 2:
        # (1 - 1/K) vanishes at K=1; there is no within-cohort spread to split.
        return {"mu2_hat": float("nan"), "sigma2_hat": float("nan"),
                "r_hat": float("nan"),
                "one_over_delta_s_predicted": float("nan"),
                "sigma2_clamped": True}

    A = sum_of_sq_norms / K_c
    B = norm_sq_of_sum / (K_c ** 2)
    sigma2 = (A - B) / (1.0 - 1.0 / K_c)
    mu2 = A - sigma2

    # sigma2 <= 0 happens when the clients are near-identical (B >= A), i.e. the
    # noise term is unresolvable at this cohort size. Clamp for r_hat only, and
    # flag the row -- never silently present the clamped value as a measurement.
    clamped = not (sigma2 > SIGMA2_EPS)
    sigma2_r = sigma2 if not clamped else SIGMA2_EPS
    r = mu2 / sigma2_r
    predicted = (K_c * r + 1.0) / (r + 1.0)
    return {"mu2_hat": mu2, "sigma2_hat": sigma2, "r_hat": r,
            "one_over_delta_s_predicted": predicted,
            "sigma2_clamped": bool(clamped)}


def assert_decomposition(pred: float, measured: float, clamped: bool,
                         underflow: bool, tag: str) -> None:
    """The predicted ratio is an identity given the estimates; a gap is a bug."""
    if clamped or underflow or not math.isfinite(pred):
        return
    if abs(pred - measured) > DECOMP_TOL * max(abs(measured), 1.0):
        raise RuntimeError(
            f"decomposition identity violated ({tag}): predicted 1/Delta_S = "
            f"{pred!r} vs measured {measured!r}. (K*r+1)/(r+1) must collapse to "
            "K*B/A exactly; a gap means decompose() and _ratio() disagree."
        )


def diversity(deltas: list[torch.Tensor], n_k: list[int]) -> dict[str, Any]:
    """
    Both variants of the cohort diversity ratio.

    deltas -- one flat float32 tensor per cohort member, D_k = w_b - w_k.
    n_k    -- that client's local sample count, used only by the weighted variant.
    """
    if len(deltas) != len(n_k) or not deltas:
        raise RuntimeError("deltas and n_k must be non-empty and the same length")
    counts = np.asarray(n_k, dtype=np.float64)
    scale = counts / counts.mean()          # n_k / mean_j(n_j), mean 1 by construction

    K_c = len(deltas)
    out: dict[str, Any] = {}
    for tag, sc in (("unweighted", None), ("weighted", scale)):
        r = _ratio(deltas, sc)
        d = decompose(r["sum_of_sq_norms"], r["norm_sq_of_sum"], K_c)
        assert_decomposition(d["one_over_delta_s_predicted"], r["inv_delta_s"],
                             d["sigma2_clamped"], r["denominator_underflow"], tag)
        for key, val in r.items():
            out[f"{key}_{tag}"] = val
        # The unweighted variant is the theory one, so it carries the bare column
        # names; the weighted counterparts are suffixed like every other column.
        for key, val in d.items():
            out[key if tag == "unweighted" else f"{key}_{tag}"] = val
    return out


def assert_bounds(inv: float, K_c: int, tag: str, underflow: bool) -> bool:
    """
    Assert 1/Delta_S in (0, K_c] up to tolerance; return the cancellation flag.

    The upper bound is Cauchy-Schwarz and a violation raises. Below it only
    positivity is asserted: a value in (0, 1) is a real net-cancelling cohort,
    so it is reported as True rather than raised on -- see BOUNDS_NOTE.
    """
    if underflow:
        return False                        # flagged row; the ratio is not a measurement
    if not inv > 0.0:
        raise RuntimeError(
            f"gradient-diversity bounds violated ({tag}): 1/Delta_S = {inv!r} "
            "is not positive. ||sum_k D_k||^2 > 0 on any row not flagged as "
            "denominator underflow, so this is a bug in this file. The offending "
            "round has already been written to the JSONL."
        )
    if inv > K_c * (1.0 + BOUNDS_TOL):
        raise RuntimeError(
            f"gradient-diversity bounds violated ({tag}): 1/Delta_S = {inv!r} "
            f"exceeds K_c = {K_c} (rel tol {BOUNDS_TOL:g}). Cauchy-Schwarz makes "
            "this unambiguously a bug in this file -- see BOUNDS_NOTE. The "
            "offending round has already been written to the JSONL."
        )
    return inv < CANCELLATION_THRESHOLD


def cancellation_flag(stats: dict[str, Any]) -> bool:
    """
    True if either variant measured 1/Delta_S < 1 on a row that is a measurement.

    Pure: it never raises, so the flag can be stamped on the record before the
    record is flushed and the bounds are asserted. See BOUNDS_NOTE.
    """
    return any(not stats[f"denominator_underflow_{tag}"]
               and stats[f"inv_delta_s_{tag}"] < CANCELLATION_THRESHOLD
               for tag in ("unweighted", "weighted"))


# ---------------------------------------------------------------------------
# Aggregation -- the single insertion point for future mask modes
# ---------------------------------------------------------------------------

def aggregate(deltas: list[torch.Tensor], n_k: list[int], server_lr: float,
              w_broadcast: torch.Tensor) -> torch.Tensor:
    """
    FedAvg server step: w <- w_b - server_lr * sum_k (n_k / sum_j n_j) * D_k.

    Kept as one function on purpose: a mask mode drops in here (transform the
    per-client deltas, or the aggregate, before the server step) without touching
    the training loop or the measurement.
    """
    counts = torch.tensor(n_k, dtype=torch.float64, device=w_broadcast.device)
    weights = counts / counts.sum()
    agg = torch.zeros_like(w_broadcast, dtype=torch.float64)
    for k, d in enumerate(deltas):
        agg += d.to(torch.float64) * weights[k]
    return (w_broadcast.to(torch.float64) - server_lr * agg).to(w_broadcast.dtype)


# ---------------------------------------------------------------------------
# Reproducible randomness
# ---------------------------------------------------------------------------

def keyed_generator(*key: int) -> torch.Generator:
    """
    A CPU torch.Generator seeded from an integer key and nothing else.

    CPU on purpose: torch.Generator(device='cuda') seeded with s produces a
    DIFFERENT stream than a CPU generator seeded with s, so any draw made
    device-side would make batch order and augmentation machine-specific. Draws
    happen here and the results are moved to the compute device.

    Keyed on purpose: no dependence on global RNG state, whose consumption order
    can shift with device, thread count, or library version.
    """
    s = int(np.random.default_rng([int(k) for k in key]).integers(1 << 62))
    g = torch.Generator()
    g.manual_seed(s)
    return g


# ---------------------------------------------------------------------------
# Data: whole train set preloaded once, index-based batching, no DataLoader
# ---------------------------------------------------------------------------

class DeviceCifar:
    """
    CIFAR-10 held as uint8 (~150 MB train, ~30 MB test).

    `store` is where the pooled tensor lives (the compute device by default, CPU
    when VRAM is tight) and `device` is where batches are computed. Batches are
    unconditionally `.to(device)`d, which is a no-op when the two coincide -- so
    there is one code path, not a per-device branch.
    """

    def __init__(self, root: str, device: torch.device, store: torch.device | None = None):
        from torchvision.datasets import CIFAR10

        tr = CIFAR10(root=root, train=True, download=True)
        te = CIFAR10(root=root, train=False, download=True)
        self.device = device
        self.store = store if store is not None else device
        self.xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous().to(self.store)
        self.ytr = torch.tensor(tr.targets, dtype=torch.long).to(self.store)
        self.xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous().to(self.store)
        self.yte = torch.tensor(te.targets, dtype=torch.long).to(self.store)
        self.labels_np = np.asarray(tr.targets, dtype=np.int64)
        self.mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    def batch(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x[idx.to(x.device)].to(self.device)

    def normalise(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.to(self.device).float().div_(255.0) - self.mean) / self.std

    def augment(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """
        Random 32x32 crop from 4-px zero padding, then random horizontal flip.

        All three draws are made on the CPU generator and moved across, so the
        augmentation stream is identical on every device.
        """
        B = x.shape[0]
        oy = torch.randint(0, 9, (B,), generator=gen).to(x.device)
        ox = torch.randint(0, 9, (B,), generator=gen).to(x.device)
        flip = (torch.rand(B, generator=gen) < 0.5).to(x.device)

        xp = F.pad(x, (4, 4, 4, 4))
        ar = torch.arange(32, device=x.device)
        rows = (oy[:, None] + ar[None, :])[:, :, None]          # (B, 32, 1)
        cols = (ox[:, None] + ar[None, :])[:, None, :]          # (B, 1, 32)
        b = torch.arange(B, device=x.device)[:, None, None]
        # advanced index with a slice in the middle -> (B, 32, 32, C)
        out = xp[b, :, rows, cols].permute(0, 3, 1, 2).contiguous()
        return torch.where(flip[:, None, None, None], out.flip(-1), out)


# ---------------------------------------------------------------------------
# Per-class Dirichlet partition (Hsu et al. 2019)
# ---------------------------------------------------------------------------

def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        seed: int, min_samples: int, max_retries: int
                        ) -> list[np.ndarray]:
    """
    For each class c draw p_c ~ Dir(alpha * 1_K) and split class c's indices across
    the K clients in those proportions. Redraw the whole partition until every
    client holds at least `min_samples`; fail loudly if that never happens.

    numpy-only and keyed on (seed, attempt), so the partition is identical on
    every machine.
    """
    classes = np.unique(labels)
    for attempt in range(max_retries):
        rng = np.random.default_rng([seed, attempt])
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, part in enumerate(np.split(idx, cuts)):
                buckets[k].append(part)
        shards = [np.concatenate(b) for b in buckets]
        if min(len(s) for s in shards) >= min_samples:
            for s in shards:
                rng.shuffle(s)
            return shards
    raise RuntimeError(
        f"could not satisfy min {min_samples} samples/client for alpha={alpha}, "
        f"K={num_clients} in {max_retries} redraws"
    )


# ---------------------------------------------------------------------------
# Numerics / device setup
# ---------------------------------------------------------------------------

def apply_dtype(dtype: str) -> None:
    """
    fp32 is the comparable default and explicitly disables TF32 -- torch enables
    cuDNN TF32 by default, which would silently change the numerics between an
    Ampere-class card and everything else.
    """
    if dtype not in DTYPES:
        raise ValueError(f"--dtype must be one of {DTYPES}, got {dtype!r}")
    tf32 = dtype == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def autocast_ctx(device: torch.device, dtype: str):
    """amp == bfloat16 autocast: works on CPU and CUDA alike, and needs no scaler."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=(dtype == "amp"))


def oom_message(exc: BaseException, cfg: Config) -> RuntimeError:
    return RuntimeError(
        f"out of memory at --local-batch-size {cfg.local_batch_size!r} on "
        f"{cfg.device}. Local batch size is an experiment axis, not a memory "
        "knob: reducing it to fit would change what is being measured and make "
        "this run incomparable to the rest of the grid. Free VRAM instead -- "
        "lower --per-gpu, use --data-device cpu, or run this configuration on a "
        "larger card. Original error: " + str(exc)
    )


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(model: nn.Module, fwd, data: DeviceCifar, idx: torch.Tensor,
                cfg: Config, gen: torch.Generator, device: torch.device) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    n = idx.numel()
    bs = resolve_batch_size(cfg.local_batch_size, n)
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(n, generator=gen)          # CPU draw, see keyed_generator
        for s in range(0, n, bs):
            sel = idx[perm[s:s + bs].to(idx.device)]
            xb = data.augment(data.batch(data.xtr, sel), gen)
            yb = data.batch(data.ytr, sel)
            with autocast_ctx(device, cfg.dtype):
                loss = F.cross_entropy(fwd(data.normalise(xb)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate(model: nn.Module, fwd, data: DeviceCifar, cfg: Config,
             device: torch.device, batch: int = 500) -> float:
    model.eval()
    correct = 0
    n = data.xte.shape[0]
    for s in range(0, n, batch):
        xb = data.normalise(data.xte[s:s + batch].to(device))
        with autocast_ctx(device, cfg.dtype):
            pred = fwd(xb).argmax(1)
        correct += int((pred == data.yte[s:s + batch].to(device)).sum())
    return correct / n


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def git_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def environment(device: torch.device, cfg: Config) -> dict[str, Any]:
    """Everything needed to tell which machine a pooled result came from."""
    return {
        "hostname": socket.gethostname(),
        "device_arg": cfg.device,
        "device_resolved": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda"
                        else "cpu"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        "dtype": cfg.dtype,
        "compile": cfg.compile,
        "data_device": cfg.data_device,
        "num_threads": torch.get_num_threads(),
        "per_gpu": cfg.per_gpu,
        "git_hash": git_hash(),
        "python": sys.version.split()[0],
    }


def warn_version_drift(out_dir: str) -> None:
    """
    Compare this interpreter's torch/CUDA against what earlier JSONL headers in
    the same output directory recorded. Pooled results from mismatched stacks are
    not directly comparable, and the mismatch is otherwise invisible at plot time.
    """
    import glob
    seen = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl")))[:200]:
        try:
            with open(path) as fh:
                env = json.loads(fh.readline()).get("env", {})
        except Exception:
            continue
        key = (env.get("torch_version"), env.get("cuda_version"))
        if key != (None, None):
            seen.setdefault(key, []).append(os.path.basename(path))
    mine = (torch.__version__, torch.version.cuda)
    others = {k: v for k, v in seen.items() if k != mine}
    if others:
        print(f"[fed_diversity] WARNING: this run uses torch {mine[0]} / CUDA "
              f"{mine[1]}, but {out_dir} already holds results from:", file=sys.stderr)
        for (tv, cv), files in others.items():
            print(f"    torch {tv} / CUDA {cv}  ({len(files)} file(s), "
                  f"e.g. {files[0]})", file=sys.stderr)
        print("    Numerics can differ across stacks; check before pooling.",
              file=sys.stderr)


def run(cfg: Config, data: DeviceCifar | None = None) -> str:
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    apply_dtype(cfg.dtype)
    # Deterministic convolutions: identical clients must produce bit-identical
    # deltas (test 2), and a paper number should not depend on algo autotuning.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Global seeding covers module init; the round loop uses keyed generators only.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))
    torch.manual_seed(cfg.seed)

    if data is None:
        store = torch.device("cpu") if cfg.data_device == "cpu" else device
        data = DeviceCifar(cfg.data_root, device, store)

    shards = dirichlet_partition(data.labels_np, cfg.num_clients, cfg.alpha,
                                 cfg.seed, cfg.min_client_samples,
                                 cfg.max_partition_retries)
    if cfg.debug_identical_clients:
        shards = [shards[0].copy() for _ in range(cfg.num_clients)]
    if cfg.max_samples_per_client > 0:
        shards = [s[:cfg.max_samples_per_client] for s in shards]
    shard_t = [torch.from_numpy(s).to(device) for s in shards]
    n_all = [int(len(s)) for s in shards]

    model = ResNet18GN().to(device)
    names = param_names(model)
    # Compile wraps the forward only; parameter names and identities stay those
    # of the eager module, so the flat-vector order is unaffected by --compile.
    fwd = torch.compile(model) if cfg.compile else model
    w = get_flat(model, names).clone()
    d_dim = int(w.numel())

    out_path = cfg.out or default_out_path(cfg)
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    warn_version_drift(out_dir)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fh = open(out_path, "w", buffering=1)

    fh.write(json.dumps({
        "record": "header",
        "config": asdict(cfg),
        "env": environment(device, cfg),
        "num_params": d_dim,
        "num_param_tensors": len(names),
        "param_names": names,
        "client_sizes": n_all,
        "started_at": time.time(),
    }) + "\n")

    t0 = time.time()
    acc = None
    n_cancelling = 0
    try:
        for rnd in range(cfg.rounds):
            r_t0 = time.time()
            # Cohort from (seed, round) alone -- no global RNG, no carried state.
            cohort = sorted(np.random.default_rng([cfg.seed, rnd, 0xC0]).choice(
                cfg.num_clients, size=cfg.cohort_size, replace=False).tolist())
            K_c = len(cohort)

            deltas, n_k = [], []
            for k in cohort:
                set_flat(model, names, w)
                # In the identical-clients debug mode the client id is dropped,
                # so every client sees exactly the same batches and crops.
                key = ((cfg.seed, rnd) if cfg.debug_identical_clients
                       else (cfg.seed, rnd, k))
                local_train(model, fwd, data, shard_t[k], cfg,
                            keyed_generator(*key), device)
                deltas.append((w - get_flat(model, names)).clone())
                n_k.append(n_all[k])

            stats = diversity(deltas, n_k)
            w = aggregate(deltas, n_k, cfg.server_lr, w)
            del deltas

            if cfg.eval_every > 0 and ((rnd + 1) % cfg.eval_every == 0
                                       or rnd == cfg.rounds - 1):
                set_flat(model, names, w)
                acc = evaluate(model, fwd, data, cfg, device)
            else:
                acc = None

            cancelling = cancellation_flag(stats)
            n_cancelling += cancelling

            rec = {
                "record": "round",
                "cancellation_regime": cancelling,
                "round": rnd,
                "alpha": cfg.alpha,
                "seed": cfg.seed,
                "cohort": cohort,
                "K_c": K_c,
                "n_k": n_k,
                "test_acc": acc,
                "round_seconds": time.time() - r_t0,
                "elapsed_seconds": time.time() - t0,
            }
            rec.update(stats)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            # After the row is on disk, so a violation leaves the evidence behind.
            assert_bounds(stats["inv_delta_s_unweighted"], K_c, "unweighted",
                          stats["denominator_underflow_unweighted"])
            assert_bounds(stats["inv_delta_s_weighted"], K_c, "weighted",
                          stats["denominator_underflow_weighted"])
    except torch.cuda.OutOfMemoryError as e:                 # pragma: no cover
        fh.close()
        raise oom_message(e, cfg) from e
    except RuntimeError as e:                                # pragma: no cover
        fh.close()
        if "out of memory" in str(e).lower():
            raise oom_message(e, cfg) from e
        raise

    total = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2 ** 20
               if device.type == "cuda" else None)
    fh.write(json.dumps({
        "record": "footer", "done": True, "rounds": cfg.rounds,
        "cancellation_rounds": n_cancelling,
        "cancellation_fraction": n_cancelling / max(cfg.rounds, 1),
        "final_test_acc": acc, "total_seconds": total,
        "seconds_per_round": total / max(cfg.rounds, 1),
        "peak_memory_mb": peak_mb,
    }) + "\n")
    fh.close()
    # Printed so --per-gpu can be raised from measurement rather than guesswork.
    print(f"[fed_diversity] {os.path.basename(out_path)}: {total / 60:.1f} min "
          f"({total / max(cfg.rounds, 1):.1f} s/round)"
          + (f", peak GPU {peak_mb:.0f} MiB" if peak_mb is not None else ""))
    # Sub-unit 1/Delta_S is a real net-cancelling cohort, not an error (BOUNDS_NOTE).
    # Reported once here rather than warned about round by round.
    print(f"[fed_diversity] {os.path.basename(out_path)}: cancellation regime "
          f"(1/Delta_S < 1) in {n_cancelling}/{cfg.rounds} rounds "
          f"({n_cancelling / max(cfg.rounds, 1):.1%})")
    return out_path


def config_stem(alpha: float, seed: int, local_batch_size: str,
                local_epochs: int) -> str:
    """Filename identity of a config. Batch and E are grid axes, so they belong here."""
    return f"alpha{alpha:g}_seed{seed}_b{local_batch_size}_e{local_epochs}"


def default_out_path(cfg: Config) -> str:
    return os.path.join(cfg.out_dir, config_stem(
        cfg.alpha, cfg.seed, cfg.local_batch_size, cfg.local_epochs) + ".jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    e = p.add_argument_group(
        "experiment (defines the result; never derived from hardware)")
    e.add_argument("--alpha", type=float, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--rounds", type=int, default=200)
    e.add_argument("--num-clients", type=int, default=100)
    e.add_argument("--cohort-size", type=int, default=10)
    e.add_argument("--local-epochs", type=int, default=2)
    e.add_argument("--lr", type=float, default=0.01)
    e.add_argument("--local-batch-size", default="32",
                   help='int, or the literal "full" for one batch per epoch')
    e.add_argument("--weight-decay", type=float, default=0.0)
    e.add_argument("--server-lr", type=float, default=1.0)
    e.add_argument("--eval-every", type=int, default=10)
    e.add_argument("--max-samples-per-client", type=int, default=0)

    m = p.add_argument_group("machine (may differ per host; never changes the result)")
    m.add_argument("--device", default=default_device(),
                   help='"cpu", "cuda", or "cuda:N"')
    m.add_argument("--dtype", choices=list(DTYPES), default="fp32",
                   help="fp32 is the comparable default; tf32/amp are opt-in and "
                        "change the numerics slightly")
    m.add_argument("--compile", action="store_true",
                   help="torch.compile the forward. Like tf32/amp this perturbs "
                        "the numerics slightly (operator fusion changes summation "
                        "order), so it is opt-in for exploratory runs; leave it "
                        "off for results that will be pooled across machines")
    m.add_argument("--data-device", choices=["device", "cpu"], default="device",
                   help="where the preloaded CIFAR-10 tensor lives")
    m.add_argument("--num-threads", type=int, default=0,
                   help="torch.set_num_threads; 0 leaves torch's default")
    m.add_argument("--data-root", default="./data")
    m.add_argument("--out-dir", default="./results")
    m.add_argument("--out", default="", help="explicit file path, overrides --out-dir")
    m.add_argument("--per-gpu", type=int, default=0,
                   help="recorded in the header for provenance only")

    p.add_argument("--smoke", action="store_true",
                   help="5 rounds / 20 clients / 3 sampled / 200 samples per client")
    p.add_argument("--debug-identical-clients", action="store_true",
                   help="test-only: identical shard and RNG stream for every client")
    return p


def cfg_from_args(a: argparse.Namespace) -> Config:
    cfg = Config(
        alpha=a.alpha, seed=a.seed, rounds=a.rounds, num_clients=a.num_clients,
        cohort_size=a.cohort_size, local_epochs=a.local_epochs, lr=a.lr,
        local_batch_size=a.local_batch_size, weight_decay=a.weight_decay,
        server_lr=a.server_lr, eval_every=a.eval_every,
        max_samples_per_client=a.max_samples_per_client,
        device=a.device, dtype=a.dtype, compile=a.compile,
        data_device=a.data_device, num_threads=a.num_threads,
        data_root=a.data_root, out_dir=a.out_dir, out=a.out, per_gpu=a.per_gpu,
        debug_identical_clients=a.debug_identical_clients,
    )
    if a.smoke:
        cfg.rounds = 5
        cfg.num_clients = 20
        cfg.cohort_size = 3
        cfg.max_samples_per_client = 200
        cfg.eval_every = 5
        cfg.extra["smoke"] = True
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = cfg_from_args(build_parser().parse_args(argv))
    print(f"[fed_diversity] torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"python {sys.version.split()[0]}")
    path = run(cfg)
    print(f"[fed_diversity] alpha={cfg.alpha} seed={cfg.seed} "
          f"b={cfg.local_batch_size} E={cfg.local_epochs} "
          f"device={cfg.device} dtype={cfg.dtype} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
