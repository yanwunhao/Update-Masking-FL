# fed_diversity — runbook

Measures per-round gradient diversity of FedAvg pseudo-gradients on CIFAR-10.

```
Δ_S = Σ_{k∈C} ‖Δ_k‖² / ‖Σ_{k∈C} Δ_k‖²        Δ_k = w_broadcast − w_k_final
```

Reported as `1/Δ_S ∈ [1, K_c]`. `K_c` means every client returned the same
pseudo-gradient; `1` means they are mutually orthogonal.

---

## Read this before interpreting any number

**Test 2 is the correctness control.** Identical shard + identical RNG stream for
every client, E=1 → every `Δ_k` identical → `1/Δ_S` is *exactly* `K_c`. It catches
parameter-ordering, buffer-inclusion and cohort-selection bugs in one shot. If it
fails, stop; nothing else means anything.

**α=100 is a noise-floor measurement, not a correctness check.** It does *not*
land near `K_c`. Measured on this code:

| what | value |
|---|---|
| α=100, 10 rounds, K_c=10 | `1/Δ_S` ≈ **3.25** |
| α=100, 200 rounds, K_c=10 | `1/Δ_S` ≈ **1.19** (last 50 rounds) |

The decomposition explains why: `‖μ̂‖²` is flat from round ~25 (0.0047 → 0.0040)
while `σ̂²` grows monotonically (0.052 → 0.186) as accuracy rises to 68%. **Local
update noise, not data heterogeneity, pins the ratio to its floor.** Anyone
expecting α=100 ≈ `K_c` will conclude the code is broken; it is not.

That is why `local-steps` is the primary preset and α is demoted to a robustness
check.

---

## Order of operations

### 0. Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### 1. Data — once per machine, before anything else

```bash
python prepare_data.py --data-root ./data
```

*~2 min on a good link; the upstream host often throttles to ~250 KB/s, so allow
15 min.* No shared filesystem between the 4060 box and the A6000 box, so run it on
both, or copy the `cifar-10-batches-py/` directory it prints.

**Do not skip this.** `run_sweep.py` refuses to launch without it, deliberately:
concurrent first-time torchvision downloads interleave writes and corrupt the
archive, and the failure then looks like a data bug.

### 2. `make check` — the gate

```bash
make check                 # CPU
make check DEVICE=cuda:0   # GPU
```

*CPU ~4 min, GPU ~2 min.* Runs data verification, the full test suite, and the
smoke path end to end. One `PASS`/`FAIL` line, non-zero exit on any failure. First
thing to run on a new machine, and after any edit.

Expected: **11/11 tests pass**, `1/Δ_S == K_c` exactly in test 2, cross-device
agreement at round 0 ≈ 1.6e-6.

### 3. Smoke — the whole path in under a minute

```bash
python fed_diversity.py --alpha 1.0 --seed 0 --smoke --device cpu
```

5 rounds, 20 clients, 3 sampled, 200 samples per client. Laptop-CPU friendly.
Not a measurement.

### 4. Plan the sweep

```bash
python run_sweep.py --preset local-steps --dry-run
```

Lists every config with an estimated runtime and what is already complete.
Estimates scale a measured 10.5 s/round reference (RTX 4060, fp32, eager, batch 32,
E=2, ~500 samples/client → 34.9 min for 200 rounds) by local step count.

### 5. Run

```bash
tmux new -s sweep
python run_sweep.py --preset local-steps --devices cuda:0 --per-gpu 3
```

**Use tmux or screen.** These runs last hours; an SSH drop kills the whole process
tree otherwise and you lose every in-flight config.

**Check `nvidia-smi` before choosing `--devices` and `--per-gpu`.** These are
shared machines. Each process needs ~1.2–1.5 GB plus activations; `--per-gpu 3`
fits an 8 GB 4060. On a 48 GB A6000 raise it — but raise it from the measured
peak-memory numbers the sweep prints per run, not by guessing.

Then the secondary grid:

```bash
python run_sweep.py --preset alpha-robustness --devices cuda:0 --per-gpu 3
```

Progress from another shell:

```bash
python run_sweep.py --status
```

Resume is automatic and config-level: a JSONL ending in a completion record is
skipped. There is no mid-run checkpointing, so a kill loses at most one config.

### 6. Merge across machines

```bash
python merge_results.py results_4060 results_a6000 --out merged
```

Reports config collisions with the measured disagreement and refuses to merge
until you choose (`--on-collision first|longest`). Also prints every distinct
torch/CUDA/dtype/compile environment found — runs under `tf32`, `amp` or
`--compile` perturb `Δ_S` at ~7e-4 relative and must not be pooled with
default-mode results.

### 7. Plots

```bash
python plot_primary.py --results-dir merged     # fig_local_steps.png
python plot_alpha.py   --results-dir merged     # fig_alpha.png
```

`fig_local_steps.png` panel (b) carries the explanation — the ratio alone cannot
show *which* term moves, only the `‖μ̂‖²` / `σ̂²` split can.

---

## Presets

| preset | grid | runs |
|---|---|---|
| `local-steps` **[primary]** | α=100, 200 rounds, seeds {0,1,2}, batch {32,128,full} × E {1,2,5} | **27** |
| `alpha-robustness` [secondary] | α {100,1,0.1}, 200 rounds, seeds {0,1,2}, batch 32, E=2 | 9 |
| `smoke` | α {100,0.1}, 5 rounds, seed 0 | 2 |

`local-steps` varies `r` in a known direction: fewer local steps and larger
batches mean less accumulated local noise, hence larger `r` and higher `1/Δ_S`.
The `full` + E=1 corner is minimum noise and should sit highest. **If it does not
rise materially there, the decomposition model is wrong** — which is worth knowing
before anything else runs.

`alpha-robustness` expects all three near 1.1–1.3.

---

## Arguments that must never track the machine

Learning rate, client count, cohort size, rounds, α, seed, local batch size, local
epochs. Local batch size is an experiment *axis*, but it is **never auto-reduced to
fit memory** — an OOM raises with an explicit message pointing at `--per-gpu` and
`--data-device cpu` instead.

Machine-shaped and safe to vary per host: `--device`, `--devices`, `--per-gpu`,
`--dtype`, `--compile`, `--data-device`, `--num-threads`, `--data-root`,
`--out-dir`.

Determinism does not depend on hardware: every random draw is made on a CPU
generator keyed on `(seed, round)` or `(seed, round, client)` and moved to the
device. Cohorts and shard sizes match across CPU and CUDA **exactly**. Δ_S agrees
to 1.6e-6 at round 0, drifting to ~5e-3 by round 2 — float32 conv kernels differ
in the last bits and SGD amplifies that ~30×/round. In float64 a single
forward+backward agrees to 2e-15, so this is arithmetic, not RNG.

---

## Known behaviour worth not being surprised by

- **`1/Δ_S ≥ 1` is not a theorem.** Only `≤ K_c` is (Cauchy–Schwarz). The lower
  bound holds iff the cross terms are non-negative on average. The bounds
  assertion raises on violation as specified — no clamping — and the offending row
  is flushed to the JSONL *before* the raise. α=100 bottoms out at 1.0778 with
  only 0.078 of margin, so low-α runs tripping this is a live possibility. If it
  happens, read the row before deciding it is a bug; `LOWER_BOUND` in
  `fed_diversity.py` is the deliberate knob.
- **`sigma2_clamped`** flags rounds where clients were near-identical and `σ̂²`
  came out non-positive. `r_hat` is clamped there; the flag is never hidden.
- **`Infinity`** appears in `delta_s_*` on denominator-underflow rows. Valid for
  `json.loads`, and the row carries `denominator_underflow_*: true`.
