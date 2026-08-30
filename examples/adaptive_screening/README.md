# Adaptive screening: go/no-go pilot

Implements the **non-neural pilot and gate in Section 11** of
[the experiment specification](../../adaptive_screening_experiment.md).
Everything lives in this example; the parent TestBED package is unchanged.
The pilot runs locally on CPU with JAX float64. It does not need an HPC or GPU.
Neural architectures, imitation, reinforcement learning and final learned-policy
evaluation are outside this go/no-go deliverable.

## Install and test

From this directory, create an isolated environment and install:

```bash
uv sync --extra dev
uv run pytest -q
```

Alternatively, in an activated Python 3.11–3.13 environment:

```bash
pip install -e '.[dev]'
pytest -q
```

`uv.lock` pins the local dependencies. No changes to the parent environment are
required. All CLI entry points explicitly select the CPU backend.

## Full preregistered pilot

```bash
uv run python scripts/run_pilot.py \
  --config configs/main.yaml --output outputs/pilot

uv run python scripts/check_gate.py \
  --pilot-dir outputs/pilot --output outputs/pilot/pilot_gate.json
```

The first command runs the entire validation tuning stage on 20,000 episodes
(seed 271828): 21 UCB coefficients, 496 explore-then-certify configurations per
κ, and all 16 static subset sizes. It freezes all selections **on disk before
generating either test dataset**, then evaluates each frozen policy under all
six deployment values on 50,000 episodes per test seed (314159 and 161803).
It will not overwrite a nonempty output directory. `--batch-size` controls CPU
memory and does not change the generated episodes, grids or protocol.

The second command prints `PILOT GATE: PASS` (exit 0), or
`PILOT GATE: FAIL -- NEURAL TRAINING DISABLED` (exit 1).
It checks both allowed pairs in order: (2.0, 2.5), then (2.5, 3.0).
Every requirement is recorded individually, including seed-specific gains,
pooled paired intervals, adaptive/static and two-stage advantages, default
rates, reduced search breadth and EIG action disagreement.

**If the gate fails, stop and seek scientific review. Do not recalibrate the
environment or start neural training.** Smoke tests never open the primary gate.

## Reproducibility and guard

Every global episode ID has independent JAX PRNG streams for its latent
utilities, candidate-specific potential observations, policy uniforms and
candidate permutation. The complete arrays are saved once per dataset. All
policies use the same permutation for tie-breaking and the same candidate's
nth observation for its nth measurement. Static allocations give extra
observations to the first candidates in the random permutation, which also
randomizes the extra-observation recipients within each subset.

The static and two-stage validation kernels calculate the terminal posterior
directly from potential-observation prefix sums. This is exactly the specified
policy, not an approximation. Tests compare this optimized path against the
full sequential simulator. Frozen test evaluation always uses the sequential
simulator, including trajectory diagnostics.

Return is the selected **posterior mean**; realized latent utility is reported
separately. The single shared prior-maximum integral uses SciPy `quad` at
1e-10 absolute/relative tolerances. `risk = Bayes component + incompatibility`
is checked at float64 tolerance. All policy pairs have paired episode-level
return and search-breadth intervals, separately by seed and pooled.

The environment hash includes the exact configuration, seeds, dataset sizes,
all tuning grids, tie tolerance, protocol version, and Git HEAD. A separate
full source snapshot hash records the actual experiment code, including
uncommitted files, for audit; it also detects edits made while a pilot is
running. It is not part of the post-gate environment comparison. The result
manifest separately hashes the resolved configuration, runtime metadata,
validation tables, frozen selections, and aggregate test evidence required by
the gate. Adding neural architecture, optimizer or training files, or training
dependencies therefore does not invalidate certified pilot evidence. The gate
retains the Git revision and full source snapshot that produced the pilot;
later commits do not rewrite that provenance. Changing the requested
environment configuration does invalidate the gate.

The training boundary can be checked without initializing a neural policy:

```bash
uv run python scripts/train_policy.py --config configs/main.yaml \
  --gate outputs/pilot/pilot_gate.json --verify-only
```

It rejects an absent, failed, altered or mismatched gate and verifies the
underlying evidence again. Without `--verify-only` it also refuses to train:
this example deliberately contains no neural trainer. A future trainer must
call `screening.gate.require_passing_gate` before initializing neural code.
GPU training, if subsequently implemented, belongs on the HPC via Slurm;
CPU evaluation remains local.

## Outputs

`outputs/pilot/` contains the specified resolved configuration, environment
hash, three complete validation CSVs, test metrics by seed, paired comparisons,
summary and gate JSON. Failure also produces `pilot_failure_report.md`.
Additional artifacts include frozen hyperparameters, runtime metadata, an
integrity manifest, the complete datasets and paired episode metrics (`.npz`),
an overview plot and exhaustive tuning plots. Large `.npz` files are retained
locally and ignored by Git. They support deeper trajectory audits but are
reproducible from the recorded seeds and are not needed to verify the gate.
All CSVs carry environment, commit and source hashes.

Budget and last-six allocation fractions use zero for default-selection
episodes. Selected SD is conditional on candidate deployment. First-leader
round is conditional on deployment and ever becoming the posterior-mean
leader among candidates, after observations in rounds 1..T. A separate
never-leader rate handles eligible candidates that were never the mean leader;
empty conditional groups are blank, not fabricated zeros.

Tests cover conjugacy, predictive moments, 100,000 exact zero-mismatch rule
comparisons, eligibility and tie boundaries, quadrature/Monte Carlo agreement,
Rao–Blackwellization for static and adaptive designs, all baseline contracts,
batch-invariant common randomness, every gate condition, provenance failures,
and a complete 256-episode d=4, T=6 smoke pilot with all deployment values.
