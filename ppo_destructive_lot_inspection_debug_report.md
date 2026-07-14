# PPO destructive lot-inspection debugging report

## Summary

The destructive lot-inspection task is not degenerate. Simple fixed and
adaptive hand policies get well below random allocation, so the original PPO
failure was not caused by an impossible objective.

The main failure was that PPO had to learn a sparse, thresholded terminal
objective through a weak state representation. The generic categorical policy
encoded each inspected zone as a normalized scalar. That made the useful
structure hard to recover: the suspicious set is non-contiguous
`{0, 4, 8, 12}`, and the adaptive rule depends on per-zone visit and defect
counts.

The fix was to keep the fixed downstream decision policy, but improve the PPO
training signal and state representation:

- train on posterior expected terminal loss under the experimenter's prior;
- evaluate reported deployed risk against sampled true lot quality;
- expose per-zone visit fraction, defect rate, defect load, and elapsed
  fraction to the actor and critic;
- initialize actor logits from the known fixed decision-maker location prior;
- use actor learning rate `2.5e-4` instead of `0.01`.

With these changes, the CLI reproduction reaches about `0.312` deployed risk
after only 200 PPO iterations, close to the simple adaptive heuristic around
`0.305`.

## Current setup

The working-tree defaults in `examples/ppo_destructive_lot_inspection.py` use:

- 16 zones on a ring, horizon `T=30`, and inspection batch size `m=8`;
- experimenter good-lot probability 0.6;
- DM good-lot probability 0.4 and acceptance threshold 0.8;
- equally spaced suspicious zones `{0, 4, 8, 12}` with suspicion weight 0.75;
- 256 PPO environments;
- `GAE(lambda=1)`, no entropy bonus, actor LR `2.5e-4`, critic LR `2.5e-4`;
- stochastic categorical sampling during evaluation.

The experimenter has equal false-accept and false-reject costs. Its reported
deployed risk is

```text
P(false accept) + P(false reject).
```

Always accepting has risk 0.4 and always rejecting has risk 0.6. Random
allocation is already close to the better constant action, so absolute gains
can look modest even when the inspection policy is meaningfully better.

## Baselines

All results use the experimenter prior and the same fixed conservative DM rule.
Conditional rates are `P(accept | good)` and `P(accept | bad)`.

| Policy | Episodes | Risk | Accept given good | Accept given bad |
|---|---:|---:|---:|---:|
| Random | 10,000 | 0.4040 +/- 0.0049 | 35.2% | 3.9% |
| Old PPO, seed 1 | 5,000 | 0.3934 +/- 0.0069 | 38.2% | 4.7% |
| Old PPO, seed 2 | 5,000 | 0.3902 +/- 0.0069 | 38.9% | 4.6% |
| Suspicious cycle | 10,000 | 0.3321 +/- 0.0047 | 48.6% | 6.3% |
| Broad then fixed | 10,000 | 0.3554 +/- 0.0048 | 43.6% | 4.8% |
| Broad then adaptive | 10,000 | **0.3051 +/- 0.0046** | **53.1%** | 6.3% |

Important comparisons:

- The old PPO policy improved over random by only about `0.011--0.014` risk.
- The suspicious cycle improved over random by about `0.072`.
- The simple adaptive heuristic improved over the suspicious cycle by about
  `0.027`.
- The fixed broad schedule was worse than the suspicious cycle. Broad coverage
  alone was not the gain; outcome-adaptive follow-up was the gain.

## What changed

### Posterior expected training reward

The old reward was a binary terminal loss after 30 inspections. Every action in
an episode received the same noisy terminal outcome, and one action only
affected reward if the complete history crossed the DM posterior threshold.

The new training reward still uses the same fixed DM accept/reject decision,
but replaces sampled true-lot loss with the posterior expected experimenter
loss under the experimenter's prior:

```text
if DM accepts:  false_accept_cost * P_exp(bad | history)
if DM rejects:  false_reject_cost * P_exp(good | history)
```

This is Rao-Blackwellization. It preserves the expected objective but removes
unnecessary Bernoulli noise from the PPO return. Evaluation intentionally still
reports empirical deployed risk against sampled true lot quality, so the final
number remains comparable to the hand baselines.

### Lot-specific actor and critic features

The generic categorical policy used a DeepSets encoder over `(normalized zone,
outcome)` pairs. That representation can in principle learn the task, but it
has to infer basic sufficient statistics from scratch:

- how often each zone has been inspected;
- how many defects have appeared in each zone;
- the defect rate by zone;
- how far through the horizon the episode is;
- which zones belong to the non-contiguous suspicious set.

The new actor and critic use explicit summary features:

```text
[per-zone visit fraction,
 per-zone defect rate,
 per-zone defect load,
 elapsed fraction]
```

This gives PPO direct access to the variables used by the simple adaptive
heuristic, while still letting the policy learn the allocation rule.

### Prior-biased initialization

The actor's initial logits are biased by the fixed decision-maker location
prior. This starts the policy near the obvious suspicious-zone preference
instead of requiring sparse policy gradients to discover that `{0, 4, 8, 12}`
matter.

This is not changing the downstream policy. It uses information already present
in the fixed DM prior and makes PPO start from a sensible exploration
distribution.

### Learning rate

The example default actor learning rate was `0.01`, which is too aggressive for
this PPO setup. Probes used `2.5e-4`, which kept KL and clipping modest. The
default actor LR is now `2.5e-4`, matching the critic LR.

## Probe results

These short CPU probes used 200 PPO iterations and stochastic evaluation.

| Variant | Risk | Suspicious-zone mass at iter 200 | Notes |
|---|---:|---:|---|
| Old generic network, sampled binary terminal reward | ~0.393 | ~0.47 | Barely learned the easy suspicious-zone preference. |
| Posterior expected reward + one-hot zone identity | ~0.363 | ~0.62 | Better, but still below the suspicious-cycle baseline. |
| Add prior-biased initialization | ~0.354 | ~0.86 | Starts near suspicious zones but still lacks good adaptive state summaries. |
| Add explicit per-zone summaries | **0.3120 +/- 0.0085** | ~0.90 | Close to the simple adaptive heuristic. |

The final CLI check was:

```sh
uv run examples/ppo_destructive_lot_inspection.py \
  --no-save \
  --no-cuda \
  --num-training-iterations 200 \
  --eval-episodes 3000 \
  --log-frequency 1000
```

It reported:

```text
deployed Bayes risk: 0.3120 +/- 0.0085
fixed-rule decision rates: accept=0.3507, false_accept=0.0283,
false_reject=0.2837, accept_given_good=0.5319, accept_given_bad=0.0719
```

## Original always-reject failure

The initially staged defaults used `T=24`, `m=4`, adjacent suspicious zones
`{0,1,2,3}`, and suspicion weight 0.5. Under that configuration, random
allocation accepted only about 2.3% of good lots. PPO therefore saw an almost
constant terminal reward.

Evaluation also used deterministic categorical argmax. An approximately
uniform categorical policy then repeatedly selected one zone and caused the DM
to reject every lot. The reported risk was approximately 0.609, essentially the
always-reject risk.

The current calibration and stochastic evaluation fixed that first pathology.
The representation and reward changes above fixed the remaining weak-learning
pathology.

## Debugging intuition

The useful mental model is that PPO was being asked to solve three problems at
once:

1. discover that four non-contiguous suspicious zones matter;
2. learn to balance repeated inspections across those zones;
3. adapt follow-up inspections based on observed defect counts.

With only a terminal binary reward and scalar zone IDs, the policy-gradient
signal mostly says "this whole 30-step trajectory was slightly better or
worse." It does not clearly say which zone should have been inspected more
often. The critic faces a similarly hard problem because the value is a
thresholded posterior decision from a partial history.

The successful changes remove avoidable difficulty without changing the actual
deployment objective:

- expected terminal loss removes outcome noise;
- explicit per-zone summaries expose the state variables the decision actually
  depends on;
- prior initialization gives the policy the easy fixed preference at step 0;
- lower LR keeps PPO updates inside the regime where clipping is meaningful.

## Verification

After the changes:

```sh
uv run pytest
uv run ruff check examples/ppo_destructive_lot_inspection.py \
  tests/test_ppo_destructive_lot_inspection.py \
  tests/test_recipes.py
git diff --check
```

Results:

- `229 passed`;
- Ruff passed;
- `git diff --check` passed.

## Reproduction command

For a quick check:

```sh
uv run examples/ppo_destructive_lot_inspection.py \
  --no-save \
  --no-cuda \
  --num-training-iterations 200 \
  --eval-episodes 3000 \
  --log-frequency 1000
```

For the default longer run:

```sh
uv run examples/ppo_destructive_lot_inspection.py \
  --no-save \
  --no-cuda \
  --eval-episodes 5000
```

Change `--seed` for additional replications. On a CUDA system, omit `--no-cuda`.
