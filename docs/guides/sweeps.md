---
orphan: true
---

# Seed Sweeps

Reporting mean ± std means running a recipe at several seeds, tracking which seeds
finished, and re-running the ones that crashed. TestBED delegates that bookkeeping to
[W&B sweeps](https://docs.wandb.ai/guides/sweeps): a sweep is a queue held on W&B's
servers, and each agent that connects is handed the next seed nobody has claimed.

This builds on [](wandb.md) — a sweep run is an ordinary W&B run, so everything on that
page still applies.

## The pieces

| Path | Role |
| --- | --- |
| `testbed/recipes/experiments/<task>/<method>.yaml` | The recipe. Declares `wandb.project` (the task) and `wandb.group` (the method). |
| `sweeps/<task>_<method>.yaml` | The queue: a grid over the seeds still to run. Generated, and committed. |
| `scripts/sweep_run.py` | Wrapper the agent invokes once per seed; turns a queue item into Hydra overrides. |
| `scripts/generate_seed_sweeps.py` | Rebuilds the queue from what W&B says has finished. |

Sweep YAMLs deliberately live outside `testbed/recipes/`: they describe *work queues*,
not configuration.

## Running a sweep

Install the extra and log in once per machine:

```sh
pip install -e ".[wandb]"
wandb login
export WANDB_ENTITY=your-team   # omit to use your default entity
```

Generate the queue, create the sweep, then start agents:

```sh
# 1. Write sweeps/location_finding_2src_actionbed.yaml with the missing seeds
python scripts/generate_seed_sweeps.py \
    --recipe experiments/location_finding_2src/actionbed

# 2. Register the queue with W&B; prints a sweep id
wandb sweep sweeps/location_finding_2src_actionbed.yaml

# 3. Drain it — on as many machines as you like, all at once
wandb agent your-team/location_finding_2src/<sweep_id>
```

Every agent pointed at the same sweep id pulls from the same queue, and W&B hands each
seed out exactly once. That is the whole mechanism: to run "the next seed that has not
been run", start another agent. Run agents from the repo root, since the sweep YAML
names `scripts/sweep_run.py` relative to the working directory.

Several recipes sweep independently, one sweep id each, so you can give a heavier method
its own machines:

```sh
python scripts/generate_seed_sweeps.py \
    --recipe experiments/location_finding_2src/{actionbed,dad,rlboed}
```

## Evaluation after training

Each sweep run evaluates its own checkpoint once training finishes, in the same job.
`scripts/sweep_run.py` fixes the W&B run id before training starts, so
`scripts/eval_run.py` can reopen that run and add `eval/lower_bound`,
`eval/upper_bound` and their standard errors as columns beside the training metrics.
The same numbers are written to `<run_dir>/eval.json`.

Evaluation is best-effort: if it fails, the failure is reported but the run is not
marked failed, since the checkpoint is already on disk. Re-run it by hand against any
run directory:

```sh
python scripts/eval_run.py --run-dir runs/location_finding_2src_actionbed_seed3_...
```

The run id is read back from `<run_dir>/wandb_run_id`, so a later manual eval still
lands on the original W&B run. Pass `--no-wandb` to only write the JSON.

Evaluation is configured by the `final_eval` group, which every method recipe composes
alongside `wandb`:

```yaml
final_eval:
  enabled: true
  checkpoint: final_step.pt
  n_outer: 2_048
  n_contrastive: 1_000_000
  outer_batch_size: 256
  contrastive_chunk_size: 200_000
  use_raw_weights: false
```

 Add `final_eval.n_outer=32` to a sweep's `command` block when the defaults do not fit the
time available, and `final_eval.enabled=false` to skip evaluation for that recipe.
These are the values a run is evaluated with in its own job: they are read back from the
run's saved config, which is also how `python -m testbed.eval.eval_policy <run_dir>`
sizes itself. `eval_run.py` takes `--n-outer` and `--n-contrastive` on top of them, for
re-scoring a finished checkpoint at a size the recipe did not ask for, and
`--checkpoint` for scoring a different file in the run directory, such as one of the
periodic `step_*.pt` policies a run writes when `checkpoint_every` is set. That one
namespaces its own output, so a mid-training score cannot overwrite the run's `eval/*`
columns.

## Re-evaluating a finished sweep

An sPCE lower bound cannot exceed `log(L+1)`, so a strong policy evaluated at too small
an `n_contrastive` reports the ceiling rather than its gain. Re-scoring at a larger L
tells the two apart. `scripts/sweep_reeval.py` takes a sweep id, resolves each run to
its run directory, and calls `eval_run.py` on it:

```sh
python scripts/sweep_reeval.py <sweep_id> --project location_finding_2src --list
python scripts/sweep_reeval.py <sweep_id> --project location_finding_2src \
    --n-contrastive 100000000 --n-outer 512 --prefix eval_L1e8
```

`--prefix` namespaces both the metric keys and the JSON file, so the new bounds land
beside the original `eval/*` columns on the same W&B run instead of overwriting them.
`scripts/reeval.sbatch` runs one array task per run, as `agent.sbatch` does for training:

```sh
sbatch --array=0-4 scripts/reeval.sbatch <entity>/<project>/<sweep_id> \
    --n-contrastive 100000000 --n-outer 512 --prefix eval_L1e8
```

Cost scales as `n_outer x n_contrastive x T`, while the ceiling only grows as `log(L+1)`,
so buy L with a smaller `n_outer`: the lower bound's standard error is well under 0.05 at
`n_outer=512`, far finer than the nats a larger L adds.

## Picking up after crashes

`generate_seed_sweeps.py` asks W&B which seeds have `state == "finished"` in the
recipe's project and group, and writes a grid containing only the rest. A crashed,
killed or preempted run is not `finished`, so it comes back in the next generation:

```sh
python scripts/generate_seed_sweeps.py \
    --recipe experiments/location_finding_2src/actionbed
wandb sweep sweeps/location_finding_2src_actionbed.yaml   # new id, only the stragglers
```

When every seed is done the YAML is deleted rather than left behind as an empty queue.

## How a swept parameter reaches the training run

The agent appends the sampled parameters to the `command` block through the
`${args_no_hyphens}` macro, which writes them as `seed=3 lr=1e-3`, i.e. Hydra override
syntax. `scripts/sweep_run.py` forwards them untouched and runs:

```sh
python testbed/methods/<method>.py --config-name experiments/<task>/<method> \
    wandb.enabled=true 'wandb.tags=[seed<n>]' \
    'hydra.run.dir=<save_dir>/<task>_<method>_seed<n>_<timestamp>_<jobid>_<taskid>' \
    seed=<n> [other swept parameters]
```

The seed is identifiable three ways in W&B: as a `seed` column in the runs table
(the whole config is snapshotted, and that is what the generator queries), in the run
name, and as a `seed<n>` tag. The run directory carries the same name as the W&B run,
so local output and W&B line up one-to-one.



## Sweeping something other than seeds

Nothing in the wrapper is seed-specific: every swept parameter arrives as a Hydra
override, so any config key can be swept. `seed` must be among the parameters, since it
names the run. To sweep learning rate across seeds, write the YAML by hand:

```yaml
name: location_finding_2src-actionbed-lr
program: scripts/sweep_run.py
method: grid
entity: rainml-testbed
project: location_finding_2src

parameters:
  seed:
    values: [0, 1, 2]
  lr:
    values: [1e-4, 5e-4, 1e-3]

command:
  - ${interpreter}
  - ${program}
  - --recipe
  - experiments/location_finding_2src/actionbed
  - ${args_no_hyphens}
```

Then `wandb sweep` and `wandb agent` as usual, for all nine runs.

`scripts/generate_seed_sweeps.py` will not build this queue for you: it diffs on
`run.config["seed"]` alone, so it cannot work out which learning rates are still
missing. A hyperparameter sweep is written by hand and re-run in full if it is
interrupted.

For a quick one-off grid on a single machine, Hydra's own multirun is simpler and needs
no sweep at all:

```sh
python testbed/methods/actionbed.py --multirun \
    --config-name experiments/location_finding_2src/actionbed \
    lr=1e-4,5e-4,1e-3
```

Reach for a W&B sweep instead when several machines share the work, or when runs need to
survive crashes.

## Adding a new recipe to the system

Give the recipe a `wandb:` block and select its method as a defaults group:

```yaml
defaults:
  - /method: actionbed    # group syntax, not `- /method/actionbed`
  - override /task: <task>
  - _self_

wandb:
  project: <task>    # one project per task; methods separate by group
  group: <method>
```

The group syntax is what records the choice in `hydra.runtime.choices`, which is how
`scripts/sweep_run.py` finds the entry point to run. Writing `- /method/actionbed`
instead includes the same file but records no choice, and the wrapper will refuse the
recipe. Taking it from the config means the filename need not match the method:
`var_post_scorebed_repro.yaml` correctly runs `testbed/methods/var_post.py`.

Leave `entity` alone: it stays `null` in `testbed/recipes/wandb/default.yaml` so that
anyone using TestBED logs to their own account, and `WANDB_ENTITY` supplies yours.