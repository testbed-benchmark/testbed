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


Size the evaluation with `--eval-n-outer` and `--eval-n-contrastive` on
`scripts/sweep_run.py` when the defaults do not fit the time available.

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

## How a seed reaches the training run

The agent appends the sampled seed to the `command` block in the sweep YAML through the
`${args}` macro, which expands to `--seed=<value>`. `scripts/sweep_run.py` then runs:

```sh
python testbed/methods/<method>.py --config-name experiments/<task>/<method> \
    seed=<n> wandb.enabled=true 'wandb.tags=[seed<n>]' \
    'hydra.run.dir=${save_dir}/<task>_<method>_seed<n>_${now:%Y-%m-%d_%H-%M-%S}'
```

so the seed is identifiable three ways in W&B: as a `seed` column in the runs table
(the whole config is snapshotted, and that is what the generator queries), in the run
name, and as a `seed<n>` tag. The run directory carries the same name as the W&B run,
so local output and W&B line up one-to-one.

```{warning}
Use `${args}` in the `command` block, never `${seed}`. The latter is not a W&B macro —
it is silently dropped, and every run in the sweep then trains with the recipe's default
seed.
```

## Sweeping something other than seeds

Nothing here is seed-specific except the generated grid. To sweep a hyperparameter,
write a sweep YAML by hand with extra `parameters` and pass them through as Hydra
overrides, or use Hydra's own multirun for a quick local grid:

```sh
python testbed/methods/actionbed.py --multirun \
    --config-name experiments/location_finding_2src/actionbed \
    lr=1e-4,5e-4,1e-3
```

Hydra multirun is simpler for a one-off on a single machine; a W&B sweep is what you
want when several machines share the work or when runs need to survive crashes.

## Adding a new recipe to the system

Give the recipe a `wandb:` block and it works with no further changes:

```yaml
wandb:
  project: <task>    # one project per task; methods separate by group
  group: <method>
```

Leave `entity` alone — it stays `null` in `testbed/recipes/wandb/default.yaml` so that
anyone using TestBED logs to their own account, and `WANDB_ENTITY` supplies yours.
`scripts/sweep_run.py` derives the entry point from the recipe's basename; pass
`--method` when the file is named something other than its method.