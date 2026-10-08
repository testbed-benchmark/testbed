---
orphan: true
---

# Weights & Biases Logging

Every run managed by {py:class}`~testbed.core.manager.Manager` writes metrics to a
`metrics.jsonl` file in the run directory by default. You can *additionally* mirror
those metrics to [Weights & Biases](https://wandb.ai) (W&B) — the JSONL log stays in
place, and W&B is layered on top via a
{py:class}`~testbed.core.logging.MultiLogger`.

W&B logging is **opt-in and off by default**, so nothing changes until you enable it.

## Install

`wandb` is an optional dependency. Install the extra:

```sh
pip install -e ".[wandb]"
# or install everything with:  pip install -e ".[full]"
```

## Authenticate

Log in once on the machine that will run training:

```sh
wandb login
```

Alternatively, set the `WANDB_API_KEY` environment variable (handy on clusters / in CI).

## Enable it via the recipe config

Every recipe composes the shared `testbed/recipes/wandb/default.yaml` group, which supplies
the `wandb:` block. Flip `enabled` to `true` there (to affect all recipes), or override it per
run on the CLI:

```yaml
wandb:
  enabled: true             # opt-in; off by default
  project: testbed          # W&B project to log to
  entity: null              # your W&B team/user; null uses the default
  mode: online              # online | offline | disabled
  tags: []                  # optional list of run tags
  group: null               # optional W&B run group
```

The block is optional — if a config has no `wandb:` section at all, W&B logging simply
stays off.

## Enable / override on the CLI

The Hydra entry points (e.g. `testbed/cli/dad.py`) merge dotted CLI overrides into the
composed config, so you can flip W&B on without editing the YAML:

```sh
python -m testbed.cli.dad \
    save_dir=./runs \
    wandb.enabled=true \
    wandb.project=my-project \
    wandb.tags=[baseline,seed0]
```

## What gets logged

- **Run config** — the full experiment config is snapshotted as the W&B run config
  (via {py:func}`~testbed.core.logging.config_to_dict`, which handles OmegaConf,
  dataclass, and plain-dict configs).
- **Metrics** — each metric dict passed to the logger is forwarded to `wandb.log`. A
  `step` key, if present, is popped and used as the W&B x-axis; torch tensors are
  coerced to Python scalars automatically.
- **Run name** — the `Manager` run name (timestamp + optional name) is used as the W&B
  run display name, so the W&B run lines up with the local run directory.

The `metrics.jsonl` file is written in parallel the whole time, so you never lose the
local record by enabling W&B.

## Using it programmatically

If you write your own training loop instead of using the recipe entry points, you can
build the loggers directly:

```python
from testbed.core.logging import JSONLLogger, MultiLogger, WandbLogger

wandb_logger = WandbLogger(
    cfg,
    run_name="my-run",
    project="testbed",
    mode="online",
)
logger = MultiLogger([JSONLLogger("metrics.jsonl"), wandb_logger])

# ... in your loop:
logger.log({"step": step, "loss": loss})
# ... at the end:
logger.close()
```

You can also hand a ready-made logger to the `Manager` via its `logger=` argument, in
which case the `Manager` uses it as-is and skips the `wandb:`-block wiring.

## Running a recipe across seeds

W&B also coordinates multi-seed runs: see [](sweeps.md) for the sweep setup that hands
each unrun seed to the next available agent and rebuilds the queue after crashes.
