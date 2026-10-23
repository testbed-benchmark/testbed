---
orphan: true
---

# Configuration recipes

Training entry points (`python testbed/methods/<method>.py`) use
[Hydra](https://hydra.cc/) recipes from `testbed/recipes/`. Hydra is an optional
dependency; install it with `pip install -e ".[train]"`. Configuration is split into three layers:

- `task/` recipes own the task constructor settings, the default horizon `T`,
  and task-specific ranges used by methods.
- `method/` recipes such as `method/actionbed.yaml` own only method and training
  defaults. They select a task recipe so that the entry point remains directly
  runnable, but do not redefine its settings.
- `experiments/<task>/<method>.yaml` recipes contain only overrides needed by
  a particular method-task combination.

For example, the files specifying the configuration for solving the two-source
location finding task with ActionBED are the following:

```text
recipes/
  method/actionbed.yaml                  # ActionBED defaults
  task/location_finding_2src.yaml        # task settings and default horizon
  experiments/location_finding_2src/
    actionbed.yaml                       # task–method overrides
```

The file `experiments/location_finding_2src/actionbed.yaml` starts from
`/method/actionbed`, changes the task selection, and then applies only its local
method-task settings. `_self_` is last, so those settings override the method
and task defaults.

```yaml
# @package _global_
defaults:
  - /method/actionbed
  - override /task: location_finding_2src
  - override /net@action_net: null
  - _self_

seed: 42
```

Run an experiment recipe by selecting it with Hydra:

```sh
python testbed/methods/actionbed.py \
  --config-name experiments/location_finding_2src/actionbed
```

For a one-off change, use CLI overrides instead of editing a recipe:

```sh
python testbed/methods/actionbed.py seed=42 lr=5e-4 wandb.enabled=true
```
