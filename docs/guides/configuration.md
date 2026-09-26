---
orphan: true
---

# Configuration recipes

Training entry points use [Hydra](https://hydra.cc/) recipes from
`testbed/recipes/`. Configuration is split into three layers:

- `task/` recipes own the task constructor settings, the default horizon `T`,
  and task-specific ranges used by methods.
- `method/` recipes such as `method/actionbed.yaml` own only method and training
  defaults. They select a task recipe so that the entry point remains directly
  runnable, but do not redefine its settings.
- `experiments/` recipes contain only the overrides needed by a particular
  method-task combination. The paper's benchmark lives in
  `experiments/benchmarking/<task>/<method>.yaml`, and the architecture
  ablations in `experiments/architecture/<task>/`, each of which starts from
  the corresponding benchmark recipe.

For example, the files specifying the configuration for solving the two-source
location finding task with DAD are the following:

```text
recipes/
  method/dad.yaml                        # DAD defaults
  task/location_finding_2src.yaml        # task settings and default horizon
  experiments/benchmarking/location_finding_2src/
    dad.yaml                             # task–method overrides
```

The file `experiments/benchmarking/location_finding_2src/dad.yaml` starts from
`/method: dad`, changes the task selection, and then applies only its local
method-task settings. `_self_` is last, so those settings override the method
and task defaults.

```yaml
# @package _global_
defaults:
  - /method: dad
  - override /task: location_finding_2src
  - _self_

train_time_budget_seconds: 43_200  # 12 hours
n_outer: 2_048
```

Run an experiment recipe by selecting it with Hydra:

```sh
python testbed/methods/dad.py \
  --config-name experiments/benchmarking/location_finding_2src/dad
```

For a one-off change, use CLI overrides instead of editing a recipe:

```sh
python testbed/methods/actionbed.py seed=42 lr=5e-4 wandb.enabled=true
```
