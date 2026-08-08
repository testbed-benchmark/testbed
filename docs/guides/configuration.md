---
orphan: true
---

# Configuration recipes

Training entry points use [Hydra](https://hydra.cc/) recipes from
`testbed/recipes/`. A root method recipe supplies the defaults; an experiment
recipe composes it and specifies only what changes. For example, the files
specifying the configuration for solving the two-source location finding task
with ActionBED are the following:

```text
recipes/
  actionbed.yaml                         # ActionBED defaults
  task/location_finding_2src.yaml        # task variant
  experiments/location_finding_2src/
    actionbed.yaml                       # task–method overrides
```

The file `experiments/location_finding_2src/actionbed.yaml` starts from
`/actionbed`, changes the task selection, and then applies its local settings.
`_self_` is last, so those local settings override the base recipe.

```yaml
# @package _global_
defaults:
  - /actionbed
  - override /task: location_finding_2src
  - override /net@action_net: null
  - _self_

seed: 42
T: 30
```

Run an experiment recipe by selecting it with Hydra:

```sh
python testbed/methods/actionbed.py \
  --config-name experiments/location_finding_2src/actionbed
```

For a one-off change, use a dotted CLI override instead of editing a recipe:

```sh
python testbed/methods/actionbed.py seed=42 lr=5e-4 wandb.enabled=true
```
