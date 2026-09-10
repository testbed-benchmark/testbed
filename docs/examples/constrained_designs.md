---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Constrained Designs

```{code-cell} ipython3
:tags: [remove-output]

import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal

from testbed.design_objectives import PCE
from testbed.designers import SequentialDesigner
from testbed.tasks import HyperbolicTemporalDiscounting

torch.manual_seed(2)

task = HyperbolicTemporalDiscounting()
```

Many tasks have constraints that the designs must respect. For example, in {py:class}`~testbed.tasks.hyperbolic_temporal_discounting.HyperbolicTemporalDiscounting`, designs $x = (D_1, R_0)$ are constrained such that $D_1 \ge 0$ and $0 \le R_0 \le 100$. 

Generally, a {py:class}`~testbed.tasks.base.Task` expects valid designs to be passed into it, and will not automatically constrain or transform the designs. If we try to pass an invalid design into our `Task` we'll get an error.

```{code-cell} ipython3
:tags: [raises-exception]

theta = task.sample_prior()
design = torch.tensor([-1., -1.])

task.sample_outcome(theta, design)
```

We see this error relies on {py:class}`~testbed.tasks.hyperbolic_temporal_discounting.HTDDesignSpace`, which is a subclass of {py:class}`~testbed.core.design_spaces.DesignSpace`. A `DesignSpace` provides methods for checking if designs are valid, as well as for transforming raw (or latent) designs to be valid designs. This is provided by the `to_design()` method, and its inverse is given by the `to_raw()` method. For the `HyperbolicTemporalDiscounting` task, raw designs are reparametrized under the maps $D_1 \mapsto \exp(D_1 - \log \mathbb{E}[k])$ and $R_0 \mapsto 100 * \mathrm{sigmoid}(R_0)$.

```{code-cell} ipython3
design_space = task.design_space # Each task has a default design_space attribute

theta = task.sample_prior()

raw_design = torch.tensor([-1., -1.])
design = design_space.to_design(raw_design)
inverse_raw_design = design_space.to_raw(design)

y = task.sample_outcome(theta, design)

print(f'design={design}, y={y}')
print(f'raw_design==inverse_raw_design? {torch.allclose(raw_design, inverse_raw_design)}')
```

## Constraints and Designers

Let's make a toy `SequentialDesigner` for this task and create a rollout. If we look at the resulting designs, we see that they are indeed valid. This works because, by default, any designer has `transform_designs=True`. In this case, the `SequentialDesigner` will automatically call `to_design()` to do the transformation during the design optimisation process. Whenever a designer computes a raw design, the rollout also carries it as `rollout.extra["raw_design_history"]`.

```{code-cell} ipython3
T = 5

task = HyperbolicTemporalDiscounting()
pce = PCE(task, n_outer=32, n_inner=32)
designer = SequentialDesigner(task, T, objective=pce, lr=5e-3, n_step_opt=5)

theta = task.sample_prior()
rollout = designer.rollout(theta)

print(f'designs: {rollout.design_history}')
print(f'raw designs: {rollout.extra["raw_design_history"]}')
```

There may be cases in which you want to skip this automatic transformation coming from the design space. You can do this by setting `transform_designs=False` in the construction of your designer.  For example, you might be using a {py:class}`~testbed.designers.amortized.PolicyDesigner` with a neural network design policy that is already constrained to output valid designs. 

```{warning}
`PolicyDesigner` always receives a `DesignProposal` from its `DesignNet`. Normally,
the proposal's `raw_design` is transformed with `to_design()` and, for stochastic
policies, the corresponding change-of-variables correction is applied to its log
probability. For numerical stability, a policy can condition on raw/latent designs
rather than physical designs by setting `use_raw_history=True`. That setting is an
advanced opt-in: it is incompatible with policies that normalize physical design
histories. With its default `use_raw_history=False`, a policy receives physical
designs after the first rollout step. When `transform_designs=False`, the
`use_raw_history` flag has no effect.

Some policies, such as `CategoricalPolicyNet`, select directly from a finite set of
physical designs. Such a policy declares this in its network contract, so
`PolicyDesigner` does not apply a design-space transform or density correction.
```
