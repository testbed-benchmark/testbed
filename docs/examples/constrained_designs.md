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
from testbed.tasks import TemporalDiscounting

torch.manual_seed(2)

task = TemporalDiscounting()
```

Many tasks have constraints that the designs must respect. For example, in {py:class}`~testbed.tasks.temporal_discounting.TemporalDiscounting`, designs $x = (D_1, R_0)$ are constrained such that $D_1 \ge 0$ and $0 \le R_0 \le 100$. 

Generally, a {py:class}`~testbed.tasks.base.Task` expects valid designs to be passed into it, and will not automatically constrain or transform the designs. If we try to pass an invalid design into our `Task` we'll get an error.

```{code-cell} ipython3
:tags: [raises-exception]

theta = task.sample_prior()
design = torch.tensor([-1., -1.])

task.sample_outcome(theta, design)
```

Valid designs are defined by the support of the task's design prior, `task.design_prior.support`, which is a [torch constraint](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints). 

TestBED provides tools for parametrising constrained designs, which work by transforming a raw or latent design into the design space. Each task has a  `design_transform` function, which is a torch `Transform` that maps raw designs to valid designs. Its inverse is `design_transform.inv`. These transforms are aliases for the same functions defined by the design prior. Design priors can provide a `transform` function, corresponding to a design parametrisation. Otherwise, the transformation defaults to torch's `biject_to(task.design_prior.support)`, except that a box uses a tanh squash (see {py:func}`~testbed.core.constraints.default_design_transform`). 

For example, the `TemporalDiscounting` task's design prior defines {py:class}`~testbed.tasks.temporal_discounting.TDDesignTransform`, which reparametrizes raw designs under the maps $D_1 \mapsto \exp(D_1 - \log \mathbb{E}[k])$ and $R_0 \mapsto 100 * \mathrm{sigmoid}(R_0)$.

```{code-cell} ipython3
transform = task.design_transform  # each task has a design transform

theta = task.sample_prior()

raw_design = torch.tensor([-1., -1.])
design = transform(raw_design)
inverse_raw_design = transform.inv(design)

y = task.sample_outcome(theta, design)

print(f'design={design}, y={y}')
print(f'raw_design==inverse_raw_design? {torch.allclose(raw_design, inverse_raw_design)}')
```

### Changing the parametrisation

To parametrise the same design space differently, write a design prior with its own `transform`. Here is a uniform prior on a box, as with {py:class}`~testbed.core.distributions.Box`, whose raw designs map through a sigmoid rather than the default tanh:

```{code-cell} ipython3
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    IndependentTransform,
    SigmoidTransform,
)

from testbed.core import Box


class SigmoidBox(Box):
    @property
    def transform(self):
        squash = ComposeTransform([SigmoidTransform(), AffineTransform(self.low, self.high - self.low)])
        return IndependentTransform(squash, 1)


low, high = [0.5, 1.0], [200.0, 99.0]
tanh_task = TemporalDiscounting(design_prior=Box(low, high, event_shape=[2]))
sigmoid_task = TemporalDiscounting(design_prior=SigmoidBox(low, high, event_shape=[2]))

raw_design = torch.tensor([-1., -1.])
print(f'tanh: {tanh_task.design_transform(raw_design)}')
print(f'sigmoid: {sigmoid_task.design_transform(raw_design)}')
```

For stochastic policies the transform also needs `log_abs_det_jacobian`, which torch's transforms provide.

## Constraints and Designers

Let's make a toy `SequentialDesigner` for this task and create a rollout. If we look at the resulting designs, we see that they are indeed valid. This works because, by default, any designer has `transform_designs=True`. In this case, the `SequentialDesigner` will automatically apply `task.design_transform` to do the transformation during the design optimisation process. Whenever a designer computes a raw design, the rollout also carries it as `rollout.extra["raw_design_history"]`.

```{code-cell} ipython3
T = 5

task = TemporalDiscounting()
pce = PCE(task, n_outer=32, n_inner=32)
designer = SequentialDesigner(task, T, objective=pce, lr=5e-3, n_step_opt=5)

theta = task.sample_prior()
rollout = designer.rollout(theta)

print(f'designs: {rollout.design_history}')
print(f'raw designs: {rollout.extra["raw_design_history"]}')
```

There may be cases in which you want to skip this automatic transformation. You can do this by setting `transform_designs=False` in the construction of your designer.  For example, you might be using a {py:class}`~testbed.designers.amortized.PolicyDesigner` with a neural network design policy that is already constrained to output valid designs. 

```{warning}
`PolicyDesigner` always receives a `DesignProposal` from its `DesignNet`. Normally,
the proposal's `raw_design` is transformed with `task.design_transform` and, for stochastic
policies, the corresponding change-of-variables correction is applied to its log
probability. For numerical stability, a policy can condition on raw/latent designs
rather than physical designs by setting `use_raw_history=True`. That setting is an
advanced opt-in: it is incompatible with policies that normalize physical design
histories. With its default `use_raw_history=False`, a policy receives physical
designs after the first rollout step. When `transform_designs=False`, the
`use_raw_history` flag has no effect.

Some policies, such as `CategoricalPolicyNet`, select directly from a finite set of
physical designs. Such a policy declares this in its network contract, so
`PolicyDesigner` does not apply the design transform or density correction.
```
