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

# Implementing New Tasks

This guide covers how to implement a new {py:class}`~testbed.tasks.base.Task` in TestBED. A task is a Bayesian model, consisting of a prior $p(\theta)$ and a likelihood $p(y \mid \theta, x, h_t)$, where $x$ is the design and $h_t = (x_{1:t}, y_{1:t})$ is the history of previous designs and outcomes. Every task subclasses {py:class}`~testbed.tasks.base.Task` and provides:

- metadata passed to `Task.__init__`, i.e. the event shapes, the prior, and flags describing the model (e.g. whether $y$ is discrete, or whether the likelihood depends on $h_t$),
- the property `reparameterizable_outcomes`, stating whether outcomes can be sampled with a pathwise gradient with respect to the design,
- the method `_sample_outcome`, which draws $y \sim p(y \mid \theta, x, h_t)$, and
- the method `_outcome_logp`, which evaluates $\log p(y \mid \theta, x, h_t)$.

The base class handles everything else: broadcasting over batch dimensions, drawing multiple samples, sampling from the joint and marginal, scoring whole histories, and checking that designs are valid.

## Example: Logistic Regression

We'll implement a logistic regression model with two binary outcomes per experiment. The parameter is a matrix $\theta \in \mathbb{R}^{2 \times 3}$ with rows $\theta_1, \theta_2$, the design is $x \in \mathbb{R}^3$, and the outcome is $y \in \{0, 1\}^2$. The model is

$$
\theta_{k,j} \overset{\text{iid}}{\sim} \mathcal{N}(0, 1), \qquad
y_k \mid \theta, x \overset{\text{ind}}{\sim} \mathrm{Bernoulli}\big(\sigma(\theta_k^\top x)\big), \qquad k = 1, 2,
$$

where $\sigma$ is the logistic sigmoid. The likelihood doesn't depend on $h_t$, so this task is *memoryless*.

### Event shapes

TestBED needs the *event shape* of each variable, i.e. its shape without any batch dimensions. Here these are `event_shape_theta = [2, 3]`, `event_shape_design = [3]` and `event_shape_y = [2]`. Every tensor passed to or returned by a task has shape `[*batch_shape, *event_shape]`. A scalar outcome has event shape `[1]`, not `[]`.

### Prior

The prior can be any {py:class}`~testbed.core.distributions.Distribution` whose `event_shape` equals `event_shape_theta`. Shapes follow `torch.distributions`: `sample(sample_shape)` returns `[*sample_shape, *event_shape_theta]`, and `log_prob(theta)` maps `[*batch_shape, *event_shape_theta]` to `[*batch_shape]`, i.e. it sums over every entry of $\theta$ and returns one value per batch element.

You can wrap a `torch.distributions` object in {py:class}`~testbed.core.distributions.TorchDistribution`. The wrapper folds the wrapped distribution's batch dimensions into the event (via `Independent`), so `Normal(torch.zeros(2, 3), torch.ones(2, 3))`, whose own `log_prob` returns one value per entry, becomes a prior over $2 \times 3$ matrices whose `log_prob` sums over all six entries:

```python
prior = TorchDistribution(Normal(torch.zeros(2, 3), torch.ones(2, 3)))
```

If `torch.distributions` doesn't have the prior you need, subclass `Distribution` directly, as {py:class}`~testbed.tasks.location_finding.LocationFindingPrior` does: pass the event shape to `Distribution.__init__`, and make `sample` and `log_prob` work for any batch shape, including none. Register any tensors as buffers so that `.to(device)` moves them. `Task.__init__` draws one prior sample and raises an error if its shape or the shape of its `log_prob` is wrong.

### Likelihood

The base class calls `_sample_outcome` and `_outcome_logp` only after broadcasting their inputs and flattening them to a single batch dimension `b`. Hence, when implementing a custom task, all inputs to these methods can be assumed to have a shared batch dimension, and neither method needs to broadcast anything itself. In this example, `theta` has shape `[b, 2, 3]`, `design` has shape `[b, 3]`, and `y` has shape `[b, 2]`.  

`_sample_outcome` returns `[b, *event_shape_y]`, and `_outcome_logp` returns `[b]`. Both methods always receive `y_history` and `design_history`. Memoryless tasks simply ignore these arguments.

Outcomes here are Bernoulli, so we can't sample them with a pathwise gradient, and `reparameterizable_outcomes` is `False`. Pathwise objectives such as `SPCE` then miss the gradient through the outcomes, so gradient-based training should use a score-function objective such as `SPCEReinforce` (see [Implementing Custom Methods](custom_methods.md)). A task that can reparameterize (as {py:class}`~testbed.tasks.location_finding.LocationFinding` does with `loc + scale * randn_like(loc)`) should return `True`. The base class raises an error if a task declares `True` but its samples carry no gradient. `Task.__init__` reads this property, so any attributes it depends on must be set before calling `super().__init__`.

The `design_space` argument, which tracks any constraints on the designs, is passed through to the base class; it is covered in Custom Design Spaces below.

```{code-cell} ipython3
:tags: [remove-stderr]

import torch
from torch.distributions import Bernoulli, Normal

from testbed.core.distributions import TorchDistribution
from testbed.tasks import Task

torch.manual_seed(0)


class MultiLogistic(Task):
    @property
    def reparameterizable_outcomes(self) -> bool:
        return False

    def __init__(self, design_space=None):

        prior = TorchDistribution(Normal(torch.zeros(2, 3), torch.ones(2, 3)))

        super().__init__(
            event_shape_y=[2],
            event_shape_theta=[2, 3],
            event_shape_design=[3],
            prior=prior,
            discrete_y=True,
            design_space=design_space,
        )

    def _likelihood(self, theta, design):
        logits = (theta @ design.unsqueeze(-1)).squeeze(-1)  # [b, 2]
        return Bernoulli(logits=logits)

    def _outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        return self._likelihood(theta, design).log_prob(y).sum(-1)  # [b]

    def _sample_outcome(self, theta, design, y_history=None, design_history=None):
        return self._likelihood(theta, design).sample()  # [b, 2]
```

We set `discrete_y=True` here. The other flags keep their defaults: $\theta$ is continuous, the task is memoryless, and the likelihood is explicit. The table below lists all the flags; see the {py:class}`~testbed.tasks.base.Task` API for full details.

| Argument | Default | Meaning |
|---|---|---|
| `discrete_y` | `False` | Outcomes are discrete. |
| `discrete_theta` | `False` | Parameters are discrete. |
| `memoryless` | `True` | The likelihood doesn't depend on $h_t$. |
| `implicit_likelihood` | `False` | The likelihood can be sampled from but not evaluated. |
| `design_space` | `None` | A {py:class}`~testbed.core.design_spaces.DesignSpace` of valid designs. `None` means unconstrained. |

If the likelihood depends on $h_t$, set `memoryless=False` and read `y_history` and `design_history`, which have shapes `[b, t, *event_shape_y]` and `[b, t, *event_shape_design]`. At the first step $t = 0$. See {py:class}`~testbed.tasks.stochastic_pendulum.StochasticPendulum` for an example. If the likelihood is implicit, set `implicit_likelihood=True` and have `_outcome_logp` raise `NotImplementedError`.

### Using the task

Always call the public methods (`sample_outcome`, `outcome_logp`, `sample_joint`, ...) rather than the underscore methods we implemented. The public methods broadcast their inputs and validate designs.

```{code-cell} ipython3
task = MultiLogistic()

theta = task.sample_prior()             # [2, 3]
design = torch.randn(3)                 # [3]
y = task.sample_outcome(theta, design)  # [2]
print(theta.shape, design.shape, y.shape)
```

Batch dimensions broadcast in the same way as in `torch.distributions`:

```{code-cell} ipython3
theta = task.sample_prior(16)                # [16, 2, 3]
designs = torch.randn(4, 1, 3)               # [4, 1, 3]
y = task.sample_outcome(theta, designs)      # [4, 16, 2]
logp = task.outcome_logp(y, theta, designs)  # [4, 16]
print(y.shape, logp.shape)

y, theta = task.sample_joint(design, sample_shape=8)  # [8, 2], [8, 2, 3]
print(y.shape, theta.shape)
```

## Custom Design Spaces

The [Constrained Designs](../examples/constrained_designs.md) example shows how a task uses its {py:class}`~testbed.core.design_spaces.DesignSpace` to validate designs and to map unconstrained "raw" designs to valid ones. TestBED provides {py:class}`~testbed.core.design_spaces.DefaultDesignSpace` (unconstrained), {py:class}`~testbed.core.design_spaces.BoxDesignSpace` and {py:class}`~testbed.core.design_spaces.FiniteDesignSpace`. For other constraints or parametrizations, subclass `DesignSpace`.

Only `_contains` is required. Implement the other methods if the designers you plan to use need them:

| Method | Purpose | Needed by |
|---|---|---|
| `_contains(design)` | Checks the value constraints. `contains` checks the shape first. | Design validation in every task |
| `sample(batch_shape, device=...)` | Draws random valid designs | `RandomDesigner` and anything else that needs a design prior |
| `project(design)` | Maps an arbitrary tensor into the space, e.g. by clipping | Callers that project designs, e.g. `BEDEnv` with `project_designs` |
| `_to_design(raw)` / `_to_raw(design)` | Smooth map from unconstrained raw designs to valid designs, and its inverse | Designers with `transform_designs=True` (the default) |
| `_log_det_jacobian(raw)` | Log-determinant of the Jacobian of `_to_design` | Stochastic policies that pass a log-probability through `to_design` |

### Example

In our logistic model, larger designs $x$ push $\sigma(\theta_k^\top x)$ towards 0 or 1. Without a constraint, design optimization tends to send $\lVert x \rVert$ off to infinity, so we restrict designs to the unit ball $\{x : \lVert x \rVert \le 1\}$. Analogously to the tanh squash in `BoxDesignSpace`, we reparametrize radially with $x = \tanh(\lVert u \rVert)\, u / \lVert u \rVert$ for raw designs $u \in \mathbb{R}^3$:

```{code-cell} ipython3
from testbed.core import DesignSpace


class UnitBall(DesignSpace):
    def __init__(self, dim):
        super().__init__(event_shape=[dim], is_discrete=False)

    def sample(self, batch_shape=(), *, device=None):
        raw = torch.randn(*batch_shape, *self.event_shape, device=device)
        return self.to_design(raw)

    def _contains(self, design):
        return bool((design.norm(dim=-1) <= 1 + 1e-6).all())

    def project(self, design):
        return design / design.norm(dim=-1, keepdim=True).clamp_min(1.0)

    def _to_design(self, raw_design):
        r = raw_design.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return raw_design * torch.tanh(r) / r

    def _to_raw(self, design):
        s = design.norm(dim=-1, keepdim=True).clamp(1e-12, 1 - 1e-6)
        return design * torch.atanh(s) / s
```

`_contains` allows a small tolerance because `project` and `_to_design` can land a rounding error outside the boundary. We haven't implemented `_log_det_jacobian`, so this space can't be used with stochastic policies that need the density correction.

`MultiLogistic` already forwards `design_space` to `Task.__init__`, so using the space just means passing it in. The task now rejects designs outside the ball:

```{code-cell} ipython3
:tags: [raises-exception]

task = MultiLogistic(design_space=UnitBall(3))
theta = task.sample_prior()

task.sample_outcome(theta, torch.tensor([2.0, 0.0, 0.0]))
```

Designers work with unconstrained raw designs and map each one through `to_design`, so the designs they produce always lie in the ball. For example, a {py:class}`~testbed.designers.non_adaptive.StaticDesigner` initializes its raw designs from a standard normal:

```{code-cell} ipython3
from testbed.designers import StaticDesigner

designer = StaticDesigner(task, T=5)
with torch.no_grad():
    rollout = designer.rollout(theta.unsqueeze(0))
    print(rollout.extra["raw_design_history"].norm(dim=-1))
    print(rollout.design_history.norm(dim=-1))
```

## Contributing New Tasks

We welcome pull requests that add tasks of interest to the community. Please follow the [contributing guidelines](https://github.com/testbed-benchmark/testbed/blob/main/CONTRIBUTING.md). A new task should usually include:

- a module `testbed/tasks/<task>.py`, with the task exported from `testbed/tasks/__init__.py` and a reference to the source paper in its module docstring,
- tests in `tests/test_<task>.py` (see `tests/test_location_finding.py`), and
- a row in the task table in `docs/index.md`.
