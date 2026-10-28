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

# Using Tasks

This guide gives a quick overview of how to use a {py:class}`~testbed.tasks.base.Task`. A task is a Bayesian model with a prior $p(\theta)$ and a likelihood $p(y \mid \theta, x, h_t)$, where $x$ is the design and $h_t = (x_{1:t}, y_{1:t})$ is the history of previous designs and outcomes. See the {py:class}`~testbed.tasks.base.Task` API for full details of each method.

As a running example, we use {py:class}`~testbed.tasks.location_finding.LocationFinding` with two sources in three dimensions. The parameter $\theta \in \mathbb{R}^{2 \times 3}$ holds the source locations, a design $x \in \mathbb{R}^3$ is a point at which we measure the total signal, and the outcome $y \in \mathbb{R}$ is a noisy measurement of that signal.

```{code-cell} ipython3
:tags: [remove-stderr]

import torch

from testbed.tasks import LocationFinding
from testbed.tasks.location_finding import LocationFindingPrior

torch.manual_seed(0)

task = LocationFinding(n_sources=2, dim_theta=3)
print(task.event_shape_theta, task.event_shape_design, task.event_shape_y)
```

The *event shapes* are the shapes of single values of $\theta$, $x$ and $y$. Every tensor passed to or returned by a task has shape `[*batch_shape, *event_shape]`.

## Sampling

{py:meth}`~testbed.tasks.base.Task.sample_outcome` draws $y \sim p(y \mid \theta, x, h_t)$:

```{code-cell} ipython3
theta = task.sample_prior()             # [2, 3]
design = torch.zeros(3)                 # [3]
y = task.sample_outcome(theta, design)  # [1]
print(theta.shape, design.shape, y.shape)
```

Tasks also provide helper methods for the other distributions in the model:

| Method | Samples from | Returns |
|---|---|---|
| `sample_prior(n_samples)` | $p(\theta)$, the original prior | `theta` |
| `sample_current_prior(n_samples)` | the current prior (see Updating the Prior) | `theta` |
| `sample_outcome(theta, design, ...)` | $p(y \mid \theta, x, h_t)$ | `y` |
| `sample_joint(design, ...)` | $p(y, \theta \mid x, h_t)$, using the current prior | `(y, theta)` |
| `sample_marginal(design, ...)` | $p(y \mid x, h_t)$, using the current prior | `y` |

The `n_samples` argument adds a leading sample dimension when it is greater than 1.

```{code-cell} ipython3
y, theta = task.sample_joint(design, n_samples=100)  # [100, 1], [100, 2, 3]
print(y.shape, theta.shape)
```

## Broadcasting

Batch dimensions broadcast in the same way as in `torch.distributions`. For example, we can sample outcomes for every pair of 100 parameters and 5 designs:

```{code-cell} ipython3
theta = task.sample_prior(n_samples=100)  # [100, 2, 3]
designs = torch.randn(5, 1, 3)            # [5, 1, 3]

y = task.sample_outcome(theta, designs)                     # [5, 100, 1]
y_many = task.sample_outcome(theta, designs, n_samples=10)  # [10, 5, 100, 1]
print(y.shape, y_many.shape)
```

## Likelihoods

{py:meth}`~testbed.tasks.base.Task.outcome_logp` evaluates the log-likelihood $\log p(y \mid \theta, x, h_t)$ of individual outcomes and broadcasts in the same way as sampling. The result has the broadcast batch shape:

```{code-cell} ipython3
logp = task.outcome_logp(y, theta, designs)  # [5, 100]
print(logp.shape)
```

{py:meth}`~testbed.tasks.base.Task.history_logp` evaluates the log-likelihood of a whole history of $T$ designs and outcomes,

$$
\log p(y_{1:T} \mid \theta, x_{1:T}) = \sum_{t=1}^T \log p(y_t \mid \theta, x_t, h_{t-1}).
$$

Histories have shape `[*batch_shape, T, *event_shape]`, and `theta` broadcasts against the batch shape:

```{code-cell} ipython3
T = 4
theta_true = task.sample_prior()
design_history = torch.randn(T, 3)                                                   # [4, 3]
y_history = torch.stack([task.sample_outcome(theta_true, x) for x in design_history])  # [4, 1]

logp = task.history_logp(y_history, design_history, theta)  # [100]
print(logp.shape)
```

`history_logp` is correct for both memoryless and non-memoryless tasks. It loops over $t$ and passes the history $h_{t-1}$ to the likelihood at each step, and memoryless tasks ignore it. For a memoryless task like this one, the result equals summing `outcome_logp` over the $T$ axis:

```{code-cell} ipython3
manual = task.outcome_logp(y_history, theta.unsqueeze(1), design_history).sum(-1)
print(torch.allclose(logp, manual, atol=1e-5))
```

Passing `fused=True` replaces the Python loop over $t$ with compiled kernels via `torch.compile`. This can speed up repeated calls, but the gain depends on the task and hardware, and the first call pays the compilation cost. `history_logp` can also condition on an earlier context without scoring it, through the `y_context` and `design_context` arguments; see the API for details.

## Updating the Prior

A task keeps two priors: `task.prior`, the original prior, and `task.current_prior`, which starts out equal to it. {py:meth}`~testbed.tasks.base.Task.update_prior` replaces the current prior, for example with a posterior after observing some data. This is how myopic and sequential designers work: {py:class}`~testbed.designers.sequential.SequentialDesigner` updates the current prior after each step, and objectives such as PCE sample from it.

`sample_current_prior`, `sample_joint` and `sample_marginal` use the current prior, while `sample_prior` always uses the original one. Here we replace the current prior with a narrower Gaussian centered on `theta_true`:

```{code-cell} ipython3
posterior = LocationFindingPrior(
    n_sources=2, dim_theta=3, theta_loc=theta_true, theta_covmat=0.1 * torch.eye(3)
)
task.update_prior(posterior)

print(task.sample_prior(1000).std(0).mean())          # ~1.0
print(task.sample_current_prior(1000).std(0).mean())  # ~0.32
```

The update modifies the task in place, so call `task.update_prior(task.prior)` to reset it before reusing the task.

```{code-cell} ipython3
task.update_prior(task.prior)
```
