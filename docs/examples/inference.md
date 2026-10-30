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

# Posterior Inference

Posterior inference over $\theta$ is typically used one of in two ways: first, as a final estimate on $\theta$ after collecting data, and second, as an intermediate representation for the design process, as in a {py:class}`~testbed.designers.sequential.SequentialDesigner`. In TestBED, an inference method is any function

```python
inference(task, y_history, design_history) -> Distribution
```

that conditions `task.prior` on a history. 

Providing a general-purpose toolkit for Bayesian inference is not one of the goals of TestBED. Instead, we encourage users to implement and use their own inference methods which are appropriate for the BED task at hand. That being said, TestBED provides a basic importance sampling method and allows you to interface with Pyro. To write your own inference method, see [Custom Inference](../guides/inference.md).

This example runs both importance sampling and MCMC on the same location-finding history, checks that they agree, and then uses Pyro's NUTS inside a {py:class}`~testbed.designers.sequential.SequentialDesigner`. 


```{code-cell} ipython3
:tags: [remove-stderr]

from functools import partial

import matplotlib.pyplot as plt
import torch

from testbed.design_objectives import PCE
from testbed.designers import RandomDesigner, SequentialDesigner
from testbed.inference import run_importance_sampling, run_mcmc
from testbed.tasks import LocationFinding

torch.manual_seed(1)

task = LocationFinding(n_sources=1, dim_theta=2)
theta = task.sample_prior(1)
```

## A history to condition on

We measure the signal at 10 random designs:

```{code-cell} ipython3
rollout = RandomDesigner(task, T=10).rollout(theta)
y_history, design_history = rollout.y_history[0], rollout.design_history[0]

print(y_history.shape, design_history.shape)
```

## Importance sampling and NUTS

Both functions take the task and a single history:

```{code-cell} ipython3
is_posterior = run_importance_sampling(task, y_history, design_history)
nuts_posterior = run_mcmc(task, y_history, design_history, n_samples=300, n_warmup=200)
```

Each returns an {py:class}`~testbed.core.distributions.EmpiricalDistribution`, a set of weighted points. Importance sampling draws 50,000 proposals from the prior and weights them by their likelihood. NUTS draws an MCMC chain. Both posteriors are centred near the true source, and they agree with each other:

```{code-cell} ipython3
def plot_posterior(ax, posterior, title, designs):
    samples = posterior.sample((2000,)).reshape(-1, 2)
    ax.scatter(*samples.T, s=2, alpha=0.2, label="posterior samples")
    ax.scatter(*designs.T, marker="x", c="k", label="designs")
    ax.scatter(*theta.reshape(2), marker="*", s=200, c="r", label="true source")
    ax.set(title=title, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal")


fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
plot_posterior(axes[0], is_posterior, "Importance sampling", design_history)
plot_posterior(axes[1], nuts_posterior, "NUTS", design_history)
axes[0].legend(loc="lower left")
plt.show()

for name, posterior in [("IS", is_posterior), ("NUTS", nuts_posterior)]:
    samples = posterior.sample((5000,)).reshape(-1, 2)
    print(f"{name:4s} mean {samples.mean(0)}, std {samples.std(0)}")
```

Importance sampling's weights degenerate as the posterior narrows relative to the prior. Its effective sample size, `is_posterior.ess`, is a quick check of how far to trust it:

```{code-cell} ipython3
print(is_posterior.ess)
```

## Inside a sequential designer

A `SequentialDesigner` runs its `inference` after every step to update `task.current_prior`, which its objective (here PCE) samples from. Use `functools.partial` to configure the method:

```{code-cell} ipython3
:tags: [remove-stderr]

designer = SequentialDesigner(
    task,
    T=4,
    objective=PCE(task, n_outer=64, n_inner=64),
    lr=0.05,
    n_step_opt=200,
    inference=partial(run_mcmc, n_samples=300, n_warmup=200),
    store_posterior_history=True,
)
rollout = designer.rollout(theta)
```

The posterior after each step is in `rollout.extra["posterior_history"]`, and the final one is in `rollout.extra["posterior"]`. Once the rollout ends, the task's `current_prior` is reset to what it was before, so the task can be reused.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for t, (ax, posterior) in enumerate(zip(axes, rollout.extra["posterior_history"])):
    plot_posterior(ax, posterior, f"after step {t + 1}", rollout.design_history[0, : t + 1])
plt.show()

print(task.current_prior is task.prior)
```
