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

# Location Finding

In this example we will walk through how to use a few different {py:class}`~testbed.designers.Designer` objects to produce designs for the {py:class}`~testbed.tasks.LocationFinding` task. This also serves as a general introduction to TestBED by way of example.

```{code-cell} ipython3
:tags: [remove-output]

import matplotlib.pyplot as plt

import torch
from torch.distributions import MultivariateNormal

from testbed.tasks import LocationFinding

torch.manual_seed(2)

# Task settings
n_sources = 1
dim_theta = 2
T         = 5  # Number of experiments to design
```

```{code-cell} ipython3
:tags: [hide-input]
from matplotlib.lines import Line2D

def plot(rollout):
    # Note: only plots the first rollout if it's batched

    posterior_handle = Line2D(
        [0],
        [0],
        marker='o',
        linestyle='',
        markerfacecolor='gray',
        markeredgecolor='gray',
        markersize=6,
        alpha=1.0,
        label='Posterior Samples',
    )

    designs = rollout.design_history[0].squeeze()
    theta = rollout.theta[0].squeeze()

    fig, ax = plt.subplots(1)

    ax.scatter(designs[:, 0], designs[:, 1], zorder=1, label='Designs')
    ax.scatter(theta[0], theta[1], marker='x', c='r', s=128, zorder=2, label='Source')

    if 'posterior' in rollout.extra:
        posterior_samples = rollout.extra['posterior'].sample(5_000).squeeze()
        ax.scatter(posterior_samples[:, 0], posterior_samples[:, 1], 
        alpha=0.1, s=0.25, c='gray', zorder=0, label='Posterior Samples')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    if 'posterior' in rollout.extra:
        # Handle hard-to-see legend for posterior
        handles, labels = ax.get_legend_handles_labels()
        idx = labels.index("Posterior Samples")
        handles[idx] = posterior_handle
        ax.legend(handles, labels)
    else:
        ax.legend()

```

## RandomDesigner

A {py:class}`~testbed.designers.RandomDesigner` produces designs by randomly sampling from a fixed distribution. With no distribution given, it samples from the task's own {py:class}`~testbed.core.design_spaces.DesignSpace`, which for `LocationFinding` is a standard Gaussian over $\mathbb{R}^2$.

```{code-cell} ipython3
:tags: [remove-output]
from testbed.designers import RandomDesigner

task = LocationFinding(n_sources, dim_theta)

random_designer = RandomDesigner(task, T)
```

To sample from a distribution of your own instead, pass it as `dist`. This can be any {py:class}`torch.distributions.Distribution` object, as long as its shape is compatible with the {py:class}`~testbed.tasks.Task`. For instance, if we change our task to `LocationFinding(n_sources=3, dim_theta=5)`, our `design_dist` would need to return samples of shape `[..., 3, 5]`. You may also use a `testbed.core.distributions.Distribution` object.

```{code-cell} ipython3
:tags: [remove-output]

design_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
random_designer_with_dist = RandomDesigner(task, T, dist=design_dist)
```

The key functionality of any {py:class}`~testbed.designers.Designer` object is its `rollout` method. This method accepts a parameter `theta` (or a batch of parameters) and produces a sequence of designs and observations under those parameters. Depending on the nature of the {py:class}`~testbed.designers.Designer`, extra additional information may be returned which is useful downstream. 

Below, we sample a batch of ten `theta` values from the task's prior and roll out the {py:class}`~testbed.designers.RandomDesigner`. Note that this returns a `Rollout` object containing the `theta` used to generate the rollout, as well as `y_history` which contains the observations and `design_history` which contains the designs. Anything else lives in `extra`, which is empty here: a design space exposes no density, so there are no design log-probs to report. Had we rolled out `random_designer_with_dist`, we would also get `extra["policy_log_probs"]`. Let's also plot the first of these rollouts. 


```{code-cell} ipython3
theta = task.sample_prior(10)

rollout = random_designer.rollout(theta)

print(rollout)
plot(rollout)
```

## SequentialDesigner

Picking designs randomly isn't very fun. Let's use {py:class}`~testbed.designers.SequentialDesigner` to produce a sequence of designs by optimising the {py:class}`~testbed.design_objectives.PCE` at each step. TestBED also supports other objectives, like {py:class}`~testbed.design_objectives.NMC`, and enables you to easily implement your own objectives as needed.

By default, {py:class}`~testbed.designers.SequentialDesigner` optimises its objective (here, `PCE`) by doing a basic gradient-ascent procedure using Adam. While this is often effective, there may be certain tasks where a more tailored approach is desired (e.g., different optimisers, initialisation schemes, random restarts...). You can achieve this in TestBED by either subclassing `SequentialDesigner` or by writing your own custom `Designer`. See our guide on [Implementing Custom Methods](../guides/custom_methods.md).

Furthermore, `PCE` requires us to draw posterior samples at each design iteration. This is done using importance sampling by default (see `testbed.inference`). As solving Bayesian inference is not the goal of our library, determining (and implementing) an appropriate inference method is largely up to you, the user. In TestBED, the recommended way to use custom inference methods are supported using hooks; see [Using Hooks](using_hooks.md). Alternatively, you may write your own `Designer` class.

```{warning}
Some designers (e.g., `SequentialDesigner`) change the state of `task` during the design process. It is highly recommended to regularly create a fresh instantiation of your `task` as needed.
```

```{code-cell} ipython3
from testbed.designers import SequentialDesigner
from testbed.design_objectives import PCE

task = LocationFinding(n_sources, dim_theta)
pce = PCE(task, n_outer=128, n_inner=128)

spce_designer = SequentialDesigner(task, T, pce,
                                   lr=7e-3, n_step_opt=1000)

theta = task.sample_prior()
rollout = spce_designer.rollout(theta)

print(rollout)
plot(rollout)
```

Notice that we now have `posterior = rollout.extra["posterior"]` available to us. Using importance sampling, this is an {py:class}`~testbed.core.distributions.EmpiricalDistribution` object, which is represented a set of weighted samples. We can use `posterior.sample(sample_shape)` to draw from this distribution, which we used to create the visualization in the above plot. We see that the posterior is fairly concentrated around the true source location. 

## What's Next?

See [Using Hooks](using_hooks.md) for how to add posterior inference to a `RandomDesigner` rollout.