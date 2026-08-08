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

# Using Hooks

In TestBED, `Designer` objects automate the design process for you and produces a rollout. Doing things with this rollout after it has completed, e.g., estimating posteriors or calculating losses, is relatively straightforward. But what if we want to make modifications during the design process itself? We could implement an entirely new, custom `Designer` subclass, but this would not make use of the many `Designer`s that are already implemented that we might hope to build on. This is preicsely what hooks are for: we can make modifications to the rollout process in order to customize existing `Designer`s. 

## Example: posterior inference with a random designer

To illustrate how you can use hooks, we will augment the `RandomDesigner` to do posterior inference at each step of the design process (which, on its own, does no inference and just samples designs from a fixed distribution). A {py:class}`~testbed.core.hooks.RolloutHook` lets us bolt that on without writing a new `Designer`. Below, a {py:class}`~testbed.core.hooks.PosteriorUpdateHook` runs importance sampling after every step, updating `task.current_prior` and (with `store_history=True`) keeping the posterior from each step in `rollout.extra["posterior_history"]`.

```{code-cell} ipython3
import torch
from torch.distributions import MultivariateNormal

from testbed.designers import RandomDesigner
from testbed.core.hooks import PosteriorUpdateHook
from testbed.tasks import LocationFinding

torch.manual_seed(0)

task = LocationFinding(n_sources=1, dim_theta=2)
design_dist = MultivariateNormal(torch.zeros(2), torch.eye(2))

designer = RandomDesigner(
    task,
    T=6,
    dist=design_dist,
    hooks=[PosteriorUpdateHook(store_history=True)],
)

theta = task.sample_prior(1)
rollout = designer.rollout(theta)

posterior_history = rollout.extra["posterior_history"]
```

We can visualize how the posterior contracts over the course of the rollout by plotting each entry of `posterior_history` alongside the designs made up to that point. As the design process proceeds, the posterior contracts towards the true source location, although not very efficiently as we are selecting designs at random.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_posterior_history(rollout, ncols=3):
    # Note: only plots the first rollout if it's batched
    posterior_handle = Line2D(
        [0],
        [0],
        marker='o',
        linestyle='',
        markerfacecolor='gray',
        markeredgecolor='gray',
        markersize=8,
        alpha=1.0,
        label='Posterior Samples',
    )

    designs = rollout.design_history[0].squeeze()
    theta = rollout.theta[0].squeeze()
    posterior_history = rollout.extra['posterior_history']

    T = len(posterior_history)
    nrows = -(-T // ncols)  # ceil division

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for t, ax in enumerate(axes):
        if t >= T:
            ax.axis('off')
            continue

        # Designs made up through step t, and the posterior after step t
        designs_so_far = designs[: t + 1]
        posterior_samples = posterior_history[t].sample(5_000).squeeze()

        ax.scatter(posterior_samples[:, 0], posterior_samples[:, 1],
                   alpha=0.1, s=0.25, c='gray', zorder=0, label='Posterior Samples')
        ax.scatter(designs_so_far[:, 0], designs_so_far[:, 1], zorder=1, label='Designs')
        ax.scatter(theta[0], theta[1], marker='x', c='r', s=128, zorder=2, label='Source')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Step {t + 1}')
        # Only show y-ticks on the first column and x-ticks on the last row
        ax.label_outer()

    # Handle hard-to-see legend for posterior
    handles, labels = axes[0].get_legend_handles_labels()
    idx = labels.index('Posterior Samples')
    handles[idx] = posterior_handle
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.96])

plot_posterior_history(rollout)
```


