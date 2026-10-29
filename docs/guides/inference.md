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

# Custom Inference

TestBED isn't an inference library. Instead, it lets you plug in whichever inference method you want as a component of the experimental design process. For convenience, we provide a basic importance sampler, as well as tools for interfacing with Pyro.  The [Posterior Inference](../examples/inference.md) example shows the built-in methods at work. You are encouraged to implement your own custom inference methods when needed.

The main way that TestBED interacts with inference is through designers that estimate beliefs over $\theta$ before producing designs. These access a {py:class}`~testbed.tasks.base.Task`'s `current_prior` over $\theta$, which can be updated through inference. The only requirement is that the current prior is a {py:class}`~testbed.core.distributions.Distribution` object (see [Distributions](distributions.md)).

## The interface

An inference method is any function

```python
inference(task, y_history, design_history) -> Distribution
```

The inputs are the histories of a single rollout, having shapes  `[t, *event_shape_y]` and `[t, *event_shape_design]`. The `Distribution` output is then the task's posterior given that history, using the original prior specified in the task. Any distribution whose `event_shape` is `task.event_shape_theta` is valid. The main examples are:
  - an {py:class}`~testbed.core.distributions.EmpiricalDistribution` for sample-based inference methods
  - a {py:class}`~testbed.core.distributions.TorchDistribution` for inference methods that work with parametric distributions, like variational inference


We note that, during a sequential rollout, `task.current_prior` holds the previous step's posterior. An inference method, e.g., one based on sequential Monte Carlo, can use this information in addition to the history.

TestBED provides:

| Function | Method |
|---|---|
| {py:func}`~testbed.inference.is_sampler.run_importance_sampling` | Importance sampling using the prior as a proposal. The default. |
| {py:func}`~testbed.inference.pyro_sampler.run_mcmc` | Pyro's NUTS or HMC. Needs `pip install 'testbed[pyro]'`. |

## Using an inference method

`SequentialDesigner(..., inference=...)` runs an inference method after every step. Pass `inference=None` to turn this off. To add inference to any other designer, give it a {py:class}`~testbed.core.hooks.PosteriorUpdateHook`. The easiest way to configure an inference method's options is with `functools.partial`.

```{code-cell} ipython3
:tags: [remove-stderr]

from functools import partial

import torch

from testbed.core.hooks import PosteriorUpdateHook
from testbed.designers import RandomDesigner
from testbed.inference import run_importance_sampling
from testbed.tasks import LocationFinding

torch.manual_seed(0)

task = LocationFinding(n_sources=1, dim_theta=2)
hook = PosteriorUpdateHook(partial(run_importance_sampling, n_proposal=10_000))
rollout = RandomDesigner(task, T=5, hooks=[hook]).rollout(task.sample_prior(1))

print(rollout.extra["posterior"])
print(task.current_prior is task.prior)  # restored after the rollout
```

The final posterior is in `rollout.extra["posterior"]`. With `store_history=True`, the hook also returns the posterior after every step, in `rollout.extra["posterior_history"]`. Once the rollout ends, the task's `current_prior` is reset to what it was before.

## Writing your own

Most inference methods need only the unnormalised log posterior. {py:class}`~testbed.inference.log_posterior.LogPosterior` provides this in the raw, unconstrained coordinates of $\theta$:
- **Raw coordinates** `z` are unconstrained and flat, of shape `[*b, dim]`.
- **`log_post.transform`** maps them onto the prior's support.
- **`log_post(z)`** is the unnormalised log posterior of `z`, with the Jacobian included.

So an optimizer or sampler never has to handle constraints. As an example, here is mean-field Gaussian variational inference in about 15 lines. It returns the fitted Gaussian pushed through `log_post.transform`, which is a torch distribution over $\theta$:

```{code-cell} ipython3
from torch.distributions import Independent, Normal, TransformedDistribution

from testbed.inference import LogPosterior


def gaussian_vi(task, y_history, design_history, n_steps=500, lr=0.05):
    log_post = LogPosterior(task, y_history, design_history)
    loc = torch.zeros(log_post.dim, requires_grad=True)
    log_scale = torch.zeros(log_post.dim, requires_grad=True)
    optimizer = torch.optim.Adam([loc, log_scale], lr=lr)
    for _ in range(n_steps):
        q = Independent(Normal(loc, log_scale.exp()), 1)
        z = q.rsample((64,))
        loss = (q.log_prob(z) - log_post(z)).mean()  # the negative ELBO
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    q = Independent(Normal(loc.detach(), log_scale.detach().exp()), 1)
    return TransformedDistribution(q, [log_post.transform])
```

On the history from the [Posterior Inference](../examples/inference.md) example, this Gaussian VI method gives us results similar to importance sampling:

```{code-cell} ipython3
torch.manual_seed(1)
history = RandomDesigner(task, T=10).rollout(task.sample_prior(1))
y_history, design_history = history.y_history[0], history.design_history[0]

for name, inference in [("VI", gaussian_vi), ("IS", run_importance_sampling)]:
    samples = inference(task, y_history, design_history).sample((5000,)).reshape(-1, 2)
    print(f"{name} mean {samples.mean(0)}, std {samples.std(0)}")
```

To use it in a designer, pass `inference=gaussian_vi`. Hooking up another library works the same way. For example, `run_mcmc` uses `LogPosterior` like this:
- it hands `-log_post` to Pyro as the potential energy;
- it starts the chain at `log_post.to_raw(task.current_prior.sample())`;
- it maps the samples back with `log_post.to_theta`, returning them as an `EmpiricalDistribution`.
