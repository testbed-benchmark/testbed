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

# Distributions

All distributions, including priors and posteriors on $\theta$, as well as design priors, are {py:class}`~testbed.core.distributions.Distribution` objects. A `Distribution` is two things at once:

- a `torch.distributions.Distribution`, so it has torch's `sample`, `log_prob` and `support`, and works with the rest of torch's machinery;
- an `nn.Module`, so it moves with `task.to(device)` and `task.to(dtype)`.

There are two ways to make a distirbution. Usually, you will wrap a torch distribution in {py:class}`~testbed.core.distributions.TorchDistribution`. When torch doesn't have the distribution you need, you subclass `Distribution` and write `sample` and `log_prob` yourself.

Some conventions followed by distributions are:

- **`sample(sample_shape)`** returns `[*sample_shape, *event_shape]`. An int `n` for the sample shape is equivalent to `(n,)`.
- **`log_prob(value)`** returns the joint log density of each draw, mapping `[*b, *event_shape]` to `[*b]`.
- **`support`** is the set draws lie in, as a [torch constraint](https://pytorch.org/docs/stable/distributions.html#module-torch.distributions.constraints).
- A task's prior must have `event_shape == task.event_shape_theta`.

## Wrapping a torch distribution

Any torch distribution works once wrapped. Here, we show how a standard Gaussian prior can be used for the location finding task (with three sources in two dimensions).

```{code-cell} ipython3
:tags: [remove-stderr]

import torch
from torch.distributions import MultivariateNormal

from testbed.core.distributions import TorchDistribution
from testbed.tasks import LocationFinding

prior = TorchDistribution(MultivariateNormal(torch.zeros(3, 2), torch.eye(2)))
theta = prior.sample((4,))

print(prior.event_shape)  # torch.Size([3, 2])
print(theta.shape, prior.log_prob(theta).shape)  # [4, 3, 2] and [4]

task = LocationFinding(n_sources=3, dim_theta=2, prior=prior)
print(task.sample_prior(5).shape)
```

By convention, `log_prob` is `-inf` outside the support of a distribution.

```{code-cell} ipython3
from torch.distributions import Gamma

gamma_dist = TorchDistribution(Gamma(torch.full((2,), 2.0), 1.0))

print(gamma_dist.support)
print(gamma_dist.log_prob(torch.tensor([[1.0, 2.0], [-1.0, 2.0]])))
```

## Moving devices and datatypes

The main reason that TestBED wraps torch distributions is that it allows us to modify the device/dtype of a distribution (and any tasks that use it) with ease. Methods like `.to()`, `.cuda()` and `.double()` move the wrapped distribution in place. For example, here we change the dtype; `.to("cuda")` works the same way.

```{code-cell} ipython3
prior = TorchDistribution(MultivariateNormal(torch.zeros(3, 2), torch.eye(2)))
prior.to(torch.float64)

print(prior.dist.base_dist.loc.dtype, prior.sample().dtype)
```

Moving a task moves its prior and design prior with it:

```{code-cell} ipython3
task = LocationFinding(n_sources=3, dim_theta=2).to(torch.float64)

print(task.sample_prior().dtype, task.design_prior.sample().dtype)
```

The wrapped distribution's parameters are read through torch's own attributes, e.g. `prior.dist.base_dist.loc` above.

## Writing your own distribution

Some distributions aren't in torch, e.g., priors that consist of smaller sub-components. To implement custom distributions that aren't just wrappers of an existing torch distribution, subclass `Distribution` and implement `sample` and `log_prob`, as well as any constraints (if needed).

| | Required? | Purpose |
|---|---|---|
| `sample(sample_shape)` | yes | Draws of shape `[*sample_shape, *event_shape]` |
| `log_prob(value)` | yes | The joint log density of each draw, `[*b, *event_shape]` to `[*b]`. Raise `NotImplementedError` if there is no density. |
| `support` | if constrained | The parameter or design space. Defaults to unconstrained. |


The prior of {py:class}`~testbed.tasks.ces.CES`, for example, concatenates three independent pieces into $\theta = (\rho, \alpha_1, \alpha_2, \alpha_3, u)$:
- $\rho \in (0, 1)$, Beta;
- $\alpha$ on the simplex, Dirichlet;
- $u > 0$, log-normal.

Torch has each piece but not their concatenation. 

The pieces can be `TorchDistribution`s. Assigning them as attributes registers them as submodules, so they move with `.to()` along with the whole distribution. Any tensor a distribution holds directly should be a buffer (`self.register_buffer`) instead, for the same reason.

The support uses {py:class}`blocks <testbed.core.constraints._Blocks>`, which concatenates the pieces' supports along the last dimension.

```{code-cell} ipython3
from torch.distributions import Beta, Dirichlet, LogNormal

from testbed.core.constraints import blocks
from testbed.core.distributions import Distribution
from testbed.tasks import CES


class CESPrior(Distribution):
    """Independent rho, alpha and u, concatenated as (rho, alpha_1, alpha_2, alpha_3, u)."""

    sizes = [1, 3, 1]

    def __init__(self):
        super().__init__(event_shape=(5,))
        self.rho = TorchDistribution(Beta(torch.ones(1), torch.ones(1)))
        self.alpha = TorchDistribution(Dirichlet(torch.ones(3)))
        self.u = TorchDistribution(LogNormal(torch.ones(1), 3.0))

    @property
    def parts(self):
        return [self.rho, self.alpha, self.u]

    @property
    def support(self):
        return blocks([part.support for part in self.parts], self.sizes)

    def sample(self, sample_shape=()):
        return torch.cat([part.sample(sample_shape) for part in self.parts], dim=-1)

    def log_prob(self, theta):
        # the pieces are independent, so their log densities add
        pieces = theta.split(self.sizes, dim=-1)
        return sum(part.log_prob(piece) for part, piece in zip(self.parts, pieces))


prior = CESPrior()
theta = prior.sample((4,))

print(theta.shape, prior.log_prob(theta).shape)
print(prior.double().sample().dtype)  # the pieces moved with it
```

It can then be used like any other distribution:

```{code-cell} ipython3
task = CES(prior=CESPrior())

print(task.sample_prior(3))
```

Because the prior declares its support, torch's `biject_to` maps unconstrained raw parameters onto it. Four raw coordinates become a valid $\theta$: stick-breaking turns two of them into the three weights of $\alpha$.

```{code-cell} ipython3
from torch.distributions import biject_to

raw = torch.randn(3, 4)
theta = biject_to(task.prior.support)(raw)

print(theta)
print(task.prior.support.check(theta))
```

## Distributions TestBED provides

Besides `TorchDistribution`, TestBED provides:

| Class | What it is |
|---|---|
| {py:class}`~testbed.core.distributions.Box` | Uniform on a box `low <= x <= high`. |
| {py:class}`~testbed.core.distributions.EmpiricalDistribution` | Weighted points: a posterior from importance sampling or MCMC, or a finite set of designs. It has no density, so `log_prob` raises. |
| {py:class}`~testbed.core.distributions.LogNormal` | Multivariate log-normal (torch's `LogNormal` is univariate). |
