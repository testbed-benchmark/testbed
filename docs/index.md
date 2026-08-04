# TestBED

TestBED is a PyTorch library for Bayesian Experimental Design (BED).

## Quickstart

TestBED provides an interface for experimental design problems, implemented as a {py:class}`~testbed.tasks.Task`, as well as a suite of {py:class}`~testbed.designers.Designer` objects which produce designs for the task at hand. Here's an example where we solve the {py:class}`~testbed.tasks.LocationFinding` task by sequentially optimizing the PCE objective.

```{code-block} Python
import torch

from testbed.tasks import LocationFinding
from testbed.design_objectives import PCE
from testbed.designers import SequentialDesigner

task = LocationFinding(n_sources=1, dim_theta=2)
pce = PCE(task, n_outer=512, n_inner=512)

designer = SequentialDesigner(task, 
                              T=5, 
                              objective=pce, 
                              lr=5e-3, 
                              n_step_opt=1_000)

theta = task.sample_prior()

# a sequence of T designs/observations & final posterior estimate
rollout = designer.rollout(theta)
```

## Tasks

Every task implemented in TestBED is listed below. See our guide on [Using Tasks](guides/using_tasks.md).

TestBED was designed to allow you to implement your own task easily. For some pointers on how to do this, see our guide on [Implementing New Tasks](guides/custom_tasks.md).

| `testbed.tasks.X` | Description | API |
|---|---|---|
| `CES` | Constant elasticity of substitution | [API](autoapi/testbed/tasks/constant_elasticity_of_substitution/index) |
| `HyperbolicTepmoralDiscounting` | Hyperbolic temporal discounting (note: discrete outcomes) | [API](autoapi/testbed/tasks/hyperbolic_temporal_discounting/index) |
| `LinearRegression` | Linear regression | [API](autoapi/testbed/tasks/linear_regression/index) |
| `LocationFinding` | Source location finding | [API](autoapi/testbed/tasks/location_finding/index) |
| `MichaelisMenten` | Michaelis-Menten| [API](autoapi/testbed/tasks/michaelis_menten/index) |

## Designers

TestBED also provides a number of implementations of commonly used experimental design methods. See our guide on [Using Designers](guides/using_designers.md).


Note that some of these are meant only as references or reasonable defaults; for many tasks, you may be interested in customizing the behavior of your {py:class}`~testbed.designers.Designer` to the specific {py:class}`~testbed.tasks.Task` at hand. For a guide on how to do this and how to implement your own custom methods, see [Implementing Custom Methods](guides/custom_methods.md).

| `testbed.designers.X` | Description | API |
|---|---|---|
| `DeterministicPolicyDesigner` | Uses a deterministic neural network based policy to produce designs | [API](autoapi/testbed/designers/amortized/index) |
| `RandomDesigner` | Samples designs from a fixed distribution | [API](autoapi/testbed/designers/non_adaptive/index) |
| `SequentialDesigner` | Produces designs by optimizing an objective at each design step; estimates a posterior at each step | [API](autoapi/testbed/designers/sequential/index) |
| `StaticDesigner` | Produces all designs up-front (i.e., non-adaptively) by optimizing an objective | [API](autoapi/testbed/designers/non_adaptive/index) |
| `StochasticPolicyDesigner` | Uses a stochastic neural network based policy to produce designs | [API](autoapi/testbed/designers/amortized/index) |


```{toctree}
---
hidden:
caption: Examples
maxdepth: 1
---

examples/location_finding
examples/constrained_designs
examples/using_hooks
```


```{toctree}
---
hidden:
caption: Guides
maxdepth: 1
---

guides/configuration
guides/custom_tasks
guides/custom_methods
guides/using_designers
guides/using_tasks
guides/wandb
```

```{toctree}
---
hidden:
caption: API
maxdepth: 2
---
API Reference<autoapi/testbed/index>
```
