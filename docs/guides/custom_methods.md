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
mystnb:
  execution_timeout: 300
---

# Implementing Custom Methods

TestBED is designed as a set of modular building blocks: it provides tasks, designers that produce rollouts, and objectives that score them, and leaves the training loop to you. A new method is usually a short script that combines these pieces, and you are encouraged to write your own training loops, experiment managers, logging, etc. in whatever way suits your research.

Which piece to customize depends on what your method changes:

| Your method changes... | Customize | See |
|---|---|---|
| how an existing designer is fitted | the training loop | [Example](#custom-methods-dad) |
| what happens during a rollout, e.g. posterior updates or actions | a {py:class}`~testbed.core.hooks.RolloutHook` | [Using Hooks](../examples/using_hooks.md) |
| the architecture of a policy | a {py:class}`~testbed.design_nets.base.DesignNet` | [Custom design networks](#custom-methods-design-nets) |
| how each design is chosen | a {py:class}`~testbed.designers.base.Designer` | [Custom designers](#custom-methods-designers) |
| the training signal | an objective | [Custom objectives](#custom-methods-objectives) |

The methods in [`testbed/methods/`](https://github.com/testbed-benchmark/testbed/tree/main/testbed/methods) are worked examples of each of these:

| Method | Built from |
|---|---|
| `dad.py` (DAD; Foster et al., 2021) | `PolicyDesigner` trained on `SPCE`, `SNMC` or `SPCEReinforce` |
| `static.py` | `StaticDesigner` trained on the same objectives, with random restarts |
| `var_post.py` | `PolicyDesigner` trained on a Barber–Agakov bound with a variational posterior network |
| `actionbed.py` (ActionBED; Rossa et al., 2026) | `PolicyDesigner` with an `ActionHook`, trained jointly on a loss |
| `posthoc_action.py` | an `ActionHook` trained on rollouts from a frozen designer |
| `aline.py` (ALINE; Huang et al., 2025) | a custom `AlineDesigner` that overrides `rollout` |
| `rlboed.py` (RL-BOED; Blau et al., 2022) | soft actor-critic on a {py:class}`~testbed.envs.bed_env.BEDEnv`, evaluated through a `PolicyDesigner` |

(custom-methods-dad)=
## Example: Training a DAD Policy

Deep Adaptive Design (DAD) trains a policy network that maps the history $h_{t-1}$ to the next design $x_t$, by maximizing the sPCE lower bound on the expected information gain of the whole experiment,

$$
\mathcal{L}_T(\pi, L) = \mathbb{E}\left[\log \frac{p(h_T \mid \theta_0, \pi)}{\frac{1}{L+1} \sum_{\ell=0}^L p(h_T \mid \theta_\ell, \pi)}\right],
$$

where $\theta_0, \dots, \theta_L \sim p(\theta)$ and $h_T$ is a rollout of $\pi$ under $\theta_0$ (Foster et al., 2021). In TestBED, this takes three pieces:

- a {py:class}`~testbed.designers.amortized.PolicyDesigner`, which wraps a policy network and produces a {py:class}`~testbed.core.rollout.Rollout` for a batch of $\theta_0$,
- an {py:class}`~testbed.design_objectives.spce.SPCE` objective, which estimates $\mathcal{L}_T$ from a rollout, and
- any PyTorch optimizer.

```{code-cell} ipython3
:tags: [remove-stderr]

import torch

from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SNMC, SPCE
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.tasks import LocationFinding

torch.manual_seed(0)

task = LocationFinding(n_sources=1, dim_theta=2)
T = 5

policy = DeepSetPolicyNet(
    event_shape_y=task.event_shape_y,
    event_shape_design=task.event_shape_design,
    event_shape_out=task.event_shape_design,
)
designer = PolicyDesigner(task, T, policy)
objective = SPCE(task, n_contrastive=255)
```

The training loop samples $\theta_0$ from the prior, rolls out the policy, and takes a gradient step on $-\mathcal{L}_T$:

```{code-cell} ipython3
optimizer = torch.optim.Adam(designer.parameters(), lr=1e-3)

for step in range(1_000):
    theta = task.sample_prior(256)
    rollout = designer.rollout(theta)
    loss = -objective.estimate_gain(rollout)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"step {step}: sPCE = {-loss.item():.3f}")
```

A designer is an `nn.Module`, so `designer.parameters()` collects every trainable parameter in the rollout: the policy here, the design itself for a {py:class}`~testbed.designers.non_adaptive.StaticDesigner`, and any network owned by a hook. With a different designer or loss, the same loop trains static designs (as in `static.py`), or a policy jointly with an action network (as in `actionbed.py`).

To evaluate, we estimate the sPCE lower bound and the sNMC upper bound with a larger $L$ than in training, since sPCE can never exceed $\log(L+1)$. For comparison, a {py:class}`~testbed.designers.non_adaptive.RandomDesigner` samples designs from the task's design space:

```{code-cell} ipython3
@torch.no_grad()
def evaluate(designer, n_outer=1024, n_contrastive=10_000):
    rollout = designer.rollout(task.sample_prior(n_outer))
    lower = SPCE(task, n_contrastive, chunk_size=1_000).estimate_gain(rollout)
    upper = SNMC(task, n_contrastive, chunk_size=1_000).estimate_gain(rollout)
    return f"[{lower.item():.3f}, {upper.item():.3f}]"


print("DAD:   ", evaluate(designer))
print("Random:", evaluate(RandomDesigner(task, T)))
```

For larger evaluations, {py:func}`~testbed.eval.base.eval_spce_snmc` batches over the outer samples and chunks over the contrastive samples.

### From this loop to `dad.py`

`testbed/methods/dad.py` is this loop with the additions you would want for real experiments: Hydra configuration, a learning rate schedule, gradient clipping, a time budget, checkpointing and resuming, logging, and a choice of objective. None of these are required to use TestBED, and you can replace any of them with your own tooling.

The choice of objective matters for tasks whose outcomes can't be reparameterized, e.g. discrete outcomes. Backpropagating through `SPCE` gives the full gradient only when the whole rollout is differentiable: the policy is deterministic or reparameterized, and the outcomes are reparameterized, i.e. `task.reparameterize_outcomes` is `True` (see [Implementing New Tasks](custom_tasks.md)). Otherwise, construct the task with `reparameterize_outcomes=False` and train on {py:class}`~testbed.design_objectives.reinforce.SPCEReinforce`, which adds the missing score-function terms. `SPCE` doesn't check this for you. As in `dad.py`, pass `stochastic_design_policy=designer.requires_policy_log_probs`, so the estimator matches the policy:

```python
objective = SPCEReinforce(
    task,
    n_contrastive=255,
    stochastic_design_policy=designer.requires_policy_log_probs,
    baseline="leave_one_out",
)
```

(custom-methods-designers)=
## Custom Designers

{py:meth}`Designer.rollout <testbed.designers.base.Designer.rollout>` implements the loop shared by every designer. At each step $t$ it runs the hooks, asks the designer for the next design, samples $y_t$ from the task, and appends both to the history. A new designer only implements {py:meth}`~testbed.designers.base.Designer._step_design`, which receives the current {py:class}`~testbed.core.rollout.RolloutState` and returns a tuple `(design, raw_design, log_prob)`:

- `design` is the physical design, with shape `[b, *event_shape_design]`.
- `raw_design` is the pre-transform value the design was mapped from by `task.design_space.to_design`, or `None`.
- `log_prob` is the log probability of a stochastic design, with shape `[b]`, or `None`. Rollouts collect these in `rollout.extra["policy_log_probs"]` for score-function objectives.

`state.y_history` and `state.design_history` hold the steps before $t$, so they are empty at $t = 0$. The rest of the state, including the task and anything hooks have stored in `state.scratch`, is also available.

As an example, here is a designer for location finding that measures at a sample from the current posterior over the source location. It relies on a {py:class}`~testbed.core.hooks.PosteriorUpdateHook` to update `task.current_prior` after each step:

```{code-cell} ipython3
from testbed.core.hooks import PosteriorUpdateHook
from testbed.designers import Designer


class PosteriorSampleDesigner(Designer):
    def _step_design(self, state):
        b = state.theta.shape[0]
        theta = state.task.current_prior.sample(b)  # [b, 1, 2]
        return theta[:, 0], None, None


designer = PosteriorSampleDesigner(task, T, hooks=[PosteriorUpdateHook(n_proposal=10_000)])

theta = task.sample_prior(1)  # [1, 1, 2]
rollout = designer.rollout(theta)
print(rollout.design_history.squeeze(0))
print(theta.squeeze(0))

task.update_prior(task.prior)  # reset the posterior left by the hook
```

The designs move towards the source as the posterior concentrates. A designer that needs control over the whole rollout, e.g. to select designs from a pool without replacement, can override `rollout` itself, as {py:class}`~testbed.designers.aline.AlineDesigner` does.

(custom-methods-design-nets)=
### Custom design networks

To try a new policy architecture with the existing training code, subclass {py:class}`~testbed.design_nets.base.DesignNet` instead of `Designer`, and pass it to a `PolicyDesigner`. A design network implements `propose(y_history, design_history, mask=None, reparameterize_policy=False)`, which returns a {py:class}`~testbed.design_nets.base.DesignProposal` holding a raw design and, for stochastic policies, its log probability. `PolicyDesigner` then maps proposals through the task's design space, including the change-of-variables correction to the log probability. Class attributes such as `produces_policy_log_probs` and `supports_reparameterized_proposals` tell the designer which gradient estimators the network supports. See the existing networks in `testbed.design_nets` for examples.

(custom-methods-objectives)=
## Custom Objectives

An objective subclasses {py:class}`~testbed.design_objectives.base_objective.RolloutObjective`, scoring a whole rollout as in the training loop above, or {py:class}`~testbed.design_objectives.base_objective.DesignObjective`, scoring a single design given the history as {py:class}`~testbed.designers.sequential.SequentialDesigner` requires. Both implement a single `estimate_gain` method that returns a scalar to maximize. Since a `Rollout` carries anything that hooks and designers put in `rollout.extra`, a rollout objective can use, e.g., an action or posterior computed during the rollout without changing any other code.
