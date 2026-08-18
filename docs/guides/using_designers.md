# Using Designers

Every designer creates a `Rollout` from a task and one or more parameter values.
The rollout contains the physical `design_history` and `y_history`; implementations
can put additional results in `rollout.extra`.

## Policy designers

Use {py:class}`~testbed.designers.amortized.PolicyDesigner` to run a learned
{py:class}`~testbed.design_nets.base.DesignNet`. All policy networks implement the
same `propose(y_history, design_history, mask=...)` interface and return a
{py:class}`~testbed.design_nets.base.DesignProposal`. Deterministic networks return
a proposal without a log probability. Stochastic networks return a sampled proposal
and its log probability, allowing score-function objectives to train the policy.

```python
from testbed.design_nets import DeepSetPolicyNet
from testbed.designers import PolicyDesigner

policy = DeepSetPolicyNet(
    event_shape_y=task.event_shape_y,
    event_shape_design=task.event_shape_design,
    event_shape_out=task.event_shape_design,
)
designer = PolicyDesigner(task, T=5, design_net=policy)
```

By default, `PolicyDesigner` treats proposals as raw coordinates, transforms them
through `task.design_space`, and stores both coordinate systems: physical designs in
`rollout.design_history` and raw designs in `rollout.extra["raw_design_history"]`.
For stochastic raw proposals it also corrects the proposal log probability for that
transform. Set `transform_designs=False` only when a policy already returns valid
physical designs. A categorical policy over `FiniteDesignSpace` does this by design;
it declares its proposals as physical and is handled automatically.

`GaussianPolicyNet` supports `reparameterize_policy=True`, which requests pathwise
samples. The flag is ignored by deterministic policies and categorical policies,
which are trained through their proposal log probabilities.
