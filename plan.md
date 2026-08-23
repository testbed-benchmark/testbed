# Remove RLBOED-owned design spaces

## Goal

Require every task configuration used with RLBOED to construct an appropriate
bounded `task.design_space`. RLBOED and its policies should consume that space,
not create a fallback space or treat the policy as the space's owner.

This change must preserve the existing unbounded Location Finding task used by
DAD and ActionBED.

## Implementation plan

### 1. Allow Location Finding configurations to inject a design space

- Add `design_space: DesignSpace | None = None` to
  `testbed/tasks/location_finding.py::LocationFinding.__init__` and document it.
- Pass it to `Task.__init__`. When omitted, retain today's
  `DefaultDesignSpace`, so the ordinary `location_finding` and
  `location_finding_2src` task recipes remain unbounded.
- Validate in `Task.__init__` that `design_space.event_shape` equals
  `event_shape_design`, giving all injectable task spaces the same early shape
  check.

### 2. Simplify stochastic policy construction

- Replace `make_rlboed_policy` with a method-agnostic factory such as
  `make_deepset_stochastic_policy` in `testbed/design_nets/stochastic.py`.
- Pass the already constructed `task.design_space` into the factory. Its only
  space-dependent responsibility is the inherent implementation dispatch:
  - `BoxDesignSpace` -> `DeepSetGaussianActor` + `GaussianPolicyNet`.
  - `FiniteDesignSpace` -> `DeepSetCategoricalActor` +
    `CategoricalPolicyNet`.
- Remove the factory's `task`, `box_low`, `box_high`, and `event_shape_out`
  parameters, `DefaultDesignSpace` handling, fallback box construction, and
  duplicate task/runtime shape checks.
- Keep RLBOED validation that the supplied task space is a supported bounded
  space (`BoxDesignSpace` or `FiniteDesignSpace`). The error should tell the
  user to provide a bounded task configuration.
- Update the export in `testbed/design_nets/__init__.py`. Recipe wiring is out
  of scope for this plan and will be handled separately.

Use one small construction helper for training and standalone evaluation so
both inject the already instantiated task's `event_shape_y`,
`event_shape_design`, and `design_space` into targets that declare those
arguments. Deterministic policies must not be forced to accept a design-space
parameter. Avoid broader DAD/ActionBED API changes beyond using this helper if
needed to preserve generic checkpoint evaluation.

### 3. Remove `self.design_space` from stochastic policies

- In `GaussianPolicyNet`, stop assigning `self.design_space`. Retain the
  registered `design_low` and `design_high` buffers needed for physical-history
  normalization.
- In `CategoricalPolicyNet`, stop assigning `self.design_space`. Retain
  `design_values`, `design_low`, and `design_high`; the values and their ordering
  are part of the categorical policy's learned semantics. Store `n_designs`
  directly or derive it from `design_values.shape[0]` when validating logits.
- Keep these tensors in `state_dict` for checkpoint compatibility and stable
  categorical action ordering. Removing the Python `DesignSpace` reference does
  not mean removing space-derived policy constants.

### 4. Make RLBOED consistently use `task.design_space`

- In `testbed/methods/rlboed.py::make_networks`, set
  `design_space = task.design_space`, pass it into policy construction, and use
  it to select/build the critics. Never read `actor.design_space`.
- `_SACNetworks.design_space` may remain as a convenience reference, but it must
  be the exact same object as `task.design_space`, not a separately resolved or
  reconstructed space.
- Replace remaining policy-owned lookups:
  - Encode categorical replay actions with `nets.design_space.encode(...)`.
  - Obtain categorical `n_designs` for target entropy from the training-owned
    space rather than `nets.actor.design_space`.
- Construct `BEDEnv` and `PolicyDesigner` with that same task-owned space.
- In `testbed/eval/eval_policy.py`, reconstruct the task first, build the policy
  using its space, and pass `task.design_space` to `PolicyDesigner`. Remove the
  `getattr(design_net, "design_space", None)` fallback.
- Leave continuous/discrete critics, replay behavior, entropy logic, and other
  SAC-specific structures unchanged.

## Non-goals

- Do not make Location Finding bounded by default.
- Do not add or reorganize recipes; bounded RLBOED task recipes will be added
  separately.
- Do not add an RLBOED fallback for `DefaultDesignSpace`.
- Do not put a `DesignSpace` object back onto a policy for convenience.
- Do not unify continuous and discrete SAC implementations.
- Documentation explaining that RLBOED requires a bounded task configuration
  will be added separately.
