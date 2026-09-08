"""Tests for the Designer extension contract: _step_design(RolloutState)."""

import pytest
import torch

from testbed.core.rollout import RolloutState
from testbed.design_nets import DesignNet, DesignProposal
from testbed.design_nets.deterministic import DeepSetPolicyNet
from testbed.designers import (
    Designer,
    PolicyDesigner,
    RandomDesigner,
    StaticDesigner,
)
from testbed.tasks import HyperbolicTemporalDiscounting, LocationFinding


def _htd():
    return HyperbolicTemporalDiscounting()


def _policy_designer(task, **kwargs):
    net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    return PolicyDesigner(task, T=3, design_net=net, **kwargs)


class _HistorySpy(DesignNet):
    """Records the design history each net call receives."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seen = []

    @property
    def device(self):
        return self.inner.device

    def forward(self, y_history, design_history, *, mask=None):
        self.seen.append(design_history.clone())
        return self.inner(y_history, design_history, mask=mask)

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask=None,
        reparameterize_policy=False,
    ):
        del reparameterize_policy
        return DesignProposal(raw_design=self(y_history, design_history, mask=mask))


# ---------------------------------------------------------------------------
# _step_design is the single extension point
# ---------------------------------------------------------------------------


def test_step_design_is_required_of_a_subclass():
    class NoStepDesign(Designer):
        @property
        def device(self):
            return self.task.device

    with pytest.raises(TypeError):
        NoStepDesign(_htd(), T=2)


def test_a_designer_needs_only_step_design():
    # The whole point: implement one method taking the state, get rollout()
    # for free.
    class ConstantDesigner(Designer):
        @property
        def device(self):
            return self.task.device

        def _step_design(self, state):
            b = state.y_history.shape[0]
            design = torch.full((b, *self.task.event_shape_design), float(state.t) + 1)
            return design, None, None

    task = _htd()
    designer = ConstantDesigner(task, T=3)

    rollout = designer.rollout(task.sample_prior(4))

    assert rollout.design_history.shape == (4, 3, 2)
    # Step t produced value t + 1.
    for t in range(3):
        assert torch.all(rollout.design_history[:, t] == t + 1)


def test_step_design_sees_the_step_index_and_prior_histories():
    seen = []

    class RecordingDesigner(Designer):
        @property
        def device(self):
            return self.task.device

        def _step_design(self, state):
            seen.append((state.t, state.y_history.shape[1], state.T))
            b = state.y_history.shape[0]
            return torch.ones(b, *self.task.event_shape_design), None, None

    task = _htd()
    RecordingDesigner(task, T=3).rollout(task.sample_prior(2))

    # Histories hold the steps completed strictly before t, and T is visible.
    assert seen == [(0, 0, 3), (1, 1, 3), (2, 2, 3)]


def test_step_design_can_read_hook_scratch():
    # The reason for passing the state: a designer can depend on what a hook
    # recorded, with no signature change anywhere.
    from testbed.designers import RolloutHook

    class CountingHook(RolloutHook):
        def after_step(self, state):
            state.scratch["n_steps"] = state.scratch.get("n_steps", 0) + 1

    observed = []

    class ScratchReadingDesigner(Designer):
        @property
        def device(self):
            return self.task.device

        def _step_design(self, state):
            observed.append(state.scratch.get("n_steps", 0))
            b = state.y_history.shape[0]
            return torch.ones(b, *self.task.event_shape_design), None, None

    task = _htd()
    designer = ScratchReadingDesigner(task, T=3, hooks=[CountingHook()])
    designer.rollout(task.sample_prior(2))

    assert observed == [0, 1, 2]


# ---------------------------------------------------------------------------
# _step_design is a thin adapter over each designer's semantic method
# ---------------------------------------------------------------------------


def _step_state(designer, task, b=4):
    """Builds a t=0 RolloutState for a one-off _step_design call."""
    y_history, design_history = task.null_history(b=b)
    return RolloutState(
        task=task,
        theta=None,
        T=designer.T,
        t=0,
        y_history=y_history,
        design_history=design_history,
    )


@pytest.mark.parametrize("kind", ["static", "random", "policy"])
def test_step_design_returns_a_design_of_the_right_shape(kind):
    task = _htd()
    designer = {
        "static": lambda: StaticDesigner(task, T=3),
        "random": lambda: RandomDesigner(task, T=3),
        "policy": lambda: _policy_designer(task),
    }[kind]()

    design, _, _ = designer._step_design(_step_state(designer, task))

    assert isinstance(design, torch.Tensor)
    assert design.shape == (4, *task.event_shape_design)


def test_step_design_is_a_thin_adapter_over_a_semantic_method():
    # Each designer's real logic lives in a method with explicit, documented
    # inputs; _step_design only pulls those pieces off the state. Calling the
    # semantic method directly must give the same answer.
    task = _htd()

    static = StaticDesigner(task, T=3)
    via_state = static._step_design(_step_state(static, task))
    via_semantic = static.design_at_step(t=0, batch_size=4)
    assert torch.equal(via_state[0], via_semantic[0])
    assert torch.equal(via_state[1], via_semantic[1])

    policy = _policy_designer(task)
    y_history, design_history = task.null_history(b=4)
    via_state = policy._step_design(_step_state(policy, task))
    via_semantic = policy.design_from_history(y_history, design_history)
    assert torch.equal(via_state[0], via_semantic[0])
    assert torch.equal(via_state[1], via_semantic[1])
    assert via_state[2] is via_semantic[2] is None


# ---------------------------------------------------------------------------
# use_raw_history lives on PolicyDesigner, and still routes correctly
# ---------------------------------------------------------------------------


def test_stochastic_designer_base_class_is_gone():
    # RandomDesigner and PolicyDesigner returning a log-prob from _step_design
    # is all that "stochastic" needs to mean.
    from testbed import designers

    assert not hasattr(designers, "StochasticDesigner")
    assert isinstance(RandomDesigner(_htd(), T=2), Designer)


def test_use_raw_history_is_not_a_general_designer_concern():
    # Only policy designers condition on the design history at all.
    assert not hasattr(Designer, "use_raw_history")
    assert not hasattr(StaticDesigner(_htd(), T=2), "use_raw_history")

    with pytest.raises(TypeError):
        StaticDesigner(_htd(), T=2, use_raw_history=True)


@pytest.mark.parametrize("use_raw_history", [True, False])
def test_policy_designer_conditions_on_the_requested_history(use_raw_history):
    # HTD's constrained designs satisfy D_1 > 0 and 0 < R_0 < 100; raw ones
    # are unconstrained, so the two are easy to tell apart.
    torch.manual_seed(0)
    task = _htd()
    net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    spy = _HistorySpy(net)
    designer = PolicyDesigner(
        task, T=3, design_net=spy, use_raw_history=use_raw_history
    )

    designer.rollout(task.sample_prior(2))

    history_at_t1 = spy.seen[1]
    in_design_space = task.design_space.contains(history_at_t1)
    assert in_design_space is (not use_raw_history)


def test_policy_designer_gets_the_constrained_history_at_t0():
    # No design exists yet at t=0, so there is no raw history to prefer.
    torch.manual_seed(0)
    task = LocationFinding(n_sources=1, dim_theta=2)
    net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    spy = _HistorySpy(net)
    designer = PolicyDesigner(task, T=2, design_net=spy, use_raw_history=True)

    designer.rollout(task.sample_prior(2))

    assert spy.seen[0].shape[1] == 0
