"""Tests for SemiAmortizedDesigner.continue_from / remaining_horizon_gain."""

import pytest
import torch

from testbed.core.hooks import RolloutHook
from testbed.design_nets import DeepSetPolicyNet
from testbed.designers import PolicyDesigner, SemiAmortizedDesigner
from testbed.tasks import HyperbolicTemporalDiscounting, LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def test_inherited_rollout_matches_policy_designer():
    # The continuation loop is a deliberate duplicate of the base template loop.
    # This pins the inherited path: if base.py ever changes, the two must still
    # agree, and this is the test that catches the copy drifting.
    task = _make_task()

    torch.manual_seed(0)
    net = _make_net(task)
    plain = PolicyDesigner(task, T=4, design_net=net)
    torch.manual_seed(1)
    plain_rollout = plain.rollout(task.sample_prior(5))

    torch.manual_seed(0)
    net_copy = _make_net(task)
    semi = SemiAmortizedDesigner(task, T=4, design_net=net_copy)
    torch.manual_seed(1)
    semi_rollout = semi.rollout(task.sample_prior(5))

    torch.testing.assert_close(semi_rollout.y_history, plain_rollout.y_history)
    torch.testing.assert_close(
        semi_rollout.design_history, plain_rollout.design_history
    )


def test_continue_from_returns_only_the_new_steps():
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=6, design_net=_make_net(task))
    observed = designer.rollout(task.sample_prior(1))

    theta = task.sample_prior(7)
    rollout = designer.continue_from(
        theta, observed.y_history, observed.design_history, n_steps=2
    )

    # 6 total steps were configured, 6 observed, but the continuation is scoped
    # by n_steps alone and reports only what it simulated.
    assert rollout.y_history.shape == (7, 2, *task.event_shape_y)
    assert rollout.design_history.shape == (7, 2, *task.event_shape_design)
    assert rollout.extra["raw_design_history"].shape == (
        7,
        2,
        *task.event_shape_design,
    )


def test_continue_from_conditions_the_policy_on_the_observed_history():
    # The policy must see [observed, simulated so far], not an empty history.
    torch.manual_seed(0)
    task = _make_task()

    seen = []

    class _SpyNet(DeepSetPolicyNet):
        def propose(self, y_history, design_history, **kwargs):
            seen.append((y_history.shape, design_history.shape))
            return super().propose(y_history, design_history, **kwargs)

    net = _SpyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    designer = SemiAmortizedDesigner(task, T=5, design_net=net)
    observed = designer.rollout(task.sample_prior(1))
    S = observed.T
    seen.clear()

    designer.continue_from(
        task.sample_prior(4), observed.y_history, observed.design_history, n_steps=3
    )

    # Step t of the continuation sees S observed steps plus the t it has added.
    assert [shapes[0][1] for shapes in seen] == [S, S + 1, S + 2]
    assert [shapes[1][1] for shapes in seen] == [S, S + 1, S + 2]
    # And it is batched to theta, not to the single observed trajectory.
    assert all(shapes[0][0] == 4 for shapes in seen)


def test_continue_from_does_not_run_hooks():
    # Refinement runs continuations from inside a hook, so a continuation that
    # fired hooks would recurse into itself.
    torch.manual_seed(0)
    task = _make_task()

    class _CountingHook(RolloutHook):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def after_step(self, state):
            self.calls += 1

    hook = _CountingHook()
    designer = SemiAmortizedDesigner(
        task, T=3, design_net=_make_net(task), hooks=[hook]
    )
    observed = designer.rollout(task.sample_prior(1))
    assert hook.calls == 3

    designer.continue_from(
        task.sample_prior(2), observed.y_history, observed.design_history, n_steps=2
    )
    assert hook.calls == 3


def test_single_observed_trajectory_broadcasts_to_the_continuation_batch():
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=3, design_net=_make_net(task))
    observed = designer.rollout(task.sample_prior(1))

    rollout = designer.continue_from(
        task.sample_prior(6), observed.y_history, observed.design_history, n_steps=2
    )
    assert rollout.b == 6


def test_mismatched_history_batch_raises():
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=3, design_net=_make_net(task))
    observed = designer.rollout(task.sample_prior(3))

    with pytest.raises(ValueError, match="History batch dim must be 1 or 5"):
        designer.continue_from(
            task.sample_prior(5), observed.y_history, observed.design_history, n_steps=1
        )


def test_n_steps_must_be_positive():
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=3, design_net=_make_net(task))
    observed = designer.rollout(task.sample_prior(1))

    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        designer.continue_from(
            task.sample_prior(2), observed.y_history, observed.design_history, n_steps=0
        )


def test_use_raw_history_requires_a_raw_history():
    # Silently falling back to transformed designs would feed an HTD policy
    # coordinates it was never trained on.
    torch.manual_seed(0)
    task = HyperbolicTemporalDiscounting()
    net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    designer = SemiAmortizedDesigner(task, T=3, design_net=net, use_raw_history=True)
    observed = designer.rollout(task.sample_prior(1))

    with pytest.raises(ValueError, match="no raw_design_history was supplied"):
        designer.continue_from(
            task.sample_prior(2), observed.y_history, observed.design_history, n_steps=1
        )

    # With it, the policy conditions on raw coordinates throughout.
    rollout = designer.continue_from(
        task.sample_prior(2),
        observed.y_history,
        observed.design_history,
        n_steps=1,
        raw_design_history=observed.extra["raw_design_history"],
    )
    torch.testing.assert_close(
        task.design_space.to_design(rollout.extra["raw_design_history"]),
        rollout.design_history,
    )
