"""Tests for the RolloutHook mechanism (hook ordering, extra-key collisions, ActionHook)."""

import pytest
import torch

from testbed.designers import ActionHook, Designer, RolloutHook
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


class _CountingDesigner(Designer):
    """Deterministic designer: step t's design is filled with value t."""

    @property
    def device(self):
        return self.task.device

    def _step_design(self, state):
        b = state.y_history.shape[0]
        design = torch.full((b, *self.task.event_shape_design), float(state.t))
        return design, None, None


class _RecordingHook(RolloutHook):
    """Records callback order and the history length seen at each callback."""

    def __init__(self):
        self.calls = []

    def before_propose(self, state):
        self.calls.append(("before_propose", state.y_history.shape[1]))

    def after_propose(self, state):
        self.calls.append(("after_propose", state.y_history.shape[1]))

    def after_step(self, state):
        self.calls.append(("after_step", state.y_history.shape[1]))


class _OverwriteDesignHook(RolloutHook):
    """Replaces the proposed design after it is proposed."""

    def __init__(self, value):
        self.value = value

    def after_propose(self, state):
        state.design = torch.full_like(state.design, self.value)


class _ConstantExtraHook(RolloutHook):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def on_rollout_end(self, state):
        return {self.key: self.value}


def test_hook_callback_ordering_and_history_shapes():
    task = _make_task()
    hook = _RecordingHook()
    designer = _CountingDesigner(task, T=3, hooks=[hook])
    theta = task.sample_prior(2)

    designer.rollout(theta)

    kinds = [call[0] for call in hook.calls]
    assert kinds == ["before_propose", "after_propose", "after_step"] * 3

    # before_propose/after_propose see history strictly before step t;
    # after_step sees history including step t.
    for t in range(3):
        before_propose_len, after_propose_len, after_step_len = (
            hook.calls[3 * t][1],
            hook.calls[3 * t + 1][1],
            hook.calls[3 * t + 2][1],
        )
        assert before_propose_len == t
        assert after_propose_len == t
        assert after_step_len == t + 1


def test_after_propose_can_overwrite_the_design_used():
    """A design rewritten in after_propose must govern the outcome and the history."""
    task = _make_task()
    hook = _OverwriteDesignHook(7.0)
    designer = _CountingDesigner(task, T=3, hooks=[hook])
    theta = task.sample_prior(2)

    rollout = designer.rollout(theta)

    assert torch.equal(
        rollout.design_history, torch.full_like(rollout.design_history, 7.0)
    )


def test_per_step_state_is_cleared_between_steps():
    """Per-step fields hold no stale values from step t-1 when step t begins."""
    task = _make_task()
    seen = []

    class _PeekHook(RolloutHook):
        def before_propose(self, state):
            seen.append(
                (state.design, state.raw_design, state.y, state.policy_log_prob)
            )

    designer = _CountingDesigner(task, T=3, hooks=[_PeekHook()])
    designer.rollout(task.sample_prior(2))

    assert seen == [(None, None, None, None)] * 3


def test_on_rollout_end_key_collision_raises():
    task = _make_task()
    hooks = [_ConstantExtraHook("x", 1), _ConstantExtraHook("x", 2)]
    designer = _CountingDesigner(task, T=2, hooks=hooks)
    theta = task.sample_prior(2)

    with pytest.raises(ValueError):
        designer.rollout(theta)


def test_action_hook_default_calls_once_on_full_history():
    task = _make_task()

    seen_lengths = []

    def action_net(y_history, design_history):
        seen_lengths.append(y_history.shape[1])
        return y_history.sum(dim=1)

    designer = _CountingDesigner(task, T=3, hooks=[ActionHook(action_net)])
    theta = task.sample_prior(2)
    rollout = designer.rollout(theta)

    assert seen_lengths == [3]
    assert rollout.extra["action"].shape[0] == 2


def test_action_hook_every_step_shape_and_causality():
    task = _make_task()

    seen_lengths = []

    def action_net(y_history, design_history):
        seen_lengths.append(y_history.shape[1])
        return y_history.sum(dim=1)

    designer = _CountingDesigner(
        task, T=3, hooks=[ActionHook(action_net, every_step=True)]
    )
    theta = task.sample_prior(2)
    rollout = designer.rollout(theta)

    # Each step's action only sees history up to and including that step.
    assert seen_lengths == [1, 2, 3]
    assert rollout.extra["action_history"].shape[:2] == (2, 3)


def test_hook_instance_reused_across_differently_batched_rollouts():
    task = _make_task()

    def action_net(y_history, design_history):
        return y_history.sum(dim=1)

    hook = ActionHook(action_net, every_step=True)
    designer = _CountingDesigner(task, T=2, hooks=[hook])

    rollout_a = designer.rollout(task.sample_prior(2))
    rollout_b = designer.rollout(task.sample_prior(5))

    assert rollout_a.extra["action_history"].shape[0] == 2
    assert rollout_b.extra["action_history"].shape[0] == 5


def test_hooks_default_to_empty_and_are_appendable():
    task = _make_task()
    designer = _CountingDesigner(task, T=2)
    assert designer.hooks == []

    seen = []
    designer.hooks.append(ActionHook(lambda y, d: seen.append(1) or y.sum(dim=1)))
    designer.rollout(task.sample_prior(2))
    assert seen == [1]
