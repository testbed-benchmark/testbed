"""Tests for the unified amortized PolicyDesigner."""

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets import (
    CategoricalPolicyNet,
    DeepSetPolicyNet,
    TanhGaussianPolicyNet,
)
from testbed.designers import PolicyDesigner, build_policy_designer
from testbed.tasks import HyperbolicTemporalDiscounting


def _make_task():
    return HyperbolicTemporalDiscounting()


def _make_policy_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_stochastic_net():
    space = BoxDesignSpace(low=[5.0, 20.0], high=[40.0, 80.0], event_shape=(2,))
    return TanhGaussianPolicyNet(space)


def test_default_transform_designs_matches_task_design_space():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = PolicyDesigner(task, T=3, design_net=policy_net)

    assert designer.transform_designs is True

    theta = task.sample_prior(4)
    rollout = designer.rollout(theta)

    raw_design_history = rollout.extra["raw_design_history"]
    torch.testing.assert_close(
        task.design_space.to_design(raw_design_history), rollout.design_history
    )
    # HTDDesignSpace.to_design is a real (non-identity) transform.
    assert not torch.allclose(raw_design_history, rollout.design_history)


def test_transform_designs_false_opts_out():
    torch.manual_seed(0)
    task = _make_task()
    # With transform_designs=False, the net's raw forward output is used
    # directly as the design, which won't satisfy HTDDesignSpace's real
    # domain constraints (D1 > 0, 0 < R0 < 100) -- disable validation since
    # this test is only about the designer-level opt-out.
    task.validate_designs = False
    policy_net = _make_policy_net(task)
    designer = PolicyDesigner(task, T=3, design_net=policy_net, transform_designs=False)

    theta = task.sample_prior(4)
    rollout = designer.rollout(theta)

    # No transform applied: the net's raw forward output is used directly as
    # the design.
    torch.testing.assert_close(
        rollout.extra["raw_design_history"], rollout.design_history
    )


def test_raw_design_history_is_always_exposed():
    # The net's raw output is computed either way, so it is always reported.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = PolicyDesigner(task, T=2, design_net=policy_net)

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" in rollout.extra


def test_build_policy_designer_forwards_transform_designs_to_deterministic():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = build_policy_designer(
        task, T=2, design_net=policy_net, transform_designs=False
    )

    assert type(designer) is PolicyDesigner
    assert designer.transform_designs is False


def test_build_policy_designer_forwards_transform_designs_to_stochastic():
    task = _make_task()
    task.validate_designs = False
    stochastic_net = _make_stochastic_net()
    designer = build_policy_designer(
        task, T=2, design_net=stochastic_net, transform_designs=False
    )

    rollout = designer.rollout(task.sample_prior(3))

    assert designer.transform_designs is False
    torch.testing.assert_close(
        rollout.design_history, rollout.extra["raw_design_history"]
    )


def test_build_policy_designer_stochastic_exposes_raw_design():
    torch.manual_seed(0)
    task = _make_task()
    stochastic_net = _make_stochastic_net()
    designer = build_policy_designer(task, T=2, design_net=stochastic_net)

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" in rollout.extra


# ============================================================================
# design history convention
# ============================================================================


def test_physical_history_is_the_uniform_default():
    task = _make_task()

    deterministic = build_policy_designer(task, T=2, design_net=_make_policy_net(task))
    stochastic = build_policy_designer(task, T=2, design_net=_make_stochastic_net())

    assert deterministic.use_raw_history is False
    assert stochastic.use_raw_history is False


def test_use_raw_history_true_is_honored_for_a_stochastic_net():
    task = _make_task()
    stochastic_net = _make_stochastic_net()

    designer = build_policy_designer(
        task, T=2, design_net=stochastic_net, use_raw_history=True
    )

    assert type(designer) is PolicyDesigner
    assert designer.use_raw_history is True


def test_use_raw_history_false_is_honored_for_a_stochastic_net():
    task = _make_task()

    designer = build_policy_designer(
        task, T=2, design_net=_make_stochastic_net(), use_raw_history=False
    )

    assert designer.use_raw_history is False


def test_categorical_policy_net_can_condition_on_design_shaped_raw_history():
    task = _make_task()
    space = FiniteDesignSpace(torch.tensor([[5.0, 20.0], [40.0, 80.0]]))
    net = CategoricalPolicyNet(space)

    designer = build_policy_designer(task, T=2, design_net=net, use_raw_history=True)
    rollout = designer.rollout(task.sample_prior(3))

    assert designer.use_raw_history is True
    assert rollout.extra["raw_design_history"].shape == rollout.design_history.shape


def test_policy_log_prob_requirement_is_a_designer_capability():
    task = _make_task()

    deterministic = PolicyDesigner(task, T=2, design_net=_make_policy_net(task))
    score_function = PolicyDesigner(task, T=2, design_net=_make_stochastic_net())
    pathwise = PolicyDesigner(
        task,
        T=2,
        design_net=_make_stochastic_net(),
        reparameterize_policy=True,
    )

    assert deterministic.requires_policy_log_probs is False
    assert score_function.requires_policy_log_probs is True
    assert pathwise.requires_policy_log_probs is False


def test_categorical_policy_still_requires_log_probs_when_reparameterization_requested():
    task = _make_task()
    space = FiniteDesignSpace(torch.tensor([[5.0, 20.0], [40.0, 80.0]]))
    designer = PolicyDesigner(
        task,
        T=2,
        design_net=CategoricalPolicyNet(space),
        reparameterize_policy=True,
    )

    assert designer.requires_policy_log_probs is True


def test_removed_designer_subclasses_are_not_exported():
    from testbed import designers

    assert not hasattr(designers, "DeterministicPolicyDesigner")
    assert not hasattr(designers, "StochasticPolicyDesigner")
