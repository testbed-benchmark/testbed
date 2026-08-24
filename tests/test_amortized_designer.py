"""Tests for the unified amortized PolicyDesigner."""

import torch

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets import (
    CategoricalPolicyNet,
    DeepSetPolicyNet,
    DesignProposal,
    GaussianPolicyNet,
)
from testbed.designers import PolicyDesigner
from testbed.tasks import HyperbolicTemporalDiscounting, LocationFinding


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
    return GaussianPolicyNet(space, event_shape_y=_make_task().event_shape_y)


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


def test_task_design_space_transforms_and_corrects_log_prob(monkeypatch):
    design_space = BoxDesignSpace(low=-2.0, high=3.0, event_shape=(2,))
    task = LocationFinding(n_sources=1, dim_theta=2, design_space=design_space)
    policy_net = GaussianPolicyNet(design_space, event_shape_y=task.event_shape_y)
    proposal = DesignProposal(
        raw_design=torch.tensor([[0.25, -0.5]]),
        log_prob=torch.tensor([-1.5]),
    )
    monkeypatch.setattr(policy_net, "propose", lambda *args, **kwargs: proposal)
    designer = PolicyDesigner(task, T=2, design_net=policy_net)
    y_history, design_history = designer.make_empty_history(1)

    assert not hasattr(designer, "design_space")
    design, raw_design, log_prob = designer.design_from_history(
        y_history, design_history
    )
    expected_design, expected_log_prob = task.design_space.to_design(
        proposal.raw_design, proposal.log_prob
    )

    torch.testing.assert_close(raw_design, proposal.raw_design)
    torch.testing.assert_close(design, expected_design)
    torch.testing.assert_close(log_prob, expected_log_prob)
    assert task.design_space.contains(design)


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


def test_policy_designer_forwards_transform_designs_to_stochastic():
    task = _make_task()
    task.validate_designs = False
    stochastic_net = _make_stochastic_net()
    designer = PolicyDesigner(
        task, T=2, design_net=stochastic_net, transform_designs=False
    )

    rollout = designer.rollout(task.sample_prior(3))

    torch.testing.assert_close(
        rollout.design_history, rollout.extra["raw_design_history"]
    )


def test_categorical_policy_physical_proposals_bypass_task_transform():
    torch.manual_seed(0)
    task = _make_task()
    policy_space = FiniteDesignSpace(torch.tensor([[5.0, 20.0], [40.0, 80.0]]))
    policy_net = CategoricalPolicyNet(policy_space, event_shape_y=task.event_shape_y)
    designer = PolicyDesigner(task, T=2, design_net=policy_net)
    y_history, design_history = designer.make_empty_history(8)

    design, raw_design, log_prob = designer.design_from_history(
        y_history, design_history
    )

    # CategoricalPolicyNet returns an element of its finite physical space;
    # applying HTDDesignSpace.to_design here would alter both its action and
    # its categorical log-mass.
    torch.testing.assert_close(design, raw_design)
    assert policy_space.contains(design)
    expected_log_prob = torch.log_softmax(
        policy_net.logits(y_history, design_history), dim=-1
    )[torch.arange(design.shape[0]), policy_space.encode(design)]
    torch.testing.assert_close(log_prob, expected_log_prob)


# ============================================================================
# design history convention
# ============================================================================


def test_categorical_policy_net_can_condition_on_design_shaped_raw_history():
    task = _make_task()
    space = FiniteDesignSpace(torch.tensor([[5.0, 20.0], [40.0, 80.0]]))
    net = CategoricalPolicyNet(space, event_shape_y=task.event_shape_y)

    designer = PolicyDesigner(task, T=2, design_net=net, use_raw_history=True)
    rollout = designer.rollout(task.sample_prior(3))

    assert designer.use_raw_history is True
    assert rollout.extra["raw_design_history"].shape == rollout.design_history.shape
