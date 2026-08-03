"""Tests for the amortized policy designers, including the expose_raw_design and
transform_designs flags on PolicyDesigner/DeterministicPolicyDesigner/
StochasticPolicyDesigner/build_policy_designer."""

import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.design_nets import DeepSetPolicyNet, TanhGaussianPolicyNet
from testbed.designers import DeterministicPolicyDesigner, build_policy_designer
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
    designer = DeterministicPolicyDesigner(task, T=3, design_net=policy_net)

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
    designer = DeterministicPolicyDesigner(
        task, T=3, design_net=policy_net, transform_designs=False
    )

    theta = task.sample_prior(4)
    rollout = designer.rollout(theta)

    # No transform applied: the net's raw forward output is used directly as
    # the design.
    torch.testing.assert_close(
        rollout.extra["raw_design_history"], rollout.design_history
    )


def test_expose_raw_design_defaults_true():
    # Unlike StaticDesigner/SequentialDesigner (default False), PolicyDesigner
    # subclasses default expose_raw_design to True.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = DeterministicPolicyDesigner(task, T=2, design_net=policy_net)

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" in rollout.extra


def test_expose_raw_design_false_omits_from_extra():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = DeterministicPolicyDesigner(
        task, T=2, design_net=policy_net, expose_raw_design=False
    )

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" not in rollout.extra


def test_build_policy_designer_forwards_transform_designs_to_deterministic():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_policy_net(task)
    designer = build_policy_designer(
        task, T=2, design_net=policy_net, transform_designs=False
    )

    assert isinstance(designer, DeterministicPolicyDesigner)
    assert designer.transform_designs is False


def test_build_policy_designer_rejects_transform_designs_false_for_stochastic_net():
    task = _make_task()
    stochastic_net = _make_stochastic_net()

    try:
        build_policy_designer(
            task, T=2, design_net=stochastic_net, transform_designs=False
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "expected TypeError for transform_designs=False with a StochasticDesignNet"
        )


def test_build_policy_designer_stochastic_expose_raw_design_defaults_true():
    torch.manual_seed(0)
    task = _make_task()
    stochastic_net = _make_stochastic_net()
    designer = build_policy_designer(task, T=2, design_net=stochastic_net)

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" in rollout.extra


def test_build_policy_designer_stochastic_expose_raw_design_false_omits_from_extra():
    torch.manual_seed(0)
    task = _make_task()
    stochastic_net = _make_stochastic_net()
    designer = build_policy_designer(
        task, T=2, design_net=stochastic_net, expose_raw_design=False
    )

    rollout = designer.rollout(task.sample_prior(3))

    assert "raw_design_history" not in rollout.extra
