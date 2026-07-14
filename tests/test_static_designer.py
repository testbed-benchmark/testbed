"""Tests for StaticDesigner, including the expose_raw_design flag."""

import torch

from testbed.designers import RolloutHook, StaticDesigner
from testbed.tasks import HyperbolicTemporalDiscounting, LocationFinding


def test_rollout_shapes():
    task = LocationFinding(n_sources=1, dim_theta=2)
    b, T = 5, 3
    designer = StaticDesigner(task, T=T)
    theta = task.sample_prior(b)

    rollout = designer.rollout(theta)

    assert rollout.design_history.shape == (b, T, *task.event_shape_design)
    assert rollout.y_history.shape == (b, T, *task.event_shape_y)
    assert "raw_design_history" not in rollout.extra


def test_raw_design_not_exposed_by_default():
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert "raw_design_history" not in rollout.extra


def test_expose_raw_design_matches_design_parameter():
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3, expose_raw_design=True)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert "raw_design_history" in rollout.extra
    raw_design_history = rollout.extra["raw_design_history"]
    assert raw_design_history.shape == (4, 3, 2)
    # Every batch entry mirrors the same learnable raw parameter.
    assert torch.allclose(raw_design_history[0], designer.raw_design)
    assert torch.allclose(raw_design_history[1], designer.raw_design)
    # The exposed raw design is what design_history is the forward transform of.
    assert torch.allclose(
        task.design_space.to_design(raw_design_history), rollout.design_history
    )


def test_expose_raw_design_agrees_between_vectorized_and_per_step_paths():
    # StaticDesigner.rollout() has a vectorized fast path (no hooks) and a
    # per-step fallback (with hooks); both must expose the same raw designs.
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3, expose_raw_design=True)
    theta = task.sample_prior(4)

    vectorized = designer.rollout(theta)

    designer.hooks.append(RolloutHook())
    per_step = designer.rollout(theta)

    assert torch.allclose(
        vectorized.extra["raw_design_history"], per_step.extra["raw_design_history"]
    )
    assert torch.allclose(vectorized.design_history, per_step.design_history)
