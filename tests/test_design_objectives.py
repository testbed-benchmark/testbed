"""Tests for design objectives that don't have a more specific home."""

import torch

from testbed.design_objectives import PCE
from testbed.tasks import TemporalDiscounting


def _make_task():
    return TemporalDiscounting()


def test_sample_joint_keeps_batch_dim_when_total_samples_is_one():
    # Regression: samplers used to squeeze their leading dim whenever the total
    # sample count was one. That silently produced an unbatched theta paired
    # against a batched design, crashing (or worse, silently misaligning)
    # downstream.
    task = _make_task()
    design = task.design_space.sample(batch_shape=(1,))  # [1, *event_shape_design]

    y, theta = task.sample_joint(design)

    assert y.shape == (1, *task.event_shape_y)
    assert theta.shape == (1, *task.event_shape_theta)

    y, theta = task.sample_joint(design, sample_shape=1)

    assert y.shape == (1, 1, *task.event_shape_y)
    assert theta.shape == (1, 1, *task.event_shape_theta)


def test_pce_n_outer_one_does_not_crash_and_has_scalar_gain():
    # Regression: PCE(n_outer=1, ...) used to crash with an IndexError inside the
    # task's likelihood, because sample_joint drops the outer batch dimension
    # whenever n_outer == 1. n_inner is irrelevant to this bug -- n_outer=1 alone
    # triggers it.
    torch.manual_seed(0)
    task = _make_task()
    design = task.design_space.sample()

    for n_inner in (1, 4):
        objective = PCE(task, n_outer=1, n_inner=n_inner)
        gain = objective(design)
        assert gain.shape == ()
        assert torch.isfinite(gain)
