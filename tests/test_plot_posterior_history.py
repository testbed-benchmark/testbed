"""Tests for the location-finding posterior plot."""

import pytest
import torch

from testbed.core.distributions import EmpiricalDistribution
from testbed.core.rollout import Rollout

pytest.importorskip("matplotlib", reason="plotting needs the full extra")

from testbed.plotting import plot_posterior_history  # noqa: E402


def _make_rollout(n_steps=6, n_sources=2, dim_theta=2, at=(2, 4)):
    """A batch-1 rollout carrying a posterior at each step in ``at``."""
    torch.manual_seed(0)
    history = {
        step: EmpiricalDistribution(
            locs=torch.randn(500, n_sources, dim_theta),
            weights=torch.rand(500),
        )
        for step in at
    }
    return Rollout(
        theta=torch.randn(1, n_sources, dim_theta),
        y_history=torch.randn(1, n_steps, 1),
        design_history=torch.randn(1, n_steps, dim_theta),
        extra={"posterior_history": history},
    )


def test_writes_one_figure_covering_every_inference_step(tmp_path):
    path = tmp_path / "posterior.png"
    returned = plot_posterior_history(_make_rollout(), path, n_plot=50)
    assert returned == path
    assert path.stat().st_size > 0


def test_plots_a_slice_of_a_higher_dimensional_search_space(tmp_path):
    # The task may search in more than two dimensions; the plot takes a slice.
    rollout = _make_rollout(dim_theta=5)
    path = tmp_path / "slice.png"
    plot_posterior_history(rollout, path, dims=(2, 4), n_plot=50)
    assert path.stat().st_size > 0


def test_rejects_dims_the_search_space_does_not_have(tmp_path):
    with pytest.raises(ValueError, match="only 2 components"):
        plot_posterior_history(_make_rollout(), tmp_path / "x.png", dims=(0, 3))


def test_rejects_a_batched_rollout(tmp_path):
    # Trajectories are refined independently, so a batch has no single truth.
    rollout = _make_rollout()
    rollout.theta = rollout.theta.repeat(3, 1, 1)
    with pytest.raises(ValueError, match="single trajectory"):
        plot_posterior_history(rollout, tmp_path / "x.png")


def test_explains_a_rollout_with_no_posterior_history(tmp_path):
    rollout = _make_rollout()
    rollout.extra = {}
    with pytest.raises(ValueError, match="store_posterior_history"):
        plot_posterior_history(rollout, tmp_path / "x.png")
