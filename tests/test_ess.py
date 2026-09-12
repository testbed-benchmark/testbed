"""Tests for the ESS diagnostic recorded on every posterior-inference call."""

import torch

from testbed.core.distributions import EmpiricalDistribution, TorchDistribution
from testbed.core.hooks import PosteriorUpdateHook, RefinementHook
from testbed.design_nets import DeepSetPolicyNet
from testbed.designers import SemiAmortizedDesigner
from testbed.inference import effective_sample_size
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_designer(task, T, hook):
    designer = SemiAmortizedDesigner(
        task,
        T=T,
        design_net=DeepSetPolicyNet(
            event_shape_y=task.event_shape_y,
            event_shape_design=task.event_shape_design,
            event_shape_out=task.event_shape_design,
        ),
    )
    designer.hooks.append(hook)
    return designer


def test_equal_weights_give_the_full_particle_count():
    locs = torch.zeros(8, 2)
    weights = torch.full((8,), 1 / 8)
    assert effective_sample_size(EmpiricalDistribution(locs, weights)) == 8.0


def test_a_degenerate_posterior_gives_one():
    # The failure this diagnostic exists to catch: all mass on one particle.
    locs = torch.zeros(8, 2)
    weights = torch.zeros(8)
    weights[3] = 1.0
    assert effective_sample_size(EmpiricalDistribution(locs, weights)) == 1.0


def test_a_posterior_without_weights_gives_none():
    # Not an error: a variational or analytic prior installed via update_prior
    # has no weights, and callers skip the record rather than crashing.
    normal = TorchDistribution(
        torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    )
    assert effective_sample_size(normal) is None


def test_posterior_update_hook_records_one_ess_per_step():
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(task, T=3, hook=PosteriorUpdateHook(n_proposal=200))

    rollout = designer.rollout(task.sample_prior(1))

    ess_history = rollout.extra["ess_history"]
    assert [record["step"] for record in ess_history] == [1, 2, 3]
    for record in ess_history:
        assert record["n_particles"] == 200
        assert 1.0 <= record["ess"] <= 200
        assert record["ess_fraction"] == record["ess"] / 200


def test_refinement_hook_records_ess_under_its_own_key():
    # A distinct key from PosteriorUpdateHook's, so the two cannot collide in
    # Rollout.extra if both are ever present.
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(
        task,
        T=4,
        hook=RefinementHook([2], lambda state, posterior: None, n_proposal=200),
    )

    rollout = designer.rollout(task.sample_prior(1))

    assert "ess_history" not in rollout.extra
    # refine_at is 1-indexed, and so is the record's step.
    assert [record["step"] for record in rollout.extra["refinement_ess"]] == [2]
