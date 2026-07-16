"""Tests for the prey-population task."""

import pytest
import torch
from torch.distributions import Binomial, LogNormal

from testbed.envs import CategoricalDesignSpace
from testbed.tasks import PreyPopulation
from testbed.tasks.prey_population import PreyPopulationPrior


def test_event_shapes_flags_and_design_space():
    task = PreyPopulation()

    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([1])
    assert task.event_shape_y == torch.Size([1])
    assert task.discrete_y
    assert isinstance(task.design_space, CategoricalDesignSpace)
    assert task.design_space.contains(torch.tensor([[1.0], [300.0]]))
    assert not task.design_space.contains(torch.tensor([[0.0], [1.5]]))
    assert torch.equal(
        task.design_space.decode(torch.tensor([0, 299])),
        torch.tensor([[1.0], [300.0]]),
    )


def test_prior_matches_independent_lognormals():
    prior = PreyPopulationPrior()
    theta = prior.sample(32)
    reference = LogNormal(-1.4, 1.35)

    assert theta.shape == (32, 2)
    assert (theta > 0).all()
    assert torch.allclose(prior.log_prob(theta), reference.log_prob(theta).sum(dim=-1))


def test_population_solver_matches_closed_form_solution():
    task = PreyPopulation()
    theta = torch.tensor([[0.2, 0.3]])
    design = torch.tensor([[50.0]])

    final_population = task.population_after(theta, design)

    assert torch.allclose(final_population, torch.tensor([[0.5398429]]), atol=1e-6)


def test_population_solver_handles_tail_parameters():
    task = PreyPopulation()
    theta = torch.tensor([[1.0, 0.001]])
    design = torch.tensor([[300.0]])

    final_population = task.population_after(theta, design)
    probability = task.make_consumption_probability(theta, design)

    assert torch.allclose(final_population, torch.tensor([[0.0421881]]), atol=1e-6)
    assert torch.allclose(probability, torch.tensor([[0.9998594]]), atol=1e-6)
    assert torch.isfinite(task.outcome_logp(torch.tensor([[299.0]]), theta, design)).all()


def test_consumption_probability_and_log_prob_match_binomial():
    task = PreyPopulation()
    theta = torch.tensor([[0.2, 0.3], [0.5, 0.1]])
    design = torch.tensor([[50.0], [100.0]])
    y = torch.tensor([[47.0], [90.0]])

    probability = task.make_consumption_probability(theta, design)
    expected = Binomial(design.squeeze(-1), probability.squeeze(-1)).log_prob(
        y.squeeze(-1)
    )

    assert (probability >= 0).all() and (probability <= 1).all()
    assert torch.allclose(task.outcome_logp(y, theta, design), expected)


def test_sampled_outcomes_are_integer_counts_within_design():
    task = PreyPopulation()
    theta = task.sample_prior(64)
    design = torch.full((64, 1), 300.0)

    y = task.sample_outcome(theta, design)

    assert y.shape == (64, 1)
    assert torch.equal(y, y.round())
    assert (y >= 0).all() and (y <= design).all()


def test_sample_joint_and_broadcasted_log_prob_shapes():
    task = PreyPopulation()
    design = torch.tensor([[10.0], [25.0], [100.0], [300.0]])

    y, theta = task.sample_joint(design, n_samples=3)
    logp = task.outcome_logp(y, theta, design)

    assert y.shape == (3, 4, 1)
    assert theta.shape == (3, 4, 2)
    assert logp.shape == (3, 4)
    assert torch.isfinite(logp).all()


def test_non_positive_duration_is_rejected():
    with pytest.raises(ValueError):
        PreyPopulation(duration=0.0)
