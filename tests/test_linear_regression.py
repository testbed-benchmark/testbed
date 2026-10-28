import math

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.tasks import LinearRegression
from testbed.tasks.linear_regression import LinearRegressionPrior


def make_prior(dim, **kwargs):
    return LinearRegressionPrior(dim, **kwargs)


def make_task(dim, **kwargs):
    return LinearRegression(dim, **kwargs)


def test_event_shapes():
    task = make_task(5)
    assert task.event_shape_theta == torch.Size([6])
    assert task.event_shape_design == torch.Size([5])
    assert task.event_shape_y == torch.Size([1])

    task = make_task(1, dim_y=3)
    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([1])
    assert task.event_shape_y == torch.Size([3])


def test_task_uses_default_prior_with_correct_dim():
    task = make_task(3)
    assert isinstance(task.prior, LinearRegressionPrior)
    assert task.prior.dim_theta == 4


def test_prior_default_covariance_is_diag_1_and_quarter():
    prior = make_prior(4)
    expected_var = torch.tensor([1.0, 0.25, 0.25, 0.25])
    assert torch.allclose(torch.diagonal(prior.cov), expected_var)


def test_prior_sample_shapes():
    prior = make_prior(5)

    assert prior.sample(1).shape == (1, 5)
    assert prior.sample(10).shape == (10, 5)


def test_prior_logprob_matches_multivariate_normal():
    torch.manual_seed(0)
    dim = 4
    loc = torch.randn(dim)
    a = torch.randn(dim, dim)
    covmat = a @ a.T + dim * torch.eye(dim)  # guaranteed positive definite

    prior = make_prior(dim, theta_loc=loc, theta_covmat=covmat)
    reference = torch.distributions.MultivariateNormal(loc, covariance_matrix=covmat)

    theta = torch.randn(10, dim)
    assert torch.allclose(prior.log_prob(theta), reference.log_prob(theta), atol=1e-4)


def test_make_mean_matches_linear_formula():
    task = make_task(2)
    theta = torch.tensor([[1.0, 2.0, -3.0]])  # bias, w1, w2
    design = torch.tensor([[0.5, 1.0]])

    mean = task.make_mean(theta, design)
    expected = 1.0 + 2.0 * 0.5 + (-3.0) * 1.0

    assert mean.shape == (1, 1)
    assert torch.allclose(mean, torch.tensor([[expected]]))


def test_make_mean_adds_quadratic_misspecification():
    task = make_task(2, quad=0.5)
    theta = torch.tensor([[0.0, 0.0, 0.0]])
    design = torch.tensor([[3.0, 4.0]])  # ||design||^2 = 25

    mean = task.make_mean(theta, design)
    expected = 0.5 * (25.0 - 1.0)

    assert torch.allclose(mean, torch.tensor([[expected]]))


def test_outcome_logp_matches_analytic_gaussian_density():
    noise_scale = 0.7
    task = make_task(2, noise_scale=noise_scale)
    theta = torch.tensor([[1.0, 2.0, -3.0]])
    design = torch.tensor([[0.5, 1.0]])
    y = torch.tensor([[0.1]])

    logp = task.outcome_logp(y, theta, design)

    mean = 1.0 + 2.0 * 0.5 + (-3.0) * 1.0
    expected = -0.5 * (
        ((0.1 - mean) ** 2) / noise_scale**2 + math.log(2 * math.pi * noise_scale**2)
    )

    assert torch.allclose(logp, torch.tensor([expected]), atol=1e-5)


def test_sample_joint_shapes():
    task = make_task(3, dim_y=2)
    design = task.design_space.sample((5,))

    y, theta = task.sample_joint(design, sample_shape=4)
    assert y.shape == (4, 5, 2)
    assert theta.shape == (4, 5, 4)


def test_sample_outcome_is_reparametrized():
    task = make_task(3)
    design = task.design_space.sample((8,)).requires_grad_(True)
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_mean_matches_conditional_mean():
    torch.manual_seed(0)
    task = make_task(2, noise_scale=0.1)
    theta = torch.tensor([[1.0, 2.0, -3.0]])
    design = torch.tensor([[0.5, 1.0]])

    samples = task.sample_outcome(theta, design, sample_shape=20_000)
    empirical_mean = samples.mean()
    expected_mean = task.make_mean(theta, design).squeeze()

    assert torch.allclose(empirical_mean, expected_mean, atol=0.02)


def test_default_design_space_is_the_unit_box():
    task = make_task(3)

    assert isinstance(task.design_space, BoxDesignSpace)
    assert task.design_space.event_shape == torch.Size([3])
    assert torch.equal(task.design_space.low, -torch.ones(3))
    assert torch.equal(task.design_space.high, torch.ones(3))


def test_design_space_tracks_dim():
    assert make_task(1).design_space.event_shape == torch.Size([1])
    assert make_task(7).design_space.event_shape == torch.Size([7])


def test_design_space_accepts_the_closed_box_and_rejects_outside():
    space = make_task(2).design_space

    assert space.contains(torch.tensor([-1.0, 1.0]))
    assert space.contains(torch.zeros(2))
    assert not space.contains(torch.tensor([-1.01, 0.0]))
    assert not space.contains(torch.tensor([0.0, 1.01]))


def test_task_rejects_designs_outside_the_box():
    task = make_task(3)
    theta = task.sample_prior(1)
    design = torch.full((1, 3), 2.0)

    with pytest.raises(ValueError, match="Design lies outside"):
        task.sample_outcome(theta, design)


def test_design_space_reparametrisation_maps_into_the_box():
    torch.manual_seed(0)
    space = make_task(3).design_space

    assert space.contains(space.sample((64,)))
    assert space.contains(space.project(torch.randn(64, 3) * 10))
    assert space.contains(space.to_design(torch.randn(64, 3) * 10))


def test_box_attains_the_norm_the_quadratic_term_is_centred_on():
    # quad * (||x||^2 - 1) vanishes at ||x|| = 1, which must be reachable.
    task = make_task(2, quad=1.0)
    design = torch.tensor([[1.0, 0.0]])

    assert task.design_space.contains(design)
    theta = torch.zeros(1, 3)
    assert torch.allclose(task.make_mean(theta, design), torch.zeros(1, 1))
