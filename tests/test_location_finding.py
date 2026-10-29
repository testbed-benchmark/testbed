import math

import torch

from testbed.tasks import LocationFinding
from testbed.tasks.location_finding import LocationFindingPrior


def make_prior(n_sources, dim_theta, **kwargs):
    return LocationFindingPrior(n_sources, dim_theta, **kwargs)


def make_task(n_sources, dim_theta, **kwargs):
    return LocationFinding(n_sources, dim_theta, **kwargs)


def test_event_shapes():
    task = make_task(3, 2)
    assert task.event_shape_theta == torch.Size([3, 2])
    assert task.event_shape_design == torch.Size([2])
    assert task.event_shape_y == torch.Size([1])

    task = make_task(1, 4, dim_y=5)
    assert task.event_shape_theta == torch.Size([1, 4])
    assert task.event_shape_design == torch.Size([4])
    assert task.event_shape_y == torch.Size([5])


def test_task_uses_default_prior_with_correct_shape():
    task = make_task(3, 2)
    assert isinstance(task.prior, LocationFindingPrior)
    assert task.prior.dist.mean.shape == (3, 2)
    assert torch.allclose(task.prior.dist.mean, torch.zeros(3, 2))
    assert torch.allclose(task.prior.dist.base_dist.covariance_matrix, torch.eye(2))


def test_prior_sample_shapes():
    prior = make_prior(3, 2)

    assert prior.sample(1).shape == (1, 3, 2)
    assert prior.sample(10).shape == (10, 3, 2)


def test_prior_logprob_matches_independent_multivariate_normal():
    n_sources, dim_theta = 3, 2
    prior = make_prior(n_sources, dim_theta)
    reference = torch.distributions.MultivariateNormal(
        torch.zeros(dim_theta), torch.eye(dim_theta)
    )

    theta = torch.randn(10, n_sources, dim_theta)
    expected = reference.log_prob(theta).sum(dim=-1)  # sources are i.i.d.

    assert torch.allclose(prior.log_prob(theta), expected, atol=1e-4)


def test_make_mean_matches_formula_single_source():
    task = make_task(1, 2)  # default base_signal=0.1, max_signal=1e-4, exponent=2
    theta = torch.tensor([[[1.0, 0.0]]])  # [b=1, n_sources=1, dim_theta=2]
    design = torch.tensor([[0.0, 0.0]])  # [b=1, dim_theta=2]

    mean = task.make_mean(theta, design)
    sq_dist = 1.0  # ||(0,0) - (1,0)||^2
    expected = math.log(0.1 + 1.0 / (1e-4 + sq_dist))

    assert mean.shape == (1, 1)
    assert torch.allclose(mean, torch.tensor([[expected]]), atol=1e-5)


def test_make_mean_sums_signal_over_sources():
    task = make_task(2, 1)
    theta = torch.tensor([[[0.0], [2.0]]])  # [b=1, n_sources=2, dim_theta=1]
    design = torch.tensor([[1.0]])  # [b=1, dim_theta=1]

    mean = task.make_mean(theta, design)
    sq_dist = 1.0  # both sources are distance 1 from the design
    expected = math.log(0.1 + 2.0 / (1e-4 + sq_dist))

    assert torch.allclose(mean, torch.tensor([[expected]]), atol=1e-5)


def test_outcome_logp_matches_analytic_gaussian_density():
    noise_scale = 0.3
    task = make_task(1, 2, noise_scale=noise_scale)
    theta = torch.tensor([[[1.0, 0.0]]])
    design = torch.tensor([[0.0, 0.0]])
    y = torch.tensor([[0.2]])

    logp = task.outcome_logp(y, theta, design)

    mean = math.log(0.1 + 1.0 / (1e-4 + 1.0))
    expected = -0.5 * (
        ((0.2 - mean) ** 2) / noise_scale**2 + math.log(2 * math.pi * noise_scale**2)
    )

    assert torch.allclose(logp, torch.tensor([expected]), atol=1e-5)


def test_sample_joint_shapes():
    task = make_task(2, 3, dim_y=2)
    design = torch.randn(5, 3)

    y, theta = task.sample_joint(design, sample_shape=4)
    assert y.shape == (4, 5, 2)
    assert theta.shape == (4, 5, 2, 3)


def test_sample_outcome_is_reparametrized():
    task = make_task(2, 3)
    design = torch.randn(8, 3, requires_grad=True)
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_mean_matches_conditional_mean():
    torch.manual_seed(0)
    task = make_task(1, 2, noise_scale=0.1)
    theta = torch.tensor([[[1.0, 0.0]]])
    design = torch.tensor([[0.0, 0.0]])

    samples = task.sample_outcome(theta, design, sample_shape=20_000)
    empirical_mean = samples.mean()
    expected_mean = task.make_mean(theta, design).squeeze()

    assert torch.allclose(empirical_mean, expected_mean, atol=0.02)
