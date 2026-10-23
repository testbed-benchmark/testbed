"""Tests for the pharmacokinetic task."""

import math

import torch

from testbed.designers import RandomDesigner, StaticDesigner
from testbed.tasks import Pharmacokinetic
from testbed.tasks.pharmacokinetic import PKPrior


def test_event_shapes_and_flags():
    task = Pharmacokinetic()

    assert task.event_shape_y == torch.Size([1])
    assert task.event_shape_theta == torch.Size([3])
    assert task.event_shape_design == torch.Size([1])
    assert task.memoryless
    assert task.design_space.contains(torch.tensor([[0.0], [24.0]]))
    assert not task.design_space.contains(torch.tensor([[24.5]]))


def test_default_noise_scales_match_reference():
    """iDAD uses epsilon_scale=sqrt(0.01) and nu_scale=sqrt(0.1)."""
    task = Pharmacokinetic()

    assert task.mult_noise_scale == math.sqrt(0.01)
    assert task.add_noise_scale == math.sqrt(0.1)
    assert task.decay_const == 400.0


def test_prior_matches_lognormal_with_jacobian():
    torch.manual_seed(0)
    prior = PKPrior()
    theta = prior.sample(1000)

    assert theta.shape == (1000, 3)
    assert (theta > 0).all()

    log_theta = torch.log(theta)
    base_logp = torch.distributions.MultivariateNormal(
        torch.tensor([1.0, 0.1, 20.0]).log(), covariance_matrix=0.05 * torch.eye(3)
    ).log_prob(log_theta)
    expected = base_logp - log_theta.sum(dim=-1)  # exp() change-of-variables term

    assert torch.allclose(prior.log_prob(theta), expected, atol=1e-4)


def test_prior_keeps_absorption_faster_than_elimination():
    """The reference asserts k_a > k_e; the prior should make that overwhelmingly likely."""
    torch.manual_seed(0)
    theta = PKPrior().sample(10_000)

    assert (theta[:, 0] > theta[:, 1]).all()


def test_concentration_matches_closed_form():
    task = Pharmacokinetic()
    theta = torch.tensor([[1.2, 0.15, 18.0]])  # ka, ke, V
    design = torch.tensor([[3.0]])

    ka, ke, v, t = 1.2, 0.15, 18.0, 3.0
    expected = (
        (task.decay_const / v)
        * (ka / (ka - ke))
        * (math.exp(-ke * t) - math.exp(-ka * t))
    )

    assert torch.allclose(task.concentration(theta, design), torch.tensor([[expected]]))


def test_concentration_is_zero_at_time_zero_and_batches_elementwise():
    task = Pharmacokinetic()
    theta = task.sample_prior(7)

    assert torch.allclose(
        task.concentration(theta, torch.zeros(7, 1)), torch.zeros(7, 1), atol=1e-7
    )
    # elementwise over the batch, not an outer product of theta against design
    assert task.concentration(theta, torch.rand(7, 1) * 24).shape == (7, 1)


def test_obs_std_combines_multiplicative_and_additive_noise():
    task = Pharmacokinetic()
    conc = torch.tensor([[0.0], [10.0]])

    expected = torch.sqrt((task.mult_noise_scale * conc) ** 2 + task.add_noise_scale**2)

    assert torch.allclose(task.obs_std(conc), expected)
    # noise grows with the concentration, so it is not homoscedastic
    assert task.obs_std(conc)[1] > task.obs_std(conc)[0]


def test_outcome_logp_matches_analytic_gaussian_density():
    task = Pharmacokinetic()
    theta = torch.tensor([[1.2, 0.15, 18.0]])
    design = torch.tensor([[3.0]])
    y = torch.tensor([[12.0]])

    logp = task.outcome_logp(y, theta, design)

    loc = task.concentration(theta, design)
    expected = (
        torch.distributions.Normal(loc, task.obs_std(loc)).log_prob(y).sum(dim=-1)
    )

    assert torch.allclose(logp, expected, atol=1e-5)


def test_sample_outcome_is_reparametrized():
    task = Pharmacokinetic()
    theta = task.sample_prior(8)
    design = task.design_space.sample((8,)).requires_grad_(True)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_matches_conditional_moments():
    torch.manual_seed(0)
    task = Pharmacokinetic()
    theta = torch.tensor([[1.2, 0.15, 18.0]]).expand(200_000, -1)
    design = torch.tensor([[3.0]]).expand(200_000, -1)

    samples = task.sample_outcome(theta, design)
    loc = task.concentration(theta[:1], design[:1])
    std = task.obs_std(loc)

    assert torch.allclose(samples.mean(), loc.squeeze(), atol=0.02)
    assert torch.allclose(samples.std(), std.squeeze(), rtol=0.02)


def test_random_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = Pharmacokinetic()
    designer = RandomDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 1)
    assert rollout.design_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()


def test_static_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = Pharmacokinetic()
    designer = StaticDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()
