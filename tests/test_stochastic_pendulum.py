"""Tests for the stochastic pendulum task."""

import math

import torch

from testbed.designers import RandomDesigner, StaticDesigner
from testbed.tasks import StochasticPendulum
from testbed.tasks.stochastic_pendulum import StochasticPendulumPrior


def test_event_shapes_and_flags():
    task = StochasticPendulum()

    assert task.event_shape_y == torch.Size([2])
    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([1])
    assert not task.memoryless
    assert task.design_space.contains(torch.tensor([[-1.0], [1.0]]))
    assert not task.design_space.contains(torch.tensor([[-1.5]]))


def test_prior_matches_lognormal_with_jacobian():
    torch.manual_seed(0)
    prior = StochasticPendulumPrior()
    theta = prior.sample(1000)

    assert theta.shape == (1000, 2)
    assert (theta > 0).all()

    log_theta = torch.log(theta)
    base_logp = torch.distributions.MultivariateNormal(
        torch.zeros(2), covariance_matrix=0.01 * torch.eye(2)
    ).log_prob(log_theta)
    expected = base_logp - log_theta.sum(dim=-1)  # exp() change-of-variables term

    assert torch.allclose(prior.log_prob(theta), expected, atol=1e-4)


def test_drift_func_matches_closed_form():
    task = StochasticPendulum()
    y = torch.tensor([[0.3, -0.2]])
    theta = torch.tensor([[1.2, 0.8]])  # m, l
    design = torch.tensor([[0.5]])

    drift = task.drift_func(y, theta, design)

    m, length = 1.2, 0.8
    q, qdot = 0.3, -0.2
    xi = 0.5
    expected_qdotdot = -3 * task.g / (2 * length) * math.sin(q) + 3 * (
        xi - task.damping * qdot
    ) / (m * length**2)

    assert torch.allclose(drift, torch.tensor([[qdot, expected_qdotdot]]), atol=1e-6)


def test_make_mean_uses_zero_initial_condition_when_no_history():
    task = StochasticPendulum()
    theta = torch.tensor([[1.0, 1.0]])
    design = torch.tensor([[0.5]])
    expected = task.dt * task.drift_func(torch.zeros(1, 2), theta, design)

    assert torch.allclose(task.make_mean(theta, design, None), expected)
    assert torch.allclose(task.make_mean(theta, design, torch.empty(1, 0, 2)), expected)


def test_make_mean_uses_last_observation_when_history_present():
    task = StochasticPendulum()
    theta = torch.tensor([[1.0, 1.0]])
    design = torch.tensor([[0.5]])
    y_history = torch.tensor([[[0.1, 0.2], [0.3, -0.1]]])  # [b=1, t=2, 2]

    mean = task.make_mean(theta, design, y_history)

    prev_y = y_history[:, -1, :]
    expected = prev_y + task.dt * task.drift_func(prev_y, theta, design)

    assert torch.allclose(mean, expected)


def test_outcome_logp_matches_analytic_gaussian_density():
    task = StochasticPendulum(noise_scale=0.2)
    theta = torch.tensor([[1.0, 1.0]])
    design = torch.tensor([[0.5]])
    y_history = torch.tensor([[[0.1, 0.2]]])
    y = torch.tensor([[0.15, 0.25]])

    logp = task.outcome_logp(y, theta, design, y_history=y_history)

    mean = task.make_mean(theta, design, y_history)
    std = math.sqrt(task.dt) * task.noise_scale
    expected = torch.distributions.Normal(mean, std).log_prob(y).sum(dim=-1)

    assert torch.allclose(logp, expected, atol=1e-5)


def test_sample_outcome_is_reparametrized():
    task = StochasticPendulum()
    theta = task.sample_prior(8)
    design = (torch.rand(8, 1) * 1.8 - 0.9).requires_grad_(True)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_mean_matches_conditional_mean():
    torch.manual_seed(0)
    task = StochasticPendulum(noise_scale=0.05)
    theta = torch.tensor([[1.0, 1.0]]).expand(20_000, -1)
    design = torch.tensor([[0.3]]).expand(20_000, -1)
    y_history = torch.tensor([[[0.1, -0.1]]]).expand(20_000, -1, -1)

    samples = task.sample_outcome(theta, design, y_history=y_history)
    expected_mean = task.make_mean(theta[:1], design[:1], y_history[:1])

    assert torch.allclose(samples.mean(dim=0, keepdim=True), expected_mean, atol=0.01)


def test_history_logp_matches_manual_sequential_computation():
    torch.manual_seed(0)
    task = StochasticPendulum(noise_scale=0.1)
    b, T = 3, 4
    theta = task.sample_prior(b)
    design_history = torch.rand(b, T, 1) * 1.8 - 0.9
    y_history = torch.randn(b, T, 2) * 0.1  # arbitrary realized trajectory

    logp = task.history_logp(y_history, design_history, theta)

    manual = torch.zeros(b)
    std = math.sqrt(task.dt) * task.noise_scale
    for t in range(T):
        mean_t = task.make_mean(theta, design_history[:, t], y_history[:, :t])
        manual += (
            torch.distributions.Normal(mean_t, std)
            .log_prob(y_history[:, t])
            .sum(dim=-1)
        )

    assert torch.allclose(logp, manual, atol=1e-5)


def test_random_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = StochasticPendulum()
    designer = RandomDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 2)
    assert rollout.design_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()


def test_static_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = StochasticPendulum()
    designer = StaticDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 2)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()
