"""Tests for the cart pole task."""

import math

import torch

from testbed.designers import RandomDesigner, StaticDesigner
from testbed.tasks import CartPole
from testbed.tasks.cart_pole import CartPolePrior


def test_event_shapes_and_flags():
    task = CartPole()

    assert task.event_shape_y == torch.Size([4])
    assert task.event_shape_theta == torch.Size([3])
    assert task.event_shape_design == torch.Size([1])
    assert not task.memoryless
    assert task.contains_design(torch.tensor([[-5.0], [5.0]]))
    assert not task.contains_design(torch.tensor([[-5.5]]))


def test_prior_matches_lognormal_with_jacobian():
    torch.manual_seed(0)
    prior = CartPolePrior()
    theta = prior.sample(1000)

    assert theta.shape == (1000, 3)
    assert (theta > 0).all()

    log_theta = torch.log(theta)
    base_logp = torch.distributions.MultivariateNormal(
        torch.zeros(3), covariance_matrix=0.01 * torch.eye(3)
    ).log_prob(log_theta)
    expected = base_logp - log_theta.sum(dim=-1)  # exp() change-of-variables term

    assert torch.allclose(prior.log_prob(theta), expected, atol=1e-4)


def test_drift_func_matches_closed_form():
    task = CartPole()
    y = torch.tensor([[0.3, -0.2, 0.4, 0.1]])
    theta = torch.tensor([[1.2, 0.8, 0.9]])  # mp, mc, l
    design = torch.tensor([[0.5]])

    drift = task.drift_func(y, theta, design)

    s, q, sdot, qdot = 0.3, -0.2, 0.4, 0.1
    mp, mc, length = 1.2, 0.8, 0.9
    xi, g = 0.5, task.g
    k1, k2 = task.stiffness_c, task.stiffness_p
    d1, d2 = task.damping_c, task.damping_p
    denom = mc + mp * math.sin(q) ** 2

    expected_sdotdot = (
        xi
        + mp * math.sin(q) * (length * qdot**2 + g * math.cos(q))
        - (k1 * s + d1 * sdot)
        - (k2 * q + d2 * qdot) * math.cos(q) / length
    ) / denom
    expected_qdotdot = (
        -xi * math.cos(q)
        - mp * length * qdot**2 * math.cos(q) * math.sin(q)
        - (mc + mp) * g * math.sin(q)
        - (k1 * s + d1 * sdot) * math.cos(q)
        - (k2 * q + d2 * qdot) * math.cos(q) ** 2 / length
    ) / (length * denom)

    expected = torch.tensor([[sdot, qdot, expected_sdotdot, expected_qdotdot]])
    assert torch.allclose(drift, expected, atol=1e-6)


def test_noise_scale_defaults_to_isotropic_and_accepts_per_coordinate():
    assert torch.equal(CartPole().noise_scale, torch.full((4,), 0.1))

    # the diffusion of Iqbal et al. (2024), noise on the cart velocity only
    task = CartPole(noise_scale=[1e-4, 1e-4, 0.1, 1e-4])

    assert torch.allclose(task.noise_scale, torch.tensor([1e-4, 1e-4, 0.1, 1e-4]))


def test_per_coordinate_noise_scale_reaches_samples_and_density():
    torch.manual_seed(0)
    scale = [1e-3, 1e-3, 0.2, 1e-3]
    task = CartPole(noise_scale=scale)
    theta = torch.ones(20_000, 3)
    design = torch.zeros(20_000, 1)

    samples = task.sample_outcome(theta, design)
    expected_std = math.sqrt(task.dt) * torch.tensor(scale)

    assert torch.allclose(samples.std(dim=0), expected_std, rtol=0.05)

    y = torch.randn(1, 4) * 0.01
    logp = task.outcome_logp(y, theta[:1], design[:1])
    mean = task.make_mean(theta[:1], design[:1], None)
    expected = torch.distributions.Normal(mean, expected_std).log_prob(y).sum(dim=-1)

    assert torch.allclose(logp, expected, atol=1e-4)


def test_drift_func_broadcasts_over_batch():
    task = CartPole()
    y = torch.randn(7, 4)
    theta = task.sample_prior(7)
    design = torch.rand(7, 1) * 10 - 5

    assert task.drift_func(y, theta, design).shape == (7, 4)


def test_make_mean_uses_zero_initial_condition_when_no_history():
    task = CartPole()
    theta = torch.tensor([[1.0, 1.0, 1.0]])
    design = torch.tensor([[0.5]])
    expected = task.dt * task.drift_func(torch.zeros(1, 4), theta, design)

    assert torch.allclose(task.make_mean(theta, design, None), expected)
    assert torch.allclose(task.make_mean(theta, design, torch.empty(1, 0, 4)), expected)


def test_make_mean_uses_last_observation_when_history_present():
    task = CartPole()
    theta = torch.tensor([[1.0, 1.0, 1.0]])
    design = torch.tensor([[0.5]])
    y_history = torch.tensor([[[0.1, 0.2, 0.0, 0.1], [0.3, -0.1, 0.2, 0.0]]])

    mean = task.make_mean(theta, design, y_history)

    prev_y = y_history[:, -1, :]
    expected = prev_y + task.dt * task.drift_func(prev_y, theta, design)

    assert torch.allclose(mean, expected)


def test_outcome_logp_matches_analytic_gaussian_density():
    task = CartPole(noise_scale=0.2)
    theta = torch.tensor([[1.0, 1.0, 1.0]])
    design = torch.tensor([[0.5]])
    y_history = torch.tensor([[[0.1, 0.2, 0.0, 0.1]]])
    y = torch.tensor([[0.15, 0.25, 0.05, 0.1]])

    logp = task.outcome_logp(y, theta, design, y_history=y_history)

    mean = task.make_mean(theta, design, y_history)
    std = math.sqrt(task.dt) * task.noise_scale
    expected = torch.distributions.Normal(mean, std).log_prob(y).sum(dim=-1)

    assert torch.allclose(logp, expected, atol=1e-5)


def test_sample_outcome_is_reparametrized():
    task = CartPole()
    theta = task.sample_prior(8)
    design = (torch.rand(8, 1) * 9 - 4.5).requires_grad_(True)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_mean_matches_conditional_mean():
    torch.manual_seed(0)
    task = CartPole(noise_scale=0.05)
    theta = torch.tensor([[1.0, 1.0, 1.0]]).expand(20_000, -1)
    design = torch.tensor([[0.3]]).expand(20_000, -1)
    y_history = torch.tensor([[[0.1, -0.1, 0.0, 0.2]]]).expand(20_000, -1, -1)

    samples = task.sample_outcome(theta, design, y_history=y_history)
    expected_mean = task.make_mean(theta[:1], design[:1], y_history[:1])

    assert torch.allclose(samples.mean(dim=0, keepdim=True), expected_mean, atol=0.01)


def test_history_logp_matches_manual_sequential_computation():
    torch.manual_seed(0)
    task = CartPole(noise_scale=0.1)
    b, T = 3, 4
    theta = task.sample_prior(b)
    design_history = torch.rand(b, T, 1) * 9 - 4.5
    y_history = torch.randn(b, T, 4) * 0.1

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
    task = CartPole()
    designer = RandomDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 4)
    assert rollout.design_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()


def test_static_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = CartPole()
    designer = StaticDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 4)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()


def test_noise_jitter_defaults_off_and_regularizes_a_singular_diffusion():
    task = CartPole()

    assert task.noise_jitter == 0.0
    assert torch.allclose(task.obs_std, math.sqrt(task.dt) * task.noise_scale)

    # the literal diffusion of Iqbal et al. (2024), regularized as their code does
    # (covar = L**2 * dt + 1e-8) rather than left singular
    task = CartPole(noise_scale=[0.0, 0.0, 0.1, 0.0], noise_jitter=1e-8)
    expected = torch.sqrt(task.dt * task.noise_scale**2 + 1e-8)

    assert torch.allclose(task.obs_std, expected)
    assert (task.obs_std > 0).all()

    theta = task.sample_prior(4)
    design = task.design_prior.sample((4,))
    logp = task.outcome_logp(task.sample_outcome(theta, design), theta, design)

    assert torch.isfinite(logp).all()


def test_noise_scale_is_a_buffer_that_round_trips_through_state_dict():
    task = CartPole()

    assert "noise_scale" in task.state_dict()
    # a broadcast view would be stride-0, which load_state_dict cannot copy into
    assert task.noise_scale.is_contiguous()

    other = CartPole(noise_scale=[0.2, 0.3, 0.4, 0.5])
    other.load_state_dict(task.state_dict())

    assert torch.equal(other.noise_scale, task.noise_scale)
