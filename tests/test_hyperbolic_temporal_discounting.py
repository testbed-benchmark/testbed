"""Tests for the hyperbolic temporal-discounting task."""

import torch
from torch.distributions import Bernoulli

from testbed.design_nets import DeepSetPolicyNet
from testbed.designers import PolicyDesigner
from testbed.tasks import HyperbolicTemporalDiscounting


def _make_task(**kwargs):
    return HyperbolicTemporalDiscounting(**kwargs)


def _make_inputs(b=64, seed=0):
    torch.manual_seed(seed)
    task = _make_task()
    theta = task.sample_prior(b)  # [b, 2]
    design = torch.stack(
        [torch.rand(b) * 100, torch.rand(b) * 100], dim=-1
    )  # [b, 2] = (D_1, R_0)
    return task, theta, design


def test_event_shapes_and_flags():
    task = _make_task()
    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([2])
    assert task.event_shape_y == torch.Size([1])
    assert task.discrete_y
    assert not task.discrete_theta
    assert not task.discrete_design


def test_prior_sample_shape_and_support():
    task = _make_task()
    theta = task.sample_prior(128)
    assert theta.shape == (128, 2)
    assert (theta > 0).all()  # k lognormal, alpha half-normal


def test_make_prob_is_valid_probability():
    task, theta, design = _make_inputs()
    p = task.make_prob(theta, design)
    assert p.shape == (theta.shape[0], 1)
    assert (p > 0).all()
    assert (p < 1).all()


def test_make_prob_decreasing_in_delay():
    # Larger delay D_1 shrinks the delayed value V_1, so P(y=1) must not increase.
    task, theta, _ = _make_inputs()
    b = theta.shape[0]
    R0 = torch.full((b,), 50.0)

    p_short = task.make_prob(theta, torch.stack([torch.full((b,), 1.0), R0], dim=-1))
    p_long = task.make_prob(theta, torch.stack([torch.full((b,), 100.0), R0], dim=-1))

    assert (p_long <= p_short).all()


def test_outcome_logp_matches_bernoulli():
    task, theta, design = _make_inputs()
    p = task.make_prob(theta, design).squeeze(-1)

    for y_val in (0.0, 1.0):
        y = torch.full((theta.shape[0], 1), y_val)
        logp = task._outcome_logp(y, theta, design)
        expected = Bernoulli(p).log_prob(torch.full_like(p, y_val))
        # Loose tolerance: Bernoulli.log_prob goes via logits, which loses
        # precision at the clamped extremes of p.
        assert torch.allclose(logp, expected, rtol=1e-4, atol=1e-6)


def test_outcome_logp_normalized():
    task, theta, design = _make_inputs()
    b = theta.shape[0]

    logp_y0 = task._outcome_logp(torch.zeros(b, 1), theta, design)
    logp_y1 = task._outcome_logp(torch.ones(b, 1), theta, design)

    total = logp_y0.exp() + logp_y1.exp()
    assert torch.allclose(total, torch.ones_like(total))


def test_unconstrained_design_transform():
    torch.manual_seed(0)
    task_raw = _make_task()
    task_unc = _make_task(unconstrained_design=True)
    theta = task_unc.sample_prior(64)

    # (xi_d, xi_r) = 0 maps to D_1 = exp(-log E[k]) = 1 / E[k], R_0 = 50
    xi = torch.zeros(64, 2)
    p_unc = task_unc.make_prob(theta, xi)
    D1 = torch.full((64,), float(torch.exp(torch.tensor(-task_unc.log_k_mean))))
    R0 = torch.full((64,), 50.0)
    p_raw = task_raw.make_prob(theta, torch.stack([D1, R0], dim=-1))
    assert torch.allclose(p_unc, p_raw)


def test_unconstrained_design_has_gradients():
    # Near-zero designs saturate the raw task's likelihood (zero gradient); the
    # unconstrained parameterization must keep gradients alive.
    torch.manual_seed(0)
    task = _make_task(unconstrained_design=True)
    theta = task.sample_prior(32)
    design = torch.zeros(32, 2, requires_grad=True)
    y = torch.ones(32, 1)

    logp = task._outcome_logp(y, theta, design)
    (grad,) = torch.autograd.grad(logp.sum(), design)
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_sample_outcome_is_binary():
    task, theta, design = _make_inputs()
    y = task.sample_outcome(theta, design)
    assert y.shape == (theta.shape[0], 1)
    assert ((y == 0) | (y == 1)).all()


def test_relaxed_sampling_is_reparameterized():
    task = _make_task(temperature=0.5)
    theta = task.sample_prior(8)
    design = torch.rand(8, 2, requires_grad=True) * 100
    y = task.sample_outcome(theta, design)
    assert y.requires_grad
    assert (y >= 0).all() and (y <= 1).all()


def test_rollout_shapes():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    b, T = 8, 4
    designer = PolicyDesigner(task, T=T, policy_net=policy_net)
    theta0 = task.sample_prior(b)
    y_obs, design_obs = designer.rollout(theta0)

    assert y_obs.shape == (b, T, 1)
    assert design_obs.shape == (b, T, 2)
    assert ((y_obs == 0) | (y_obs == 1)).all()
