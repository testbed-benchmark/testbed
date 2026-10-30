import math

import pytest
import torch

from testbed.core import Box
from testbed.envs.bed_env import _project
from testbed.tasks import MichaelisMenten
from testbed.tasks.michaelis_menten import MichaelisMentenPrior


def make_prior(**kwargs):
    return MichaelisMentenPrior(**kwargs)


def make_task(**kwargs):
    return MichaelisMenten(**kwargs)


def test_event_shapes():
    task = make_task()
    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([1])
    assert task.event_shape_y == torch.Size([1])


def test_task_uses_default_prior_with_correct_bounds():
    task = make_task()
    assert isinstance(task.prior, MichaelisMentenPrior)
    assert torch.equal(task.prior.dist.base_dist.low, torch.tensor([20.0, 20.0]))
    assert torch.equal(task.prior.dist.base_dist.high, torch.tensor([200.0, 200.0]))


def test_prior_sample_shapes_and_bounds():
    prior = make_prior()

    samples = prior.sample(64)
    assert samples.shape == (64, 2)
    assert (samples >= 20).all()
    assert (samples <= 200).all()


def test_prior_logprob_is_uniform_in_support():
    prior = make_prior(vmax_lb=0, vmax_ub=10, vhalf_lb=0, vhalf_ub=20)
    expected = -math.log(10.0) - math.log(20.0)

    theta = torch.tensor([[1.0, 2.0], [9.9, 19.9], [0.0, 0.0]])
    logp = prior.log_prob(theta)

    assert torch.allclose(logp, torch.full((3,), expected), atol=1e-5)


def test_prior_logprob_is_neg_inf_outside_support():
    prior = make_prior(vmax_lb=0, vmax_ub=10, vhalf_lb=0, vhalf_ub=20)

    theta = torch.tensor([[10.1, 5.0], [5.0, -0.1], [-1.0, 25.0]])
    logp = prior.log_prob(theta)

    assert (logp == -torch.inf).all()


def test_make_mean_matches_formula():
    task = make_task()  # default exponent=1.0
    theta = torch.tensor([[120.0, 40.0]])  # vmax, vhalf
    design = torch.tensor([[80.0]])

    mean = task.make_mean(theta, design)
    expected = 120.0 * 80.0 / (40.0 + 80.0)

    assert mean.shape == (1, 1)
    assert torch.allclose(mean, torch.tensor([[expected]]))


def test_make_mean_is_half_vmax_at_design_equal_vhalf():
    # num = vmax * vhalf^s, denom = 2 * vhalf^s, so mean = vmax / 2
    # regardless of the Hill exponent s.
    for exponent in (0.5, 1.0, 2.0, 3.0):
        task = make_task(exponent=exponent)
        theta = torch.tensor([[100.0, 30.0]])
        design = torch.tensor([[30.0]])

        mean = task.make_mean(theta, design)
        assert torch.allclose(mean, torch.tensor([[50.0]]), atol=1e-4)


def test_outcome_logp_matches_analytic_gaussian_density():
    noise_scale = 1.5
    task = make_task(noise_scale=noise_scale)
    theta = torch.tensor([[120.0, 40.0]])
    design = torch.tensor([[80.0]])
    y = torch.tensor([[75.0]])

    logp = task.outcome_logp(y, theta, design)

    mean = 120.0 * 80.0 / (40.0 + 80.0)
    expected = -0.5 * (
        ((75.0 - mean) ** 2) / noise_scale**2 + math.log(2 * math.pi * noise_scale**2)
    )

    assert torch.allclose(logp, torch.tensor([expected]), atol=1e-5)


def test_sample_joint_shapes():
    task = make_task()
    design = torch.rand(5, 1) * 100 + 20

    y, theta = task.sample_joint(design, sample_shape=4)
    assert y.shape == (4, 5, 1)
    assert theta.shape == (4, 5, 2)


def test_sample_outcome_is_reparametrized():
    task = make_task()
    design = (torch.rand(8, 1) * 100 + 20).requires_grad_(True)
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_mean_matches_conditional_mean():
    torch.manual_seed(0)
    task = make_task(noise_scale=0.5)
    theta = torch.tensor([[120.0, 40.0]])
    design = torch.tensor([[80.0]])

    samples = task.sample_outcome(theta, design, sample_shape=20_000)
    empirical_mean = samples.mean()
    expected_mean = task.make_mean(theta, design).squeeze()

    assert torch.allclose(empirical_mean, expected_mean, atol=0.05)


def test_default_design_space_is_the_concentration_box():
    task = make_task()

    assert isinstance(task.design_prior, Box)
    assert task.design_prior.event_shape == torch.Size([1])
    assert torch.equal(task.design_prior.low, torch.zeros(1))
    assert torch.equal(task.design_prior.high, torch.full((1,), 400.0))


def test_design_space_accepts_the_closed_box_and_rejects_outside():
    task = make_task()

    assert task.contains_design(torch.zeros(1))
    assert task.contains_design(torch.full((1,), 400.0))
    assert not task.contains_design(torch.full((1,), -1e-3))
    assert not task.contains_design(torch.full((1,), 400.001))


def test_task_rejects_negative_and_oversized_concentrations():
    task = make_task()
    theta = task.sample_prior(1)

    with pytest.raises(ValueError, match="Design lies outside"):
        task.sample_outcome(theta, torch.full((1, 1), -10.0))
    with pytest.raises(ValueError, match="Design lies outside"):
        task.sample_outcome(theta, torch.full((1, 1), 500.0))


def test_design_space_sample_and_project_stay_in_the_box():
    torch.manual_seed(0)
    task = make_task()
    space = task.design_prior

    assert task.contains_design(space.sample((64,)))
    assert task.contains_design(_project(space.support, torch.randn(64, 1) * 1000))


def test_reparametrisation_reaches_the_bounds_under_saturation():
    # tanh is open in exact arithmetic but saturates to exactly +/-1 in float32,
    # so a large raw value maps to exactly 0.0 or 400.0 -- the bounds are
    # reachable through the squash, not just through project(). That matters
    # for the x = 0 gradient singularity covered below.
    torch.manual_seed(0)
    task = make_task()

    design = task.design_transform(torch.randn(256, 1) * 10)

    assert task.contains_design(design)
    assert (design == 0.0).any()
    assert (design == 400.0).any()
    # Inverting a saturated design stays finite rather than blowing up.
    assert torch.isfinite(task.design_transform.inv(design)).all()


def test_mean_gradient_at_zero_concentration_depends_on_the_hill_exponent():
    # x = 0 is a valid design, but x^s has an infinite derivative there when
    # s < 1. Finite for the default s = 1, so this is only a hazard for a
    # fractional Hill exponent -- and x = 0 is reachable both by project() and
    # by a saturated to_design(), per the test above.
    theta = torch.tensor([[120.0, 40.0]])

    for exponent, expect_finite in ((1.0, True), (2.0, True), (0.5, False)):
        task = make_task(exponent=exponent)
        design = torch.zeros(1, 1, requires_grad=True)

        mean = task.make_mean(theta, design)
        mean.sum().backward()

        assert torch.allclose(mean, torch.zeros(1, 1))
        assert torch.isfinite(design.grad).all() == expect_finite
