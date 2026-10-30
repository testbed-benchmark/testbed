"""Tests for the gravimetry task."""

import math

import torch

from testbed.designers import RandomDesigner, StaticDesigner
from testbed.tasks import Gravimetry
from testbed.tasks.gravimetry import GravimetryPrior


def test_event_shapes_and_flags():
    task = Gravimetry()

    assert task.event_shape_y == torch.Size([1])
    assert task.event_shape_theta == torch.Size([3])
    assert task.event_shape_design == torch.Size([1])
    assert task.memoryless
    # designs are unbounded measurement locations along a line
    assert task.contains_design(torch.tensor([[-500.0], [500.0]]))


def test_kappa_coefficient_matches_the_physical_mapping():
    """kappa = G (4/3) pi R^3 delta_rho, rescaled from m/s^2 to microGal."""
    prior = GravimetryPrior()

    expected = 6.674e-11 * (4 / 3) * math.pi * -2200.0 / 1e-8

    assert prior.kappa_coeff == expected
    # the paper quotes kappa in [-3940, -210] microGal m^2 over R in [1.5, 4]
    assert math.isclose(prior.kappa_coeff * 4.0**3, -3936.2, rel_tol=1e-3)
    assert math.isclose(prior.kappa_coeff * 1.5**3, -207.6, rel_tol=1e-3)


def test_transform_matches_closed_form():
    prior = GravimetryPrior()
    u = torch.tensor([[0.5, -0.25, 0.75]])

    theta = prior.transform(u)

    phi = 0.5 * (1 + math.erf(0.75 / math.sqrt(2)))
    r = 1.5 + (4.0 - 1.5) * phi
    q = prior.z_floor_cdf + (1 - prior.z_floor_cdf) * 0.5 * (
        1 + math.erf(-0.25 / math.sqrt(2))
    )
    z = math.exp(math.log(10.0) + 0.5 * torch.special.ndtri(torch.tensor(q)).item())
    expected = torch.tensor([[30.0 * 0.5, z, prior.kappa_coeff * r**3]])

    assert torch.allclose(theta, expected, atol=1e-5)


def test_depth_is_truncated_not_clamped():
    """Truncation keeps a density, so no draw should pile up exactly at the floor."""
    torch.manual_seed(0)
    prior = GravimetryPrior()

    z = prior.sample(50_000)[:, 1]

    assert (z > prior.z_floor).all()  # strictly above, no atom
    assert z.min() < 4.0  # but the lower tail is still reached
    # the untruncated law puts ~0.8% below the floor; truncation removes exactly that
    assert math.isclose(prior.z_floor_cdf, 0.00802, abs_tol=1e-4)


def test_prior_sample_shapes_and_support():
    torch.manual_seed(0)
    prior = GravimetryPrior()

    theta = prior.sample(20_000)
    x0, z, kappa = theta[:, 0], theta[:, 1], theta[:, 2]
    r = (kappa / prior.kappa_coeff) ** (1 / 3)

    assert theta.shape == (20_000, 3)
    assert (z >= prior.z_floor).all()
    assert (kappa < 0).all()  # a void is a mass deficit
    assert (r >= prior.r_min - 1e-5).all() and (r <= prior.r_max + 1e-5).all()
    assert torch.allclose(x0.mean(), torch.tensor(0.0), atol=0.7)
    assert torch.allclose(x0.std(), torch.tensor(30.0), rtol=0.05)
    assert torch.allclose(z.median(), torch.tensor(10.0), rtol=0.05)


def _x0_density(prior, x0):
    return torch.exp(
        torch.distributions.Normal(prior.x0_loc, prior.x0_scale).log_prob(
            torch.tensor(x0)
        )
    )


def _z_density(prior, z):
    """Truncated log-normal density of the depth."""
    return torch.exp(
        torch.distributions.Normal(prior.z_loc, prior.z_scale).log_prob(
            torch.log(torch.tensor(z))
        )
        - math.log(z)
        - math.log1p(-prior.z_floor_cdf)
    )


def _kappa_density(prior, kappa):
    """Uniform radius pushed through kappa = c R^3."""
    r = (kappa / prior.kappa_coeff) ** (1 / 3)
    return 1.0 / ((prior.r_max - prior.r_min) * 3 * abs(prior.kappa_coeff) * r**2)


def test_log_prob_normalizes_over_the_depth():
    """A valid density is the point of truncating, so check it integrates to one.

    The density factorizes, so integrating over one component with the others
    held fixed should leave exactly the other two factors behind.
    """
    prior = GravimetryPrior()
    kappa0 = prior.kappa_coeff * 2.0**3
    z = torch.linspace(prior.z_floor, 400.0, 400_001)
    theta = torch.stack([torch.zeros_like(z), z, torch.full_like(z, kappa0)], dim=-1)

    integral = torch.trapz(torch.exp(prior.log_prob(theta)), z)

    assert torch.allclose(
        integral, _x0_density(prior, 0.0) * _kappa_density(prior, kappa0), rtol=2e-3
    )


def test_log_prob_normalizes_over_the_source_strength():
    prior = GravimetryPrior()
    lo = prior.kappa_coeff * prior.r_max**3
    hi = prior.kappa_coeff * prior.r_min**3
    kappa = torch.linspace(lo, hi, 200_001)[1:-1]
    theta = torch.stack(
        [torch.zeros_like(kappa), torch.full_like(kappa, 10.0), kappa], dim=-1
    )

    integral = torch.trapz(torch.exp(prior.log_prob(theta)), kappa)

    assert torch.allclose(
        integral, _x0_density(prior, 0.0) * _z_density(prior, 10.0), rtol=1e-3
    )


def test_log_prob_is_minus_inf_outside_the_support():
    prior = GravimetryPrior()
    good = prior.sample(1)

    too_shallow = good.clone()
    too_shallow[:, 1] = 1.0  # below the floor
    too_strong = good.clone()
    too_strong[:, 2] = prior.kappa_coeff * (prior.r_max + 1) ** 3

    assert torch.isfinite(prior.log_prob(good)).all()
    assert torch.isinf(prior.log_prob(too_shallow)).all()
    assert torch.isinf(prior.log_prob(too_strong)).all()


def test_make_mean_matches_closed_form():
    task = Gravimetry()
    theta = torch.tensor([[10.0, 8.0, -1200.0]])  # x0, z, kappa
    design = torch.tensor([[25.0]])

    x0, z, kappa, x = 10.0, 8.0, -1200.0, 25.0
    expected = kappa * z / ((x - x0) ** 2 + z**2) ** 1.5

    assert torch.allclose(task.make_mean(theta, design), torch.tensor([[expected]]))


def test_signal_peaks_directly_above_the_anomaly_and_decays_with_offset():
    task = Gravimetry()
    theta = torch.tensor([[0.0, 10.0, -1200.0]]).expand(4, -1)
    design = torch.tensor([[0.0], [10.0], [50.0], [200.0]])

    g = task.make_mean(theta, design).abs().squeeze(-1)

    assert (g[:-1] > g[1:]).all()  # strictly decays as we move away
    assert g[-1] < 1e-2  # far away the anomaly is invisible against sigma = 1


def test_outcome_logp_matches_analytic_gaussian_density():
    task = Gravimetry(noise_scale=2.0)
    theta = torch.tensor([[10.0, 8.0, -1200.0]])
    design = torch.tensor([[25.0]])
    y = torch.tensor([[-0.5]])

    logp = task.outcome_logp(y, theta, design)

    loc = task.make_mean(theta, design)
    expected = torch.distributions.Normal(loc, 2.0).log_prob(y).sum(dim=-1)

    assert torch.allclose(logp, expected, atol=1e-5)


def test_sample_outcome_is_reparametrized():
    task = Gravimetry()
    theta = task.sample_prior(8)
    design = (torch.rand(8, 1) * 100 - 50).requires_grad_(True)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_sample_outcome_matches_conditional_moments():
    torch.manual_seed(0)
    task = Gravimetry()
    theta = torch.tensor([[10.0, 8.0, -1200.0]]).expand(100_000, -1)
    design = torch.tensor([[25.0]]).expand(100_000, -1)

    samples = task.sample_outcome(theta, design)
    loc = task.make_mean(theta[:1], design[:1])

    assert torch.allclose(samples.mean(), loc.squeeze(), atol=0.02)
    assert torch.allclose(samples.std(), torch.tensor(1.0), rtol=0.02)


def test_random_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = Gravimetry()
    designer = RandomDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 1)
    assert rollout.design_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()


def test_static_designer_rollout_end_to_end():
    torch.manual_seed(0)
    task = Gravimetry()
    designer = StaticDesigner(task, T=5)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert rollout.y_history.shape == (4, 5, 1)
    logp = task.history_logp(rollout.y_history, rollout.design_history, rollout.theta)
    assert torch.isfinite(logp).all()
