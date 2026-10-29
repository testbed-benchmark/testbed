"""Tests for the distribution adapters in testbed.core.distributions."""

import pytest
import torch
from torch.distributions import Gamma, Normal

from testbed.core.distributions import LogNormal, TorchDistribution


def make_lognormal():
    """A non-isotropic, non-zero-mean case, so no symmetry can hide a bug."""
    loc = torch.tensor([0.5, -1.0, 2.0])
    cov = torch.tensor([[0.20, 0.05, 0.00], [0.05, 0.10, 0.02], [0.00, 0.02, 0.30]])
    return LogNormal(loc, cov), loc, cov


def test_sample_shape_and_positivity():
    torch.manual_seed(0)
    dist, _, _ = make_lognormal()

    samples = dist.sample(1000)

    assert dist.event_shape == (3,)
    assert samples.shape == (1000, 3)
    assert (samples > 0).all()


def test_log_prob_matches_change_of_variables():
    torch.manual_seed(0)
    dist, loc, cov = make_lognormal()
    theta = dist.sample(500)

    log_theta = torch.log(theta)
    expected = torch.distributions.MultivariateNormal(loc, cov).log_prob(
        log_theta
    ) - log_theta.sum(dim=-1)  # exp() Jacobian

    assert torch.allclose(dist.log_prob(theta), expected, atol=1e-4)


def test_sample_moments_match_analytic_lognormal():
    torch.manual_seed(0)
    dist, loc, cov = make_lognormal()

    samples = dist.sample(400_000)
    expected_mean = torch.exp(loc + 0.5 * torch.diagonal(cov))

    assert torch.allclose(samples.mean(dim=0), expected_mean, rtol=0.02)


def test_to_converts_the_wrapped_distribution():
    dist, loc, _ = make_lognormal()

    dist.double()

    assert dist.dist.base_dist.loc.dtype == torch.float64
    assert dist.sample(4).dtype == torch.float64
    torch.testing.assert_close(dist.dist.base_dist.loc, loc.double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_to_moves_the_wrapped_distribution_and_its_device():
    dist = TorchDistribution(Normal(torch.zeros(2), 1.0)).to("cuda")

    assert dist.device.type == "cuda"
    assert dist.dist.base_dist.loc.device.type == "cuda"
    assert dist.sample(3).device.type == "cuda"
    assert dist.log_prob(torch.zeros(3, 2, device="cuda")).device.type == "cuda"


def test_log_prob_is_minus_inf_outside_the_support():
    dist, _, _ = make_lognormal()

    log_prob = dist.log_prob(torch.tensor([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]))

    assert log_prob[0] == -torch.inf
    assert torch.isfinite(log_prob[1])


def test_accepts_non_tensor_parameters():
    dist = LogNormal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    assert dist.event_shape == (2,)
    assert dist.sample(4).shape == (4, 2)


def test_wrapping_a_validating_distribution_gives_minus_inf_rather_than_raising():
    # validate explicitly: torch.compile turns torch's default off process-wide.
    # The wrapper also folds Gamma's batch dim via Independent.
    dist = TorchDistribution(Gamma(torch.full((2,), 2.0), 1.0, validate_args=True))

    log_prob = dist.log_prob(torch.tensor([[1.0, 2.0], [-1.0, 2.0]]))

    assert torch.isfinite(log_prob[0])
    assert log_prob[1] == -torch.inf
