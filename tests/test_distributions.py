"""Tests for the distribution adapters in testbed.core.distributions."""

import torch

from testbed.core.distributions import LogNormal


def make_lognormal():
    """A non-isotropic, non-zero-mean case, so no symmetry can hide a bug."""
    loc = torch.tensor([0.5, -1.0, 2.0])
    cov = torch.tensor([[0.20, 0.05, 0.00], [0.05, 0.10, 0.02], [0.00, 0.02, 0.30]])
    return LogNormal(loc, cov), loc, cov


def test_sample_shape_and_positivity():
    torch.manual_seed(0)
    dist, _, _ = make_lognormal()

    samples = dist.sample(1000)

    assert dist.dim == 3
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


def test_parameters_are_buffers_so_the_module_moves_and_serializes():
    dist, loc, cov = make_lognormal()

    assert sorted(dict(dist.named_buffers())) == ["cov", "loc"]

    other = LogNormal(torch.zeros(3), torch.eye(3))
    other.load_state_dict(dist.state_dict())

    assert torch.equal(other.loc, loc)
    assert torch.equal(other.cov, cov)


def test_accepts_non_tensor_parameters():
    dist = LogNormal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    assert dist.dim == 2
    assert dist.sample(4).shape == (4, 2)
