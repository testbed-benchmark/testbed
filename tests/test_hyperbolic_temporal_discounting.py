import math

import torch

from testbed.tasks import HyperbolicTemporalDiscounting
from testbed.tasks.hyperbolic_temporal_discounting import HTDPrior


def make_prior(**kwargs):
    return HTDPrior(**kwargs)


def make_task(**kwargs):
    return HyperbolicTemporalDiscounting(**kwargs)


def test_event_shapes_and_discreteness():
    task = make_task()
    assert task.event_shape_theta == torch.Size([2])
    assert task.event_shape_design == torch.Size([2])
    assert task.event_shape_y == torch.Size([1])
    assert task.discrete_y is True


def test_task_uses_default_prior():
    task = make_task()
    assert isinstance(task.prior, HTDPrior)
    assert torch.allclose(task.prior.k_loc, torch.tensor(-4.25))
    assert torch.allclose(task.prior.k_scale, torch.tensor(math.sqrt(1.5)))
    assert torch.allclose(task.prior.alpha_scale, torch.tensor(math.sqrt(2)))


def test_prior_sample_shapes():
    prior = make_prior()

    assert prior.sample(1).shape == (1, 2)
    assert prior.sample(10).shape == (10, 2)


def test_prior_logprob_matches_lognormal_and_halfnormal():
    prior = make_prior()
    reference_k = torch.distributions.LogNormal(-4.25, math.sqrt(1.5))
    reference_alpha = torch.distributions.HalfNormal(math.sqrt(2))

    theta = prior.sample(10)
    k, alpha = theta[:, 0], theta[:, 1]
    expected = reference_k.log_prob(k) + reference_alpha.log_prob(alpha)

    assert torch.allclose(prior.log_prob(theta), expected, atol=1e-5)


def test_make_prob_at_indifference_point():
    # V1 = 100 / (1 + (k * D1)) = 100 / (1 + 1.0) = 50 = R0, so cdf_in = 0
    # regardless of alpha, giving p(y=1) = epsilon + (1 - 2*epsilon) * 0.5 = 0.5.
    task = make_task()
    theta = torch.tensor([[0.05, 1.0]])  # k, alpha
    design = torch.tensor([[20.0, 50.0]])  # D1, R0

    p_y1 = task.make_prob(theta, design)
    assert torch.allclose(p_y1, torch.tensor([[0.5]]), atol=1e-6)


def test_make_prob_matches_analytic_formula():
    epsilon = 0.01
    task = make_task(epsilon=epsilon)
    k, alpha, D1, R0 = 0.02, 2.0, 10.0, 80.0
    theta = torch.tensor([[k, alpha]])
    design = torch.tensor([[D1, R0]])

    V1 = 100.0 / (1.0 + k * D1)
    cdf_in = (V1 - R0) / alpha
    phi = 0.5 * (1.0 + math.erf(cdf_in / math.sqrt(2)))
    expected = epsilon + (1.0 - 2 * epsilon) * phi

    p_y1 = task.make_prob(theta, design)
    assert torch.allclose(p_y1, torch.tensor([[expected]]), atol=1e-6)


def test_make_prob_bounded_by_epsilon():
    torch.manual_seed(0)
    task = make_task()
    theta = task.sample_prior(64)
    design = torch.randn(64, 2) * 50

    p_y1 = task.make_prob(theta, design)
    assert (p_y1 >= task.epsilon - 1e-6).all()
    assert (p_y1 <= 1.0 - task.epsilon + 1e-6).all()


def test_outcome_logp_matches_bernoulli_logpmf_for_batch():
    # Regression test: previously, squeezing y to [b] while p_y1/log_p_y1/log_p_y0
    # stayed [b, 1] made torch.where broadcast to [b, b] instead of [b].
    epsilon = 0.01
    task = make_task(epsilon=epsilon)
    theta = torch.tensor([[0.05, 1.0], [0.05, 1.0], [0.02, 2.0], [0.02, 2.0]])
    design = torch.tensor([[20.0, 50.0], [20.0, 50.0], [10.0, 80.0], [10.0, 80.0]])
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    logp = task.outcome_logp(y, theta, design)
    assert logp.shape == (4,)

    p_indiff = 0.5
    V1 = 100.0 / (1.0 + 0.02 * 10.0)
    phi = 0.5 * (1.0 + math.erf(((V1 - 80.0) / 2.0) / math.sqrt(2)))
    p_asym = epsilon + (1.0 - 2 * epsilon) * phi

    expected = torch.tensor(
        [
            math.log(p_indiff),
            math.log(1.0 - p_indiff),
            math.log(p_asym),
            math.log(1.0 - p_asym),
        ]
    )
    assert torch.allclose(logp, expected, atol=1e-5)


def test_sample_outcome_returns_binary_values():
    torch.manual_seed(0)
    task = make_task()
    theta = task.sample_prior(50)
    design = torch.randn(50, 2)

    y = task.sample_outcome(theta, design)
    assert y.shape == (50, 1)
    assert set(y.unique().tolist()) <= {0.0, 1.0}


def test_sample_outcome_mean_matches_bernoulli_probability():
    torch.manual_seed(0)
    task = make_task()
    theta = torch.tensor([[0.05, 1.0]])
    design = torch.tensor([[20.0, 50.0]])  # indifference point, p(y=1) = 0.5

    samples = task.sample_outcome(theta, design, n_samples=20_000)
    assert torch.allclose(samples.mean(), torch.tensor(0.5), atol=0.02)


def test_sample_outcome_relaxed_bernoulli_is_reparametrized():
    # Designs are kept near the indifference point (D1=20, R0=50 for k=0.05):
    # far from it, normal_cdf saturates and the clamp kills the gradient.
    task = make_task(temperature=0.5)
    theta = torch.tensor([[0.05, 1.0]] * 4)
    design = torch.tensor(
        [[20.0, 50.0], [21.0, 49.0], [19.0, 51.0], [20.5, 50.5]],
        requires_grad=True,
    )

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()
