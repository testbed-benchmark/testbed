"""Tests for the Distribution shape contract and the Task samplers built on it."""

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal, biject_to

from testbed.core.distributions import (
    Distribution,
    EmpiricalDistribution,
    LogNormal,
    TorchDistribution,
    as_sample_shape,
)
from testbed.tasks import (
    CES,
    CartPole,
    DoubleLink,
    Gravimetry,
    LinearRegression,
    LocationFinding,
    MichaelisMenten,
    Pharmacokinetic,
    PreyPopulation,
    StochasticPendulum,
    Task,
    TemporalDiscounting,
)

TASKS = [
    CES,
    CartPole,
    DoubleLink,
    Gravimetry,
    lambda: LinearRegression(dim=3),
    lambda: LocationFinding(n_sources=2, dim_theta=3),
    MichaelisMenten,
    Pharmacokinetic,
    PreyPopulation,
    StochasticPendulum,
    TemporalDiscounting,
]
SAMPLE_SHAPES = [(), 1, (4,), (2, 3)]


@pytest.mark.parametrize("make_task", TASKS)
@pytest.mark.parametrize("sample_shape", SAMPLE_SHAPES)
def test_prior_honours_shape_contract(make_task, sample_shape):
    task = make_task()
    prior = task.prior

    theta = task.sample_prior(sample_shape)

    assert prior.event_shape == task.event_shape_theta
    assert theta.shape == (*as_sample_shape(sample_shape), *task.event_shape_theta)
    assert prior.log_prob(theta).shape == as_sample_shape(sample_shape)


@pytest.mark.parametrize("make_task", TASKS)
def test_prior_is_a_torch_distribution_whose_samples_lie_in_its_support(make_task):
    task = make_task()
    theta = task.sample_prior((256,))

    assert isinstance(task.prior, torch.distributions.Distribution)
    assert task.prior.support.check(theta).all()
    assert not task.discrete_theta


@pytest.mark.parametrize("make_task", TASKS)
def test_support_bijection_round_trips_through_raw_coordinates(make_task):
    torch.manual_seed(0)
    task = make_task()
    theta = task.sample_prior((256,))
    transform = biject_to(task.prior.support)

    raw = transform.inv(theta)
    raw_shape = transform.inverse_shape(task.event_shape_theta)

    assert raw.shape == (256, *raw_shape)
    assert torch.isfinite(raw).all()
    torch.testing.assert_close(transform(raw), theta, rtol=1e-4, atol=1e-4)
    # Every raw point maps back into the support, so raw coordinates are free.
    assert task.prior.support.check(transform(3 * torch.randn(256, *raw_shape))).all()


def test_ces_support_bijection_maps_four_raw_coordinates_onto_the_simplex():
    transform = biject_to(CES().prior.support)
    raw = torch.randn(64, 4)

    theta = transform(raw)

    assert transform.inverse_shape((5,)) == (4,)
    torch.testing.assert_close(theta[:, 1:4].sum(-1), torch.ones(64))
    assert ((theta[:, 0] > 0.01) & (theta[:, 0] < 1.0)).all()


@pytest.mark.parametrize("make_task", TASKS)
def test_prior_log_prob_agrees_across_batch_shapes(make_task):
    # A log_prob that indexes a fixed batch dim (e.g. theta[:, 0]) returns the
    # wrong values once the batch has more than one dim.
    task = make_task()
    theta = task.sample_prior((2, 3))

    nested = task.prior.log_prob(theta)
    flat = task.prior.log_prob(theta.reshape(6, *task.event_shape_theta))

    torch.testing.assert_close(nested.reshape(6), flat)


@pytest.mark.parametrize("make_task", TASKS)
@pytest.mark.parametrize("sample_shape", SAMPLE_SHAPES)
def test_samplers_prepend_sample_shape_and_never_squeeze(make_task, sample_shape):
    task = make_task()
    shape = as_sample_shape(sample_shape)
    design = task.design_prior.sample((5,))
    theta = task.sample_prior(5)

    y = task.sample_outcome(theta, design, sample_shape=sample_shape)
    y_joint, theta_joint = task.sample_joint(design, sample_shape=sample_shape)

    assert y.shape == (*shape, 5, *task.event_shape_y)
    assert y_joint.shape == (*shape, 5, *task.event_shape_y)
    assert theta_joint.shape == (*shape, 5, *task.event_shape_theta)
    assert task.sample_current_prior(sample_shape).shape == (
        *shape,
        *task.event_shape_theta,
    )


def test_torch_distribution_folds_batch_dims_into_the_event():
    # One MVN per source; the prior over all sources sums their log-probs.
    loc = torch.randn(3, 2)
    mvn = MultivariateNormal(loc, torch.eye(2))
    prior = TorchDistribution(mvn)
    theta = prior.sample((5,))

    assert prior.event_shape == (3, 2)
    assert prior.batch_shape == ()
    assert theta.shape == (5, 3, 2)
    torch.testing.assert_close(prior.log_prob(theta), mvn.log_prob(theta).sum(-1))


def test_empirical_distribution_honours_sample_shape():
    dist = EmpiricalDistribution(torch.randn(10, 2, 3))

    assert dist.event_shape == (2, 3)
    assert dist.sample().shape == (2, 3)
    assert dist.sample((4, 5)).shape == (4, 5, 2, 3)


class _PriorTask(Task):
    reparameterizable_outcomes = True

    def __init__(self, prior):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[3, 2],
            event_shape_design=[2],
            prior=prior,
        )

    def _outcome_logp(self, y, theta, design, y_history, design_history):
        raise NotImplementedError

    def _sample_outcome(self, theta, design, y_history, design_history):
        raise NotImplementedError


class _UnreducedPrior(Distribution):
    """Returns one log-prob per source instead of summing them."""

    def __init__(self):
        super().__init__(event_shape=(3, 2))

    def sample(self, sample_shape=()):
        return torch.randn(self._extended_shape(sample_shape))

    def log_prob(self, theta):
        return Normal(0.0, 1.0).log_prob(theta).sum(-1)


def test_task_accepts_a_batched_torch_prior():
    task = _PriorTask(
        TorchDistribution(MultivariateNormal(torch.zeros(3, 2), torch.eye(2)))
    )

    assert task.prior.log_prob(task.sample_prior(4)).shape == (4,)


def test_task_rejects_a_prior_with_the_wrong_event_shape():
    prior = TorchDistribution(MultivariateNormal(torch.zeros(2), torch.eye(2)))

    with pytest.raises(ValueError, match="event_shape"):
        _PriorTask(prior)


def test_task_rejects_a_prior_with_a_batch_shape():
    # a bare torch distribution can have one; testbed's own cannot
    prior = MultivariateNormal(torch.zeros(3, 2), torch.eye(2))

    with pytest.raises(ValueError, match="batch_shape"):
        _PriorTask(prior)


def test_task_rejects_a_prior_whose_log_prob_keeps_event_dims():
    with pytest.raises(ValueError, match="reduce over all event dims"):
        _PriorTask(_UnreducedPrior())


def test_update_prior_checks_the_event_shape():
    task = _PriorTask(TorchDistribution(Normal(torch.zeros(3, 2), 1.0)))

    with pytest.raises(ValueError, match="event_shape"):
        task.update_prior(TorchDistribution(Normal(torch.zeros(2), 1.0)))


def test_update_prior_checks_one_draw():
    task = _PriorTask(TorchDistribution(Normal(torch.zeros(3, 2), 1.0)))

    with pytest.raises(ValueError, match="reduce over all event dims"):
        task.update_prior(_UnreducedPrior())


def test_lognormal_samples_its_event_shape():
    dist = LogNormal(torch.zeros(2), torch.eye(2))

    assert dist.batch_shape == ()
    assert dist.event_shape == (2,)
    assert dist.sample((5,)).shape == dist._extended_shape((5,)) == (5, 2)


def test_constructing_a_task_leaves_the_rng_untouched():
    torch.manual_seed(0)
    expected = torch.rand(3)

    torch.manual_seed(0)
    LocationFinding(n_sources=2, dim_theta=3)

    torch.testing.assert_close(torch.rand(3), expected)
