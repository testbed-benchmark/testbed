"""Tests that MyopicBarberAgakov conditions q on the whole history.

The bound subtracts ``task.current_prior``, i.e. ``p(theta | h)`` once a
``PosteriorUpdateHook`` has conditioned it on the history ``h``. So ``q`` has to
see ``h`` too:

    EIG(d | h) >= E[ log q(theta | h, d, y) - log p(theta | h) ]

Both sides must condition on the same thing. Feeding ``q`` only the new step
leaves the bound valid but slack, and feeds a net trained on full histories an
input it never saw in training.

Linear-Gaussian conjugacy makes that checkable: ``ExactPosterior`` returns the
true posterior for whatever history it is handed, which makes the bound tight,
so a correct estimator hits the analytic EIG exactly.
"""

import math

import pytest
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

from testbed.core.distributions import TorchDistribution
from testbed.design_objectives.barber_agakov import MyopicBarberAgakov
from testbed.tasks.base import Task

SIGMA = 0.7
DIM = 2
PRIOR_LOC = torch.zeros(DIM)
PRIOR_COV = torch.eye(DIM)


class LinearGaussian(Task):
    """y = theta . x + N(0, SIGMA^2), with a standard normal prior over theta."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[DIM],
            event_shape_design=[DIM],
            prior=TorchDistribution(Independent(Normal(PRIOR_LOC, torch.ones(DIM)), 1)),
        )

    def _outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        loc = (theta * design).sum(-1, keepdim=True)
        return Normal(loc, SIGMA).log_prob(y).sum(-1)

    def _sample_outcome(self, theta, design, y_history=None, design_history=None):
        loc = (theta * design).sum(-1, keepdim=True)
        return loc + SIGMA * torch.randn_like(loc)


def _posterior(y_history, design_history):
    """Exact Gaussian posterior from the standard normal prior; batched over rows."""
    X, y = design_history, y_history
    precision = torch.eye(DIM) + X.transpose(-1, -2) @ X / SIGMA**2
    cov = torch.linalg.inv(precision)
    mean = (cov @ (X.transpose(-1, -2) @ y) / SIGMA**2).squeeze(-1)
    return mean, cov


class ExactPosterior:
    """Posterior net that is exactly right, making the Barber-Agakov bound tight."""

    def log_prob(self, theta, y_history, design_history, mask=None):
        mean, cov = _posterior(y_history, design_history)
        return MultivariateNormal(mean, cov).log_prob(theta)


class SpyPosterior:
    """Records the histories it is asked to condition on."""

    def __init__(self):
        self.calls = []

    def log_prob(self, theta, y_history, design_history, mask=None):
        self.calls.append((y_history.clone(), design_history.clone()))
        return torch.zeros(theta.shape[0])


def _history(S, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, S, 1), torch.randn(1, S, DIM)


def _analytic_eig(design, cov):
    """0.5 * log(1 + x^T Sigma x / sigma^2), Sigma being the current posterior covariance."""
    return 0.5 * math.log(1 + (design @ cov @ design).item() / SIGMA**2)


@pytest.mark.parametrize("S", [1, 3])
def test_q_conditions_on_the_whole_history_not_just_the_new_step(S):
    task = LinearGaussian()
    spy = SpyPosterior()
    y_history, design_history = _history(S)
    design = torch.tensor([0.7, -0.4])

    MyopicBarberAgakov(task, spy, n_outer=5).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )

    seen_y, seen_design = spy.calls[-1]
    assert seen_y.shape == (5, S + 1, 1)
    assert seen_design.shape == (5, S + 1, DIM)
    # The history comes first, the new step last.
    assert torch.equal(seen_y[:, :S], y_history.expand(5, S, 1))
    assert torch.equal(seen_design[:, S], design.expand(5, DIM))


def test_call_forwards_the_history_to_estimate_gain():
    task = LinearGaussian()
    spy = SpyPosterior()
    y_history, design_history = _history(2)
    design = torch.tensor([0.7, -0.4])

    MyopicBarberAgakov(task, spy, n_outer=4)(
        design, y_history=y_history, design_history=design_history
    )

    seen_y, _ = spy.calls[-1]
    assert seen_y.shape == (4, 3, 1)


def test_no_history_still_conditions_on_the_single_step():
    task = LinearGaussian()
    spy = SpyPosterior()
    design = torch.tensor([0.7, -0.4])

    MyopicBarberAgakov(task, spy, n_outer=4).estimate_gain(design)

    seen_y, seen_design = spy.calls[-1]
    assert seen_y.shape == (4, 1, 1)
    assert seen_design.shape == (4, 1, DIM)


@pytest.mark.parametrize("S", [1, 4])
def test_bound_is_tight_against_analytic_eig_with_an_exact_posterior(S):
    # With q exact the Barber-Agakov gap is zero, so the estimate must land on
    # the analytic EIG under the *posterior given the history*. It only does so
    # if q conditions on that history as well.
    torch.manual_seed(0)
    task = LinearGaussian()
    y_history, design_history = _history(S, seed=1)
    mean_h, cov_h = _posterior(y_history, design_history)
    task.update_prior(TorchDistribution(MultivariateNormal(mean_h[0], cov_h[0])))

    design = torch.tensor([0.7, -0.4])
    gain = MyopicBarberAgakov(task, ExactPosterior(), n_outer=20000).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )

    assert gain.item() == pytest.approx(_analytic_eig(design, cov_h[0]), abs=0.02)


def test_conditioning_q_on_only_the_new_step_understates_the_bound():
    # Guards the direction of the fix: the same estimate with q given only the
    # new step is a strictly looser (smaller) bound, so the two cannot be
    # confused by a test that only checks finiteness.
    torch.manual_seed(0)
    task = LinearGaussian()
    S = 4
    y_history, design_history = _history(S, seed=1)
    mean_h, cov_h = _posterior(y_history, design_history)
    task.update_prior(TorchDistribution(MultivariateNormal(mean_h[0], cov_h[0])))
    design = torch.tensor([0.7, -0.4])

    class LastStepOnly(ExactPosterior):
        def log_prob(self, theta, y_history, design_history, mask=None):
            return super().log_prob(theta, y_history[:, -1:], design_history[:, -1:])

    torch.manual_seed(0)
    full = MyopicBarberAgakov(task, ExactPosterior(), n_outer=20000).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )
    torch.manual_seed(0)
    last = MyopicBarberAgakov(task, LastStepOnly(), n_outer=20000).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )

    assert last.item() < full.item()
    assert full.item() == pytest.approx(_analytic_eig(design, cov_h[0]), abs=0.02)
