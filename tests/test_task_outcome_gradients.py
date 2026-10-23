"""Tests that every public sampler gates the outcome's gradient to the design."""

import pytest
import torch
from torch.distributions import Independent, Normal

from testbed.core.distributions import TorchDistribution
from testbed.tasks import (
    CES,
    LinearRegression,
    LocationFinding,
    MichaelisMenten,
    PreyPopulation,
    TemporalDiscounting,
)
from testbed.tasks.base import Task


class _ReparameterizableTask(Task):
    """Gaussian likelihood whose sampler reparameterizes via ``rsample``."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self, reparameterize_outcomes=True, detached_sampler=False):
        prior = TorchDistribution(Independent(Normal(torch.zeros(2), torch.ones(2)), 1))
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[2],
            prior=prior,
            reparameterize_outcomes=reparameterize_outcomes,
        )
        self.detached_sampler = detached_sampler

    def _loc(self, theta, design):
        return theta.sum(dim=-1) + design.sum(dim=-1)

    def _outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        return Normal(self._loc(theta, design), 1.0).log_prob(y.squeeze(-1))

    def _sample_outcome(self, theta, design, y_history=None, design_history=None):
        distr = Normal(self._loc(theta, design), 1.0)
        draw = distr.sample() if self.detached_sampler else distr.rsample()
        return draw.unsqueeze(-1)


def _design(batch_shape=(4,)):
    return torch.randn(*batch_shape, 2, requires_grad=True)


@pytest.mark.parametrize("n_samples", [1, 3])
def test_sample_joint_detaches_when_reparameterize_is_off(n_samples):
    task = _ReparameterizableTask(reparameterize_outcomes=False)
    y, _ = task.sample_joint(_design(), n_samples=n_samples)
    assert not y.requires_grad


@pytest.mark.parametrize("n_samples", [1, 3])
def test_sample_joint_keeps_gradient_when_reparameterize_is_on(n_samples):
    task = _ReparameterizableTask(reparameterize_outcomes=True)
    design = _design()
    y, _ = task.sample_joint(design, n_samples=n_samples)
    assert y.requires_grad
    y.sum().backward()
    assert design.grad is not None and design.grad.abs().sum() > 0


def test_sample_joint_raises_when_sampler_is_not_reparameterized():
    task = _ReparameterizableTask(detached_sampler=True)
    with pytest.raises(ValueError, match="no gradient to the design"):
        task.sample_joint(_design())


def test_sample_marginal_detaches_when_reparameterize_is_off():
    task = _ReparameterizableTask(reparameterize_outcomes=False)
    assert not task.sample_marginal(_design()).requires_grad


def test_sample_outcome_and_sample_joint_agree_on_gating():
    task = _ReparameterizableTask(reparameterize_outcomes=False)
    theta = task.sample_prior(n_samples=4)
    design = _design()
    assert not task.sample_outcome(theta, design).requires_grad
    assert not task.sample_joint(design)[0].requires_grad


# Every pathwise task takes the flag, so any of them can be trained with
# REINFORCE for comparison.
_PATHWISE_TASKS = [
    lambda **kw: LocationFinding(n_sources=1, dim_theta=2, **kw),
    lambda **kw: LinearRegression(dim=2, **kw),
    lambda **kw: MichaelisMenten(**kw),
    lambda **kw: CES(**kw),
    lambda **kw: TemporalDiscounting(temperature=0.1, **kw),
]

# Tasks with no pathwise sampler, which must reject an explicit True.
_NON_PATHWISE_TASKS = [
    lambda **kw: PreyPopulation(**kw),
    lambda **kw: TemporalDiscounting(temperature=0.0, **kw),
]


@pytest.mark.parametrize("make_task", _PATHWISE_TASKS)
def test_pathwise_tasks_accept_reparameterize_outcomes(make_task):
    assert make_task().reparameterize_outcomes
    assert make_task(reparameterize_outcomes=True).reparameterize_outcomes
    assert not make_task(reparameterize_outcomes=False).reparameterize_outcomes


@pytest.mark.parametrize("make_task", _NON_PATHWISE_TASKS)
def test_non_pathwise_tasks_reject_reparameterize_outcomes(make_task):
    # None follows the task's own capability rather than raising.
    assert not make_task().reparameterize_outcomes
    assert not make_task(reparameterize_outcomes=False).reparameterize_outcomes
    with pytest.raises(ValueError, match="not reparameterizable"):
        make_task(reparameterize_outcomes=True)
