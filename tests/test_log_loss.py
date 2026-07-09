"""Tests for the LogLoss design objective."""

import pytest
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

from testbed.core.distributions import TorchDistribution
from testbed.design_objectives import LogLoss
from testbed.tasks import LocationFinding
from testbed.tasks.base import Task


def _gaussian_design_distr(dim):
    return TorchDistribution(MultivariateNormal(torch.zeros(dim), torch.eye(dim)))


def _make_location_finding_setup(n_sources=1):
    task = LocationFinding(n_sources=n_sources, dim_theta=2)
    design_distr = _gaussian_design_distr(task.event_shape_design.numel())
    return task, design_distr


class _ToyGaussianTask(Task):
    """Minimal task with a closed-form Gaussian likelihood and a *multi-dim*
    ``event_shape_design`` (unlike every real task in this repo, which is
    flat). Used to regression-test that LogLoss doesn't collapse the design's
    event shape via a bare ``reshape(..., -1)``.
    """

    def __init__(self):
        prior = TorchDistribution(Independent(Normal(torch.zeros(2), torch.ones(2)), 1))
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[2, 3],
            prior=prior,
        )

    def _loc(self, theta, design):
        return theta.sum(dim=-1) + design.flatten(start_dim=-2).sum(dim=-1)

    def _outcome_logp(self, y, theta, design):
        return Normal(self._loc(theta, design), 1.0).log_prob(y.squeeze(-1))

    def _sample_outcome(self, theta, design):
        return Normal(self._loc(theta, design), 1.0).sample().unsqueeze(-1)


def _toy_design_distr():
    return TorchDistribution(
        Independent(Normal(torch.zeros(2, 3), torch.ones(2, 3)), 2)
    )


# ============================================================================
# Validation
# ============================================================================


def test_rejects_invalid_reduction():
    task, design_distr = _make_location_finding_setup()
    with pytest.raises(ValueError, match="Unsupported reduction"):
        LogLoss(design_distr, task, n_outer_samples=4, reduction="max")


def test_rejects_mismatched_shapes():
    task, design_distr = _make_location_finding_setup()
    loss_fn = LogLoss(design_distr, task, n_outer_samples=4)
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(torch.randn(2, 1, 2), torch.randn(2, 2))


# ============================================================================
# Shapes across reductions and theta event shapes
# ============================================================================


@pytest.mark.parametrize("n_sources", [1, 2])
@pytest.mark.parametrize(
    "reduction,expected_shape_fn",
    [
        ("mean", lambda b, n: ()),
        ("sum", lambda b, n: ()),
        ("none", lambda b, n: (b, n)),
    ],
)
def test_shapes_across_reductions_and_theta_event_shape(
    n_sources, reduction, expected_shape_fn
):
    torch.manual_seed(0)
    task, design_distr = _make_location_finding_setup(n_sources=n_sources)
    b, n_outer = 3, 5
    loss_fn = LogLoss(design_distr, task, n_outer_samples=n_outer, reduction=reduction)

    theta_true = task.sample_prior(b)
    theta_hat = theta_true + 0.1 * torch.randn_like(theta_true)

    loss = loss_fn(theta_hat, theta_true)

    assert loss.shape == expected_shape_fn(b, n_outer)
    assert torch.isfinite(loss).all()


def test_none_reduction_is_consistent_with_mean_and_sum():
    torch.manual_seed(1)
    task, design_distr = _make_location_finding_setup()
    b, n_outer = 4, 8

    theta_true = task.sample_prior(b)
    theta_hat = theta_true + 0.1 * torch.randn_like(theta_true)

    torch.manual_seed(2)
    none_loss = LogLoss(design_distr, task, n_outer, reduction="none")(
        theta_hat, theta_true
    )
    torch.manual_seed(2)
    mean_loss = LogLoss(design_distr, task, n_outer, reduction="mean")(
        theta_hat, theta_true
    )
    torch.manual_seed(2)
    sum_loss = LogLoss(design_distr, task, n_outer, reduction="sum")(
        theta_hat, theta_true
    )

    torch.testing.assert_close(none_loss.mean(), mean_loss)
    torch.testing.assert_close(none_loss.sum(), sum_loss)


# ============================================================================
# Gradients
# ============================================================================


@pytest.mark.parametrize("n_sources", [1, 2])
def test_gradient_flows_through_input(n_sources):
    torch.manual_seed(3)
    task, design_distr = _make_location_finding_setup(n_sources=n_sources)
    loss_fn = LogLoss(design_distr, task, n_outer_samples=6)

    theta_true = task.sample_prior(4)
    theta_hat = (theta_true + 0.1 * torch.randn_like(theta_true)).requires_grad_(True)

    loss_fn(theta_hat, theta_true).backward()

    assert theta_hat.grad is not None
    assert torch.isfinite(theta_hat.grad).all()


def test_target_does_not_require_grad_from_loss():
    # target is the "true" theta; nothing about LogLoss should force a graph
    # through it (it's only used to sample y, not differentiated).
    torch.manual_seed(4)
    task, design_distr = _make_location_finding_setup()
    loss_fn = LogLoss(design_distr, task, n_outer_samples=4)

    theta_true = task.sample_prior(3)
    theta_hat = theta_true.clone().requires_grad_(True)

    loss = loss_fn(theta_hat, theta_true)

    assert not theta_true.requires_grad
    assert loss.requires_grad


# ============================================================================
# Correctness
# ============================================================================


def test_matches_manual_computation():
    # Directly pins down the broadcasting/shape contract: LogLoss's forward
    # should be exactly -task.outcome_logp(y, theta_hat, design) for y sampled
    # at the true theta over independently-drawn designs.
    torch.manual_seed(5)
    task, design_distr = _make_location_finding_setup()
    b, n_outer = 5, 7

    theta_true = task.sample_prior(b)
    theta_hat = theta_true + 0.2 * torch.randn_like(theta_true)

    torch.manual_seed(6)
    loss = LogLoss(design_distr, task, n_outer, reduction="none")(theta_hat, theta_true)

    torch.manual_seed(6)
    design_samples = design_distr.sample(b * n_outer).reshape(
        b, n_outer, *task.event_shape_design
    )
    target = theta_true.unsqueeze(1)
    y_samples = task.sample_outcome(target, design_samples)
    input_ = theta_hat.unsqueeze(1)
    expected = -task.outcome_logp(y_samples, input_, design_samples)

    torch.testing.assert_close(loss, expected)


def test_prefers_correct_theta_over_wrong_theta():
    # A theta_hat equal to the true theta should, in expectation, achieve a
    # lower (better) loss than a theta_hat drawn from the prior independently
    # of the true theta.
    torch.manual_seed(7)
    task, design_distr = _make_location_finding_setup()
    loss_fn = LogLoss(design_distr, task, n_outer_samples=256, reduction="mean")

    theta_true = task.sample_prior(64)
    theta_wrong = task.sample_prior(64)

    correct_loss = loss_fn(theta_true, theta_true)
    wrong_loss = loss_fn(theta_wrong, theta_true)

    assert correct_loss.item() < wrong_loss.item()


# ============================================================================
# Multi-dim event_shape_design (regression test for the reshape fix)
# ============================================================================


def test_supports_multi_dim_event_shape_design():
    # Every real task in this repo has a flat event_shape_design, so this
    # exercises a path a naive `design_samples.reshape(b, n, -1)` would get
    # wrong (or crash on) once the design's event shape has more than one dim.
    torch.manual_seed(8)
    task = _ToyGaussianTask()
    design_distr = _toy_design_distr()
    b, n_outer = 4, 6
    loss_fn = LogLoss(design_distr, task, n_outer_samples=n_outer, reduction="none")

    theta_true = task.sample_prior(b)
    theta_hat = (theta_true + 0.1 * torch.randn_like(theta_true)).requires_grad_(True)

    loss = loss_fn(theta_hat, theta_true)

    assert loss.shape == (b, n_outer)
    assert torch.isfinite(loss).all()

    loss.mean().backward()
    assert torch.isfinite(theta_hat.grad).all()
