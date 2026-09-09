"""Tests that myopic PCE/NMC treat a prior history as conditioning context.

The contrastive draws come from ``task.current_prior``, which a
``PosteriorUpdateHook`` has already conditioned on the history. The history
must therefore be conditioned on but *not* scored: scoring it re-applies
``p(y_{1:S} | theta)`` to draws that already carry it.

``ShiftedLinear`` makes that checkable. Its mean shifts by a known multiple of
the last observation, so conditioning on any history leaves the EIG at the
memoryless ``0.5 * log(1 + ||x||^2 / sigma^2)``: a correct estimator returns
the same number for every history, and one that mis-handles the history does not.
"""

import math

import pytest
import torch
from torch.distributions import Independent, Normal

from testbed.core.distributions import TorchDistribution
from testbed.core.rollout import Rollout
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.design_objectives.snmc import NMC
from testbed.design_objectives.spce import PCE
from testbed.tasks.base import Task

SIGMA = 0.5
SHIFT = 1.7


class ShiftedLinear(Task):
    r"""y_t ~ N(theta . x + SHIFT * y_{t-1}, SIGMA^2), with y_0 := 0.

    The history enters only as a known additive shift, so it changes the mean
    but carries no information about theta: EIG(x | history) is the memoryless
    value for every history.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[2],
            prior=TorchDistribution(
                Independent(Normal(torch.zeros(2), torch.ones(2)), 1)
            ),
            memoryless=False,
        )

    def _loc(self, theta, design, y_history):
        if y_history is None or y_history.shape[-2] == 0:
            prev = torch.zeros(theta.shape[0], 1)
        else:
            prev = y_history[..., -1, :]
        return (theta * design).sum(-1, keepdim=True) + SHIFT * prev

    def _outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        return Normal(self._loc(theta, design, y_history), SIGMA).log_prob(y).sum(-1)

    def _sample_outcome(self, theta, design, y_history=None, design_history=None):
        return self._loc(theta, design, y_history) + SIGMA * torch.randn(
            theta.shape[0], 1
        )


def _analytic_eig(design):
    """0.5 * log(1 + ||x||^2 / sigma^2), for a standard normal prior over theta."""
    return 0.5 * math.log(1 + (design @ design).item() / SIGMA**2)


def _histories(last_y):
    y_history = torch.tensor([[[0.1], [0.2], [last_y]]])
    design_history = torch.zeros(1, 3, 2)
    return y_history, design_history


# A history whose last outcome is large moves the mean far from where an
# unconditioned likelihood expects it, so a dropped or double-counted history
# shows up as a wildly wrong gain rather than a subtle bias.
@pytest.mark.parametrize("last_y", [0.0, 4.0, 40.0])
@pytest.mark.parametrize("estimator", [PCE, NMC])
def test_myopic_gain_is_invariant_to_a_purely_additive_history(estimator, last_y):
    torch.manual_seed(0)
    task = ShiftedLinear()
    design = torch.tensor([0.7, -0.4])
    y_history, design_history = _histories(last_y)

    gain = estimator(task, n_outer=1500, n_inner=1500).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )

    assert gain.item() == pytest.approx(_analytic_eig(design), abs=0.1)


def test_pce_is_a_lower_bound_and_nmc_an_upper_bound_given_a_history():
    task = ShiftedLinear()
    design = torch.tensor([0.7, -0.4])
    y_history, design_history = _histories(4.0)

    torch.manual_seed(0)
    lower = PCE(task, n_outer=1500, n_inner=1500).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )
    torch.manual_seed(0)
    upper = NMC(task, n_outer=1500, n_inner=1500).estimate_gain(
        design, y_history=y_history, design_history=design_history
    )

    assert lower.item() <= upper.item()


def test_context_free_scoring_is_unchanged():
    # The no-context path must still be exactly history_logp, so SPCE/SNMC over
    # a full rollout are untouched by the context plumbing.
    task = ShiftedLinear()
    b, T = 5, 3
    rollout = Rollout(
        theta=torch.randn(b, 2),
        y_history=torch.randn(b, T, 1),
        design_history=torch.randn(b, T, 2),
    )

    logp_joint, _ = contrastive_logsumexps(task, rollout, n_contrastive=4)

    assert torch.equal(
        logp_joint,
        task.history_logp(rollout.y_history, rollout.design_history, rollout.theta),
    )


def test_a_context_is_conditioned_on_but_never_scored():
    # Scoring the context would add log p(y_context | theta) to every term.
    task = ShiftedLinear()
    b, S = 4, 2
    theta = torch.randn(b, 2)
    y_context, design_context = torch.randn(b, S, 1), torch.randn(b, S, 2)
    y_history, design_history = torch.randn(b, 1, 1), torch.randn(b, 1, 2)

    with_context = task.history_logp(
        y_history,
        design_history,
        theta,
        y_context=y_context,
        design_context=design_context,
    )
    scored_together = task.history_logp(
        torch.cat([y_context, y_history], dim=1),
        torch.cat([design_context, design_history], dim=1),
        theta,
    )
    context_only = task.history_logp(y_context, design_context, theta)

    # Scoring the whole thing is exactly the context's own log-prob more.
    assert torch.allclose(scored_together, with_context + context_only, atol=1e-5)
    assert not torch.allclose(with_context, scored_together)


@pytest.mark.parametrize("chunk_size", [1, 7, None])
def test_chunking_is_exact_with_a_context(chunk_size):
    # Chunk boundaries must not change the reduction, only how much is
    # materialized at once. Pin the draws so the comparison isolates that.
    task = ShiftedLinear()
    b, L = 3, 12
    pool = torch.randn(b, L, 2)
    cursor = 0

    def pinned(n_samples=1):
        nonlocal cursor
        k = n_samples // b
        out = pool[:, cursor : cursor + k].reshape(b * k, 2)
        cursor += k
        return out

    task.sample_current_prior = pinned
    rollout = Rollout(
        theta=torch.randn(b, 2),
        y_history=torch.randn(b, 1, 1),
        design_history=torch.randn(b, 1, 2),
    )
    context = {
        "y_context": torch.randn(1, 2, 1),
        "design_context": torch.randn(1, 2, 2),
    }

    _, chunks = contrastive_logsumexps(task, rollout, L, chunk_size, **context)
    cursor = 0
    _, single = contrastive_logsumexps(task, rollout, L, None, **context)

    assert torch.allclose(
        torch.logsumexp(chunks, dim=1), torch.logsumexp(single, dim=1), atol=1e-5
    )
