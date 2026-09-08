"""Importance-sampling posterior inference."""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.distributions import Distribution, EmpiricalDistribution
from ..core.util import squeeze_history
from ..tasks.base import Task


def run_importance_sampling(
    task: Task,
    y_history: Tensor,
    design_history: Tensor,
    n_proposal: int = 50_000,
    proposal: Distribution | None = None,
) -> EmpiricalDistribution:
    """Estimates a posterior with importance sampling.

    Args:
        task: Task
            Any Task subclass. Works for both continuous and discrete theta.
        y_history: Tensor, shape ``[T, *event_shape_y]``
            Stack T observations along dim 0. Leading size-1 batch dims are
            squeezed automatically; non-unit leading dims raise ValueError.
        design_history: Tensor, shape ``[T, *event_shape_design]``
            Matching designs. Same squeeze behaviour as y_history.
        n_proposal: int
            Number of proposal samples. Larger values give a better IS estimate
            at the cost of memory and compute. Keep <= 10^6 for practical use.
        proposal: Distribution or None
            Proposal distribution. Must implement sample(n_samples) and, if
            different from the prior, log_prob(theta) -> [n_proposal].
            Defaults to task.prior, in which case the IS weights are proportional
            to the likelihood alone.

    Returns:
        EmpiricalDistribution
            locs: ``[n_proposal, *event_shape_theta]``; the proposal samples.
            weights: ``[n_proposal]``; normalized importance weights (sum to 1).

    Notes:
        Draw posterior samples via posterior.sample(k). ESS can be computed as
        1 / (posterior.weights**2).sum(); low ESS (< ~1% of n_proposal) indicates
        a poor proposal — increase n_proposal or supply a better proposal.
    """
    y_history = squeeze_history(y_history, task.event_shape_y)
    design_history = squeeze_history(design_history, task.event_shape_design)

    src = task.prior if proposal is None else proposal
    theta = src.sample(n_samples=n_proposal)  # [n_proposal, *event_theta]

    # theta batch=[n_proposal] broadcasts with y/design batch=[]; returns [n_proposal]
    log_lik = task.history_logp(y_history, design_history, theta)

    if proposal is None:
        log_weights = log_lik
    else:
        log_weights = log_lik + task.prior.log_prob(theta) - proposal.log_prob(theta)

    weights = torch.softmax(log_weights, dim=0)
    return EmpiricalDistribution(locs=theta, weights=weights)


def effective_sample_size(posterior) -> float | None:
    """Kish ESS of a weighted-particle posterior, or None if it carries no weights.

    Assumes normalized weights, as :func:`run_importance_sampling` produces. None
    rather than an error keeps this safe for the posteriors that have no weights
    at all (a variational or analytic one installed via ``task.update_prior``).
    """
    weights = getattr(posterior, "weights", None)
    if weights is None:
        return None
    return (1.0 / weights.pow(2).sum()).item()
