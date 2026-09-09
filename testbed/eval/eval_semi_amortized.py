"""Scoring for designers refined at deployment time (Step-DAD).

A refined policy depends on the history it was refined on, so one policy cannot
roll out a whole batch and :func:`eval_spce_snmc` does not apply directly. The
trajectories are the experiment, so the method runs them and orchestrates; the
functions here each score one thing.

Two ways to get a total. :func:`eval_end_to_end` stacks independently-refined
trajectories and scores one full-T sPCE/sNMC against the original prior, which is
directly comparable to the DAD row. Or build it per segment, as the chain-rule
sum ``EIG(0 -> tau_1) + sum_i EIG(tau_i -> tau_{i+1} | h)``; the segments carry
the ``start``/``end`` they covered, and the method adds them up.

Which measure a segment is taken under is the thing to keep straight, and it does
not depend on how many refinements there are. The opening segment runs from
``t = 0``, where nothing has been observed, so it is measured under the original
prior and is just :func:`eval_spce_snmc` on a designer whose ``T`` is that
segment's length. Every later segment is conditional on a realized history, so it
is measured under that history's posterior, which the caller installs on the task
before calling :func:`eval_segment`.
"""

import torch

from testbed.core.rollout import Rollout
from testbed.core.util import reset_prior
from testbed.eval.base import chunked_spce_snmc, summarize_bounds


def stack_rollouts(rollouts):
    """Concatenates single-trajectory rollouts along the batch dimension."""
    return Rollout(
        theta=torch.cat([rollout.theta for rollout in rollouts]),
        y_history=torch.cat([rollout.y_history for rollout in rollouts]),
        design_history=torch.cat([rollout.design_history for rollout in rollouts]),
    )


def eval_end_to_end(
    task, rollouts, n_contrastive=100_000, contrastive_chunk_size=2_000
):
    """Scores finished trajectories with one full-T sPCE/sNMC against the prior.

    Valid despite the policy differing between trajectories: each refined policy
    is still a map from history to design, and the bound holds for any design
    distribution.

    Args:
        task: The task.
        rollouts: Finished batch-1 trajectories.
        n_contrastive: Contrastive samples per outer sample.
        contrastive_chunk_size: Contrastive samples scored at once.

    Returns:
        dict: Both bounds with standard errors.
    """
    rollout = stack_rollouts(rollouts)
    with reset_prior(task), torch.no_grad():
        lower, upper = chunked_spce_snmc(
            task, rollout, n_contrastive, contrastive_chunk_size
        )
    return summarize_bounds(
        lower,
        upper,
        n_thetas=rollout.b,
        T=rollout.T,
        n_contrastive=n_contrastive,
    )


def eval_segment(
    task,
    designer,
    y_history,
    design_history,
    n_steps,
    raw_design_history=None,
    n_outer=2_048,
    n_contrastive=1_000_000,
    outer_batch_size=256,
    contrastive_chunk_size=2_000,
):
    """EIG of a designer's next ``n_steps`` given an observed history.

    The conditional counterpart of :func:`eval_spce_snmc`, batched over outer
    samples and chunked over contrastive samples the same way, so a segment's
    bounds rest on the same sample sizes as the full-horizon row they get
    compared to.

    Measured **under ``task.current_prior``**. That posterior then supplies both 
    the outer theta and the contrastive draws, which have to be under the same measure.

    The history is conditioned on but never scored: it is passed to the estimator
    as context, and the continuation being scored holds only the new steps.

    Args:
        task: The task, with the posterior for ``y_history`` installed.
        designer: A designer exposing ``continue_from``.
        y_history: Outcomes observed so far, ``[1, S, *event_shape_y]``.
        design_history: The designs alongside them.
        n_steps: Length of the segment.
        raw_design_history: Pre-transform designs, required when the designer
            conditions on a raw history.
        n_outer: Total continuations to average over.
        n_contrastive: Contrastive samples per outer sample.
        outer_batch_size: Continuations simulated at once.
        contrastive_chunk_size: Contrastive samples scored at once.

    Returns:
        dict: Both bounds with standard errors, plus the segment as
        ``start``/``end``.
    """
    start = y_history.shape[1]
    lower_gains, upper_gains = [], []
    remaining = n_outer

    with torch.no_grad():
        while remaining > 0:
            b = min(outer_batch_size, remaining)
            continuation_rollout = designer.continue_from(
                task.current_prior.sample(n_samples=b),
                y_history,
                design_history,
                n_steps,
                raw_design_history=raw_design_history,
            )
            lower, upper = chunked_spce_snmc(
                task,
                continuation_rollout,
                n_contrastive,
                contrastive_chunk_size,
                y_context=y_history,
                design_context=design_history,
            )
            lower_gains.append(lower)
            upper_gains.append(upper)
            remaining -= b

    return summarize_bounds(
        torch.cat(lower_gains),
        torch.cat(upper_gains),
        start=start,
        end=start + n_steps,
        n_contrastive=n_contrastive,
    )
