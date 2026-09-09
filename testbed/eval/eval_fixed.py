"""Scoring a design that has already been chosen.

The counterpart of :func:`testbed.eval.eval_segment` for a designer with no
``continue_from``. A policy can be asked to design a whole stretch on demand, so
a segment is scored by simulating it forward; a designer that optimizes one
design at a time cannot be re-asked without landing on a different design from
the one the trajectory used, so the design it produced is scored as it stands.

Same estimator and same measure as ``eval_segment``, so a sum of these is
comparable with a sum of segments.
"""

import torch

from testbed.core.rollout import Rollout
from testbed.eval.base import chunked_spce_snmc, summarize_bounds


def eval_fixed_design(
    task,
    design,
    y_history,
    design_history,
    n_outer=2_048,
    n_contrastive=1_000_000,
    outer_batch_size=256,
    contrastive_chunk_size=2_000,
):
    """EIG of one already-chosen design, given an observed history.

    Measured **under ``task.current_prior``**, which supplies both the outer
    theta and the contrastive draws. The history is conditioned on but never
    scored: it is passed to the estimator as context, and the step being scored
    holds only the new design and its outcome.

    Args:
        task: The task, with the posterior for ``y_history`` installed.
        design: The design to score, ``[1, *event_shape_design]``.
        y_history: Outcomes observed so far, ``[1, S, *event_shape_y]``.
        design_history: The designs alongside them.
        n_outer: Outcomes to average over.
        n_contrastive: Contrastive samples per outer sample.
        outer_batch_size: Outcomes simulated at once.
        contrastive_chunk_size: Contrastive samples scored at once.

    Returns:
        dict: Both bounds with standard errors, plus the step this covers as
        ``start``/``end``.
    """
    start = y_history.shape[1]
    lower_gains, upper_gains = [], []
    remaining = n_outer

    with torch.no_grad():
        while remaining > 0:
            b = min(outer_batch_size, remaining)
            theta = task.current_prior.sample(n_samples=b)
            # The same design for every outer sample: it was chosen before any
            # of these outcomes was drawn, so it does not vary with them.
            designs = design.expand(b, *design.shape[1:])
            outcome = task.sample_outcome(
                theta,
                designs,
                y_history=y_history,
                design_history=design_history,
            )
            step = Rollout(
                theta=theta,
                y_history=outcome.reshape(b, 1, *task.event_shape_y),
                design_history=designs.reshape(b, 1, *task.event_shape_design),
            )
            lower, upper = chunked_spce_snmc(
                task,
                step,
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
        end=start + 1,
        n_contrastive=n_contrastive,
    )
