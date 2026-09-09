"""Evaluation functions shared across different methods."""

import math

import torch
from tqdm.auto import tqdm

from testbed.design_objectives import SNMC, SPCE
from testbed.design_objectives.contrastive import contrastive_logsumexps


def sum_segments(segments, T):
    """Adds per-segment EIGs into a total, requiring that they tile ``[0, T)``.

    Additive by the chain rule for mutual information, but only if the segments
    partition the horizon: a gap drops information and an overlap double-counts
    it, and either still produces a plausible-looking number.
    """
    segments = sorted(segments, key=lambda segment: segment["start"])
    expected_start = 0
    for segment in segments:
        if segment["start"] != expected_start:
            raise ValueError(
                f"Segments must tile [0, {T}) with no gap or overlap; a segment "
                f"starts at {segment['start']} where {expected_start} was expected."
            )
        expected_start = segment["end"]
    if expected_start != T:
        raise ValueError(
            f"Segments cover [0, {expected_start}) but the horizon is {T}."
        )

    return {
        "lower_bound": sum(segment["lower_bound"] for segment in segments),
        "upper_bound": sum(segment["upper_bound"] for segment in segments),
        "segments": segments,
    }


def summarize_bounds(lower, upper, **extra):
    """Mean and standard error of both bounds, plus whatever labels the caller adds."""
    n = lower.shape[0]
    return {
        "lower_bound": lower.mean().item(),
        "lower_bound_se": (lower.std(unbiased=True) / n**0.5).item(),
        "upper_bound": upper.mean().item(),
        "upper_bound_se": (upper.std(unbiased=True) / n**0.5).item(),
        **extra,
    }


def chunked_spce_snmc(
    task, rollout, n_contrastive, chunk_size, y_context=None, design_context=None
):
    """SPCE (lower) and SNMC (upper) per-sample gains, chunked over n_contrastive.

    Both bounds come from one set of contrastive draws, so the expensive
    sampling is paid for once rather than per bound.

    A context is history already absorbed into ``task.current_prior``, conditioned
    on but not scored, as when scoring the tail of a rollout whose earlier steps a
    posterior update has already accounted for.
    """
    logp_joint, chunk_logsumexps = contrastive_logsumexps(
        task,
        rollout,
        n_contrastive,
        chunk_size,
        y_context=y_context,
        design_context=design_context,
    )
    lower = SPCE.gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive)
    upper = SNMC.gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive)
    return lower, upper


def eval_spce_snmc(
    task,
    designer,
    n_outer: int = 2_048,
    n_contrastive: int = 1_000_000,
    outer_batch_size: int = 256,
    contrastive_chunk_size: int = 2_000,
    verbose: bool = True,
):
    """Evaluates the SPCE/SNMC bounds on the EIG.

    This is both batched (outer theta samples) and chunked (inner samples).

    If verbose, shows a progress bar over the outer batches. `disable=None`
    lets tqdm silence itself when stderr is not a terminal, so batch jobs
    don't flood their logs.
    """
    lower_gains, upper_gains = [], []
    remaining = n_outer

    progress = tqdm(
        total=math.ceil(n_outer / outer_batch_size),
        desc="eval_spce_snmc",
        unit="batch",
        disable=None if verbose else True,
    )

    with torch.no_grad(), progress:
        while remaining > 0:
            b = min(outer_batch_size, remaining)
            theta0 = task.sample_prior(b)
            rollout = designer.rollout(theta0)
            lower, upper = chunked_spce_snmc(
                task, rollout, n_contrastive, contrastive_chunk_size
            )
            lower_gains.append(lower)
            upper_gains.append(upper)
            remaining -= b
            progress.update(1)

    lower_gains = torch.cat(lower_gains)
    upper_gains = torch.cat(upper_gains)

    n = lower_gains.shape[0]

    return {
        "lower_bound": lower_gains.mean().item(),
        "lower_bound_se": (lower_gains.std(unbiased=True) / n**0.5).item(),
        "upper_bound": upper_gains.mean().item(),
        "upper_bound_se": (upper_gains.std(unbiased=True) / n**0.5).item(),
        "n_outer": n,
        "n_contrastive": n_contrastive,
    }
