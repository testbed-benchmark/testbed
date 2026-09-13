"""Contrastive log-likelihoods shared by the SPCE and SNMC bounds.

Both bounds score a rollout under its own theta and under prior draws, then
differ only in how they combine the two -- see ``SPCE.gain_from_logsumexps``
and ``SNMC.gain_from_logsumexps``.

The prior draws are scored in chunks so memory stays flat as ``n_contrastive``
grows: evaluation uses L=500_000, which often does not fit as one ``[b*L, T, ...]``
tensor. Chunking is exact, via the associativity identity ``logsumexp(all) ==
logsumexp([logsumexp(chunk) for chunk in chunks])``, and the default single
chunk reduces to scoring them all at once.
"""

import torch


def _as_chunk_batch(x):
    """Adds the contrastive axis to a per-rollout tensor, giving ``[b, 1, ...]``."""
    return None if x is None else x.unsqueeze(1)


def contrastive_logsumexps(
    task,
    rollout,
    n_contrastive,
    chunk_size=None,
    y_context=None,
    design_context=None,
    fused=False,
):
    """Scores a rollout under its own theta and under contrastive draws.

    The contrastive draws come from ``task.current_prior``, so what has to be
    scored depends on what that prior already knows:

    - **No context** (sPCE/sNMC over a full rollout): the prior knows nothing
      about the rollout, so every step is scored from t=0.
    - **With a context** ``(y_{1:S}, x_{1:S})`` already absorbed into
      ``current_prior``, as ``PosteriorUpdateHook`` does: the rollout's steps
      are scored conditioned on it and the context itself is not scored.
      Scoring it again would re-apply ``p(y_{1:S} | theta)``, tilting the contrastive measure a second time.

    Args:
        task: Task supplying the prior and ``history_logp``.
        rollout: Batched rollout to score.
        n_contrastive: Number of contrastive draws per rollout (L).
        chunk_size: Draws to score at once. ``None`` scores all L together.
        y_context: Outcomes already absorbed into ``current_prior``, shape
            ``[b or 1, S, *event_shape_y]``. None means no context.
        design_context: The designs alongside them, same leading shape.
        fused: Score each chunk with one compiled call rather than a loop over
            the steps. See ``Task.history_logp``. Ignored for a non-memoryless task.

    Returns:
        ``(logp_joint, chunk_logsumexps)`` of shapes ``[b]`` and
        ``[b, n_chunks]``, each chunk entry being
        ``log(sum_{l in chunk} p(y | theta_l))``. The joint term is returned
        separately so callers can add it to the marginal (SPCE) or not (SNMC).
    """
    y_history, design_history, theta_0 = (
        rollout.y_history,
        rollout.design_history,
        rollout.theta,
    )
    b = theta_0.shape[0]
    fused = fused and task.memoryless

    logp_joint = task.history_logp(
        y_history,
        design_history,
        theta_0,
        y_context=y_context,
        design_context=design_context,
    )  # [b]

    # The rollout varies over b and the draws over L, so the histories get a
    # singleton L axis and broadcast against the draws inside history_logp.
    # Replicating them up front instead would copy [b * L, T, ...] per chunk.
    y_chunk = _as_chunk_batch(y_history)
    design_chunk = _as_chunk_batch(design_history)
    context = {
        "y_context": _as_chunk_batch(y_context),
        "design_context": _as_chunk_batch(design_context),
    }

    chunk_logsumexps = []
    remaining = int(n_contrastive)
    chunk_size = remaining if chunk_size is None else int(chunk_size)
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        # sample_current_prior drops the leading dim when asked for exactly one
        # sample, so reshape rather than trusting the returned shape.
        theta = task.sample_current_prior(b * chunk).reshape(
            b, chunk, *task.event_shape_theta
        )
        logp_chunk = task.history_logp(
            y_chunk, design_chunk, theta, fused=fused, **context
        )  # [b, chunk]
        chunk_logsumexps.append(torch.logsumexp(logp_chunk, dim=1))  # [b]
        remaining -= chunk

    return logp_joint, torch.stack(chunk_logsumexps, dim=1)  # [b], [b, n_chunks]
