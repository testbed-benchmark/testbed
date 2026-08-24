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


def contrastive_logsumexps(task, rollout, n_contrastive, chunk_size=None):
    """Scores a rollout under its own theta and under prior draws.

    Args:
        task: Task supplying the prior and ``history_logp``.
        rollout: Batched rollout to score.
        n_contrastive: Number of prior draws per rollout (L).
        chunk_size: Draws to score at once. ``None`` scores all L together.

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
    logp_joint = task.history_logp(y_history, design_history, theta_0)  # [b]

    chunk_logsumexps = []
    remaining = int(n_contrastive)
    chunk_size = remaining if chunk_size is None else int(chunk_size)
    while remaining > 0:
        chunk = min(chunk_size, remaining)
        # sample_current_prior drops the leading dim when asked for exactly one
        # sample, so reshape rather than trusting the returned shape.
        theta_flat = task.sample_current_prior(b * chunk).reshape(
            b * chunk, *task.event_shape_theta
        )
        y_rep = (
            y_history.unsqueeze(1)
            .expand(b, chunk, *y_history.shape[1:])
            .reshape(b * chunk, *y_history.shape[1:])
        )
        design_rep = (
            design_history.unsqueeze(1)
            .expand(b, chunk, *design_history.shape[1:])
            .reshape(b * chunk, *design_history.shape[1:])
        )
        logp_chunk = task.history_logp(y_rep, design_rep, theta_flat).reshape(b, chunk)
        chunk_logsumexps.append(torch.logsumexp(logp_chunk, dim=1))  # [b]
        remaining -= chunk

    return logp_joint, torch.stack(chunk_logsumexps, dim=1)  # [b], [b, n_chunks]
