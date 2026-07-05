"""Sequential prior contrastive estimation objectives."""

import numpy as np
import torch
from torch import Tensor

from ..tasks.base import Task
from .base_objective import Objective


class SPCE(Objective):
    """Sequential Prior Contrastive Estimation (SPCE) lower bound on EIG.

    See Theorem 2 of the DAD paper (https://arxiv.org/pdf/2103.02438).

    Given outer samples:
      theta_i ~ p(theta), i = 1, ..., b
      y_i^{1:T}, x_i^{1:T} ~ p(· | theta_i, pi_X)

    Estimates:
      (1/b) sum_i [ log p(y_i^{1:T} | theta_i) - log (1/L * sum_l p(y_i^{1:T} | theta_l')) ]

    where theta_0' = theta_i and theta_l' ~ p(theta) for l >= 1.
    """

    def __init__(
        self,
        task: Task,
        n_contrastive: int,
        *,
        contrastive_chunk_size: int | None = None,
        particle_budget: int = 50_000_000,
    ):
        """Initializes the SPCE estimator.

        Args:
            task: The BED task whose likelihood is scored.
            n_contrastive: Number of contrastive prior samples (the ``L - 1``
                particles beyond ``theta_0``).
            contrastive_chunk_size: Contrastive particles per outer sample to
                score per chunk. ``None`` auto-sizes each chunk so the outer
                batch times the chunk stays under ``particle_budget``,
                bounding peak memory regardless of batch size. When the whole
                estimate fits in a single chunk this reproduces the unchunked
                computation exactly.
            particle_budget: Cap on the number of contrastive particles (outer
                batch x per-env chunk) materialized at once when ``contrastive_chunk_size``
                is left unset.
        """
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)
        self.contrastive_chunk_size = (
            None if contrastive_chunk_size is None else int(contrastive_chunk_size)
        )
        self.particle_budget = particle_budget

    def __call__(
        self,
        y_history: Tensor,
        design_history: Tensor,
        theta_0: Tensor,
        *,
        reduce: bool = True,
    ):
        """Estimates gain for a simulated history."""
        return self.estimate_gain(y_history, design_history, theta_0, reduce=reduce)

    def estimate_gain(
        self,
        y_history: Tensor,
        design_history: Tensor,
        theta_0: Tensor,
        *,
        reduce: bool = True,
    ):
        """Estimates gain from batched histories.

        The contrastive particles are streamed in chunks and reduced with a
        running (associative) ``logsumexp`` so the ``[b, n_contrastive]``
        likelihood tensors are never all materialized at once. This keeps peak
        memory bounded while remaining autograd-differentiable.

        y_history: ``[b, T, *event_shape_y]``
        design_history: ``[b, T, *event_shape_design]``
        theta_0: ``[b, *event_shape_theta]``; true theta used to generate the history.
        reduce: if ``True`` (default) return the scalar mean over the batch; if
            ``False`` return the per-history ``[b]`` sPCE bound (useful for
            reporting a standard error over rollouts).
        returns: scalar SPCE estimate, or a ``[b]`` tensor when ``reduce=False``.
        """
        task = self.task
        n_contrastive = self.n_contrastive
        b = theta_0.shape[0]
        L = n_contrastive + 1

        # Particle 0 is the true theta that generated the history.
        logp_joint = task.history_logp(y_history, design_history, theta_0)  # [b]

        # Each entry is a logsumexp over a disjoint group of particles; particle 0
        # (the joint) forms its own singleton group. logsumexp is associative, so
        # the logsumexp of these partials equals the logsumexp over all L particles.
        partial_logsumexps = [logp_joint]
        chunk = self._contrastive_chunk_size(b)
        remaining = n_contrastive
        while remaining > 0:
            m = min(chunk, remaining)
            theta_chunk = task.sample_current_prior(b * m).reshape(
                b, m, *task.event_shape_theta
            )
            logp_chunk = self._history_logp_particles(
                y_history, design_history, theta_chunk
            )  # [b, m]
            partial_logsumexps.append(torch.logsumexp(logp_chunk, dim=1))  # [b]
            remaining -= m

        logsumexp_all = torch.logsumexp(torch.stack(partial_logsumexps, dim=1), dim=1)
        logp_marginal = logsumexp_all - np.log(L)

        per_history = logp_joint - logp_marginal  # [b]
        return per_history.mean() if reduce else per_history

    def _contrastive_chunk_size(self, b: int) -> int:
        """Returns the number of contrastive particles to score per env, per chunk."""
        if self.contrastive_chunk_size is not None:
            return max(1, self.contrastive_chunk_size)
        return max(1, self.particle_budget // max(1, b))

    def _history_logp_particles(
        self, y_history: Tensor, design_history: Tensor, theta: Tensor
    ) -> Tensor:
        """Scores a history against a batch of per-env contrastive particles.

        y_history: ``[b, T, *event_shape_y]``
        design_history: ``[b, T, *event_shape_design]``
        theta: ``[b, m, *event_shape_theta]``; m contrastive particles per env. returns: ``[b, m]`` log p(history | theta) for each particle.
        """
        task = self.task
        b, m = theta.shape[0], theta.shape[1]

        if task.separable:
            # Sum outcome log-probs over timesteps without ever expanding the
            # T axis across particles: each step is a [b, m] broadcast.
            T = y_history.shape[-(len(task.event_shape_y) + 1)]
            logp = theta.new_zeros(b, m)
            for t in range(T):
                y_t = y_history[:, t].unsqueeze(1)  # [b, 1, *event_shape_y]
                design_t = design_history[:, t].unsqueeze(
                    1
                )  # [b, 1, *event_shape_design]
                logp = logp + task.outcome_logp(y_t, theta, design_t)
            return logp

        # Non-separable fallback: replicate the history within the chunk and defer
        # to the task's history_logp override. Peak memory carries the T factor,
        # so chunks should be smaller for these tasks.
        y_rep = (
            y_history.unsqueeze(1)
            .expand(b, m, *y_history.shape[1:])
            .reshape(b * m, *y_history.shape[1:])
        )
        design_rep = (
            design_history.unsqueeze(1)
            .expand(b, m, *design_history.shape[1:])
            .reshape(b * m, *design_history.shape[1:])
        )
        theta_flat = theta.reshape(b * m, *task.event_shape_theta)
        return task.history_logp(y_rep, design_rep, theta_flat).reshape(b, m)


class PCE(Objective):
    """Prior Contrastive Estimation (PCE) for a single fixed design.

    A thin wrapper around SPCE that draws the outer samples (theta, y) and
    evaluates the T=1 SPCE bound. Useful when optimizing or evaluating a static design.
    """

    def __init__(self, task: Task, n_outer: int, n_inner: int):
        """Initializes the PCE estimator."""
        super().__init__(task)
        self.n_outer = int(n_outer)
        self.n_inner = int(n_inner)
        self.inner_spce = SPCE(task, n_contrastive=n_inner)

    def __call__(self, design: Tensor):
        """Estimates gain for one design."""
        return self.estimate_gain(design)

    def estimate_gain(self, design: Tensor):
        """Estimates gain for one fixed design.

        design: ``[*event_shape_design]``. Leading size-1 batch dims are tolerated and stripped.
        returns: scalar PCE estimate.
        """
        n_event_dims = len(self.task.event_shape_design)
        n_batch_dims = design.ndim - n_event_dims
        batch_shape = design.shape[:n_batch_dims]
        if any(s != 1 for s in batch_shape):
            raise ValueError(
                f"PCE expects a single design; got shape {tuple(design.shape)}"
            )
        design = design.reshape(self.task.event_shape_design)

        b = self.n_outer

        # y_outer: [b, *event_shape_y], theta_outer: [b, *event_shape_theta]
        y_outer, theta_outer = self.task.sample_joint(design, n_samples=b)

        # Build T=1 histories: [b, 1, *event_shape]
        y_history = y_outer.unsqueeze(1)
        design_history = (
            design.unsqueeze(0).expand(b, *self.task.event_shape_design).unsqueeze(1)
        )

        return self.inner_spce(y_history, design_history, theta_outer)
