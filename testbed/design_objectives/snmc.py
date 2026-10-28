"""Sequential nested Monte Carlo objectives."""

import numpy as np
import torch
from torch import Tensor

from testbed.core.rollout import Rollout
from testbed.design_objectives.base_objective import DesignObjective, RolloutObjective
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.tasks.base import Task


class SNMC(RolloutObjective):
    """Implements sequential NMC.

    See Equation 10 of the DAD paper (https://arxiv.org/pdf/2103.02438).
    """

    def __init__(
        self,
        task: Task,
        n_contrastive: int,
        chunk_size: int | None = None,
        fused: bool = False,
    ):
        """Initializes the SNMC estimator.

        chunk_size caps how many contrastive samples are scored at once; the
        default scores all of them together. Set it when n_contrastive is too
        large to materialize, as evaluation does.

        fused scores each chunk in one compiled call rather than looping over
        the T steps. See ``Task.history_logp``.
        """
        super().__init__(task, fused=fused)
        self.n_contrastive = int(n_contrastive)
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def estimate_gain(
        self,
        rollout: Rollout,
        y_context: Tensor | None = None,
        design_context: Tensor | None = None,
    ) -> Tensor:
        """Estimates gain from a batched rollout.

        returns: scalar EIG estimate.
        """
        return self.per_sample_gain(
            rollout, y_context=y_context, design_context=design_context
        ).mean()

    def per_sample_gain(
        self,
        rollout: Rollout,
        y_context: Tensor | None = None,
        design_context: Tensor | None = None,
    ) -> Tensor:
        """Estimates per-sample gains from a batched rollout.

        A context is history already absorbed into ``task.current_prior``; it is
        conditioned on but not scored. See :func:`contrastive_logsumexps`.

        returns: [b] per-sample SNMC estimates.
        """
        logp_joint, chunk_logsumexps = contrastive_logsumexps(
            self.task,
            rollout,
            self.n_contrastive,
            self.chunk_size,
            y_context=y_context,
            design_context=design_context,
            fused=self.fused,
        )
        return self.gain_from_logsumexps(
            logp_joint, chunk_logsumexps, self.n_contrastive
        )

    @staticmethod
    def gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive):
        """Per-sample SNMC from log-likelihoods produced by contrastive_logsumexps.

        The marginal averages over the L contrastive samples only, excluding
        the joint one. Split out so a caller holding the log-likelihoods (e.g.
        testbed.eval) can take both bounds from one set of draws.
        """
        logp_marginal = torch.logsumexp(chunk_logsumexps, dim=1) - np.log(
            n_contrastive
        )  # [b,]
        return logp_joint - logp_marginal


class NMC(DesignObjective):
    """Nested Monte Carlo (NMC) estimator for EIG with a single fixed design.

    A thin wrapper around SNMC that draws outer samples (theta, y) and
    evaluates the T=1 SNMC bound. Useful when optimizing or evaluating a static design.
    """

    def __init__(self, task: Task, n_outer: int, n_inner: int, fused: bool = False):
        """Initializes the NMC estimator."""
        super().__init__(task, fused=fused)
        self.n_outer = int(n_outer)
        self.n_inner = int(n_inner)
        self.inner_snmc = SNMC(task, n_contrastive=n_inner, fused=fused)

    def estimate_gain(
        self,
        design: Tensor,
        y_history: Tensor | None = None,
        design_history: Tensor | None = None,
    ) -> Tensor:
        """Estimates gain for one fixed design.

        design: ``[*event_shape_design]``. Leading size-1 batch dims are tolerated and stripped.
        returns: scalar NMC estimate.
        """
        n_event_dims = len(self.task.event_shape_design)
        n_batch_dims = design.ndim - n_event_dims
        batch_shape = design.shape[:n_batch_dims]
        if any(s != 1 for s in batch_shape):
            raise ValueError(
                f"NMC expects a single design; got shape {tuple(design.shape)}"
            )
        design = design.reshape(self.task.event_shape_design)

        b = self.n_outer

        # sample_joint drops the leading batch dim when n_samples==1, and a
        # context history contributes a batch dim of its own, so reshape
        # explicitly rather than trusting the returned shape.
        y_outer, theta_outer = self.task.sample_joint(
            design,
            y_history=y_history,
            design_history=design_history,
            n_samples=b,
        )
        y_outer = y_outer.reshape(b, *self.task.event_shape_y)
        theta_outer = theta_outer.reshape(b, *self.task.event_shape_theta)

        # Build one-step-ahead rollout: [b, 1, *event_shape]
        this_y_history = y_outer.unsqueeze(1)
        this_design_history = (
            design.unsqueeze(0).expand(b, *self.task.event_shape_design).unsqueeze(1)
        )
        rollout = Rollout(
            theta=theta_outer,
            y_history=this_y_history,
            design_history=this_design_history,
        )

        # The incoming history is passed as context, not scored: theta here is
        # drawn from current_prior, which already conditions on it.
        return self.inner_snmc.estimate_gain(
            rollout, y_context=y_history, design_context=design_history
        )
