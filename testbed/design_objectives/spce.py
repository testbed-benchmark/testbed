"""Sequential prior contrastive estimation objectives."""

import numpy as np
import torch
from torch import Tensor

from ..core.rollout import Rollout
from ..tasks.base import Task
from .base_objective import DesignObjective, RolloutObjective
from .contrastive import contrastive_logsumexps


class SPCE(RolloutObjective):
    """Sequential Prior Contrastive Estimation (SPCE) lower bound on EIG.

    See Theorem 2 of the DAD paper (https://arxiv.org/pdf/2103.02438).

    Given outer samples:
      theta_i ~ p(theta), i = 1, ..., b
      y_i^{1:T}, x_i^{1:T} ~ p(· | theta_i, pi_X)

    Estimates:
      (1/b) sum_i [ log p(y_i^{1:T} | theta_i) - log (1/L * sum_l p(y_i^{1:T} | theta_l')) ]

    where theta_0' = theta_i and theta_l' ~ p(theta) for l >= 1.
    """

    def __init__(self, task: Task, n_contrastive: int, chunk_size: int = None):
        """Initializes the SPCE estimator.

        chunk_size caps how many contrastive samples are scored at once; the
        default scores all of them together. Set it when n_contrastive is too
        large to materialize, as evaluation does.
        """
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def __call__(self, rollout: Rollout):
        """Estimates gain for a simulated rollout."""
        return self.estimate_gain(rollout)

    def estimate_gain(self, rollout: Rollout):
        """Estimates gain from a batched rollout.

        returns: scalar SPCE estimate.
        """
        return self.per_sample_gain(rollout).mean()

    def per_sample_gain(self, rollout: Rollout):
        """Estimates per-sample gains from a batched rollout.

        returns: [b] per-sample SPCE estimates.
        """
        logp_joint, chunk_logsumexps = contrastive_logsumexps(
            self.task, rollout, self.n_contrastive, self.chunk_size
        )
        return self.gain_from_logsumexps(
            logp_joint, chunk_logsumexps, self.n_contrastive
        )

    @staticmethod
    def gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive):
        """Per-sample SPCE from log-likelihoods produced by contrastive_logsumexps.

        The joint sample joins the marginal, which therefore averages over
        L+1 samples. Split out so a caller holding the log-likelihoods (e.g.
        testbed.eval) can take both bounds from one set of draws.
        """
        logp_marginal = torch.logsumexp(
            torch.cat([logp_joint.unsqueeze(1), chunk_logsumexps], dim=1), dim=1
        ) - np.log(n_contrastive + 1)
        return logp_joint - logp_marginal


class PCE(DesignObjective):
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

        # sample_joint drops the leading batch dim when n_samples==1 (its
        # contract for a single design/sample), so reshape explicitly rather
        # than trusting the returned shape -- this must hold for b==1 too.
        y_outer, theta_outer = self.task.sample_joint(design, n_samples=b)
        y_outer = y_outer.reshape(b, *self.task.event_shape_y)
        theta_outer = theta_outer.reshape(b, *self.task.event_shape_theta)

        # Build T=1 histories: [b, 1, *event_shape]
        y_history = y_outer.unsqueeze(1)
        design_history = (
            design.unsqueeze(0).expand(b, *self.task.event_shape_design).unsqueeze(1)
        )

        rollout = Rollout(
            theta=theta_outer, y_history=y_history, design_history=design_history
        )
        return self.inner_spce(rollout)
