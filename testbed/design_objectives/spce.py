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

    def __init__(self, task: Task, n_contrastive: int):
        """Initializes the SPCE estimator."""
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)

    def __call__(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Estimates gain for a simulated history."""
        return self.estimate_gain(y_history, design_history, theta_0)

    def estimate_gain(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Estimates gain from batched histories.

        y_history: ``[b, T, *event_shape_y]``
        design_history: ``[b, T, *event_shape_design]``
        theta_0: ``[b, *event_shape_theta]``; true theta used to generate the history.
        returns: scalar SPCE estimate.
        """
        task = self.task
        n_contrastive = self.n_contrastive
        b = theta_0.shape[0]
        L = n_contrastive + 1

        theta_contrastive = task.sample_current_prior(b * n_contrastive)
        theta_contrastive = theta_contrastive.reshape(
            b, n_contrastive, *task.event_shape_theta
        )

        # [b, L, *event_shape_theta]
        theta_all = torch.cat([theta_0.unsqueeze(1), theta_contrastive], dim=1)
        theta_flat = theta_all.reshape(b * L, *task.event_shape_theta)

        # Repeat each history L times: [b*L, T, *event_shape_*]
        y_rep = (
            y_history.unsqueeze(1)
            .expand(b, L, *y_history.shape[1:])
            .reshape(b * L, *y_history.shape[1:])
        )
        design_rep = (
            design_history.unsqueeze(1)
            .expand(b, L, *design_history.shape[1:])
            .reshape(b * L, *design_history.shape[1:])
        )

        logp_all = task.history_logp(y_rep, design_rep, theta_flat).reshape(
            b, L
        )  # [b, L]

        logp_joint = logp_all[:, 0]
        logp_marginal = torch.logsumexp(logp_all, dim=1) - np.log(L)

        return torch.mean(logp_joint - logp_marginal)


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
