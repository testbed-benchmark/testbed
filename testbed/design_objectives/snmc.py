"""Sequential nested Monte Carlo objectives."""

import numpy as np
import torch
from torch import Tensor

from testbed.design_objectives.base_objective import Objective
from testbed.tasks.task import Task


class SNMC(Objective):
    """Implements sequential NMC.

    See Equation 10 of the DAD paper (https://arxiv.org/pdf/2103.02438).
    """

    def __init__(self, task: Task, n_contrastive: int):
        """Initializes the SNMC estimator."""
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)

    def __call__(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Estimates gain for a simulated history."""
        return self.estimate_gain(y_history, design_history, theta_0)

    def estimate_gain(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Estimates gain from batched histories.

        y_history:      [b, T, *event_shape_y]
        design_history: [b, T, *event_shape_design]
        theta_0:        [b, *event_shape_theta]
        returns:        scalar EIG estimate.
        """
        task = self.task
        n_contrastive = self.n_contrastive
        b = theta_0.shape[0]

        logp_joint = task.history_logp(y_history, design_history, theta_0)  # [b,]

        theta_contrastive = task.sample_current_prior(b * n_contrastive)
        theta_contrastive = theta_contrastive.reshape(
            b, n_contrastive, *task.event_shape_theta
        )
        theta_flat = theta_contrastive.reshape(
            b * n_contrastive, *task.event_shape_theta
        )

        y_rep = (
            y_history.unsqueeze(1)
            .expand(b, n_contrastive, *y_history.shape[1:])
            .reshape(b * n_contrastive, *y_history.shape[1:])
        )
        design_rep = (
            design_history.unsqueeze(1)
            .expand(b, n_contrastive, *design_history.shape[1:])
            .reshape(b * n_contrastive, *design_history.shape[1:])
        )

        logp_contrastive = task.history_logp(y_rep, design_rep, theta_flat)
        logp_contrastive = logp_contrastive.reshape(
            b, n_contrastive
        )  # [b, n_contrastive]

        logp_marginal = torch.logsumexp(logp_contrastive, dim=1) - np.log(
            n_contrastive
        )  # [b,]

        return torch.mean(logp_joint - logp_marginal)


class NMC(Objective):
    """Nested Monte Carlo (NMC) estimator for EIG with a single fixed design.

    A thin wrapper around SNMC that draws outer samples (theta, y) and
    evaluates the T=1 SNMC bound. Useful when optimizing or evaluating a static design.
    """

    def __init__(self, task: Task, n_outer: int, n_inner: int):
        """Initializes the NMC estimator."""
        super().__init__(task)
        self.n_outer = int(n_outer)
        self.n_inner = int(n_inner)
        self.inner_snmc = SNMC(task, n_contrastive=n_inner)

    def __call__(self, design: Tensor):
        """Estimates gain for one design."""
        return self.estimate_gain(design)

    def estimate_gain(self, design: Tensor):
        """Estimates gain for one fixed design.

        design: [*event_shape_design]  (leading size-1 batch dims are tolerated and stripped)
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

        # y_outer: [b, *event_shape_y], theta_outer: [b, *event_shape_theta]
        y_outer, theta_outer = self.task.sample_joint(design, n_samples=b)

        # Build T=1 histories: [b, 1, *event_shape]
        y_history = y_outer.unsqueeze(1)
        design_history = (
            design.unsqueeze(0).expand(b, *self.task.event_shape_design).unsqueeze(1)
        )

        return self.inner_snmc(y_history, design_history, theta_outer)
