import numpy as np
import torch
from torch import Tensor

from ..tasks.base_task import Task
from .base_objective import Objective


class SPCE(Objective):
    """
    Implements sequential PCE.
    See Theorem 2 of the DAD paper (https://ardesignv.org/pdf/2103.02438)
    """

    def __init__(self, task: Task, n_contrastive: int):
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)

    def estimate_gain(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """
        y_history: [b, T, *event_shape_y]
        design_history: [b, T, *event_shape_design]
        theta_0: [b, *event_shape_theta] -- true theta used to generate the history
        """
        task = self.task
        n_contrastive = self.n_contrastive
        b = theta_0.shape[0]
        L = n_contrastive + 1

        theta_contrastive = task.sample_current_prior(b * n_contrastive)
        theta_contrastive = theta_contrastive.reshape(b, n_contrastive, *task.event_shape_theta)

        theta_all  = torch.cat([theta_0.unsqueeze(1), theta_contrastive], dim=1)  # [b, L, *event_shape_theta]
        theta_flat = theta_all.reshape(b * L, *task.event_shape_theta)

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

        logp_all = task.history_logp(y_rep, design_rep, theta_flat).reshape(b, L)  # [b, L]

        logp_joint    = logp_all[:, 0]
        logp_marginal = torch.logsumexp(logp_all, dim=1) - np.log(L)

        spce = torch.mean(logp_joint - logp_marginal)
        return spce
