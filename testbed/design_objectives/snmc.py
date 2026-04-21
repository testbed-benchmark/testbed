import torch
import numpy as np
from torch import Tensor

from .base_objective import Objective
from ..tasks.base_task import Task

class SNMC(Objective):
    """
    Implements sequential NMC.
    See Equation 10 of the DAD paper (https://ardesignv.org/pdf/2103.02438)
    """

    def __init__(self, task: Task, n_contrastive: int):
        super().__init__(task)
        self.n_contrastive = int(n_contrastive)

    def estimate_gain(self, y_history : Tensor, design_history : Tensor, theta_0 : Tensor):
            """
            y_history:  [b, T, *event_shape_y]
            design_history: [b, T, *event_shape_design]
            theta_0:    [b, *event_shape_theta]
            returns:    scalar EIG estimate [b, ]
            """
            task = self.task
            n_contrastive = self.n_contrastive
            b = theta_0.shape[0]

            logp_joint = task.history_logp(y_history, design_history, theta_0)  # [b,]

            theta_contrastive = task.sample_current_prior(b * n_contrastive)
            theta_contrastive = theta_contrastive.reshape(b, n_contrastive, *task.event_shape_theta)
            theta_flat = theta_contrastive.reshape(b * n_contrastive, *task.event_shape_theta)
            
            #y_rep  = y_history.repeat_interleave(n_contrastive, dim=0)
            #design_rep = design_history.repeat_interleave(n_contrastive, dim=0)

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
            logp_contrastive = logp_contrastive.reshape(b, n_contrastive)   # [b, n_contrastive]

            logp_marginal = torch.logsumexp(logp_contrastive, dim=1) - np.log(n_contrastive)  # [b,]

            snmc = torch.mean(logp_joint - logp_marginal)
            return snmc
