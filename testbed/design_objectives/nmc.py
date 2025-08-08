import torch
import numpy as np
from torch import Tensor
from einops import repeat, rearrange

from .base_objective import Objective
from ..tasks.base_task import Task

class NMC(Objective):

    def __init__(self, task: Task, n_outer: int, n_inner: int):
        super().__init__(task)
        self.n_outer = n_outer
        self.n_inner = n_inner

    def estimate_gain(self, xi: Tensor):
        # xi: [1, **dim_xi]
        # NOTE: this implementation currently only evaluates one design at a time, i.e., no batching
        assert xi.shape[0] == 1, f'NMC currently only supports a batch size of one; got {xi.shape[0]}'

        task = self.task

        y_joint, theta_joint = task.sample_joint(self.n_outer, xi)
        logprob_joint = task.outcome_logp(y_joint, theta_joint, xi) # [n_outer, ]

        theta_inner = task.sample_current_prior(n_samples = self.n_outer * self.n_inner)
        y_inner = repeat(y_joint, 'b n d -> (b k) n d', k=self.n_inner)
        logprob_inner = task.outcome_logp(y_inner, theta_inner, xi)                     # [n_outer * n_inner]
        logprob_inner = rearrange(logprob_inner, '(b k) -> b k', b=self.n_outer)        # [n_outer, n_inner]
        logprob_marginal = torch.logsumexp(logprob_inner, dim=1) - np.log(self.n_inner) # [n_outer, ]
        

        eig = torch.mean(logprob_joint - logprob_marginal)
        return eig
    