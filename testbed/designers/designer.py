import torch
from einops import rearrange, repeat

from abc import ABC, abstractmethod
from typing import Any, Protocol

class Designer(ABC):

    def __init__(self, task, T):
        super().__init__()

        self.task = task
        self.T = T

    @abstractmethod
    def propose_design(self, y_history, xi_history):
        # y_history, xi_history: [b=1, T, **dim]
        # returns: [b=1, **dim_xi]
        ...

    def rollout(self, theta):
        # Runs the designer given a true theta for T steps
        # theta: [b, dim_theta]
        # returns: y_history, xi_history: [b, T, **]

        b = theta.shape[0]
        T = self.T
        
        y_history = torch.empty(b, T, self.task.dim_y, device=self.device)
        xi_history = torch.empty(b, T, self.task.dim_xi, device=self.device)
        
        for t in range(T):
            xi = self.propose_design(y_history[:, :t, :], xi_history[:, :t, :])       # [B, dim_xi]  
            y = self.task.sample_outcome(theta, xi)                      # [1, b, dim_y]
            y = rearrange(y, '1 b d -> b d')

            y_history[:, t] = y
            xi_history[:, t] = xi

        return y_history, xi_history


class SequentialDesigner(Designer):

    def __init__(self, task, objective, T):
        super().__init__(task, T)

        self.objective = objective
    
    def rollout(self, theta):
        b = theta.shape[0]
        assert b == 1, f'Batching across theta not currently supported for SequentialDesigner; got a batch size of {b}.'
        super().rollout(theta)

    def propose_design(self, init_xi, lr, n_step_opt):
        # y_history, xi_history: [b=1, T, **dim]
        # returns: [b=1, **dim_xi]
        # Note this generally won't be something that can be parallelized over batches easily.

        # TODO this is a starter implementation; but could probably get cleaned up and improved for e.g. logging
        # TODO we're relying on the objective here to estimate the gain... but this will need to somehow handle having an updated posterior, etc.
        # -- and then the objective also probably needs to take in the task object. But we'll worry about that later...
        # -- a little odd to then have this take in y_history, xi_history; not clear how to get consistent across policy/sequential though... 

        xi = init_xi
        optimizer = torch.optim.Adam([xi], lr=lr, maximize=True)

        for step in range(n_step_opt):
            optimizer.zero_grad()
            gain = self.objective.estimate_gain(xi)

            gain.backward()
            optimizer.step()
        return xi
  

class PolicyDesigner(Designer):

    def __init__(self, task, T, policy_net):
        super().__init__(task, T)

        self.policy_net = policy_net

    def propose_design(self, y_history, xi_history):
        # y_history, xi_history: [b, T, **dim]
        # returns: [b, **dim_xi]

        # TODO enforce policy_net can handle this kind of input or otherwise handle it here
        # When t=0, the slices of y_history, xi_history will have shape [b, 0, dim]
        design = self.policy_net(y_history, xi_history) 
        return design