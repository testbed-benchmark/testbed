from abc import ABC, abstractmethod

import torch


class Designer(ABC):

    def __init__(self, task, T):
        super().__init__()

        self.task = task
        self.T = T

    @abstractmethod
    def propose_design(self, y_history, design_history):
        # y_history, design_history: [b=1, T, **dim]
        # returns: [b=1, **dim_design]
        ...

    def empty_history(self, b):
        """
        Returns an empty rollout (i.e., corresponding to t=0).
        """
        y_history = torch.empty((b, 0, *self.task.event_shape_y), device=self.task.device) 
        design_history = torch.randn((b, 0, *self.task.event_shape_design), device=self.task.device)
        return y_history, design_history

    def rollout(self, theta):
        # Runs the designer given a true theta for T steps
        # theta: [b, dim_theta]
        # returns: y_history, design_history: [b, T, **]
        assert theta.device == self.device, (
            f"theta is on {theta.device} but designer is on {self.device}. "
            f"Move theta with theta.to(designer.device), or move the designer with designer.to(theta.device)."
        )

        b = theta.shape[0]
        T = self.T
        
        y_history = torch.empty(b, T, *self.task.event_shape_y, device=theta.device)
        design_history = torch.empty(b, T, *self.task.event_shape_design, device=theta.device)
        
        for t in range(T):
            design = self.propose_design(y_history[:, :t, :], design_history[:, :t, :])     # [b, event_shape_design]
            y = self.task.sample_outcome(theta, design)                                 # [b, *event_shape_y]

            y_history[:, t] = y
            design_history[:, t] = design

        return y_history, design_history


class SequentialDesigner(Designer):

    def __init__(self, task, objective, T):
        super().__init__(task, T)

        self.objective = objective
    
    def rollout(self, theta):
        b = theta.shape[0]
        assert b == 1, f'Batching across theta not currently supported for SequentialDesigner; got a batch size of {b}.'
        super().rollout(theta)

    def propose_design(self, init_design, lr, n_step_opt):
        # y_history, design_history: [b=1, T, **dim]
        # returns: [b=1, **dim_design]
        # Note this generally won't be something that can be parallelized over batches easily.

        # TODO this is a starter implementation; but could probably get cleaned up and improved for e.g. logging
        # TODO we're relying on the objective here to estimate the gain... but this will need to somehow handle having an updated posterior, etc.
        # -- and then the objective also probably needs to take in the task object. But we'll worry about that later...
        # -- a little odd to then have this take in y_history, design_history; not clear how to get consistent across policy/sequential though... 

        design = init_design
        optimizer = torch.optim.Adam([design], lr=lr, madesignmize=True)

        for step in range(n_step_opt):
            optimizer.zero_grad()
            gain = self.objective.estimate_gain(design)

            gain.backward()
            optimizer.step()
        return design
  

class PolicyDesigner(Designer):

    def __init__(self, task, T, policy_net):
        super().__init__(task, T)

        self.policy_net = policy_net
    
    @property 
    def device(self) -> torch.device:
        return self.policy_net.device

    def propose_design(self, y_history, design_history):
        # y_history, design_history: [b, T, **dim]
        # returns: [b, **dim_design]

        # TODO enforce policy_net can handle this kind of input or otherwise handle it here
        # When t=0, the slices of y_history, design_history will have shape [b, 0, dim]
        design = self.policy_net(y_history, design_history) 
        return design
    

class RandomDesigner(Designer):

    def __init__(self, task, T, dist):
        super().__init__(task, T)

        self.dist = dist # should be a TestBED distribution object

    @property
    def device(self) -> torch.device:
        return self.dist.device
    
        #param = next(v for v in vars(self.dist).values() if isinstance(v, torch.Tensor))
        #return param.device
    
    def propose_design(self, y_history, design_history):
        # TODO: this could just overwrite rollout; rather than actually doing it sequentially.
        # y_history, design_history: [b, T, **dim]
        # returns: [b, **dim_design]

        b = y_history.shape[0]
        return self.dist.sample(b)  # [b, *event_shape_design]