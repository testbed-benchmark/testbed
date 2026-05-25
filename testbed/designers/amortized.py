from abc import ABC, abstractmethod

import torch


class Designer(ABC):

    def __init__(self, task, T):
        super().__init__()

        self.task = task
        self.T = T

    @abstractmethod
    def propose_design(self, y_history, design_history):
        # y_history, design_history: [b=1, T, **dim]
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
        # returns: y_history, design_history: [b, T, **]
        assert theta.device == self.device, (
            f"theta is on {theta.device} but designer is on {self.device}. "
            f"Move theta with theta.to(designer.device), or move the designer with designer.to(theta.device)."
        )

        b = theta.shape[0]
        T = self.T
        
        y_list = []
        design_list = []

        for _ in range(T):
            y_hist_t = torch.stack(y_list, dim=1) if y_list else torch.empty(b, 0, *self.task.event_shape_y, device=theta.device)
            d_hist_t = torch.stack(design_list, dim=1) if design_list else torch.empty(b, 0, *self.task.event_shape_design, device=theta.device)

            design = self.propose_design(y_hist_t, d_hist_t)       # [b, *event_shape_design]
            y = self.task.sample_outcome(theta, design)             # [b, *event_shape_y]

            y_list.append(y)
            design_list.append(design)

        y_history = torch.stack(y_list, dim=1)           # [b, T, *event_shape_y]
        design_history = torch.stack(design_list, dim=1) # [b, T, *event_shape_design]
        
        return y_history, design_history


class PolicyDesigner(Designer):

    def __init__(self, task, T, policy_net):
        super().__init__(task, T)

        self.policy_net = policy_net
    
    @property 
    def device(self) -> torch.device:
        return self.policy_net.device

    def propose_design(self, y_history, design_history):
        # y_history, design_history: [b, T, **dim]
        # returns: [b, **dim_design]

        # TODO enforce policy_net can handle this kind of input or otherwise handle it here
        # When t=0, the slices of y_history, design_history will have shape [b, 0, dim]
        design = self.policy_net(y_history, design_history) 
        return design
