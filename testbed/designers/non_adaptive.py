"""Non-adaptive designers."""

import torch
from torch import nn

from testbed.designers import Designer


class StaticDesigner(Designer, nn.Module):
    """A designer with a single learnable design sequence [T, *event_shape_design]."""

    def __init__(self, task, T):
        """Initializes learnable static designs."""
        Designer.__init__(self, task, T)
        self.design = nn.Parameter(torch.randn(T, *task.event_shape_design))

    @property
    def device(self) -> torch.device:
        """Returns the design parameter device."""
        return self.design.device

    def propose_design(self, y_history, design_history, *, mask=None):
        """Returns the design for the current step."""
        # y_history: [b, t, *event_shape_y]; t is the current step index
        t = y_history.shape[1]
        b = y_history.shape[0]
        return self.design[t].unsqueeze(0).expand(b, *self.task.event_shape_design)

    def rollout(self, theta):
        """Runs the static design sequence."""
        # theta: [b, *event_shape_theta]
        # returns: y_history [b, T, *event_shape_y], design_history [b, T, *event_shape_design]
        assert theta.device == self.device, (
            f"theta is on {theta.device} but designer is on {self.device}. "
            f"Move theta with theta.to(designer.device), or move the designer with designer.to(theta.device)."
        )
        b = theta.shape[0]

        y_history = self.task.sample_outcome(
            theta.unsqueeze(1), self.design.unsqueeze(0)
        )
        design_history = self.design.unsqueeze(0).expand(
            b, self.T, *self.task.event_shape_design
        )
        return y_history, design_history


class RandomDesigner(Designer):
    """Samples designs randomly."""

    def __init__(self, task, T, dist):
        """Initializes with a design distribution."""
        super().__init__(task, T)

        self.dist = dist  # should be a TestBED distribution object

    @property
    def device(self) -> torch.device:
        """Returns the distribution device."""
        return self.dist.device

    def propose_design(self, y_history, design_history, *, mask=None):
        """Samples one design per batch item."""
        # y_history, design_history: [b, T, **dim]
        # returns: [b, **dim_design]

        b = y_history.shape[0]
        return self.dist.sample(b)  # [b, *event_shape_design]

    def rollout(self, theta):
        """Runs a random design rollout."""
        # theta: [b, *event_shape_theta]
        # returns: y_history [b, T, *event_shape_y], design_history [b, T, *event_shape_design]

        b = theta.shape[0]

        design_history = self.dist.sample(b * self.T)
        design_history = design_history.reshape(
            b, self.T, *self.task.event_shape_design
        )

        theta = theta.unsqueeze(1)
        y_history = self.task.sample_outcome(theta, design_history)

        return y_history, design_history
