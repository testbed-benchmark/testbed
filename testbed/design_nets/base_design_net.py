"""Base design-network interface."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class DesignNet(torch.nn.Module, ABC):
    """Maps experimental history to a new design."""

    def __init__(self):
        """Initializes the network."""
        super().__init__()

    @property
    def device(self) -> torch.device:
        """Returns the network device."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, y_history, design_history) -> Tensor:
        """Maps past (design, outcome) pairs to the next design x_t.

        Returns:
        Tensor
            The next design x_t with shape [*batch_shape, *event_shape_design].

        """
        ...
