from abc import ABC, abstractmethod
from typing import Any

import torch


class DesignNet(torch.nn.Module, ABC):
    """
    Maps experimental history to a new design
    
    """
    
    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @abstractmethod
    def forward(self, past_designs, past_outcomes) -> Any:
        """
        
        Given past (design, outcome) pairs, return the next design ξ_t.

                Returns
        -------
        Any
            The next design ξ_t 
        
        """
        ...