from abc import ABC, abstractmethod
import torch
from typing import Any

class DesignNet(torch.nn.Module, ABC):
    """
    Maps experimental history to a new design
    
    """
    
    def __init__(self):
        super().__init__()



    @abstractmethod
    def sample_design(self, past_designs: Any, past_outcomes: Any) -> Any:
        """
        Given past (design, outcome) pairs, return the next design ξ_t.

        Parameters
        ----------
        past_designs : Any
            Designs up to time t-1 (e.g. Tensor or list).
        past_outcomes : Any
            Outcomes up to t-1 associated with past designs.

        Returns
        -------
        Any
            The next design ξ_t 
        """
        ...
    
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