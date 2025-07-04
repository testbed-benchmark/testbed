from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch

class Task(ABC):

    def __init__(self, prior, device):
        self.device = torch.device(device)
        
        self.prior = prior
        self.current_posterior = prior
    
    @abstractmethod
    def outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        returns: [batch, ]
        """
        ...
    
    @abstractmethod
    def sample_outcome(self, n_samples, theta, design):
        """
        theta, design: [b, **shape]
        returns: [n_samples, b, **shape]
        """
        ...
    
    def sample_joint(self, n_samples, design):
        theta = self.current_posterior.sample(n_samples)
        outcome = self.sample_outcome(1, theta, design)

        return outcome, theta
    
    def update_prior(self, design):
        # Potentially updates the prior
        # For use in sequential experiments
        raise NotImplementedError
    