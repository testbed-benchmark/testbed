from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch

from ..core.util import expand_along_batch


class Task(ABC):

    def __init__(self, prior, device):
        self.device = torch.device(device)
        
        self.prior = prior
        self.current_prior = prior
    
    @abstractmethod
    def _outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        returns: [batch, ]
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design, n_samples=1):
        """
        theta, design: [b, **shape]
        returns: [n_samples, b, **shape]
        """
        ...

    def outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        Also supports when any of y, theta, design only have batch=1 by broadcasting.
        """
        y_expanded, theta_expanded, design_expanded = expand_along_batch(y, theta, design)
        return self._outcome_logp(y_expanded, theta_expanded, design_expanded)
    
    def sample_outcome(self, theta, design, n_samples=1):
        theta_expanded, design_expanded = expand_along_batch(theta, design)
        return self._sample_outcome(theta_expanded, design_expanded, n_samples=n_samples)
        
    def sample_joint(self, n_samples, design):
        theta = self.sample_current_prior(n_samples)
        outcome = self.sample_outcome(theta, design)
        outcome = outcome.squeeze(0)

        return outcome, theta
    
    def sample_prior(self, n_samples):
        return self.prior.sample(n_samples=n_samples)
    
    def sample_current_prior(self, n_samples):
        return self.current_prior.sample(n_samples=n_samples)
    
    def update_prior(self, design):
        # Potentially updates the prior
        # For use in sequential experiments
        raise NotImplementedError
    