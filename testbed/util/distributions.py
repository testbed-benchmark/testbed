from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
from torch import nn, Tensor

class _DistributionLike(Protocol):
    """
    Minimal torch-style distribution interface used by the Task ABC.
    
    Written using a Protocol to allow for flexibility in the types of distributions
    that can be used, while still ensuring they implement the necessary methods.
    
    """
    def sample(self, sample_shape: torch.Size | tuple = torch.Size(),
               generator: torch.Generator | None = None) -> Any: ...
    
    def log_prob(self, value: Any) -> torch.Tensor: ...

class TaskPrior(ABC):
    """Abstract prior (or posterior) p(θ)."""

    @abstractmethod
    def sample(self, num_samples: int = 1) -> Any: ...

    @abstractmethod
    def log_prob(self, theta: Any) -> torch.Tensor: ...

class EmpiricalPrior(TaskPrior, nn.Module):
    """
    Prior distribution for:
      theta: [B, K, p] - the location of the sources
    """

    def __init__(
        self,
        device: torch.device | str,
        theta: Tensor,  # [B, K, p]
        *,
        p: int = 1,
        K: int = 2,
    ) -> None:
        TaskPrior.__init__(self)
        nn.Module.__init__(self)
        self.theta = theta.to(device)

    def sample(self, num_samples: int = 1) -> Tensor:
        """
        Sample from the empirical prior.
        
        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw, by default 1
        
        Returns
        -------
        Tensor
            Samples from the empirical prior distribution.
        """
        indices = torch.randint(0, self.theta.size(0), (num_samples,))
        return self.theta[indices]
    
    def log_prob(self, theta: Tensor) -> Tensor:
        raise NotImplementedError(
            "EmpiricalPrior does not support log_prob."
        )