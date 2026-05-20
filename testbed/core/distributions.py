from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class Distribution(ABC, nn.Module):
    """
    Abstract class for arbitrary distributions.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, n_samples: int = 1) -> Tensor: ...

    @abstractmethod
    def log_prob(self, theta: Any) -> Tensor: ...

    @property
    def device(self) -> torch.device:
        try:
            return next(self.buffers()).device
        except StopIteration:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device('cpu')

class TorchDistribution(Distribution):
    """
    Adapts torch.distributions.Distribution to our Distribution class.
    """

    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    @property
    def device(self) -> torch.device:
        params = self.dist.__dict__.values()
        for p in params:
            if isinstance(p, torch.Tensor):
                return p.device
        return torch.device('cpu')

    def sample(self, n_samples: int = 1) -> Tensor:
        return self.dist.sample((n_samples,))
    
    def log_prob(self, theta: Tensor) -> Tensor:
        return self.dist.log_prob(theta)
    

class EmpiricalDistribution(Distribution):
    """
    Base class for empirical distributions (i.e., implicitly defined by samples) 
    """

    def __init__(self, locs: Tensor, weights: Tensor | None = None) -> None:
        """
        Here, loc is [b, **shape], consisting of the locations that will be used to define the distribution
        weight is [b, ] or [b, 1] consisting of the weights associated with each location
        - Note that weight does /not/ have to sum to one; they only need to be non-negative.

        """
        super().__init__(self)
        self.locs = locs
       
        if weights is None:
            weights = torch.ones(locs.shape[0])
        assert torch.all(weights >=0), 'Weights must be non-negative.'
        self.weight = weights.squeeze()

    def sample(self, n_samples: int = 1) -> Tensor:
        """
        Sample from the empirical prior.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw, by default 1
        
        Returns
        -------
        Tensor
            Samples from the empirical distribution.
        """
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        return self.locs[indices]
    
    def log_prob(self, z: Tensor) -> Tensor:
        raise NotImplementedError('EmpiricalPrior does not support log_prob.')