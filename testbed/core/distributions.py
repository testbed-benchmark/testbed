"""Distribution abstractions and adapters."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class Distribution(ABC, nn.Module):
    """Abstract class for arbitrary distributions."""

    def __init__(self):
        """Initializes the distribution."""
        super().__init__()

    @abstractmethod
    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples."""
        ...

    @abstractmethod
    def log_prob(self, theta: Any) -> Tensor:
        """Evaluates log probabilities."""
        ...

    @property
    def device(self) -> torch.device:
        """Infers the module device."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")


class TorchDistribution(Distribution):
    """Adapts torch.distributions.Distribution to our Distribution class."""

    def __init__(self, dist):
        """Wraps a torch distribution."""
        super().__init__()
        self.dist = dist

    @property
    def device(self) -> torch.device:
        """Infers the wrapped distribution device."""
        params = self.dist.__dict__.values()
        for p in params:
            if isinstance(p, torch.Tensor):
                return p.device
        return torch.device("cpu")

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples from the wrapped distribution."""
        return self.dist.sample((n_samples,))

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates log probabilities under the wrapped distribution."""
        return self.dist.log_prob(theta)


class EmpiricalDistribution(Distribution):
    """Base class for empirical distributions (i.e., implicitly defined by samples)."""

    def __init__(self, locs: Tensor, weights: Tensor | None = None) -> None:
        """Initializes weighted sample locations.

        locs: [b, *shape].
        weights: [b] or [b, 1], non-negative and not necessarily normalized.
        """
        super().__init__()

        if weights is None:
            weights = torch.ones(locs.shape[0], device=locs.device)
        assert torch.all(weights >= 0), "Weights must be non-negative."
        self.register_buffer("locs", locs)
        self.register_buffer("weights", weights.squeeze())

    def sample(self, n_samples: int = 1) -> Tensor:
        """Samples from the empirical prior.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw, by default 1

        Returns:
        -------
        Tensor
            Samples from the empirical distribution.
        """
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        return self.locs[indices]

    def log_prob(self, z: Tensor) -> Tensor:
        """Raises because empirical log probabilities are unsupported."""
        raise NotImplementedError("EmpiricalPrior does not support log_prob.")
