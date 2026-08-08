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


def _wrapped_device(dist) -> torch.device:
    """Returns the device a torch.distributions object's own tensors live on."""
    for value in dist.__dict__.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


class TorchDistribution(Distribution):
    """Adapts torch.distributions.Distribution to our Distribution class.

    A torch distribution keeps its parameters as plain attributes, which
    ``nn.Module.to()`` cannot reach, so ``.to(device)`` leaves them where they
    were. This wrapper tracks the requested device with a buffer instead and
    moves what crosses its interface: samples come back on that device, and
    values handed to ``log_prob`` are sent to the parameters' device to be
    evaluated. Parameters of a design/prior distribution are small and the
    transfers are per-call, so leaving them put costs little.
    """

    def __init__(self, dist):
        """Wraps a torch distribution."""
        super().__init__()
        self.dist = dist
        # Empty, non-persistent (so it stays out of state_dict) and carries no
        # data: it exists only because .to() moves buffers, which is what gives
        # this wrapper a device to track. Starts where the wrapped
        # distribution's parameters are, so an untouched wrapper reports the
        # device it was built on.
        self.register_buffer(
            "_device_tracker",
            torch.empty(0, device=_wrapped_device(dist)),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Returns the device this distribution produces samples on."""
        return self._device_tracker.device

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples from the wrapped distribution."""
        return self.dist.sample((n_samples,)).to(self.device)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates log probabilities under the wrapped distribution."""
        log_prob = self.dist.log_prob(theta.to(_wrapped_device(self.dist)))
        return log_prob.to(self.device)


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
