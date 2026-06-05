"""Torch-first design/action spaces for BED environments."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module

import torch
from torch import Tensor


def _has_event_shape(tensor: Tensor, event_shape: torch.Size) -> bool:
    """Checks whether trailing dimensions match an event shape."""
    if len(event_shape) == 0:
        return True
    return tensor.shape[-len(event_shape) :] == event_shape


class DesignSpace(ABC):
    """Base class for environment design spaces."""

    event_shape: torch.Size
    is_discrete: bool

    @abstractmethod
    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples random designs with shape [*batch_shape, *event_shape]."""
        ...

    @abstractmethod
    def contains(self, design: Tensor) -> bool:
        """Checks whether a tensor lies in the space."""
        ...

    @abstractmethod
    def project(self, design: Tensor) -> Tensor:
        """Projects a tensor into the design space."""
        ...

    def to_gymnasium(self):
        """Converts to a Gymnasium space when Gymnasium is installed."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement Gymnasium conversion."
        )


class BoxDesignSpace(DesignSpace):
    """Continuous box design space."""

    is_discrete = False

    def __init__(self, low, high, event_shape=None):
        """Initializes lower/upper bounds."""
        low = torch.as_tensor(low)
        high = torch.as_tensor(high, device=low.device)
        dtype = torch.promote_types(low.dtype, high.dtype)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.get_default_dtype()
        low = low.to(dtype=dtype)
        high = high.to(dtype=dtype)
        if event_shape is None:
            event_shape = torch.broadcast_shapes(low.shape, high.shape)
        self.event_shape = torch.Size(event_shape)
        self.low = low.broadcast_to(self.event_shape)
        self.high = high.broadcast_to(self.event_shape)
        if torch.any(self.low > self.high):
            raise ValueError("BoxDesignSpace requires low <= high.")

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples uniformly from the box."""
        device = device or self.low.device
        low = self.low.to(device)
        high = self.high.to(device)
        batch_shape = torch.Size(batch_shape)
        u = torch.rand(*batch_shape, *self.event_shape, device=device, dtype=low.dtype)
        return low + (high - low) * u

    def contains(self, design: Tensor) -> bool:
        """Checks shape and bounds."""
        design = torch.as_tensor(design)
        if not _has_event_shape(design, self.event_shape):
            return False
        low = self.low.to(device=design.device, dtype=design.dtype)
        high = self.high.to(device=design.device, dtype=design.dtype)
        return bool(torch.all((design >= low) & (design <= high)).item())

    def project(self, design: Tensor) -> Tensor:
        """Clamps to the box bounds."""
        low = self.low.to(device=design.device, dtype=design.dtype)
        high = self.high.to(device=design.device, dtype=design.dtype)
        return design.clamp(low, high)

    def to_gymnasium(self):
        """Converts to gymnasium.spaces.Box."""
        gym = import_module("gymnasium")

        return gym.spaces.Box(
            low=self.low.cpu().numpy(),
            high=self.high.cpu().numpy(),
            shape=tuple(self.event_shape),
            dtype=self.low.cpu().numpy().dtype,
        )


class IntegerDesignSpace(DesignSpace):
    """Integer box design space with inclusive bounds."""

    is_discrete = True

    def __init__(self, low, high, event_shape=None):
        """Initializes inclusive integer bounds."""
        low = torch.as_tensor(low, dtype=torch.long)
        high = torch.as_tensor(high, dtype=torch.long)
        if event_shape is None:
            event_shape = torch.broadcast_shapes(low.shape, high.shape)
        self.event_shape = torch.Size(event_shape)
        self.low = low.broadcast_to(self.event_shape)
        self.high = high.broadcast_to(self.event_shape)
        if torch.any(self.low > self.high):
            raise ValueError("IntegerDesignSpace requires low <= high.")

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples integer designs uniformly."""
        device = device or self.low.device
        batch_shape = torch.Size(batch_shape)
        low = self.low.to(device)
        high = self.high.to(device)
        u = torch.rand(*batch_shape, *self.event_shape, device=device)
        return torch.floor(u * (high - low + 1) + low).to(torch.long)

    def contains(self, design: Tensor) -> bool:
        """Checks shape, integer values, and bounds."""
        design = torch.as_tensor(design)
        if not _has_event_shape(design, self.event_shape):
            return False
        if torch.is_floating_point(design) and not torch.all(design == design.round()):
            return False
        low = self.low.to(design.device)
        high = self.high.to(design.device)
        return bool(torch.all((design >= low) & (design <= high)).item())

    def project(self, design: Tensor) -> Tensor:
        """Rounds and clamps to the integer box."""
        low = self.low.to(device=design.device)
        high = self.high.to(device=design.device)
        return design.round().clamp(low, high).to(torch.long)

    def to_gymnasium(self):
        """Converts to a Gymnasium integer Box."""
        gym = import_module("gymnasium")

        return gym.spaces.Box(
            low=self.low.cpu().numpy(),
            high=self.high.cpu().numpy(),
            shape=tuple(self.event_shape),
            dtype=self.low.cpu().numpy().dtype,
        )


class CategoricalDesignSpace(DesignSpace):
    """Scalar categorical design space with values 0, ..., n - 1."""

    event_shape = torch.Size([])
    is_discrete = True

    def __init__(self, n: int):
        """Initializes a categorical design space."""
        self.n = int(n)
        if self.n <= 0:
            raise ValueError("CategoricalDesignSpace requires n > 0.")

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples category indices."""
        return torch.randint(self.n, torch.Size(batch_shape), device=device)

    def contains(self, design: Tensor) -> bool:
        """Checks integer category membership."""
        design = torch.as_tensor(design)
        if torch.is_floating_point(design) and not torch.all(design == design.round()):
            return False
        return bool(torch.all((design >= 0) & (design < self.n)).item())

    def project(self, design: Tensor) -> Tensor:
        """Rounds and clamps to valid categories."""
        return design.round().clamp(0, self.n - 1).to(torch.long)

    def to_gymnasium(self):
        """Converts to gymnasium.spaces.Discrete."""
        gym = import_module("gymnasium")

        return gym.spaces.Discrete(self.n)


class ProductDesignSpace(DesignSpace):
    """Flattened product of design spaces.

    Samples and projections concatenate flattened components into one vector.
    """

    def __init__(self, spaces: Sequence[DesignSpace]):
        """Initializes a product space."""
        if not spaces:
            raise ValueError("ProductDesignSpace requires at least one component.")
        self.spaces = tuple(spaces)
        self._sizes = [space.event_shape.numel() for space in self.spaces]
        self.event_shape = torch.Size([sum(self._sizes)])
        self.is_discrete = all(space.is_discrete for space in self.spaces)

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples and flattens component designs."""
        parts = [
            space.sample(batch_shape, device=device).reshape(*batch_shape, size)
            for space, size in zip(self.spaces, self._sizes)
        ]
        return torch.cat(parts, dim=-1)

    def contains(self, design: Tensor) -> bool:
        """Checks each flattened component."""
        design = torch.as_tensor(design)
        if design.shape[-1:] != self.event_shape:
            return False
        for part, space in zip(torch.split(design, self._sizes, dim=-1), self.spaces):
            part = part.reshape(*design.shape[:-1], *space.event_shape)
            if not space.contains(part):
                return False
        return True

    def project(self, design: Tensor) -> Tensor:
        """Projects each flattened component."""
        parts = []
        for part, space, size in zip(
            torch.split(design, self._sizes, dim=-1), self.spaces, self._sizes
        ):
            part = part.reshape(*design.shape[:-1], *space.event_shape)
            parts.append(space.project(part).reshape(*design.shape[:-1], size))
        return torch.cat(parts, dim=-1)
