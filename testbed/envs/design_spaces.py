"""Design spaces for BED environments."""

from abc import ABC, abstractmethod

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
        """Samples random designs with shape ``[*batch_shape, *event_shape]``."""
        ...

    @abstractmethod
    def contains(self, design: Tensor) -> bool:
        """Checks whether a tensor lies in the space."""
        ...

    @abstractmethod
    def project(self, design: Tensor) -> Tensor:
        """Projects a tensor into the design space."""
        ...


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
