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

    def project(self, design: Tensor) -> Tensor:
        """Returns a design unchanged when the space has no projection."""
        return design


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


class CategoricalDesignSpace(DesignSpace):
    """Finite set of categorical designs.

    Args:
        values: Valid physical designs with shape
            ``[n_actions, *event_shape]``.
    """

    is_discrete = True

    def __init__(self, values) -> None:
        """Initializes the finite design set.

        Raises:
            ValueError: If no actions are supplied or actions are duplicated.
        """
        values = torch.as_tensor(values)
        if values.ndim == 0 or values.shape[0] == 0:
            raise ValueError("CategoricalDesignSpace requires at least one action.")
        self.values = values
        self.event_shape = torch.Size(values.shape[1:])
        flat_values = values.reshape(values.shape[0], -1)
        if torch.unique(flat_values, dim=0).shape[0] != values.shape[0]:
            raise ValueError("CategoricalDesignSpace requires unique actions.")

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples uniformly from the valid physical designs."""
        device = device or self.values.device
        indices = torch.randint(self.n_actions, torch.Size(batch_shape), device=device)
        return self.decode(indices)

    def contains(self, design: Tensor) -> bool:
        """Checks whether every design is one of the valid actions."""
        design = torch.as_tensor(design)
        if not _has_event_shape(design, self.event_shape):
            return False
        return bool(self._matches(design).any(dim=-1).all().item())

    @property
    def n_actions(self) -> int:
        """Returns the number of valid categorical actions."""
        return self.values.shape[0]

    def encode(self, design: Tensor) -> Tensor:
        """Maps valid physical designs to categorical action indices.

        Raises:
            ValueError: If a design is not in the categorical set.
        """
        design = torch.as_tensor(design)
        if not _has_event_shape(design, self.event_shape):
            raise ValueError("Design has the wrong event shape.")
        matches = self._matches(design)
        if not matches.any(dim=-1).all():
            raise ValueError("Design is outside the categorical design space.")
        batch_shape = (
            design.shape[: -len(self.event_shape)] if self.event_shape else design.shape
        )
        return matches.to(torch.long).argmax(dim=-1).reshape(batch_shape)

    def decode(self, index: Tensor) -> Tensor:
        """Maps categorical action indices to physical designs.

        Raises:
            ValueError: If an index is outside the categorical action range.
        """
        index = torch.as_tensor(index)
        if not torch.equal(index, index.round()):
            raise ValueError("Action index must be integer-valued.")
        if torch.any((index < 0) | (index >= self.n_actions)):
            raise ValueError("Action index is outside the categorical design space.")
        return self.values.to(device=index.device)[index.long()]

    def _matches(self, design: Tensor) -> Tensor:
        """Returns action-match indicators with shape ``[*batch_shape, n_actions]``."""
        values = self.values.to(device=design.device, dtype=design.dtype)
        flat_design = design.reshape(-1, self.event_shape.numel())
        flat_values = values.reshape(self.n_actions, self.event_shape.numel())
        matches = (flat_design.unsqueeze(1) == flat_values.unsqueeze(0)).all(dim=-1)
        batch_shape = (
            design.shape[: -len(self.event_shape)] if self.event_shape else design.shape
        )
        return matches.reshape(*batch_shape, self.n_actions)
