"""Design spaces: constraints and raw/design reparametrization for Tasks."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


def _has_event_shape(tensor: Tensor, event_shape: torch.Size) -> bool:
    """Checks whether trailing dimensions match an event shape."""
    if len(event_shape) == 0:
        return True
    return tensor.shape[-len(event_shape) :] == event_shape


class DesignSpace(ABC):
    """Base class for a Task's design space.

    Beyond the space itself (sample/contains/project), a DesignSpace may
    support a raw<->design reparametrization (to_design/to_raw), used by
    designers that optimize or condition a net on a raw/latent
    parametrization rather than the design directly -- for pathwise/
    reparameterization-trick gradients, or for numerically stable history
    conditioning. Default: identity, i.e. raw and design coincide.
    """

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

    def to_design(self, raw_design: Tensor) -> Tensor:
        """Maps a raw/latent value to a design in this space.

        A pure pointwise map -- no log-det-Jacobian or density tracking --
        for pathwise/reparameterization-trick gradients (SequentialDesigner,
        DeterministicPolicyDesigner, StaticDesigner). Default: identity.
        """
        return raw_design

    def to_raw(self, design: Tensor) -> Tensor:
        """Inverse of to_design. Default: identity.

        Not every DesignSpace can support this efficiently or safely --
        override to raise if inversion isn't well-defined (e.g. because the
        forward map is lossy, or because designs should only ever be
        produced from a raw value, never inverted back to one).
        """
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


class DefaultDesignSpace(DesignSpace):
    """Every real vector of the right shape is valid.

    The default Task.design_space for tasks that declare no real space of
    their own (e.g. LocationFinding). sample() draws standard normal --
    there's no well-defined "uniform over R^d" -- matching what designers
    already assumed by default before this class existed.
    """

    is_discrete = False

    def __init__(self, event_shape):
        """Initializes with the task's design event shape."""
        self.event_shape = torch.Size(event_shape)

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Draws a standard normal sample."""
        batch_shape = torch.Size(batch_shape)
        return torch.randn(*batch_shape, *self.event_shape, device=device)

    def contains(self, design: Tensor) -> bool:
        """Checks shape only -- every value is valid."""
        return _has_event_shape(torch.as_tensor(design), self.event_shape)

    def project(self, design: Tensor) -> Tensor:
        """No-op: every value is already valid."""
        return design
