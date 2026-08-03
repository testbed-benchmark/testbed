"""Design spaces handle constraints, reparametrisation, and priors on designs."""

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .util import matches_event_shape


class DesignSpace(ABC):
    """Abstract base class for a Task's design space.

    Checks whether a tensor is a valid design, and optionally provides
    methods for reparametrising designs, projecting designs, and
    sampling from a design prior.

    Subclasses must implement :meth:`_contains`. By default,
    reparametrisation and projection correspond to no-ops.
    This is only an abstract class and should be subclassed in actual use cases,
    e.g., :class:`DefaultDesignSpace` provides a default unconstrained space.

    :meth:`project` and :meth:`to_design` both produce valid designs but are
    not the same operation: ``project`` is a typically lossy, typically
    non-differentiable correction of an already-nearly-valid design, while
    ``to_design`` is a (typically) smooth reparametrization of an unconstrained value.

    - :meth:`_contains` (**required**): the space's value constraints.
      Reached through the public :meth:`contains`, which validates shape
      first, so subclasses never re-check shape.
    - :meth:`sample`: draw a random valid design. Needed
      by RandomDesigner and by callers wanting a design prior (default: raises).
    - :meth:`project`: map an arbitrary tensor into the
      space, e.g. by clipping (default: identity).
    - :meth:`to_design` / :meth:`to_raw`: reparametrise between raw and valid
      designs, optionally carrying a log-prob through the change of variables.
      Subclasses override :meth:`_to_design` / :meth:`_to_raw` (default: identity).
    - :meth:`log_det_jacobian`: log-det of that reparametrisation; required
      whenever :meth:`_to_design` is overridden (default: zeros).

    Args:
        event_shape (iterable):
            Shape of a single design.
        is_discrete (bool):
            Whether designs take discrete values.
    """

    def __init__(self, event_shape, is_discrete: bool):
        """Initializes the design space."""
        self.event_shape = torch.Size(event_shape)
        self.is_discrete = bool(is_discrete)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"{type(self).__name__}(event_shape={tuple(self.event_shape)})"

    def contains(self, design) -> bool:
        """Checks whether a tensor lies in the space.

        A tensor of the wrong shape is not an element of the space, so this
        validates the trailing (event) dimensions before delegating the value
        constraints to :meth:`_contains`. Leading dimensions are treated as a
        batch, and the result is a single bool for the whole batch: True only
        if *every* design in it lies in the space.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape, *event_shape]``.

        Returns:
            bool:
                Whether all designs in the batch lie in the space.
        """
        design = torch.as_tensor(design)
        if not matches_event_shape(design, self.event_shape):
            return False
        return self._contains(design)

    @abstractmethod
    def _contains(self, design: Tensor) -> bool:
        """Checks whether a batch of designs satisfies the space's constraints.

        Called by :meth:`contains` once ``design`` is known to be a tensor with
        trailing shape ``event_shape``, so implementations express only the
        value constraints. A space with no value constraints returns True.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape, *event_shape]``.

        Returns:
            bool:
                Whether all designs in the batch lie in the space.
        """
        ...

    # TODO this possibly wants to be cleaned up a bit
    # batch_shape doesn't match other conventions in testbed
    # and explicitly passing in a device might not be the cleanest way to do this.
    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples random designs with shape ``[*batch_shape, *event_shape]``.

        Raises a NotImplementedError by default. Subclasses can override this
        class to specify a design prior. Implementations follow the
        ``torch.distributions`` convention of an explicit ``batch_shape``,
        rather than the ``n_samples`` convention used by
        :class:`~testbed.core.distributions.Distribution` and ``Task``:
        ``sample()`` returns a single unbatched design, and no leading
        dimension is ever squeezed away.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sampling.")

    def project(self, design: Tensor) -> Tensor:
        """Projects a tensor into the design space.

        Default: identity.
        """
        return design

    def to_design(self, raw_design: Tensor, log_prob: Tensor | None = None):
        """Maps a raw/latent value to a design in this space. Default: identity.

        With ``log_prob`` (the log-density of ``raw_design``, shape
        ``[*batch_shape]``) this is a change of variables: returns
        ``(design, log_prob - log_det_jacobian(raw_design))``, the density of
        the design. Without it, returns just the design.

        Subclasses override :meth:`_to_design` and :meth:`log_det_jacobian`,
        not this method.
        """
        design = self._to_design(raw_design)
        if log_prob is None:
            return design
        return design, log_prob - self._log_det_jacobian(raw_design)

    def to_raw(self, design: Tensor, log_prob: Tensor | None = None):
        """Inverse of :meth:`to_design`. Default: identity.

        Undoes the change of variables when ``log_prob`` (the log-density of
        ``design``) is given. Often not required even when :meth:`to_design`
        is supported.
        """
        raw_design = self._to_raw(design)
        if log_prob is None:
            return raw_design
        return raw_design, log_prob + self._log_det_jacobian(raw_design)

    def _to_design(self, raw_design: Tensor) -> Tensor:
        """Pointwise raw -> design map. Default: identity.

        Subclasses should override this method if reparametrisation is needed.
        """
        return raw_design

    def _to_raw(self, design: Tensor) -> Tensor:
        """Pointwise design -> raw map. Default: identity.

        Subclasses should override this method if reparametrisation is needed.
        """
        return design

    def log_det_jacobian(self, raw_design: Tensor) -> Tensor:
        """Log |det d(design)/d(raw)| at ``raw_design``, summed over event dims.

        Returns shape ``[*batch_shape]``, matching the ``log_prob`` that
        :meth:`to_design` corrects. Default: zeros, which is correct for the
        identity map, so any subclass overriding :meth:`_to_design` must
        override this too (:meth:`to_design` raises if it doesn't).
        """
        batch_ndim = raw_design.ndim - len(self.event_shape)
        return raw_design.new_zeros(raw_design.shape[:batch_ndim])

    def _log_det_jacobian(self, raw_design: Tensor) -> Tensor:
        """Refuses to silently apply a zero correction to a real transform."""
        cls = type(self)
        if (
            cls._to_design is not DesignSpace._to_design
            and cls.log_det_jacobian is DesignSpace.log_det_jacobian
        ):
            raise NotImplementedError(
                f"{cls.__name__} reparametrises designs but does not implement "
                "log_det_jacobian, so log-probs cannot be transformed."
            )
        return self.log_det_jacobian(raw_design)

    def _sum_event_dims(self, x: Tensor) -> Tensor:
        """Sums trailing event dims of a per-element quantity, leaving the batch."""
        if not self.event_shape:
            return x
        return x.sum(dim=tuple(range(-len(self.event_shape), 0)))


class DefaultDesignSpace(DesignSpace):
    """Unconstrained space: every real tensor of the right shape is valid.

    The default ``Task.design_space`` for tasks that declare no real space of
    their own (e.g. LocationFinding). Constrains nothing beyond shape, so it
    keeps the identity ``project``/``to_design``/``to_raw``. :meth:`sample`
    draws standard normal.

    Tasks that want to use an unconstrained design space but a custom design prior
    can subclass DefaultDesignSpace and override :meth:`sample()`.

    Args:
        event_shape (iterable):
            Shape of a single design.
    """

    def __init__(self, event_shape):
        """Initializes with the task's design event shape."""
        super().__init__(event_shape, is_discrete=False)

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Draws a standard normal sample."""
        batch_shape = torch.Size(batch_shape)
        return torch.randn(*batch_shape, *self.event_shape, device=device)

    def _contains(self, design: Tensor) -> bool:
        """Accepts any value; shape is already checked by contains()."""
        return True


class BoxDesignSpace(DesignSpace):
    """Continuous box design space, i.e. elementwise ``low <= x <= high``.

    The raw<->design reparametrisation is the tanh squash
    ``design = bias + scale * tanh(raw)``, with ``bias = (high + low) / 2`` and
    ``scale = (high - low) / 2``. This is the same transform
    :class:`~testbed.design_nets.stochastic.TanhGaussianPolicyNet` applies
    internally; that net predates this method and still owns its own copy,
    because its squash has to enter the sampled log-prob.

    Args:
        low (Tensor or float):
            Lower bounds, broadcast against ``high`` and ``event_shape``.
        high (Tensor or float):
            Upper bounds, broadcast against ``low`` and ``event_shape``.
        event_shape (iterable):
            Shape of a single design.
    """

    def __init__(self, low, high, event_shape):
        """Initializes lower/upper bounds."""
        low = torch.as_tensor(low)
        high = torch.as_tensor(high, device=low.device)

        # Integer bounds are legitimate input for a continuous box, so promote
        # to a float dtype rather than sampling/comparing in integer arithmetic.
        dtype = torch.promote_types(low.dtype, high.dtype)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.get_default_dtype()
        low = low.to(dtype=dtype)
        high = high.to(dtype=dtype)

        super().__init__(event_shape, is_discrete=False)

        self.low = low.broadcast_to(self.event_shape)
        self.high = high.broadcast_to(self.event_shape)
        if torch.any(self.low > self.high):
            raise ValueError("BoxDesignSpace requires low <= high.")

    def _bounds_like(self, design: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the bounds on the device and dtype of a design."""
        return (
            self.low.to(device=design.device, dtype=design.dtype),
            self.high.to(device=design.device, dtype=design.dtype),
        )

    def sample(self, batch_shape=(), *, device=None) -> Tensor:
        """Samples uniformly from the box."""
        device = self.low.device if device is None else device
        low, high = self.low.to(device), self.high.to(device)
        batch_shape = torch.Size(batch_shape)
        u = torch.rand(*batch_shape, *self.event_shape, device=device, dtype=low.dtype)
        return low + (high - low) * u

    def _contains(self, design: Tensor) -> bool:
        """Checks the bounds."""
        low, high = self._bounds_like(design)
        return bool(torch.all((design >= low) & (design <= high)).item())

    def project(self, design: Tensor) -> Tensor:
        """Clamps to the box bounds."""
        low, high = self._bounds_like(design)
        return design.clamp(low, high)

    def _squash_affine(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the (bias, scale) of the tanh squash, on x's device and dtype."""
        low, high = self._bounds_like(x)
        scale = (high - low) / 2.0
        if not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item()):
            raise ValueError(
                "BoxDesignSpace reparametrisation requires finite bounds with "
                f"low < high; got low={self.low}, high={self.high}."
            )
        return (high + low) / 2.0, scale

    def _to_design(self, raw_design: Tensor) -> Tensor:
        """Tanh-squashes an unconstrained value into the box."""
        bias, scale = self._squash_affine(raw_design)
        return bias + scale * torch.tanh(raw_design)

    def _to_raw(self, design: Tensor) -> Tensor:
        """Inverts the squash; a design on the boundary maps to a large finite raw."""
        bias, scale = self._squash_affine(design)
        eps = torch.finfo(design.dtype).eps
        return torch.atanh(((design - bias) / scale).clamp(-1 + eps, 1 - eps))

    def log_det_jacobian(self, raw_design: Tensor) -> Tensor:
        """Log-det of the tanh-then-affine squash."""
        _, scale = self._squash_affine(raw_design)
        # log(1 - tanh(x)^2) in a numerically stable form
        # evaluated forward so it stays finite once tanh saturates in float.
        tanh_log_det = 2.0 * (
            math.log(2.0) - raw_design - torch.nn.functional.softplus(-2.0 * raw_design)
        )
        return self._sum_event_dims(torch.log(scale) + tanh_log_det)
