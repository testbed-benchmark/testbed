"""Distribution abstractions and adapters."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.distributions as td
from torch import Tensor, nn
from torch.distributions import (
    Independent,
    MultivariateNormal,
    TransformedDistribution,
    Uniform,
    constraints,
)
from torch.distributions.transforms import ExpTransform, Transform

from .constraints import _matches, finite_set
from .util import matches_event_shape


def as_sample_shape(sample_shape: int | Iterable[int]) -> torch.Size:
    """Normalizes a sample shape, reading an int ``n`` as ``(n,)``.

    An int always adds a sample dim: ``1`` gives a leading dim of size one,
    and only ``()`` gives none.
    """
    if isinstance(sample_shape, int):
        return torch.Size((sample_shape,))
    return torch.Size(sample_shape)


class Distribution(ABC, nn.Module, td.Distribution):
    """A ``torch.distributions.Distribution`` that is also an ``nn.Module``.

    The base for distributions written by hand, which implement ``sample`` and
    ``log_prob`` and keep their tensors as buffers so they follow ``.to()``. For
    anything torch provides, wrap it in :class:`TorchDistribution` instead.

    Its ``batch_shape`` is always empty. Shapes follow ``torch.distributions``:

    - ``sample(sample_shape)`` returns ``[*sample_shape, *event_shape]``.
    - ``log_prob(value)`` returns the joint log density of each draw, mapping
      ``[*b, *event_shape]`` to ``[*b]``.

    Args:
        event_shape (iterable):
            Shape of a single draw.
    """

    def __init__(self, event_shape: Iterable[int]) -> None:
        """Initializes the distribution."""
        nn.Module.__init__(self)
        td.Distribution.__init__(
            self, torch.Size(), torch.Size(event_shape), validate_args=False
        )

    @property
    def support(self) -> constraints.Constraint:
        """The constraint draws satisfy: all of real space unless overridden."""
        return constraints.independent(constraints.real, len(self.event_shape))

    def _extended_shape(self, sample_shape: int | Iterable[int] = ()) -> torch.Size:
        """Returns ``[*sample_shape, *event_shape]``."""
        return as_sample_shape(sample_shape) + self.event_shape

    @abstractmethod
    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples of shape ``[*sample_shape, *event_shape]``."""
        ...

    @abstractmethod
    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluates log probabilities, ``[*b, *event_shape] -> [*b]``."""
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
    """Returns the device a torch.distributions object's tensors live on.

    Composite distributions (MixtureSameFamily, Independent) hold no tensors
    directly, so nested distributions are searched too.
    """
    for value in dist.__dict__.values():
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, torch.distributions.Distribution):
            return _wrapped_device(value)
    return torch.device("cpu")


def _apply_to_tensors(obj, fn, seen: set):
    """Applies ``fn`` to every tensor a torch distribution holds, in place.

    Recurses into the distributions, transforms and constraints it is built
    from, and into lists and tuples of them. Modules are skipped, since they
    move themselves.
    """
    if isinstance(obj, Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple)) and not isinstance(obj, torch.Size):
        return type(obj)(_apply_to_tensors(item, fn, seen) for item in obj)
    if (
        isinstance(obj, (td.Distribution, Transform, constraints.Constraint))
        and not isinstance(obj, nn.Module)
        and id(obj) not in seen
    ):
        seen.add(id(obj))
        for name, value in vars(obj).items():
            vars(obj)[name] = _apply_to_tensors(value, fn, seen)
    return obj


def _skip_sample_checks(dist: td.Distribution) -> None:
    """Stops a torch distribution, and those it is built from, raising on values outside the support."""
    dist._validate_args = False
    for value in vars(dist).values():
        if isinstance(value, td.Distribution) and not isinstance(value, nn.Module):
            _skip_sample_checks(value)


class TorchDistribution(Distribution):
    """Adapts any ``torch.distributions.Distribution`` so that it follows ``.to()``.

    ``nn.Module.to()`` moves only parameters and buffers, but a torch
    distribution keeps its tensors as plain attributes. This wrapper's
    ``_apply``, which ``.to()``, ``.cuda()`` and ``.double()`` go through,
    converts those as well, moving the wrapped distribution in place.

    Every batch dim of ``dist`` is folded into the event (via ``Independent``),
    so that ``log_prob`` returns one value per draw. For example,
    ``MultivariateNormal(torch.zeros(3, 2), torch.eye(2))`` has ``batch_shape
    [3]`` and ``event_shape [2]``; wrapped, it is a single distribution with
    ``event_shape [3, 2]`` whose ``log_prob`` sums the three per-row densities.

    ``log_prob`` is ``-inf`` outside the support, so the wrapped distribution's
    own check of ``log_prob``'s input, which would raise there, is turned off.
    Validating its arguments, which torch does when it is built, is unaffected.

    Args:
        dist (torch.distributions.Distribution):
            The distribution to wrap.
    """

    def __init__(self, dist: td.Distribution):
        """Wraps a torch distribution."""
        if dist.batch_shape:
            dist = Independent(dist, len(dist.batch_shape))
        _skip_sample_checks(dist)
        super().__init__(dist.event_shape)
        self.dist = dist
        # Empty and non-persistent: a buffer that .to() moves, so that this
        # module, and a task holding it, report the wrapped distribution's device.
        self.register_buffer(
            "_device_tracker",
            torch.empty(0, device=_wrapped_device(dist)),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        """Converts the wrapped distribution's tensors along with the buffers."""
        super()._apply(fn, recurse)
        _apply_to_tensors(self.dist, fn, set())
        return self

    @property
    def support(self) -> constraints.Constraint:
        """The wrapped distribution's support."""
        return self.dist.support

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples from the wrapped distribution."""
        return self.dist.sample(as_sample_shape(sample_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluates log probabilities; ``-inf`` outside the support.

        Torch's own may be NaN there instead, e.g. log of a negative number.
        """
        value = value.to(self.device)
        log_prob = self.dist.log_prob(value)
        return torch.where(self.dist.support.check(value), log_prob, -torch.inf)


def as_distribution(dist: td.Distribution) -> Distribution:
    """Returns ``dist`` as a testbed Distribution, wrapping a torch one."""
    return dist if isinstance(dist, Distribution) else TorchDistribution(dist)


class LogNormal(TorchDistribution):
    r"""Multivariate log-normal: the law of :math:`\exp(z)`, :math:`z \sim \mathcal{N}(\mu, \Sigma)`.

    Args:
        loc (Tensor):
            Mean :math:`\mu` of the underlying normal. Shape ``[d]``.
        cov (Tensor):
            Covariance :math:`\Sigma` of the underlying normal. Shape ``[d, d]``.
    """

    def __init__(self, loc: Tensor, cov: Tensor) -> None:
        """Builds an exp-transformed ``MultivariateNormal``."""
        normal = MultivariateNormal(torch.as_tensor(loc), torch.as_tensor(cov))
        super().__init__(TransformedDistribution(normal, [ExpTransform()]))


class EmpiricalDistribution(Distribution):
    """Weighted points, e.g. posterior samples or a finite set of designs.

    Its support is the finite set of points, so as a design prior it makes the
    design space discrete, with :meth:`encode` / :meth:`decode` converting
    between a design and its index.

    Args:
        locs (Tensor):
            The points, shape ``[n, *event_shape]``.
        weights (Tensor, optional):
            Non-negative weights, not necessarily normalized, shape ``[n]`` or
            ``[n, 1]``. None means uniform (default: None).
    """

    def __init__(self, locs: Tensor, weights: Tensor | None = None) -> None:
        """Initializes weighted sample locations."""
        super().__init__(event_shape=locs.shape[1:])

        if weights is None:
            weights = torch.ones(locs.shape[0], device=locs.device)
        assert torch.all(weights >= 0), "Weights must be non-negative."
        self.register_buffer("locs", locs)
        self.register_buffer("weights", weights.reshape(-1))

    @property
    def support(self) -> constraints.Constraint:
        """The finite set of points."""
        return finite_set(self.locs)

    @property
    def ess(self) -> Tensor:
        """Effective sample size, ``1 / sum(w**2)`` for normalized weights."""
        weights = self.weights / self.weights.sum()
        return 1.0 / (weights**2).sum()

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Samples from the empirical distribution.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, *event_shape]``.
        """
        sample_shape = as_sample_shape(sample_shape)
        indices = torch.multinomial(
            self.weights, sample_shape.numel(), replacement=True
        )
        return self.locs[indices].reshape(self._extended_shape(sample_shape))

    def log_prob(self, z: Tensor) -> Tensor:
        """Raises because empirical log probabilities are unsupported."""
        raise NotImplementedError("EmpiricalPrior does not support log_prob.")

    def encode(self, value: Tensor) -> Tensor:
        """Maps points in the set to their indices.

        Args:
            value: Points with trailing shape ``event_shape``.

        Returns:
            Indices with the points' batch shape.

        Raises:
            ValueError: If a point is not in the set.
        """
        value = torch.as_tensor(value)
        if not matches_event_shape(value, self.event_shape):
            raise ValueError(
                f"Value must have trailing shape {tuple(self.event_shape)}; "
                f"got shape {tuple(value.shape)}."
            )
        matches = _matches(self.locs, value)
        if not matches.any(dim=-1).all():
            raise ValueError("Value is outside the finite set.")
        return matches.to(torch.long).argmax(dim=-1)

    def decode(self, index: Tensor) -> Tensor:
        """Maps indices to points in the set.

        Args:
            index: Integer-valued indices.

        Returns:
            Points with shape ``[*index.shape, *event_shape]``.

        Raises:
            ValueError: If an index is not an integer in range.
        """
        index = torch.as_tensor(index)
        if not torch.equal(index, index.round()):
            raise ValueError("Index must be integer-valued.")
        if torch.any((index < 0) | (index >= self.locs.shape[0])):
            raise ValueError("Index is outside the finite set.")
        return self.locs.to(device=index.device)[index.long()]


class Box(TorchDistribution):
    """Uniform distribution on the box ``low <= x <= high``.

    As a design prior it makes the design space that box, which the default
    design transform maps raw designs onto with tanh.

    Args:
        low (Tensor or float):
            Lower bounds, broadcast against ``high`` and ``event_shape``.
        high (Tensor or float):
            Upper bounds, broadcast against ``low`` and ``event_shape``.
        event_shape (iterable):
            Shape of a single draw.
    """

    def __init__(self, low, high, event_shape) -> None:
        """Builds an ``Independent(Uniform)`` between the bounds."""
        event_shape = torch.Size(event_shape)
        low = torch.as_tensor(low)
        high = torch.as_tensor(high, device=low.device)

        # Integer bounds are legitimate input for a continuous box, so promote
        # to a float dtype rather than sampling/comparing in integer arithmetic.
        dtype = torch.promote_types(low.dtype, high.dtype)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.get_default_dtype()
        low = low.to(dtype).broadcast_to(event_shape).clone()
        high = high.to(dtype).broadcast_to(event_shape).clone()
        if torch.any(low > high):
            raise ValueError("Box requires low <= high.")

        # torch's Uniform rejects low == high, but a zero-width box is allowed
        uniform = Uniform(low, high, validate_args=False)
        super().__init__(Independent(uniform, len(event_shape)))

    def log_prob(self, value: Tensor) -> Tensor:
        """Minus the log volume inside the closed box, ``-inf`` outside.

        Torch's ``Uniform`` is half-open, so its density is ``-inf`` at
        ``high``, where projecting designs onto the box puts them. A
        zero-width dimension pins a coordinate rather than spreading mass over
        it, so it contributes no volume.
        """
        value = value.to(self.device)
        width = self.high - self.low
        log_volume = torch.where(width > 0, torch.log(width), 0.0).sum()
        return torch.where(self.support.check(value), -log_volume, -torch.inf)

    @property
    def low(self) -> Tensor:
        """Lower bounds, shape ``event_shape``."""
        return self.dist.base_dist.low

    @property
    def high(self) -> Tensor:
        """Upper bounds, shape ``event_shape``."""
        return self.dist.base_dist.high
