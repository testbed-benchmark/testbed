"""Distribution abstractions and adapters."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributions import (
    Independent,
    MultivariateNormal,
    TransformedDistribution,
)
from torch.distributions.transforms import ExpTransform


def as_sample_shape(sample_shape: int | Iterable[int]) -> torch.Size:
    """Normalizes a sample shape, reading an int ``n`` as ``(n,)``.

    An int always adds a sample dim: ``1`` gives a leading dim of size one,
    and only ``()`` gives none.
    """
    if isinstance(sample_shape, int):
        return torch.Size((sample_shape,))
    return torch.Size(sample_shape)


class Distribution(ABC, nn.Module):
    """Abstract class for arbitrary distributions.

    Shapes follow ``torch.distributions``:

    - ``sample(sample_shape)`` returns ``[*sample_shape, *batch_shape, *event_shape]``.
    - ``log_prob(value)`` maps ``[*b, *event_shape]`` to ``[*b]``, reducing over
      exactly the event dims. ``b`` broadcasts against ``batch_shape`` and may
      be empty.

    A prior over a task's parameters has ``event_shape == event_shape_theta``
    and an empty ``batch_shape``.

    Args:
        event_shape (iterable):
            Shape of a single draw.
        batch_shape (iterable):
            Shape of a batch of independently parameterized distributions
            (default: ()).
    """

    def __init__(
        self, event_shape: Iterable[int], batch_shape: Iterable[int] = ()
    ) -> None:
        """Initializes the distribution."""
        super().__init__()
        self.event_shape = torch.Size(event_shape)
        self.batch_shape = torch.Size(batch_shape)

    def _extended_shape(self, sample_shape: int | Iterable[int] = ()) -> torch.Size:
        """Returns ``[*sample_shape, *batch_shape, *event_shape]``."""
        return as_sample_shape(sample_shape) + self.batch_shape + self.event_shape

    @abstractmethod
    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples of shape ``[*sample_shape, *batch_shape, *event_shape]``."""
        ...

    @abstractmethod
    def log_prob(self, theta: Any) -> Tensor:
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


class TorchDistribution(Distribution):
    """Adapts torch.distributions.Distribution to our Distribution class.

    A torch distribution keeps its parameters as plain attributes, which
    ``nn.Module.to()`` cannot reach, so ``.to(device)`` leaves them where they
    were. This wrapper tracks the requested device with a buffer instead and
    moves what crosses its interface: samples come back on that device, and
    values handed to ``log_prob`` are sent to the parameters' device to be
    evaluated. Parameters of a design/prior distribution are small and the
    transfers are per-call, so leaving them put costs little.

    By default every batch dim of ``dist`` is folded into the event (via
    ``Independent``), so that ``log_prob`` returns one value per draw. For
    example, ``MultivariateNormal(torch.zeros(3, 2), torch.eye(2))`` has
    ``batch_shape [3]`` and ``event_shape [2]``; wrapped, it is a single
    distribution with ``event_shape [3, 2]`` whose ``log_prob`` sums the three
    per-row densities.

    Args:
        dist (torch.distributions.Distribution):
            The distribution to wrap.
        reinterpreted_batch_ndims (int, optional):
            How many trailing batch dims of ``dist`` to fold into the event.
            None folds all of them; pass 0 to keep a genuinely batched
            distribution, e.g. one posterior per history (default: None).
    """

    def __init__(self, dist, reinterpreted_batch_ndims: int | None = None):
        """Wraps a torch distribution."""
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(dist.batch_shape)
        if reinterpreted_batch_ndims > 0:
            dist = Independent(dist, reinterpreted_batch_ndims)
        super().__init__(dist.event_shape, dist.batch_shape)
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

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples from the wrapped distribution."""
        return self.dist.sample(as_sample_shape(sample_shape)).to(self.device)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates log probabilities under the wrapped distribution."""
        log_prob = self.dist.log_prob(theta.to(_wrapped_device(self.dist)))
        return log_prob.to(self.device)


class LogNormal(Distribution):
    r"""Multivariate log-normal distribution.

    The distribution of :math:`\exp(z)` for :math:`z \sim \mathcal{N}(\mu, \Sigma)`,
    for parameters constrained to be positive. Torch has no native multivariate
    log-normal, so this wraps a ``MultivariateNormal`` in an ``ExpTransform``.

    Args:
        loc (Tensor):
            Mean :math:`\mu` of the underlying normal. Shape ``[d]``.
        cov (Tensor):
            Covariance :math:`\Sigma` of the underlying normal. Shape ``[d, d]``.
    """

    def __init__(self, loc: Tensor, cov: Tensor) -> None:
        """Initializes the underlying normal's parameters."""
        loc = torch.as_tensor(loc)
        super().__init__(event_shape=loc.shape[-1:])

        self.register_buffer("loc", loc)
        self.register_buffer("cov", torch.as_tensor(cov))
        self.dim = loc.shape[-1]

    @property
    def lognormal(self) -> torch.distributions.Distribution:
        """The distribution as an exp-transformed ``MultivariateNormal``."""
        mvn = MultivariateNormal(self.loc, covariance_matrix=self.cov)
        return TransformedDistribution(mvn, [ExpTransform()])

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, d]``.
        """
        return self.lognormal.sample(as_sample_shape(sample_shape))

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., d]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``.
        """
        return self.lognormal.log_prob(theta)


class EmpiricalDistribution(Distribution):
    """Base class for empirical distributions (i.e., implicitly defined by samples)."""

    def __init__(self, locs: Tensor, weights: Tensor | None = None) -> None:
        """Initializes weighted sample locations.

        locs: [b, *shape].
        weights: [b] or [b, 1], non-negative and not necessarily normalized.
        """
        super().__init__(event_shape=locs.shape[1:])

        if weights is None:
            weights = torch.ones(locs.shape[0], device=locs.device)
        assert torch.all(weights >= 0), "Weights must be non-negative."
        self.register_buffer("locs", locs)
        self.register_buffer("weights", weights.reshape(-1))

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
