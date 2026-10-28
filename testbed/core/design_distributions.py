"""Design-sampling distributions to be used with RandomDesigner."""

from collections.abc import Iterable

import torch
from torch import Tensor

from testbed.core.distributions import Distribution, as_sample_shape


class RandomScaleGaussianDesignDistribution(Distribution):
    """Per-trajectory random-scale isotropic Gaussian design proposal.

    Each trajectory draws its own scale ``s ~ Uniform(scale_low, scale_high)``;
    all ``T`` design steps within that trajectory are then drawn i.i.d.
    ``N(0, s^2)``. This is the "random scale Gaussian" design sampler from
    ScoreBED (Phillips, Kerrigan, Rainforth, UAI 2026), used
    there to pretrain a score/posterior net independently of the policy on the
    location-finding task.

    Only usable with ``RandomDesigner``'s vectorized (no-hooks) rollout path,
    which draws a full batch of trajectories (``b * T`` designs) in a single
    ``sample`` call and reshapes to ``[b, T, ...]`` -- that reshape is what
    makes consecutive groups of ``T`` draws share one scale. Attaching hooks
    to a ``RandomDesigner`` using this distribution would instead sample one
    step at a time with no way to know which trajectory or step index a given
    call belongs to, breaking the intended grouping.
    """

    def __init__(
        self,
        T: int,
        event_shape_design,
        scale_low: float = 0.5,
        scale_high: float = 2.0,
        device=None,
    ):
        """Initializes.

        Args:
            T: Steps per trajectory (must match the RandomDesigner's T).
            event_shape_design: Shape of a single design.
            scale_low: Lower bound for the per-trajectory scale.
            scale_high: Upper bound for the per-trajectory scale.
            device: Device to sample on (default: cpu; moves with `.to()`).
        """
        super().__init__(event_shape=event_shape_design)
        self.T = int(T)
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self._last_scale_flat = None
        self.register_buffer(
            "_device_tracker", torch.empty(0, device=device or "cpu"), persistent=False
        )

    @property
    def device(self) -> torch.device:
        """Returns the device this distribution samples on."""
        return self._device_tracker.device

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws designs, whole trajectories' worth at once.

        Args:
            sample_shape: Shape of the batch of draws. Its total size must be
                a multiple of `T` (one full trajectory, or a batch of them,
                per call), with consecutive groups of `T` draws, in row-major
                order, sharing one scale.

        Returns:
            Tensor: Designs, shape ``[*sample_shape, *event_shape_design]``.
        """
        sample_shape = as_sample_shape(sample_shape)
        n_samples = sample_shape.numel()
        if n_samples % self.T != 0:
            raise ValueError(
                f"RandomScaleGaussianDesignDistribution requires n_samples "
                f"({n_samples}) to be a multiple of T ({self.T}): it is meant "
                "to be sampled one full trajectory (or a batch of full "
                "trajectories) at a time, via RandomDesigner's vectorized "
                "rollout path."
            )
        b = n_samples // self.T
        scale = torch.empty(b, device=self.device).uniform_(
            self.scale_low, self.scale_high
        )
        scale_flat = scale.repeat_interleave(
            self.T
        )  # [n_samples(==b*T)], shared within each trajectory
        self._last_scale_flat = scale_flat

        extra_dims = (1,) * len(self.event_shape)
        noise = torch.randn(n_samples, *self.event_shape, device=self.device)
        designs = noise * scale_flat.reshape(n_samples, *extra_dims)
        return designs.reshape(self._extended_shape(sample_shape))

    def log_prob(self, design: Tensor) -> Tensor:
        """Log density conditional on the scale drawn by the preceding `sample` call.

        Not a true marginal density (which would integrate out the random
        scale) -- `RandomDesigner` always pairs one `sample()` call with one
        `log_prob()` call on its result, (and nothing downstream in posterior specific
        calls to the implementation for pretraining consumes this value),
        so this conditional surrogate is sufficient.
        Do not call out of order or on a different-sized tensor.
        """
        batch_shape = design.shape[: design.ndim - len(self.event_shape)]
        n = batch_shape.numel()
        if self._last_scale_flat is None or n != self._last_scale_flat.shape[0]:
            raise RuntimeError(
                "log_prob must be called immediately after a sample() call "
                "producing the same number of designs."
            )
        scale_flat = self._last_scale_flat.reshape(n, 1).expand(
            n, self.event_shape.numel()
        )
        flat = design.reshape(n, -1)
        logp = torch.distributions.Normal(0.0, scale_flat).log_prob(flat).sum(-1)
        return logp.reshape(batch_shape)
