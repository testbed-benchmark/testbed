"""Destructive lot inspection with a latent weak region."""

import torch
from torch import Tensor

from ..core.distributions import Distribution
from ..envs.design_spaces import CategoricalDesignSpace
from .base import Task


class DestructiveLotInspectionPrior(Distribution):
    """Prior over lot quality, weak-region location, and peak defect rate.

    Samples are ordered as ``(G, J, rho)``, where ``G`` indicates a good lot,
    ``J`` is the weak-region centre, and ``rho`` is its peak defect probability.
    """

    def __init__(
        self,
        num_zones: int = 16,
        good_probability: float = 0.6,
        location_probs: Tensor | None = None,
        theta_base: float = 0.02,
        tau: float = 0.12,
        good_rho_loc: float = 0.08,
        good_rho_scale: float = 0.02,
        bad_rho_loc: float = 0.16,
        bad_rho_scale: float = 0.025,
    ) -> None:
        """Initializes the experimenter's prior."""
        super().__init__()

        num_zones = int(num_zones)
        if num_zones <= 0:
            raise ValueError("num_zones must be positive.")
        if not 0.0 <= good_probability <= 1.0:
            raise ValueError("good_probability must lie between 0 and 1.")
        if not 0.0 <= theta_base < tau < 1.0:
            raise ValueError(
                "theta_base and tau must satisfy 0 <= theta_base < tau < 1."
            )
        if good_rho_scale <= 0.0 or bad_rho_scale <= 0.0:
            raise ValueError("good_rho_scale and bad_rho_scale must be positive.")

        dtype = torch.get_default_dtype()
        if location_probs is None:
            location_probs = torch.full((num_zones,), 1.0 / num_zones, dtype=dtype)
        else:
            location_probs = torch.as_tensor(location_probs, dtype=dtype)
            if location_probs.shape != (num_zones,):
                raise ValueError(f"location_probs must have shape ({num_zones},).")
            if torch.any(location_probs < 0.0) or location_probs.sum() <= 0.0:
                raise ValueError(
                    "location_probs must be non-negative and have positive mass."
                )
            location_probs = location_probs / location_probs.sum()

        self.num_zones = num_zones
        self.register_buffer(
            "good_probability", torch.as_tensor(good_probability, dtype=dtype)
        )
        self.register_buffer("location_probs", location_probs)
        self.register_buffer("theta_base", torch.as_tensor(theta_base, dtype=dtype))
        self.register_buffer("tau", torch.as_tensor(tau, dtype=dtype))
        self.register_buffer("good_rho_loc", torch.as_tensor(good_rho_loc, dtype=dtype))
        self.register_buffer(
            "good_rho_scale", torch.as_tensor(good_rho_scale, dtype=dtype)
        )
        self.register_buffer("bad_rho_loc", torch.as_tensor(bad_rho_loc, dtype=dtype))
        self.register_buffer(
            "bad_rho_scale", torch.as_tensor(bad_rho_scale, dtype=dtype)
        )

    @staticmethod
    def _standard_normal(reference: Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.ones((), device=reference.device, dtype=reference.dtype),
        )

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws ``(G, J, rho)`` samples."""
        good = torch.bernoulli(self.good_probability.expand(n_samples))
        location = torch.multinomial(self.location_probs, n_samples, replacement=True)

        loc = torch.where(good.bool(), self.good_rho_loc, self.bad_rho_loc)
        scale = torch.where(good.bool(), self.good_rho_scale, self.bad_rho_scale)
        lower = torch.where(good.bool(), self.theta_base, self.tau)
        upper = torch.where(good.bool(), self.tau, torch.ones_like(self.tau))

        normal = self._standard_normal(loc)
        lower_cdf = normal.cdf((lower - loc) / scale)
        upper_cdf = normal.cdf((upper - loc) / scale)
        probabilities = lower_cdf + torch.rand_like(loc) * (upper_cdf - lower_cdf)
        eps = torch.finfo(probabilities.dtype).eps
        rho = loc + scale * normal.icdf(probabilities.clamp(eps, 1.0 - eps))

        return torch.stack((good, location.to(good.dtype), rho), dim=-1)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates joint prior log probabilities for ``(G, J, rho)``."""
        good, location, rho = theta.unbind(dim=-1)
        is_good = good == 1.0

        loc = torch.where(is_good, self.good_rho_loc, self.bad_rho_loc)
        scale = torch.where(is_good, self.good_rho_scale, self.bad_rho_scale)
        lower = torch.where(is_good, self.theta_base, self.tau)
        upper = torch.where(is_good, self.tau, torch.ones_like(self.tau))

        normal = self._standard_normal(rho)
        standardized = (rho - loc) / scale
        log_rho = normal.log_prob(standardized) - torch.log(scale)
        normalizer = normal.cdf((upper - loc) / scale) - normal.cdf(
            (lower - loc) / scale
        )
        log_rho = log_rho - torch.log(normalizer)

        location_index = location.to(torch.long)
        valid_location = (
            (location == location_index)
            & (location_index >= 0)
            & (location_index < self.num_zones)
        )
        safe_location = location_index.clamp(0, self.num_zones - 1)
        log_location = torch.log(self.location_probs[safe_location])

        log_good = torch.where(
            is_good,
            torch.log(self.good_probability),
            torch.log1p(-self.good_probability),
        )
        valid_good = (good == 0.0) | is_good
        valid_rho = torch.where(
            is_good,
            (rho >= self.theta_base) & (rho <= self.tau),
            (rho > self.tau) & (rho <= 1.0),
        )
        logp = log_good + log_location + log_rho
        valid = valid_good & valid_location & valid_rho
        return torch.where(valid, logp, torch.full_like(logp, -torch.inf))


class DestructiveLotInspection(Task):
    """Bayesian destructive inspection of a spatial latent weak region.

    At each timestep the design selects one zone on a ring. The outcome is the
    number of defective units in a fixed-size destructive inspection batch.
    """

    def __init__(
        self,
        num_zones: int = 16,
        length_scale: float = 1.5,
        theta_base: float = 0.02,
        tau: float = 0.12,
        batch_size: int = 4,
        good_probability: float = 0.6,
        location_probs: Tensor | None = None,
        good_rho_loc: float = 0.08,
        good_rho_scale: float = 0.02,
        bad_rho_loc: float = 0.16,
        bad_rho_scale: float = 0.025,
        prior: Distribution | None = None,
    ) -> None:
        """Initializes the latent weak-region task."""
        num_zones = int(num_zones)
        batch_size = int(batch_size)
        if num_zones <= 0:
            raise ValueError("num_zones must be positive.")
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if not 0.0 <= theta_base < tau < 1.0:
            raise ValueError(
                "theta_base and tau must satisfy 0 <= theta_base < tau < 1."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if prior is None:
            prior = DestructiveLotInspectionPrior(
                num_zones=num_zones,
                good_probability=good_probability,
                location_probs=location_probs,
                theta_base=theta_base,
                tau=tau,
                good_rho_loc=good_rho_loc,
                good_rho_scale=good_rho_scale,
                bad_rho_loc=bad_rho_loc,
                bad_rho_scale=bad_rho_scale,
            )

        self.num_zones = num_zones
        self.batch_size = batch_size

        super().__init__(
            event_shape_y=(1,),
            event_shape_theta=(3,),
            event_shape_design=(),
            prior=prior,
            discrete_y=True,
            discrete_theta=False,
            discrete_design=True,
            design_space=CategoricalDesignSpace(num_zones),
        )
        self.event_shape_action = torch.Size([1])

        dtype = torch.get_default_dtype()
        self.register_buffer("length_scale", torch.as_tensor(length_scale, dtype=dtype))
        self.register_buffer("theta_base", torch.as_tensor(theta_base, dtype=dtype))
        self.register_buffer("tau", torch.as_tensor(tau, dtype=dtype))

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"DestructiveLotInspection(num_zones={self.num_zones}, "
            f"batch_size={self.batch_size})"
        )

    def make_prob(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes defect probabilities at selected zones."""
        location = theta[:, 1].round().to(torch.long)
        rho = theta[:, 2]
        zone = design.reshape(-1).round().to(torch.long)
        distance = (zone - location).abs()
        ring_distance = torch.minimum(distance, self.num_zones - distance)
        kernel = torch.exp(
            -(ring_distance.to(rho.dtype) ** 2) / (2.0 * self.length_scale**2)
        )
        probability = self.theta_base + (rho - self.theta_base) * kernel
        return probability.unsqueeze(-1)

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        probability = self.make_prob(theta, design).squeeze(-1)
        count = y.squeeze(-1)
        probability = probability.clamp(
            torch.finfo(probability.dtype).tiny,
            1.0 - torch.finfo(probability.dtype).eps,
        )
        log_coefficient = (
            torch.lgamma(count.new_tensor(self.batch_size + 1.0))
            - torch.lgamma(count + 1.0)
            - torch.lgamma(self.batch_size - count + 1.0)
        )
        logp = (
            log_coefficient
            + count * torch.log(probability)
            + (self.batch_size - count) * torch.log1p(-probability)
        )
        valid = (count == count.round()) & (count >= 0.0) & (count <= self.batch_size)
        return torch.where(valid, logp, torch.full_like(logp, -torch.inf))

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        probability = self.make_prob(theta, design)
        return torch.distributions.Binomial(
            total_count=self.batch_size,
            probs=probability,
        ).sample()
