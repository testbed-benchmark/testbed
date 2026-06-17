"""Destructive lot-inspection task."""

import torch
from torch import Tensor

from ..core.distributions import Distribution
from ..envs.design_spaces import CategoricalDesignSpace
from .base_task import Task


class DestructiveLotInspectionPrior(Distribution):
    """Independent beta prior over zone-level defect probabilities."""

    def __init__(
        self,
        num_zones: int,
        prior_a: float | Tensor = 1.0,
        prior_b: float | Tensor = 19.0,
    ) -> None:
        """Initializes beta concentration parameters."""
        super().__init__()

        num_zones = int(num_zones)
        if num_zones <= 0:
            raise ValueError("num_zones must be positive.")

        concentration1 = self._expand_prior_parameter(prior_a, num_zones, "prior_a")
        concentration0 = self._expand_prior_parameter(prior_b, num_zones, "prior_b")
        if torch.any(concentration1 <= 0) or torch.any(concentration0 <= 0):
            raise ValueError("prior_a and prior_b must be positive.")

        self.register_buffer("concentration1", concentration1)
        self.register_buffer("concentration0", concentration0)

    @staticmethod
    def _expand_prior_parameter(value, num_zones: int, name: str) -> Tensor:
        tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
        if tensor.ndim == 0:
            return tensor.expand(num_zones).clone()
        if tensor.shape != (num_zones,):
            raise ValueError(f"{name} must be scalar or have shape ({num_zones},).")
        return tensor.clone()

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws defect-probability samples."""
        dist = torch.distributions.Beta(self.concentration1, self.concentration0)
        return dist.sample((n_samples,))

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities."""
        in_support = ((theta >= 0.0) & (theta <= 1.0)).all(dim=-1)
        theta = theta.clamp(
            torch.finfo(theta.dtype).tiny,
            1.0 - torch.finfo(theta.dtype).eps,
        )
        log_norm = (
            torch.lgamma(self.concentration1)
            + torch.lgamma(self.concentration0)
            - torch.lgamma(self.concentration1 + self.concentration0)
        )
        logp = (
            (self.concentration1 - 1.0) * torch.log(theta)
            + (self.concentration0 - 1.0) * torch.log1p(-theta)
            - log_norm
        ).sum(dim=-1)
        return torch.where(
            in_support,
            logp,
            torch.full_like(logp, -torch.inf),
        )


class DestructiveLotInspection(Task):
    """Bayesian destructive inspection with zone-specific defect rates.

    At each timestep the design is a scalar zone index. The outcome is a binary
    defect indicator for one destroyed unit sampled from that zone.
    """

    def __init__(
        self,
        num_zones: int = 4,
        prior_a: float | Tensor = 1.0,
        prior_b: float | Tensor = 19.0,
        prior=None,
    ) -> None:
        """Initializes the lot-inspection task."""
        num_zones = int(num_zones)
        if num_zones <= 0:
            raise ValueError("num_zones must be positive.")
        if prior is None:
            prior = DestructiveLotInspectionPrior(num_zones, prior_a, prior_b)

        self.event_shape_theta = torch.Size([num_zones])
        self.event_shape_design = torch.Size([])
        self.event_shape_y = torch.Size([1])
        self.design_space = CategoricalDesignSpace(num_zones)
        self.num_zones = num_zones

        super().__init__(
            prior,
            discrete_y=True,
            discrete_theta=False,
            discrete_design=True,
        )

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"DestructiveLotInspection(num_zones={self.num_zones})"

    def make_prob(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes selected-zone defect probabilities."""
        zone = design.reshape(-1)
        if torch.is_floating_point(zone):
            zone = zone.round()
        zone = zone.to(torch.long)
        return theta.gather(dim=-1, index=zone.unsqueeze(-1))

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        # y: [b, 1]
        # theta: [b, num_zones]
        # design: [b]
        prob = self.make_prob(theta, design)
        prob = prob.clamp(
            torch.finfo(prob.dtype).tiny,
            1.0 - torch.finfo(prob.dtype).eps,
        )
        log_p1 = torch.log(prob)
        log_p0 = torch.log1p(-prob)

        is_zero = y == 0
        is_one = y == 1
        logp = torch.full_like(prob, -torch.inf)
        logp = torch.where(is_one, log_p1, logp)
        logp = torch.where(is_zero, log_p0, logp)
        return logp.squeeze(-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        # theta: [b, num_zones]
        # design: [b]
        # returns: [b, 1]
        return torch.bernoulli(self.make_prob(theta, design))
