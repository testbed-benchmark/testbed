"""Gravimetry task.

References:
    1. https://arxiv.org/abs/2607.08335
"""

import math

import torch
from torch import Tensor

from ..core.design_spaces import DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_cdf, normal_log_prob
from .base import Task


class GravimetryPrior(Distribution):
    r"""Gravimetry prior over theta = (x0, z, kappa).

    The components are independent, with

    .. math::
        x_0 \sim \mathcal{N}(0, \tau^2), \qquad
        z \sim \mathrm{LogNormal}(m, \sigma^2)\big|_{z \geq z_{\min}}, \qquad
        R \sim \mathcal{U}(R_{\min}, R_{\max}),

    where the source strength kappa follows generatively from the radius as
    :math:`\kappa = G \tfrac{4}{3}\pi R^3 \Delta\rho / 10^{-8}`.

    Note: we use a truncated log-normal instead of a clamped log-normal,
    (as in the original implementation) to ensure a well-defined density exists.

    Args:
        x0_loc (float, optional):
            Mean of the horizontal source position (default: 0.0).
        x0_scale (float, optional):
            Standard deviation of the horizontal source position (default: 30.0).
        z_loc (float, optional):
            Mean of the log depth (default: ``log 10``).
        z_scale (float, optional):
            Standard deviation of the log depth (default: 0.5).
        z_floor (float, optional):
            Minimum depth; the depth is truncated below it (default: 3.0).
        r_min (float, optional):
            Minimum anomaly radius (default: 1.5).
        r_max (float, optional):
            Maximum anomaly radius (default: 4.0).
        density_contrast (float, optional):
            Density contrast :math:`\Delta\rho`; negative for a void
            (default: -2200.0, an air-filled void in rock).
        gravitational_const (float, optional):
            Newton's constant :math:`G` (default: 6.674e-11).
    """

    def __init__(
        self,
        x0_loc: float = 0.0,
        x0_scale: float = 30.0,
        z_loc: float = math.log(10.0),
        z_scale: float = 0.5,
        z_floor: float = 3.0,
        r_min: float = 1.5,
        r_max: float = 4.0,
        density_contrast: float = -2200.0,
        gravitational_const: float = 6.674e-11,
    ) -> None:
        """Initializes source-location prior parameters."""
        super().__init__(event_shape=(3,))

        self.x0_loc = x0_loc
        self.x0_scale = x0_scale
        self.z_loc = z_loc
        self.z_scale = z_scale
        self.z_floor = z_floor
        self.r_min = r_min
        self.r_max = r_max
        self.density_contrast = density_contrast
        self.gravitational_const = gravitational_const

        # kappa = kappa_coeff * R**3
        self.kappa_coeff = (
            gravitational_const * (4 / 3) * math.pi * density_contrast / 1e-8
        )
        # untruncated mass below the floor, which the truncation divides out
        self.z_floor_cdf = normal_cdf(
            torch.tensor((math.log(z_floor) - z_loc) / z_scale)
        ).item()
        self.dim = 3

        # exists to give .to() something to move and so
        # Distribution.device reports where samples should be drawn
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

    def transform(self, u: Tensor) -> Tensor:
        """Maps a standard normal latent to physical parameters.

        Each component is a monotone map of its own latent coordinate, so this
        is the inverse-CDF construction of the prior described above.

        Args:
            u (Tensor):
                Standard normal latents of shape ``[..., 3]``.

        Returns:
            Tensor:
                Parameters ``(x0, z, kappa)`` of shape ``[..., 3]``.
        """
        x0 = self.x0_loc + self.x0_scale * u[..., 0:1]
        # truncate by mapping the latent onto the untruncated quantiles that lie
        # above the floor, rather than clamping the draws that fall below it
        q = self.z_floor_cdf + (1.0 - self.z_floor_cdf) * normal_cdf(u[..., 1:2])
        z = torch.exp(self.z_loc + self.z_scale * torch.special.ndtri(q))
        r = self.r_min + (self.r_max - self.r_min) * normal_cdf(u[..., 2:3])
        return torch.cat([x0, z, self.kappa_coeff * r**3], dim=-1)

    def sample(self, sample_shape=()) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, 3]``
        """
        return self.transform(
            torch.randn(self._extended_shape(sample_shape), device=self.device)
        )

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Exact everywhere, and ``-inf`` outside the support.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 3]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``
        """
        x0 = theta[..., 0]
        z = theta[..., 1]
        kappa = theta[..., 2]

        logp_x0 = normal_log_prob(x0, self.x0_loc, self.x0_scale)
        # log-normal depth (normal in log z with the 1/z Jacobian), renormalized
        # over the part of its support above the floor
        logp_z = (
            normal_log_prob(torch.log(z), self.z_loc, self.z_scale)
            - torch.log(z)
            - math.log1p(-self.z_floor_cdf)
        )
        # kappa = c R**3 for R uniform, so the density picks up |dR/dkappa|
        r = (kappa / self.kappa_coeff) ** (1 / 3)
        logp_kappa = -math.log(self.r_max - self.r_min) - torch.log(
            3 * abs(self.kappa_coeff) * r**2
        )

        logp = logp_x0 + logp_z + logp_kappa
        in_support = (z >= self.z_floor) & (r >= self.r_min) & (r <= self.r_max)
        return torch.where(in_support, logp, float("-inf"))


class Gravimetry(Task):
    r"""Gravimetry task.

    A buried spherical mass anomaly of strength :math:`\kappa` at horizontal
    position :math:`x_0` and depth :math:`z` produces, at a surface measurement
    location :math:`x`, the vertical gravity perturbation

    .. math::
        \mu(\theta, x) = \kappa \frac{z}{\left((x - x_0)^2 + z^2\right)^{3/2}}

    observed under additive Gaussian noise:

    .. math::
        p(y \mid \theta, x) = \mathcal{N}(\mu(\theta, x), \sigma^2)

    The design is where along a line to take the measurement, so informative
    designs sit near but not directly above the anomaly: depth and strength are
    only separable from off-centre readings.

    Args:
        noise_scale (float, optional):
            Standard deviation of observation noise :math:`\sigma` (default: 1.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default GravimetryPrior is used.
        design_space (DesignSpace, optional):
            Measurement locations. If None, the task retains its unbounded
            default design space.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Gaussian noise added to a differentiable mean."""
        return True

    def __init__(
        self,
        noise_scale: float = 1.0,
        prior=None,
        design_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the gravimetry task."""
        if prior is None:
            prior = GravimetryPrior()

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[3],
            event_shape_design=[1],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        self.noise_scale = noise_scale

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"Gravimetry(noise_scale={self.noise_scale})"

    def make_mean(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the gravity anomaly given theta, design.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 3]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.

        Returns:
            Tensor:
                Mean of shape ``[b, 1]``.
        """
        x0 = theta[..., 0:1]
        z = theta[..., 1:2]
        kappa = theta[..., 2:3]

        sq_dist = (design - x0) ** 2
        return kappa * z / (sq_dist + z**2) ** 1.5

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Computes log probability of observations under the model.

        Args:
            y (Tensor):
                Observations of shape ``[b, 1]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 3]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(
        self,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Samples from the likelihood using reparametrization.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 3]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 1]``.
        """
        loc = self.make_mean(theta, design)  # [b, 1]
        return loc + self.noise_scale * torch.randn_like(loc)
