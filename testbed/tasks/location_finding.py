"""Location-finding task."""

import torch
from torch import Tensor

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .task import Task


class LocationFindingPrior(Distribution):
    r"""Gaussian prior for source locations.

    This assumes that the source locations are i.i.d. a priori.
    The default prior is:

    .. math::
        p(\theta_k) = \mathcal{N}\left(0, I \right) \qquad k = 1, \dots, \mathrm{n_sources}

    Args:
        n_sources (int):
            Number of source locations.
        dim_theta (int):
            Dimensionality of the search space; total dimensionality of the problem is ``n_sources * dim_theta``.
        theta_loc (Tensor, optional):
            Prior mean for each source (default: None). Shape ``[dim_theta]``.
        theta_covmat (Tensor, optional):
            Prior covariance for each source (default: None). Shape ``[dim_theta, dim_theta]``.
    """

    def __init__(
        self,
        n_sources,
        dim_theta,
        theta_loc=None,
        theta_covmat=None,
    ) -> None:
        """Initializes source-location prior parameters."""
        super().__init__()

        if theta_loc is None:
            theta_loc = torch.zeros((n_sources, dim_theta))
        if theta_covmat is None:
            theta_covmat = torch.eye(dim_theta)

        chol = torch.linalg.cholesky(theta_covmat)
        logdet = 2 * torch.log(torch.diagonal(chol)).sum()

        # All persistent tensors registered as buffers
        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.register_buffer("chol", chol)
        self.register_buffer("logdet", logdet)

        self.dim = dim_theta

    def sample(self, n_samples: int) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            n_samples (int):
                How many samples to draw.

        Returns:
            Tensor:
                Samples, shape ``[n_samples, n_sources, dim_theta]``
        """
        # [n_samples, n_sources, dim_theta]
        eps = torch.randn(n_samples, *self.loc.shape, device=self.loc.device)
        return self.loc + eps @ self.chol.T

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., n_sources, dim_theta]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``
        """
        diff = theta - self.loc

        # solve L x = diff  (more stable than inverse)
        solve = torch.linalg.solve_triangular(
            self.chol, diff.transpose(-1, -2), upper=False
        ).transpose(-1, -2)

        maha = (solve**2).sum(dim=-1)  # [..., n_sources]

        logp_per_source = -0.5 * (
            maha + self.logdet + self.dim * torch.log(torch.tensor(2 * torch.pi))
        )
        return logp_per_source.sum(dim=-1)  # [...] — independent sources


class LocationFinding(Task):
    r"""Source location-finding task.

    Likelihood:

    .. math::
        \mu(\theta, x) = b + \sum_{k=1}^K \frac{\alpha}{m + \lVert \theta_k - x \rVert^s}

    .. math::
        p(y \mid \theta, x) = \mathcal{N}(\mu(\theta,x), \sigma^2)

    Args:
        n_sources (int):
            Number of source locations.
        dim_theta (int):
            Dimensionality of the search space; total dimensionality of the problem is ``n_sources * dim_theta``.
        dim_y (int, optional):
            Number of independent observations per design (default: 1).
        base_signal (float, optional):
            Base signal strength (:math:`b`) (default: 0.1).
        max_signal (float, optional):
            Controls the maximum signal strength (:math:`m`) (default: 1e-4).
        amplitude (float, optional):
            Controls the amplitude of each signal (:math:`\alpha`) (default: 1.0).
        noise_scale (float, optional):
            Standard deviation of observation noise (:math:`\sigma`) (default: 0.5).
        exponent (float, optional):
            Exponent in the rate of decay of signal strength (:math:`s`) (default: 2.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default LocationFindingPrior is used.
    """

    def __init__(
        self,
        n_sources: int,
        dim_theta: int,
        dim_y: int = 1,
        base_signal: float = 0.1,
        max_signal: float = 1e-4,
        amplitude: float = 1.0,
        noise_scale: float = 0.5,
        exponent: float = 2.0,
        prior=None,
    ) -> None:
        """Initializes the location-finding task."""
        # Default prior
        if prior is None:
            prior = LocationFindingPrior(dim_theta=dim_theta, n_sources=n_sources)

        super().__init__(
            event_shape_y=[dim_y],
            event_shape_theta=[n_sources, dim_theta],
            event_shape_design=[dim_theta],
            prior=prior,
        )

        self.base_signal = base_signal
        self.max_signal = max_signal
        self.amplitude = amplitude
        self.noise_scale = noise_scale

        # Exponent in signal decay; 2 is the standard LF model
        self.exponent = exponent

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"LocationFinding(n_sources={self.event_shape_theta[0]}, "
            f"dim_theta={self.event_shape_theta[1]}, "
            f"dim_y={self.event_shape_y[0]}, "
            f"base_signal={self.base_signal}, "
            f"max_signal={self.max_signal}, "
            f"amplitude={self.amplitude}, "
            f"noise_scale={self.noise_scale}, "
            f"exponent={self.exponent})"
        )

    def make_mean(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the conditional mean given theta, design.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, n_sources, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_theta]``.

        Returns:
            Tensor:
                Mean of shape ``[b, 1]``.
        """
        sq_2_norm = (
            (design.unsqueeze(dim=1) - theta).pow(2).sum(dim=-1)
        )  # [B, n_sources] - sum over p dim
        signal_term = sq_2_norm.pow(self.exponent / 2)
        inv_term = (self.max_signal + signal_term).pow(-1)  # [B, n_sources]
        mean_y = torch.log(
            self.base_signal + self.amplitude * inv_term.sum(dim=-1, keepdim=True)
        )  # [B, 1]
        return mean_y

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """Computes log probability of observations under the model.

        If dim_y > 1, observations are assumed to be conditionally independent given (theta, design).

        Args:
            y (Tensor):
                Observations of shape ``[b, dim_y]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, n_sources, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_theta]``.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        """Samples from the likelihood using reparametrization.

        If dim_y > 1, observations are assumed to be conditionally independent given (theta, design).

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, n_sources, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_design]``.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, dim_y]``.
        """
        loc = self.make_mean(theta, design)  # [b, 1]
        loc = loc.expand(-1, self.event_shape_y[0])  # [b, dim_y]
        return torch.randn_like(loc) * self.noise_scale + loc
