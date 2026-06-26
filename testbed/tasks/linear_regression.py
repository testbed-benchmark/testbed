"""Linear-Gaussian regression task."""

import math

import torch
from torch import Tensor

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base_task import Task


class LinearRegressionPrior(Distribution):
    """Gaussian prior for weights in linear regression."""

    def __init__(
        self,
        dim_theta,
        theta_loc=None,
        theta_covmat=None,
    ):
        """Initialises prior for weights in linear regression."""
        super().__init__()

        self.dim_theta = dim_theta  # Dimensionality of theta, including bias; e.g., dim_theta=1 is theta0 + theta1 x

        # Default prior is Normal with mean zero and covariance diag([1.0, 0.25, ..., 0.25])
        if theta_loc is None:
            theta_loc = torch.zeros((self.dim_theta,))
        if theta_covmat is None:
            variances = torch.zeros(self.dim_theta)
            variances[0] = 1.0
            variances[1:] = 0.25
            theta_covmat = torch.diag(variances)

        chol = torch.linalg.cholesky(theta_covmat)
        logdet = 2 * torch.log(torch.diagonal(chol)).sum()

        # All persistent tensors registered as buffers
        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.register_buffer("chol", chol)
        self.register_buffer("logdet", logdet)

    def sample(self, n_samples):
        """Draws samples of parameters from the prior."""
        # [n_samples, dim_theta]
        eps = torch.randn(n_samples, *self.loc.shape, device=self.loc.device)
        return self.loc + eps @ self.chol.T

    def log_prob(self, theta):
        """Evaluates prior log probabilities."""
        # theta: [*batch_shape, dim_theta]

        diff = theta - self.loc

        # solve L x = diff  (more stable than inverse)
        solve = torch.linalg.solve_triangular(
            self.chol, diff.transpose(-1, -2), upper=False
        ).transpose(-1, -2)

        maha = (solve**2).sum(-1)

        logp = -0.5 * (maha + self.logdet + self.dim_theta * math.log(2 * math.pi))
        return logp


class LinearRegression(Task):
    """Implements a Linear-Gaussian regression task."""

    def __init__(
        self,
        dim: int,
        dim_y: int = 1,
        noise_scale: float = 0.01,
        quad: float = 0.0,
        prior=None,
    ) -> None:
        """Initializes the LinearRegression task."""
        # dim: dimensionality of designs; theta is dim+1 as it includes a bias term
        # dim_y: how many independent observations are made at each design
        # quad: degree of misspecification; constant in front of quadratic term in mean function

        # Default prior
        if prior is None:
            prior = LinearRegressionPrior(dim_theta=dim + 1)

        self.event_shape_theta = torch.Size([dim + 1])
        self.event_shape_design = torch.Size([dim])
        self.event_shape_y = torch.Size([dim_y])

        self.noise_scale = noise_scale
        self.quad = quad

        super().__init__(
            prior,
            discrete_y=False,
            discrete_theta=False,
            discrete_design=False,
        )

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"LinearRegression("
            f"dim_y={self.event_shape_design[0]}, "
            f"dim_theta={self.event_shape_theta[0]}, "
            f"dim_design={self.event_shape_design[0]}, "
            f"noise_scale={self.noise_scale})"
        )

    def make_mean(self, theta, design):
        """Computes the underlying true function, i.e., the likelihood mean."""
        # theta:  [b, dim_theta]
        # design: [b, dim_design]
        # returns: [b, 1]

        b = theta.shape[0]
        # prepend ones to design for bias terms
        design_left_ones = torch.hstack(
            (torch.ones((b, 1), device=design.device), design)
        )  # [b, dim_theta]

        # batched dot product
        mean_y = torch.einsum("b d, b d -> b", theta, design_left_ones)

        if self.quad > 0:
            design_norm_sq = torch.linalg.norm(design, dim=-1) ** 2
            mean_y = mean_y + self.quad * (design_norm_sq - 1)

        return mean_y.unsqueeze(-1)

    def _outcome_logp(self, y, theta, design):
        # y: [b, dim_y]
        # theta: [b, dim_theta]
        # design: [b, dim_design]
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        # theta:   [b,, dim_theta]
        # design:  [b, dim_design]
        # returns: [b, dim_y]
        loc = self.make_mean(theta, design)  # [b, 1]
        loc = loc.expand(-1, self.event_shape_y[0])  # [b, dim_y]
        return torch.randn_like(loc) * self.noise_scale + loc
