"""Linear-Gaussian regression task."""

import math

import torch
from torch import Tensor

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


class LinearRegressionPrior(Distribution):
    r"""Gaussian prior for weights in linear regression.

    The default prior is:

    .. math::
        p(\theta) = \mathcal{N}\left(0, \mathrm{diag}(1, 0.25, \dots, 0.25) \right)

    Args:
        dim_theta (int):
            Dimensionality of the parameters theta, including bias.
        theta_loc (Tensor, optional):
            Prior mean (default: None). Shape ``[dim_theta]``.
        theta_covmat (Tensor, optional):
            Prior covariance (default: None). Shape ``[dim_theta, dim_theta]``.
    """

    def __init__(
        self,
        dim_theta: int,
        theta_loc: Tensor = None,
        theta_covmat: Tensor = None,
    ):
        """Initializes prior for weights in linear regression."""
        super().__init__(event_shape=(dim_theta,))

        self.dim_theta = dim_theta  # Dimensionality of theta, including bias; e.g., dim_theta=2 is theta0 + theta1 x

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

    def sample(self, sample_shape=()):
        """Draws samples of parameters from the prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, dim_theta]``
        """
        eps = torch.randn(self._extended_shape(sample_shape), device=self.loc.device)
        return self.loc + eps @ self.chol.T

    def log_prob(self, theta: Tensor):
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., dim_theta]``.

        Returns:
            Tensor:
                Log probs, shape ``[...]``
        """
        diff = theta - self.loc

        # solve L x = diff  (more stable than inverse); diff as a column vector
        # so that an unbatched theta works too
        solve = torch.linalg.solve_triangular(
            self.chol, diff.unsqueeze(-1), upper=False
        ).squeeze(-1)

        maha = (solve**2).sum(-1)

        logp = -0.5 * (maha + self.logdet + self.dim_theta * math.log(2 * math.pi))
        return logp


class LinearRegression(Task):
    r"""Linear-Gaussian regression task.

    This task defines a Bayesian linear regression model with optional
    quadratic misspecification.

    Likelihood:

    .. math::
        p(y \mid \theta, x) = \theta_0 + x_1 \theta_1 + \dots + x_d \theta_d + s \left(\lvert\lvert x \rvert\rvert^2 - 1\right)

    Args:
        dim (int):
            Dimensionality of the design space. theta includes a bias term, so dim_theta = dim + 1.
        dim_y (int, optional):
            Number of independent observations per design (default: 1).
        noise_scale (float, optional):
            Standard deviation of observation noise (default: 0.5).
        quad (float, optional):
            Strength of quadratic misspecification term (default: 0.0).
            Setting this to zero gives the standard linear model.
        prior (Distribution, optional):
            Prior over parameters. If None, the default Gaussian LinearRegressionPrior is used.
        design_space (DesignSpace, optional):
            Controls the design parametrization. If None, a
            :class:`~testbed.core.design_spaces.BoxDesignSpace` over
            ``[-1, 1]^dim`` is used. Note the quadratic term above is centred
            on ``||x|| = 1``, which this box attains.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Gaussian noise added to a differentiable mean."""
        return True

    def __init__(
        self,
        dim: int,
        dim_y: int = 1,
        noise_scale: float = 0.1,
        quad: float = 0.0,
        prior=None,
        design_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the LinearRegression task."""
        # Default prior
        if prior is None:
            prior = LinearRegressionPrior(dim_theta=dim + 1)

        if design_space is None:
            design_space = BoxDesignSpace(low=-1.0, high=1.0, event_shape=[dim])

        super().__init__(
            event_shape_y=[dim_y],
            event_shape_theta=[dim + 1],
            event_shape_design=[dim],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        self.noise_scale = noise_scale
        self.quad = quad

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"LinearRegression("
            f"dim_y={self.event_shape_y[0]}, "
            f"dim_theta={self.event_shape_theta[0]}, "
            f"dim_design={self.event_shape_design[0]}, "
            f"noise_scale={self.noise_scale})"
        )

    def make_mean(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the conditional mean given theta, design.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_design]``.

        Returns:
            Tensor:
                Mean of shape ``[b, 1]``.
        """
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

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Computes log probability of observations under the model.

        If dim_y > 1, observations are assumed to be conditionally independent given (theta, design).

        Args:
            y (Tensor):
                Observations of shape ``[b, dim_y]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_design]``.
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

        If dim_y > 1, observations are assumed to be conditionally independent given (theta, design).

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, dim_theta]``.
            design (Tensor):
                Design tensor of shape ``[b, dim_design]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, dim_y]``.
        """
        loc = self.make_mean(theta, design)  # [b, 1]
        loc = loc.expand(-1, self.event_shape_y[0])  # [b, dim_y]
        return torch.randn_like(loc) * self.noise_scale + loc
