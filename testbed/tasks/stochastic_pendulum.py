"""Stochastic pendulum task.

References:
    1. https://arxiv.org/abs/2402.07868
    2. https://arxiv.org/abs/2607.08335
"""

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import ExpTransform

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


class StochasticPendulumPrior(Distribution):
    r"""Log-normal prior for theta = (m, l).

    The default prior is:

    .. math::
        p(\theta) = \mathcal{N}\left(0, 0.01 I_2 \right)

    Args:
        theta_loc (Tensor, optional):
            Prior mean (default: None). Shape ``[2,]``.
        theta_covmat (Tensor, optional):
            Prior covariance (default: None). Shape ``[2, 2]``.
    """

    def __init__(
        self,
        theta_loc=None,
        theta_covmat=None,
    ) -> None:
        """Initializes prior parameters."""
        super().__init__()

        if theta_loc is None:
            theta_loc = torch.zeros((2,))
        if theta_covmat is None:
            theta_covmat = 0.01 * torch.eye(2)

        # All persistent tensors registered as buffers
        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.dim = 2

    @property
    def theta_lognormal(self) -> torch.distributions.Distribution:
        """Distribution for theta.

        Note that torch does not natively support a multivariate LogNormal.
        """
        mvn = MultivariateNormal(self.loc, covariance_matrix=self.cov)
        multivariate_lognormal = TransformedDistribution(mvn, [ExpTransform()])
        return multivariate_lognormal

    def sample(self, n_samples: int) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            n_samples (int):
                How many samples to draw.

        Returns:
            Tensor:
                Samples, shape ``[n_samples, 2]``
        """
        dist = self.theta_lognormal
        samples = dist.sample((n_samples,))
        return samples.reshape(-1, 2)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 2]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``
        """
        dist = self.theta_lognormal
        return dist.log_prob(theta)


class StochasticPendulum(Task):
    r"""Stochastic pendulum task.

    By convention y = (q, qdot) and theta = (m, l). y_t is a noisy
    Euler-Maruyama step of the pendulum's forced dynamics from y_{t-1}:

    .. math::
        p(y_t \mid \theta, x_t, y_{t-1}) = \mathcal{N}(y_{t-1} + \mathrm{dt} \cdot \mathrm{drift}(y_{t-1}, \theta, x_t), \sigma^2)

    Args:
        prior (Distribution, optional):
            Prior over parameters. If None, the default StochasticPendulumPrior is used.
        design_space (DesignSpace, optional):
            Design space for the applied torque. If None, defaults to ``[-1, 1]``.
        dt (float):
            Step size in the Euler-Maruyama discretization (default: 0.05).
        damping (float):
            Damping coefficient (default: 0.1).
        noise_scale (float or sequence):
            Diffusion scale, broadcast over the observation (default: 0.1). Pass a
            length-2 sequence for a per-coordinate diffusion.
        noise_jitter (float):
            Added to the observation variance, so outcomes have standard deviation
            ``sqrt(dt * noise_scale**2 + noise_jitter)`` (default: 0.0). Needed only
            to regularize a singular diffusion, where some ``noise_scale`` entry is
            zero and the observation density would otherwise be degenerate.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Gaussian noise added to a differentiable mean."""
        return True

    def __init__(
        self,
        prior=None,
        design_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
        dt: float = 0.05,
        damping: float = 0.1,
        noise_scale: float | Iterable[float] = 0.1,
        noise_jitter: float = 0.0,
    ) -> None:
        """Initializes the stochastic pendulum task."""
        if prior is None:
            prior = StochasticPendulumPrior()

        if design_space is None:
            design_space = BoxDesignSpace(low=-1.0, high=1.0, event_shape=[1])

        super().__init__(
            event_shape_y=[2],
            event_shape_theta=[2],
            event_shape_design=[1],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
            memoryless=False,
        )

        self.dt = dt
        self.g = 9.81
        self.damping = damping
        scale = torch.as_tensor(noise_scale, dtype=torch.get_default_dtype())
        self.register_buffer(
            "noise_scale", scale.broadcast_to(self.event_shape_y).clone()
        )
        self.noise_jitter = noise_jitter

    @property
    def obs_std(self) -> Tensor:
        """Per-coordinate observation standard deviation."""
        return torch.sqrt(self.dt * self.noise_scale**2 + self.noise_jitter)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        scale = [round(v, 8) for v in self.noise_scale.tolist()]
        return (
            f"StochasticPendulum(dt={self.dt}, "
            f"damping={self.damping}, "
            f"noise_scale={scale}, "
            f"noise_jitter={self.noise_jitter})"
        )

    def drift_func(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """See "Bayesian Experimental Design via Score Matching", Phillips et al. 2026, Appendix D.1.2.

        y: [..., 2]
        theta: [..., 2]
        design: [..., 1]
        """
        q = y[..., 0]
        qdot = y[..., 1]

        m = theta[..., 0]
        len = theta[..., 1]

        term1 = self.g / (2 * len) * torch.sin(q)
        term2 = (design[..., 0] - self.damping * qdot) / (m * len**2)
        qdotdot = 3 * (-term1 + term2)

        return torch.stack([qdot, qdotdot], dim=-1)

    def make_mean(
        self, theta: Tensor, design: Tensor, y_history: Tensor | None = None
    ) -> Tensor:
        """Makes mean function of observations i.e. y_t + drift_func(y_t-1, design_t, theta).

        An empty or absent y_history means t=0, where y_0 = (0, 0).
        """
        if y_history is None or y_history.shape[1] == 0:
            prev_y = torch.zeros(
                theta.shape[0], 2, device=theta.device, dtype=theta.dtype
            )
        else:
            prev_y = y_history[:, -1, :]
        drift = self.drift_func(prev_y, theta, design)
        return prev_y + self.dt * drift

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
                Observations of shape ``[b, 2]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Previous observations of shape ``[b, t, 2]``.
            design_history (Tensor, optional):
                Ignored; the drift depends only on the previous observation.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        loc = self.make_mean(theta, design, y_history)
        logp = normal_log_prob(y, loc, self.obs_std)
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
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Past observations of shape ``[b, t, 2]``.
            design_history (Tensor, optional):
                Ignored; the drift depends only on the previous observation.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 2]``.
        """
        # euler-maruyama SDE step
        mean = self.make_mean(theta, design, y_history)
        return mean + self.obs_std * torch.randn_like(mean)
