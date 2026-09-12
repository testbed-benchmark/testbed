"""Double link task."""

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import ExpTransform

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


class DoubleLinkPrior(Distribution):
    r"""Log-normal prior for theta = (m1, m2, l1, l2).

    The default prior is:

    .. math::
        p(\theta) = \mathcal{N}\left(0, 0.01 I_4 \right)

    Args:
        theta_loc (Tensor, optional):
            Prior mean (default: None). Shape ``[4,]``.
        theta_covmat (Tensor, optional):
            Prior covariance (default: None). Shape ``[4, 4]``.
    """

    def __init__(
        self,
        theta_loc=None,
        theta_covmat=None,
    ) -> None:
        """Initializes prior parameters."""
        super().__init__()

        if theta_loc is None:
            theta_loc = torch.zeros((4,))
        if theta_covmat is None:
            theta_covmat = 0.01 * torch.eye(4)

        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.dim = 4

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
                Samples, shape ``[n_samples, 4]``
        """
        dist = self.theta_lognormal
        samples = dist.sample((n_samples,))
        return samples.reshape(-1, 4)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 4]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``
        """
        dist = self.theta_lognormal
        return dist.log_prob(theta)


class DoubleLink(Task):
    r"""Double link task.

    By convention y = (q1, q2, q1dot, q2dot) is the angles (q1, q2) and their velocities (q1dot, q2dot) of two links.
    Further theta = (m1, m2, l1, l2) where (m1, m2) are their masses and (l1, l2) are their lengths.
    Designs are 2D control torques applied to each link.
    The observation y_t is a noisy Euler-Maruyama step of the dynamics from y_{t-1}:

    .. math::
        p(y_t \mid \theta, x_t, y_{t-1}) = \mathcal{N}(y_{t-1} + \mathrm{dt} \cdot \mathrm{drift}(y_{t-1}, \theta, x_t), \mathrm{dt} \cdot \sigma^2)

    Args:
        prior (Distribution, optional):
            Prior over parameters. If None, the default DoubleLinkPrior is used.
        design_space (DesignSpace, optional):
            Design space for the applied torques. If None, defaults to ``[-4, 4] x [-2, 2]``.
        dt (float):
            Step size in the Euler-Maruyama discretization (default: 0.05).
        noise_scale (float or sequence):
            Diffusion scale, broadcast over the observation (default: 0.1). Pass a
            length-4 sequence for a per-coordinate diffusion.
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
        noise_scale: float | Iterable[float] = 0.1,
        noise_jitter: float = 0.0,
    ) -> None:
        """Initializes the double link task."""
        if prior is None:
            prior = DoubleLinkPrior()

        if design_space is None:
            design_space = BoxDesignSpace(
                low=[-4.0, -2.0], high=[4.0, 2.0], event_shape=[2]
            )

        super().__init__(
            event_shape_y=[4],
            event_shape_theta=[4],
            event_shape_design=[2],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
            memoryless=False,
        )

        self.dt = dt
        self.g = 9.81
        scale = torch.as_tensor(noise_scale, dtype=torch.get_default_dtype())
        self.noise_scale = scale.broadcast_to(self.event_shape_y)
        self.noise_jitter = noise_jitter

    @property
    def obs_std(self) -> Tensor:
        """Per-coordinate observation standard deviation."""
        return torch.sqrt(self.dt * self.noise_scale**2 + self.noise_jitter)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        scale = [round(v, 8) for v in self.noise_scale.tolist()]
        return (
            f"DoubleLink(dt={self.dt}, noise_scale={scale}, "
            f"noise_jitter={self.noise_jitter})"
        )

    def drift_func(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """See "Bayesian Experimental Design via Score Matching", Phillips et al. 2026, Appendix D.1.2.

        y: [..., 4]
        theta: [..., 4]
        design: [..., 2]
        """
        q1 = y[..., 0]
        q2 = y[..., 1]
        q1dot = y[..., 2]
        q2dot = y[..., 3]

        m1 = theta[..., 0]
        m2 = theta[..., 1]
        len1 = theta[..., 2]
        len2 = theta[..., 3]

        g = self.g

        # mass matrix, symmetric so entry 01 == entry 10
        a = m2 * len1 * len2 * torch.cos(q2)
        b = m2 * len2**2
        m00 = (m1 + m2) * len1**2 + b + 2 * a
        m01 = b + a
        mass_matrix = torch.stack(
            [torch.stack([m00, m01], dim=-1), torch.stack([m01, b], dim=-1)], dim=-2
        )

        # coriolis/centrifugal matrix
        a = -m2 * len1 * len2 * torch.sin(q2)
        c01 = a * (2 * q1dot + q2dot)
        c10 = -0.5 * c01
        c11 = 0.5 * a * q1dot
        coriolis_matrix = torch.stack(
            [
                torch.stack([torch.zeros_like(c01), c01], dim=-1),
                torch.stack([c10, c11], dim=-1),
            ],
            dim=-2,
        )

        # gravity term ('tau')
        # This has a sign error as written in (Phillips 2026) but implemented correctly in their code
        # Agrees with the math in (Appendix D4; Nesting Particle Filters..., Iqbal 2024)
        # However (Iqbal 2024) has an additional -k dq term in their code; see:
        # https://github.com/Sahel13/InsideOutSMC.jl/blob/main/experiments/double_pendulum/environment.jl#L47
        term2 = m2 * len2 * torch.sin(q1 + q2)
        term1 = (m1 + m2) * len1 * torch.sin(q1) + term2
        tau = -g * torch.stack([term1, term2], dim=-1)

        # M qddot + C qdot = design + tau
        qdot = torch.stack([q1dot, q2dot], dim=-1)
        rhs = design + tau - (coriolis_matrix @ qdot.unsqueeze(-1)).squeeze(-1)
        qdotdot = torch.linalg.solve(mass_matrix, rhs.unsqueeze(-1)).squeeze(-1)

        return torch.cat([qdot, qdotdot], dim=-1)

    def make_mean(
        self, theta: Tensor, design: Tensor, y_history: Tensor | None = None
    ) -> Tensor:
        """Makes mean function of observations i.e. y_t + drift_func(y_t-1, design_t, theta).

        An empty or absent y_history means t=0, where y_0 = 0.
        """
        if y_history is None or y_history.shape[1] == 0:
            prev_y = torch.zeros(
                theta.shape[0],
                *self.event_shape_y,
                device=theta.device,
                dtype=theta.dtype,
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
                Observations of shape ``[b, 4]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 4]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.
            y_history (Tensor, optional):
                Previous observations of shape ``[b, t, 4]``.
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
                Parameter tensor of shape ``[b, 4]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.
            y_history (Tensor, optional):
                Past observations of shape ``[b, t, 4]``.
            design_history (Tensor, optional):
                Ignored; the drift depends only on the previous observation.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 4]``.
        """
        # euler-maruyama SDE step
        mean = self.make_mean(theta, design, y_history)
        return mean + self.obs_std * torch.randn_like(mean)
