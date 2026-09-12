"""Cart pole task.

References:
    1. https://arxiv.org/abs/2402.07868
    2. https://arxiv.org/abs/2607.08335
"""

from collections.abc import Iterable

import torch
from torch import Tensor

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import LogNormal
from ..core.util import normal_log_prob
from .base import Task


class CartPolePrior(LogNormal):
    r"""Log-normal prior for theta = (mp, mc, l).

    The default prior is:

    .. math::
        \log \theta \sim \mathcal{N}(0, 0.01 I_3)

    Args:
        theta_loc (Tensor, optional):
            Mean of the underlying normal (default: None). Shape ``[3,]``.
        theta_covmat (Tensor, optional):
            Covariance of the underlying normal (default: None). Shape ``[3, 3]``.
    """

    def __init__(self, theta_loc=None, theta_covmat=None) -> None:
        """Initializes prior parameters."""
        if theta_loc is None:
            theta_loc = torch.zeros(3)
        if theta_covmat is None:
            theta_covmat = 0.01 * torch.eye(3)
        super().__init__(theta_loc, theta_covmat)


class CartPole(Task):
    r"""Cart pole task.

    By convention y = (s, q, sdot, qdot) is the cart position s, pole angle q, and their velocities.
    Further theta = (mp, mc, l) where mp is the pole mass, mc is the cart mass, and l is the pole length.
    Designs are a scalar control force applied to the cart.
    The observation y_t is a noisy Euler-Maruyama step of the dynamics from y_{t-1}:

    .. math::
        p(y_t \mid \theta, x_t, y_{t-1}) = \mathcal{N}(y_{t-1} + \mathrm{dt} \cdot \mathrm{drift}(y_{t-1}, \theta, x_t), \mathrm{dt} \cdot \sigma^2)

    Args:
        prior (Distribution, optional):
            Prior over parameters. If None, the default CartPolePrior is used.
        design_space (DesignSpace, optional):
            Design space for the applied force. If None, defaults to ``[-5, 5]``.
        dt (float):
            Step size in the Euler-Maruyama discretization (default: 0.05).
        stiffness_c (float):
            Cart stiffness coefficient (default: 0.01).
        stiffness_p (float):
            Pole stiffness coefficient (default: 0.01).
        damping_c (float):
            Cart damping coefficient (default: 0.01).
        damping_p (float):
            Pole damping coefficient (default: 0.01).
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
        stiffness_c: float = 0.01,
        stiffness_p: float = 0.01,
        damping_c: float = 0.01,
        damping_p: float = 0.01,
        noise_scale: float | Iterable[float] = 0.1,
        noise_jitter: float = 0.0,
    ) -> None:
        """Initializes the cart pole task."""
        if prior is None:
            prior = CartPolePrior()

        if design_space is None:
            design_space = BoxDesignSpace(low=-5.0, high=5.0, event_shape=[1])

        super().__init__(
            event_shape_y=[4],
            event_shape_theta=[3],
            event_shape_design=[1],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
            memoryless=False,
        )

        self.dt = dt
        self.g = 9.81
        self.stiffness_c = stiffness_c
        self.stiffness_p = stiffness_p
        self.damping_c = damping_c
        self.damping_p = damping_p
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
        return (
            f"CartPole(dt={self.dt}, "
            f"stiffness=({self.stiffness_c}, {self.stiffness_p}), "
            f"damping=({self.damping_c}, {self.damping_p}), "
            f"noise_scale={[round(v, 8) for v in self.noise_scale.tolist()]}, "
            f"noise_jitter={self.noise_jitter})"
        )

    def drift_func(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """See "Bayesian Experimental Design via Score Matching", Phillips et al. 2026, Appendix D.1.2.

        y: [..., 4]
        theta: [..., 3]
        design: [..., 1]
        """
        s = y[..., 0]
        q = y[..., 1]
        sdot = y[..., 2]
        qdot = y[..., 3]

        mp = theta[..., 0]
        mc = theta[..., 1]
        len = theta[..., 2]

        xi = design[..., 0]
        g = self.g
        k1 = self.stiffness_c
        k2 = self.stiffness_p
        d1 = self.damping_c
        d2 = self.damping_p

        cq = torch.cos(q)
        sq = torch.sin(q)
        denom = mc + mp * sq**2

        # acceleration of cart
        term1 = mp * sq * (len * qdot**2 + g * cq)
        term2 = -(k1 * s + d1 * sdot)
        term3 = -(k2 * q + d2 * qdot) * cq / len
        sdotdot = (xi + term1 + term2 + term3) / denom

        # angular acceleration of pole
        term1 = xi * cq
        term2 = mp * len * qdot**2 * cq * sq
        term3 = (mc + mp) * g * sq
        term4 = (k1 * s + d1 * sdot) * cq
        term5 = (k2 * q + d2 * qdot) * cq**2 / len
        qdotdot = -(term1 + term2 + term3 + term4 + term5) / (len * denom)

        return torch.stack([sdot, qdot, sdotdot, qdotdot], dim=-1)

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
