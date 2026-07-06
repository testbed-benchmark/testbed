"""Constant elasticity of substitution (CES) task.

References:
    1. https://arxiv.org/abs/1903.05480
    2. https://arxiv.org/abs/2202.00821
"""

import math

import torch
from torch import Tensor
from torch.distributions import (
    AffineTransform,
    Beta,
    Dirichlet,
    LogNormal,
    TransformedDistribution,
)

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


def _stable_log_ndtr(x: Tensor, thresh: float = -10.0) -> Tensor:
    r"""Log standard-normal CDF :math:`\log \Phi(x)`, stable and gradient-safe.

    Matches :func:`torch.special.log_ndtr` for ``x >= thresh``. Below ``thresh``
    it swaps in the first-order Gaussian-tail asymptotic
    :math:`\log \Phi(x) \approx \log f(x) - \log(-x)` (with :math:`f` the standard
    normal pdf).
    """
    if not x.requires_grad:
        return torch.special.log_ndtr(x)

    x_safe = torch.clamp(x, min=thresh)
    exact = torch.special.log_ndtr(x_safe)

    x_tail = torch.clamp(x, max=thresh)
    log_sqrt_2pi = 0.5 * math.log(2 * math.pi)
    asymptotic = -0.5 * x_tail * x_tail - torch.log(-x_tail) - log_sqrt_2pi

    tail = x < thresh
    return torch.where(tail, asymptotic, exact)


class CESPrior(Distribution):
    r"""Prior over the CES latent variables :math:`(\rho, \alpha, u)`.

    The default prior is:

    .. math::
        \begin{aligned}
        \rho &= 0.01 + 0.99\,\tilde{\rho}, \quad \tilde{\rho} \sim \mathrm{Beta}(1, 1) \\
        \alpha &\sim \mathrm{Dirichlet}([1, 1, 1]) \\
        \log u &\sim \mathcal{N}(1, 3^2)
        \end{aligned}

    The affine shift bounds :math:`\rho` to :math:`[0.01, 1]`, matching the
    RL-BOED reference, where :math:`\alpha = (\alpha_1, \alpha_2, \alpha_3)`.
    Convention: :math:`\theta = (\rho, \alpha_1, \alpha_2, \alpha_3, u)`.

    Args:
        rho_concentration (tuple, optional):
            Concentration of the underlying :math:`\mathrm{Beta}` prior over
            :math:`\rho`, before the :math:`[0.01, 1]` affine shift (default: (1.0, 1.0)).
        alpha_concentration (tuple, optional):
            Concentration parameters for the prior over :math:`\alpha` (default: (1.0, 1.0, 1.0)).
        u_loc (float, optional):
            Mean for the prior over :math:`u` (default: 1.0).
        u_scale (float, optional):
            Standard deviation for the prior over :math:`u` (default: 3.0).
    """

    def __init__(
        self,
        rho_concentration=(1.0, 1.0),
        alpha_concentration=(1.0, 1.0, 1.0),
        u_loc=1.0,
        u_scale=3.0,
    ) -> None:
        """Initializes CES prior parameters."""
        super().__init__()

        self.register_buffer("rho_c1", torch.tensor(rho_concentration[0]))
        self.register_buffer("rho_c0", torch.tensor(rho_concentration[1]))
        self.register_buffer("alpha_concentration", torch.tensor(alpha_concentration))
        self.register_buffer("u_loc", torch.tensor(u_loc))
        self.register_buffer("u_scale", torch.tensor(u_scale))

    @property
    def rho_dist(self) -> torch.distributions.Distribution:
        """Distribution for the elasticity rho: an affine-shifted Beta on [0.01, 1]."""
        return TransformedDistribution(
            Beta(self.rho_c1, self.rho_c0),
            AffineTransform(loc=0.01, scale=0.99),
        )

    @property
    def alpha_dirichlet(self) -> torch.distributions.Distribution:
        """Distribution for the weights alpha."""
        return Dirichlet(self.alpha_concentration)

    @property
    def u_lognormal(self) -> torch.distributions.Distribution:
        """Distribution for the scale u."""
        return LogNormal(self.u_loc, self.u_scale)

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples from the prior distribution.

        Args:
            n_samples (int):
                The number of samples to draw from the prior.

        Returns:
            Tensor:
                A tensor of shape ``[n_samples, 5]``.
        """
        rho = self.rho_dist.sample((n_samples, 1))
        alpha = self.alpha_dirichlet.sample((n_samples,))
        u = self.u_lognormal.sample((n_samples, 1))
        return torch.cat([rho, alpha, u], dim=-1)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                A sample from the prior distribution of shape ``[..., 5]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[..., ]``.
        """
        rho = theta[..., 0]
        alpha = theta[..., 1:4]
        u = theta[..., 4]

        logp_rho = self.rho_dist.log_prob(rho)
        logp_alpha = self.alpha_dirichlet.log_prob(alpha)
        logp_u = self.u_lognormal.log_prob(u)
        return logp_rho + logp_alpha + logp_u


class CES(Task):
    """Constant elasticity of substitution (CES) task.

    A participant compares two baskets of goods x, x' in [0, 100]^3 and rates
    the subjective difference in utility on a sliding scale from 0 to 1. The
    outcome is a censored sigmoid of a Gaussian, so much of the design space
    maps to the boundary of the observation space.

    Convention: design = (x, x') flattened to 6 dims; theta = (rho, alpha, u).

    Args:
        tau (float, optional):
            Observation noise scale hyperparameter (default: 0.005).
        epsilon (float, optional):
            Censoring bound on the sigmoid output (default: 2**-22).
        prior (Distribution, optional):
            Prior distribution of the parameters.
    """

    def __init__(
        self,
        tau: float = 0.005,
        epsilon: float = 2**-22,
        prior=None,
    ) -> None:
        """Initializes the CES task."""
        if prior is None:
            prior = CESPrior()

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[5],
            event_shape_design=[6],
            prior=prior,
        )

        self.tau = tau
        self.epsilon = epsilon

        # Logit of the censoring bounds (logit(1 - eps) = -logit(eps))
        self.logit_lower = math.log(epsilon) - math.log1p(-epsilon)
        self.logit_upper = -self.logit_lower

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"CES(tau={self.tau}, epsilon={self.epsilon})"

    def make_eta_params(self, theta: Tensor, design: Tensor) -> Tensor:
        r"""Computes the mean and scale of the latent Gaussian eta.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[..., 5]``, where the columns are
                ``(rho, alpha_1, alpha_2, alpha_3, u)``.
            design (Tensor):
                Design tensor of shape ``[..., 6]``, representing two baskets
                of goods ``(x, x')`` each in ``[0, 100]^3``, flattened.

        Returns:
            tuple[Tensor, Tensor]:
                Mean :math:`\mu` and standard deviation :math:`\sigma` of the
                latent Gaussian ``eta``, each with shape ``[..., 1]``.
        """
        rho = theta[..., 0:1]
        alpha = theta[..., 1:4]
        u = theta[..., 4:5]

        def utility(basket):
            inner = (basket.pow(rho) * alpha).sum(dim=-1, keepdim=True)
            return inner.pow(1.0 / rho)

        x = design[..., 0:3]
        x_prime = design[..., 3:6]

        u_x = utility(x)
        u_xp = utility(x_prime)

        mu = (u_x - u_xp) * u
        diff_norm = (x - x_prime).norm(dim=-1, keepdim=True)
        sigma = (1.0 + diff_norm) * self.tau * u
        return mu, sigma

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """Evaluates the log probability of an observation under the CES model.

        Args:
            y (Tensor):
                Observations of shape ``[b, 1]``, values in ``[epsilon, 1 - epsilon]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 5]``.
            design (Tensor):
                Design tensor of shape ``[b, 6]``.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        mu, sigma = self.make_eta_params(theta, design)  # [b, 1] each

        # Continuous part: y = sigmoid(eta), eta = logit(y).
        # p(y) = N(logit(y); mu, sigma^2) / (y (1 - y)).
        # Clamp y into [eps, 1 - eps] before taking logs.
        y_safe = y.clamp(self.epsilon, 1.0 - self.epsilon)
        log_y = torch.log(y_safe)
        log1m_y = torch.log1p(-y_safe)
        logit_y = log_y - log1m_y
        log_cont = normal_log_prob(logit_y, mu, sigma) - log_y - log1m_y

        log_lower = _stable_log_ndtr((self.logit_lower - mu) / sigma)
        log_upper = _stable_log_ndtr((mu - self.logit_upper) / sigma)

        lower_mask = y <= self.epsilon
        upper_mask = y >= 1.0 - self.epsilon
        logp = torch.where(
            lower_mask, log_lower, torch.where(upper_mask, log_upper, log_cont)
        )
        return logp.squeeze(-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        """Samples an outcome from the CES model.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 5]``.
            design (Tensor):
                Design tensor of shape ``[b, 6]``.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 1]``, values in ``[epsilon, 1 - epsilon]``.
        """
        mu, sigma = self.make_eta_params(theta, design)
        eta = mu + sigma * torch.randn_like(mu)  # reparameterized
        y = torch.sigmoid(eta)
        return y.clamp(self.epsilon, 1.0 - self.epsilon)
