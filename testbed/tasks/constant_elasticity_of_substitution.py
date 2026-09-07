"""Constant elasticity of substitution (CES) task.

References:
    1. https://arxiv.org/abs/1903.05480
    2. https://arxiv.org/abs/2202.00821


Note: there is some disagreement in the literature about which prior to use for u.
Using a smaller prior variance significantly simplifies the task. We thus follow SGBOED and use a prior variance of 9.
    - SGBOED uses a prior variance of 9 (https://github.com/ae-foster/pyro/blob/sgboed-reproduce/ces.py#L156)
    - PASOA uses a prior variance of 3 (https://github.com/jcopo/pasoa/blob/main/exptax/models/model_ces.py#L28)
"""

import math

import torch
from torch import Tensor
from torch.distributions import Beta, Dirichlet, LogNormal

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task

# Baskets hold between 0 and 100 units of each of 3 goods; a design is two
# such baskets flattened to 6 dims.
BASKET_LOW = 0.0
BASKET_HIGH = 100.0


def _split_theta(theta: Tensor):
    """Convenience function that splits theta = (rho, alpha, u) into its components.

    Args:
        theta (Tensor):
            Parameters of shape [..., 5].
    """
    rho = theta[..., 0:1]
    alpha = theta[..., 1:4]
    u = theta[..., 4:5]
    return rho, alpha, u


class CESPrior(Distribution):
    r"""Prior over the CES latent variables :math:`(\rho, \alpha, u)`.

    The default prior is:

    .. math::
        \begin{aligned}
        \rho &\sim \mathrm{Beta}(1, 1) \\
        \alpha &\sim \mathrm{Dirichlet}([1, 1, 1]) \\
        \log u &\sim \mathcal{N}(1, 3^2)
        \end{aligned}

    where :math:`\alpha = (\alpha_1, \alpha_2, \alpha_3)`. Convention: :math:`\theta = (\rho, \alpha_1, \alpha_2, \alpha_3, u)`.

    Args:
        rho_concentration (tuple, optional):
            Concentration parameters for the prior over :math:`\rho` (default: (1.0, 1.0)).
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
    def rho_beta(self) -> torch.distributions.Distribution:
        """Distribution for the elasticity rho."""
        return Beta(self.rho_c1, self.rho_c0)

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
        rho = self.rho_beta.sample((n_samples, 1))
        # a hack from the VBOED paper to prevent rho being too small
        # c.f. https://github.com/ae-foster/pyro/blob/vboed-reproduce/examples/contrib/oed/ces.py#L38
        # following VBOED, this is not accounted for in the prior log_prob.
        rho = 0.01 + 0.99 * rho

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
        rho, alpha, u = _split_theta(theta)

        logp_rho = self.rho_beta.log_prob(rho.squeeze(-1))
        logp_alpha = self.alpha_dirichlet.log_prob(alpha)
        logp_u = self.u_lognormal.log_prob(u.squeeze(-1))
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
        design_space (DesignSpace, optional):
            Controls the design parametrization. If None, a
            :class:`~testbed.core.design_spaces.BoxDesignSpace` over
            ``[0, 100]^6`` is used.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """The latent eta is Gaussian, so the censored sigmoid is pathwise."""
        return True

    def __init__(
        self,
        tau: float = 0.005,
        epsilon: float = 2**-22,
        prior=None,
        design_space: DesignSpace | None = None,
        action_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the CES task."""
        if prior is None:
            prior = CESPrior()

        if design_space is None:
            design_space = BoxDesignSpace(
                low=BASKET_LOW, high=BASKET_HIGH, event_shape=[6]
            )

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[5],
            event_shape_design=[6],
            prior=prior,
            design_space=design_space,
            action_space=action_space,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        self.tau = tau
        self.epsilon = epsilon

        # Logit of the censoring bounds (logit(1 - eps) = -logit(eps))
        self.logit_lower = math.log(epsilon) - math.log1p(-epsilon)
        self.logit_upper = -self.logit_lower

        # Numerical clamping (to avoid infinite gradients of x^rho at x=0)
        self.clamp_val = 1e-6

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
        rho, alpha, u = _split_theta(theta)

        def utility(basket):
            base = basket.clamp(min=self.clamp_val)
            inner = (base.pow(rho) * alpha).sum(dim=-1, keepdim=True)
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
        # Clamp y into [eps, 1 - eps] before taking logs. Boundary observations
        # select the censoring branches below, but feeding log(0) into the
        # unused continuous branch would otherwise poison the backward pass with
        # NaNs (0 * -inf) in theta.grad / design.grad.
        y_safe = y.clamp(self.epsilon, 1.0 - self.epsilon)
        log_y = torch.log(y_safe)
        log1m_y = torch.log1p(-y_safe)
        logit_y = log_y - log1m_y
        log_cont = normal_log_prob(logit_y, mu, sigma) - log_y - log1m_y

        # A small sigma (small u) drives z = (logit - mu) / sigma to ~1e8, so we floor it.
        z_min = -1e3

        # Lower censoring mass: P(sigmoid(eta) <= eps) = Phi((logit(eps) - mu) / sigma).
        log_lower = torch.special.log_ndtr(
            ((self.logit_lower - mu) / sigma).clamp(min=z_min)
        )

        # Upper censoring mass: P(sigmoid(eta) >= 1 - eps) = Phi((mu - logit(1 - eps)) / sigma).
        log_upper = torch.special.log_ndtr(
            ((mu - self.logit_upper) / sigma).clamp(min=z_min)
        )

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
