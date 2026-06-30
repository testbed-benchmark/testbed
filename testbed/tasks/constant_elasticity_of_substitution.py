"""Constant elasticity of substitution (CES) task.

References:
    1. https://arxiv.org/abs/1903.05480
    2. https://arxiv.org/abs/2202.00821
"""

import math

import torch
from torch.distributions import Beta, Dirichlet, LogNormal

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base_task import Task


class CESPrior(Distribution):
    """Prior over the CES latent variables (rho, alpha, u).

    The default prior is:
        rho   ~ Beta(1, 1)
        alpha ~ Dirichlet([1, 1, 1])
        log u ~ N(1, 3).

    Convention: theta = (rho, alpha_1, alpha_2, alpha_3, u).
    """

    def __init__(
        self,
        rho_concentration=(1.0, 1.0),
        alpha_concentration=(1.0, 1.0, 1.0),
        u_loc=1.0,
        u_scale=3.0,
    ):
        """Initializes prior parameters."""
        super().__init__()

        self.register_buffer("rho_c1", torch.tensor(rho_concentration[0]))
        self.register_buffer("rho_c0", torch.tensor(rho_concentration[1]))
        self.register_buffer("alpha_concentration", torch.tensor(alpha_concentration))
        self.register_buffer("u_loc", torch.tensor(u_loc))
        self.register_buffer("u_scale", torch.tensor(u_scale))

    @property
    def rho_beta(self):
        """Distribution for the elasticity rho."""
        return Beta(self.rho_c1, self.rho_c0)

    @property
    def alpha_dirichlet(self):
        """Distribution for the weights alpha."""
        return Dirichlet(self.alpha_concentration)

    @property
    def u_lognormal(self):
        """Distribution for the scale u."""
        return LogNormal(self.u_loc, self.u_scale)

    def sample(self, n_samples: int = 1):
        """Returns samples from the prior distribution.

        Args:
            n_samples: The number of samples to draw from the prior.

        Returns:
            A tensor of shape `[n_samples, 5]`.
        """
        rho = self.rho_beta.sample((n_samples, 1))
        alpha = self.alpha_dirichlet.sample((n_samples,))
        u = self.u_lognormal.sample((n_samples, 1))
        return torch.cat([rho, alpha, u], dim=-1)

    def log_prob(self, theta):
        """Evaluates prior log probabilities.

        Args:
            theta: A sample from the prior distribution (can have leading batch dimensions).

        Returns:
            The log probability of `theta` under the prior distribution.
        """
        rho = theta[..., 0]
        alpha = theta[..., 1:4]
        u = theta[..., 4]

        logp_rho = self.rho_beta.log_prob(rho)
        logp_alpha = self.alpha_dirichlet.log_prob(alpha)
        logp_u = self.u_lognormal.log_prob(u)
        return logp_rho + logp_alpha + logp_u


class CES(Task):
    """Implements the constant elasticity of substitution (CES) model.

    A participant compares two baskets of goods x, x' in [0, 100]^3 and rates
    the subjective difference in utility on a sliding scale from 0 to 1. The
    outcome is a censored sigmoid of a Gaussian, so much of the design space
    maps to the boundary of the observation space.

    Convention: design = (x, x') flattened to 6 dims; theta = (rho, alpha, u).
    """

    def __init__(
        self,
        tau: float = 0.005,
        epsilon: float = 2**-22,
        prior=None,
    ):
        """Initializes the task.

        Args:
            tau: Observation-noise scale hyperparameter.
            epsilon: Censoring bound on the sigmoid output.
            prior: The prior distribution of the parameters.
        """
        if prior is None:
            prior = CESPrior()

        self.tau = tau
        self.epsilon = epsilon

        # Logit of the censoring bounds (logit(1 - eps) = -logit(eps))
        self.logit_lower = math.log(epsilon) - math.log1p(-epsilon)
        self.logit_upper = -self.logit_lower

        # Numerical clamping (to avoid infinite gradients of x^rho at x=0)
        self.clamp_val = 1e-6

        self.event_shape_theta = torch.Size([5])
        self.event_shape_design = torch.Size([6])
        self.event_shape_y = torch.Size([1])

        super().__init__(
            prior,
            discrete_y=False,
            discrete_theta=False,
            discrete_design=False,
        )

    def __repr__(self):
        """Returns a compact representation."""
        return f"CES(tau={self.tau}, epsilon={self.epsilon})"

    def make_eta_params(self, theta, design):
        """Computes the mean and scale of the latent Gaussian eta.

        Args:
            theta: Parameter tensor of shape `[..., 5]`, where the columns are
                `(rho, alpha_1, alpha_2, alpha_3, u)`.
            design: Design tensor of shape `[..., 6]`, representing two baskets
                of goods `(x, x')` each in `[0, 100]^3`, flattened.

        Returns:
            A tuple `(mu, sigma)`, each of shape `[..., 1]`, giving the mean and
            standard deviation of the latent Gaussian `eta`.
        """
        rho = theta[..., 0:1]
        alpha = theta[..., 1:4]
        u = theta[..., 4:5]

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

    def _outcome_logp(self, y, theta, design):
        """Evaluates the log probability of an observation under the CES model.

        Args:
            y: Observation tensor of shape `[b, 1]`, values in
                `[epsilon, 1 - epsilon]`.
            theta: Parameter tensor of shape `[b, 5]`.
            design: Design tensor of shape `[b, 6]`.

        Returns:
            Log probability tensor of shape `[b]`.
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

        # We need to account for the fact that the distribution of y is censored.
        # Lower censoring mass: P(sigmoid(eta) <= eps) = Phi((logit(eps) - mu) / sigma).
        log_lower = torch.special.log_ndtr((self.logit_lower - mu) / sigma)

        # Upper censoring mass: P(sigmoid(eta) >= 1 - eps) = Phi((mu - logit(1 - eps)) / sigma).
        log_upper = torch.special.log_ndtr((mu - self.logit_upper) / sigma)

        lower_mask = y <= self.epsilon
        upper_mask = y >= 1.0 - self.epsilon
        logp = torch.where(
            lower_mask, log_lower, torch.where(upper_mask, log_upper, log_cont)
        )
        return logp.squeeze(-1)

    def _sample_outcome(self, theta, design):
        """Samples an outcome from the CES model.

        Args:
            theta: Parameter tensor of shape `[b, 5]`.
            design: Design tensor of shape `[b, 6]`.

        Returns:
            Observation tensor of shape `[b, 1]`, values in
            `[epsilon, 1 - epsilon]`.
        """
        mu, sigma = self.make_eta_params(theta, design)
        eta = mu + sigma * torch.randn_like(mu)  # reparameterized
        y = torch.sigmoid(eta)
        return y.clamp(self.epsilon, 1.0 - self.epsilon)
