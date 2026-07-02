"""Hyperbolic temporal-discounting task."""

import math

import torch
from torch.distributions import Bernoulli, HalfNormal, LogNormal, RelaxedBernoulli

from ..core.distributions import Distribution
from ..core.util import normal_cdf
from .task import Task


class HTDPrior(Distribution):
    """Prior from the DAD paper HTD settings.

    This follows the settings in the DAD paper, Appendix D.2
    https://arxiv.org/pdf/2103.02438.
    """

    def __init__(self, k_loc=-4.25, k_scale=math.sqrt(1.5), alpha_scale=math.sqrt(2)):
        """Initializes prior parameters."""
        super().__init__()

        self.register_buffer("k_loc", torch.tensor(k_loc))
        self.register_buffer("k_scale", torch.tensor(k_scale))
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

    @property
    def k_lognormal(self):
        """Distribution for discount rate k."""
        return LogNormal(self.k_loc, self.k_scale)

    @property
    def alpha_halfnormal(self):
        """Distribution for alpha."""
        return HalfNormal(self.alpha_scale)

    def sample(self, n_samples: int = 1):
        """Draws parameter samples."""
        # returns: [n_samples, 2]
        k_samples = self.k_lognormal.sample((n_samples, 1))
        alpha_samples = self.alpha_halfnormal.sample((n_samples, 1))

        samples = torch.hstack([k_samples, alpha_samples])
        return samples

    def log_prob(self, theta):
        """Evaluates prior log probabilities."""
        # theta: [b, 2]

        k = theta[:, 0]
        alpha = theta[:, 1]

        logp_k = self.k_lognormal.log_prob(k)
        logp_alpha = self.alpha_halfnormal.log_prob(alpha)
        logp = logp_k + logp_alpha
        return logp


class HyperbolicTemporalDiscounting(Task):
    """Implements the Hyperbolic Temporal Discounting model.

    Convention: design = (D_1, R_0); theta = (k, alpha).
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        exponent: float = 1.0,
        temperature: float = 0.0,
        prior=None,
    ):
        """Initializes the task."""
        if prior is None:
            prior = HTDPrior()

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[2],
            prior=prior,
            discrete_y=True,
        )

        # Acts as a bias towards y=1
        self.epsilon = epsilon

        # Aka "s" -- s=1 corresponds to the HTD model
        # Other values correspond to the Rachlin model
        self.exponent = exponent

        # If temperature=0, we do not have reparametrised sampling
        # Otherwise, uses a relaex categorical at sampling time
        self.temperature = temperature

        # Numerical clamping
        self.clamp_val = 1e-6

    def __repr__(self):
        """Returns a compact representation."""
        return f"HyperbolicTemporalDiscounting(exponent={self.exponent})"

    def make_prob(self, theta, design):
        """Computes response probabilities."""
        # theta:  [b, 2]
        # design: [b, 2]
        # computes P(y=1 | theta, design)

        k = theta[:, 0].unsqueeze(-1)
        alpha = theta[:, 1].unsqueeze(-1)

        D1 = design[:, 0].unsqueeze(-1)
        R0 = design[:, 1].unsqueeze(-1)

        V1 = 100 / (1 + (k * D1).pow(self.exponent))

        cdf_in = (V1 - R0) / alpha
        p_y1 = self.epsilon + (1.0 - 2 * self.epsilon) * normal_cdf(cdf_in)
        return p_y1.clamp(self.clamp_val, 1.0 - self.clamp_val)

    def _outcome_logp(self, y, theta, design):
        # y: [b, 1]
        # theta: [b, 2]
        # design: [b, 1]
        # returns: [b, ]

        p_y1 = self.make_prob(theta, design)
        log_p_y1 = torch.log(p_y1)
        log_p_y0 = torch.log1p(-p_y1)

        y = y.squeeze(-1)

        is_zero = y == 0
        is_one = y == 1

        logp = torch.full_like(p_y1, -torch.inf)
        logp = torch.where(is_one, log_p_y1, logp)
        logp = torch.where(is_zero, log_p_y0, logp)

        return logp

    def _sample_outcome(self, theta, design):
        # theta:   [b, 2]
        # design:  [b, 2]
        # returns: [b, 1]
        p_y1 = self.make_prob(theta, design)  # [b, 1]
        if self.temperature > 0:
            return RelaxedBernoulli(temperature=self.temperature, probs=p_y1).rsample()
        return Bernoulli(p_y1).sample()
