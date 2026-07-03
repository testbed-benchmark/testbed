"""Hyperbolic temporal-discounting task."""

import torch
from torch.distributions import Bernoulli, HalfNormal, LogNormal, RelaxedBernoulli

from ..core.distributions import Distribution
from ..core.util import normal_cdf
from .base_task import Task


class HTDPrior(Distribution):
    """Prior from the DAD paper HTD settings.

    This follows the settings in the DAD paper, Appendix D.2
    https://arxiv.org/pdf/2103.02438.
    """

    def __init__(self, k_loc=-4.25, k_scale=1.5, alpha_scale=2.0):
        """Initializes prior parameters.

        The scales are standard deviations, matching the official DAD code
        (log_k ~ Normal(-4.25, 1.5), alpha ~ HalfNormal(2)).
        """
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

    With unconstrained_design=True, the design is instead an unconstrained
    (xi_d, xi_r) mapped to D_1 = exp(xi_d - log E[k]) and R_0 = 100 * sigmoid(xi_r).
    Without it, near-zero policy outputs saturate the choice probability and all
    design gradients vanish.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        exponent: float = 1.0,
        temperature: float = 0.0,
        unconstrained_design: bool = False,
        prior=None,
    ):
        """Initializes the task."""
        if prior is None:
            prior = HTDPrior()

        # Acts as a bias towards y=1
        self.epsilon = epsilon

        # Aka "s" -- s=1 corresponds to the HTD model
        # Other values correspond to the Rachlin model
        self.exponent = exponent

        # If temperature=0, we do not have reparametrised sampling
        # Otherwise, uses a relaex categorical at sampling time
        self.temperature = temperature

        # Shift so k * D_1 is O(1) for xi_d near 0 (cf. Step-DAD): log E[k] under the prior
        self.unconstrained_design = unconstrained_design
        if unconstrained_design:
            self.log_k_mean = float(prior.k_loc + 0.5 * prior.k_scale**2)

        # Numerical clamping
        self.clamp_val = 1e-6

        self.event_shape_theta = torch.Size([2])
        self.event_shape_design = torch.Size([2])
        self.event_shape_y = torch.Size([1])

        super().__init__(
            prior,
            discrete_y=True,
            discrete_theta=False,
            discrete_design=False,
        )

    def __repr__(self):
        """Returns a compact representation."""
        return f"HyperbolicTemporalDiscounting(exponent={self.exponent})"

    def make_prob(self, theta, design):
        """Computes response probabilities."""
        # theta:  [b, 2]
        # design: [b, 2]
        # computes P(y=1 | theta, design)

        k = theta[:, 0].unsqueeze(-1)
        # Floor alpha: prior samples can round to exactly zero, making the
        # division below produce nan gradients.
        alpha = theta[:, 1].unsqueeze(-1).clamp(min=self.clamp_val)

        D1 = design[:, 0].unsqueeze(-1)
        R0 = design[:, 1].unsqueeze(-1)

        if self.unconstrained_design:
            D1 = torch.exp(D1 - self.log_k_mean)
            R0 = 100 * torch.sigmoid(R0)

        V1 = 100 / (1 + (k * D1).pow(self.exponent))

        # Clamp for numerical stability: normal_cdf is saturated beyond +-15
        # (phi(15) ~ 5e-50), but the unclamped backward can produce inf/nan
        # gradients via 1/alpha when alpha is tiny.
        cdf_in = ((V1 - R0) / alpha).clamp(-15.0, 15.0)
        p_y1 = self.epsilon + (1.0 - 2 * self.epsilon) * normal_cdf(cdf_in)
        return p_y1.clamp(self.clamp_val, 1.0 - self.clamp_val)

    def _outcome_logp(self, y, theta, design):
        # y: [b, 1]
        # theta: [b, 2]
        # design: [b, 1]
        # returns: [b, ]

        p_y1 = self.make_prob(theta, design).squeeze(-1)  # [b]
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
