"""Hyperbolic temporal-discounting task.

References:
    1. https://arxiv.org/abs/2103.02438
"""

import math

import torch
from torch import Tensor
from torch.distributions import Bernoulli, HalfNormal, LogNormal, RelaxedBernoulli

from ..core.distributions import Distribution
from ..core.util import normal_cdf
from .task import Task


class HTDPrior(Distribution):
    r"""Prior over the discount rate and choice-noise scale :math:`(k, \alpha)`.

    Follows the settings in the DAD paper, Appendix D.2
    (https://arxiv.org/pdf/2103.02438). The default prior is:

    .. math::
        \begin{aligned}
        \log k &\sim \mathcal{N}(k_{\mathrm{loc}}, k_{\mathrm{scale}}^2) \\
        \alpha &\sim \mathrm{HalfNormal}(\alpha_{\mathrm{scale}})
        \end{aligned}

    Convention: :math:`\theta = (k, \alpha)`.

    Args:
        k_loc (float, optional):
            Mean of the log-discount rate :math:`\log k` (default: -4.25).
        k_scale (float, optional):
            Standard deviation of the log-discount rate :math:`\log k`
            (default: :math:`\sqrt{1.5}`).
        alpha_scale (float, optional):
            Scale of the half-normal prior over :math:`\alpha`
            (default: :math:`\sqrt{2}`).
    """

    def __init__(
        self, k_loc=-4.25, k_scale=math.sqrt(1.5), alpha_scale=math.sqrt(2)
    ) -> None:
        """Initializes HTD prior parameters."""
        super().__init__()

        self.register_buffer("k_loc", torch.tensor(k_loc))
        self.register_buffer("k_scale", torch.tensor(k_scale))
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

    @property
    def k_lognormal(self) -> torch.distributions.Distribution:
        """Distribution for the discount rate k."""
        return LogNormal(self.k_loc, self.k_scale)

    @property
    def alpha_halfnormal(self) -> torch.distributions.Distribution:
        """Distribution for the choice-noise scale alpha."""
        return HalfNormal(self.alpha_scale)

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            n_samples (int):
                How many samples to draw.

        Returns:
            Tensor:
                Samples, shape ``[n_samples, 2]``.
        """
        k_samples = self.k_lognormal.sample((n_samples, 1))
        alpha_samples = self.alpha_halfnormal.sample((n_samples, 1))

        samples = torch.hstack([k_samples, alpha_samples])
        return samples

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[b, 2]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[b]``.
        """
        k = theta[:, 0]
        alpha = theta[:, 1]

        logp_k = self.k_lognormal.log_prob(k)
        logp_alpha = self.alpha_halfnormal.log_prob(alpha)
        logp = logp_k + logp_alpha
        return logp


class HyperbolicTemporalDiscounting(Task):
    r"""Hyperbolic temporal-discounting task.

    A participant chooses between a delayed reward of 100 available after
    :math:`D_1` days, and an immediate reward :math:`R_0`. The subjective
    value of the delayed reward decays hyperbolically:

    .. math::
        V_1(\theta, x) = \frac{100}{1 + (k D_1)^s}

    .. math::
        p(y=1 \mid \theta, x) = \epsilon + (1 - 2\epsilon)\,
            \Phi\!\left(\frac{V_1(\theta, x) - R_0}{\alpha}\right)

    Convention: design :math:`x = (D_1, R_0)`; :math:`\theta = (k, \alpha)`.

    Args:
        epsilon (float, optional):
            Lapse-rate bias towards random responding (default: 0.01).
        exponent (float, optional):
            Exponent :math:`s` in the discounting curve; ``s=1`` gives the
            standard hyperbolic (HTD) model, other values give the Rachlin
            model (default: 1.0).
        temperature (float, optional):
            Relaxation temperature for reparametrized sampling. If 0,
            outcomes are sampled from a (non-reparametrized) Bernoulli;
            otherwise a RelaxedBernoulli is used (default: 0.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default HTDPrior is used.
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        exponent: float = 1.0,
        temperature: float = 0.0,
        prior=None,
    ) -> None:
        """Initializes the HTD task."""
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

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"HyperbolicTemporalDiscounting(exponent={self.exponent})"

    def make_prob(self, theta: Tensor, design: Tensor) -> Tensor:
        r"""Computes the probability of choosing the delayed reward.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``, with columns ``(k, alpha)``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``, with columns ``(D_1, R_0)``.

        Returns:
            Tensor:
                Probability :math:`p(y=1 \mid \theta, x)`, shape ``[b, 1]``.
        """
        k = theta[:, 0].unsqueeze(-1)
        alpha = theta[:, 1].unsqueeze(-1)

        D1 = design[:, 0].unsqueeze(-1)
        R0 = design[:, 1].unsqueeze(-1)

        V1 = 100 / (1 + (k * D1).pow(self.exponent))

        cdf_in = (V1 - R0) / alpha
        p_y1 = self.epsilon + (1.0 - 2 * self.epsilon) * normal_cdf(cdf_in)
        return p_y1.clamp(self.clamp_val, 1.0 - self.clamp_val)

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """Computes log probability of observations under the model.

        Note:
            The returned log-probability always corresponds to a hard
            (non-relaxed) Bernoulli likelihood, regardless of ``temperature``;
            ``temperature`` only affects sampling, via :meth:`_sample_outcome`.

        Args:
            y (Tensor):
                Observations of shape ``[b, 1]``, taking values in ``{0, 1}``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        p_y1 = self.make_prob(theta, design).squeeze(-1)
        log_p_y1 = torch.log(p_y1)
        log_p_y0 = torch.log1p(-p_y1)

        y = y.squeeze(-1)

        is_zero = y == 0
        is_one = y == 1

        logp = torch.full_like(p_y1, -torch.inf)
        logp = torch.where(is_one, log_p_y1, logp)
        logp = torch.where(is_zero, log_p_y0, logp)

        return logp

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        """Samples an outcome from the Bernoulli likelihood.

        Uses a reparametrized RelaxedBernoulli when ``temperature > 0``, and a
        (non-reparametrized) Bernoulli otherwise.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 1]``, taking values in ``{0, 1}``
                (or a relaxed value in ``(0, 1)`` when ``temperature > 0``).
        """
        p_y1 = self.make_prob(theta, design)  # [b, 1]
        if self.temperature > 0:
            return RelaxedBernoulli(temperature=self.temperature, probs=p_y1).rsample()
        return Bernoulli(p_y1).sample()
