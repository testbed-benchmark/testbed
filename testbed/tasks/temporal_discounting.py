"""Hyperbolic temporal-discounting task.

References:
    1. https://arxiv.org/abs/2103.02438
"""

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.distributions import (
    Bernoulli,
    HalfNormal,
    Independent,
    LogNormal,
    Normal,
    RelaxedBernoulli,
    TransformedDistribution,
    constraints,
)
from torch.distributions.transforms import Transform

from ..core.constraints import blocks
from ..core.distributions import Distribution, TorchDistribution, as_sample_shape
from ..core.util import normal_cdf
from .base import Task

# Constants
# The delayed reward the participant is offered i.e. R_0.
# Used in several places; hardcoded here to keep in sync
DELAYED_REWARD = 100.0
# Floor for D_1 before taking a log
_D1_FLOOR = 1e-30
# Valid designs: D_1 >= 0 and 0 <= R_0 <= 100
TD_DESIGNS = blocks(
    [constraints.nonnegative, constraints.interval(0.0, DELAYED_REWARD)], [1, 1]
)


class TDDesignTransform(Transform):
    r"""Maps raw designs onto TD's designs, :math:`D_1 \ge 0` and :math:`0 \le R_0 \le 100`.

    The reparametrization is :math:`D_1 = \exp(x_d - \texttt{log_k_mean})`,
    :math:`R_0 = 100\,\sigma(x_r)`, which maps an unconstrained
    :math:`(x_d, x_r)` onto that domain smoothly.

    Args:
        log_k_mean (float, optional):
            Shift applied to :math:`D_1`, so that :math:`k D_1 \sim O(1)` at a
            raw value near zero and gradients don't vanish for an untrained
            policy. Typically ``prior.k_loc + 0.5 * prior.k_scale**2``, i.e.
            :math:`\log \mathbb{E}[k]` (default: 0.0).
    """

    domain = constraints.independent(constraints.real, 1)
    codomain = TD_DESIGNS
    bijective = True

    def __init__(self, log_k_mean: float = 0.0):
        """Initializes with the D_1 shift."""
        super().__init__()
        self.log_k_mean = float(log_k_mean)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"TDDesignTransform(log_k_mean={self.log_k_mean})"

    def _call(self, raw_design: Tensor) -> Tensor:
        """Maps raw (x_d, x_r) to valid (D_1, R_0).

        The exponent is capped so exp cannot overflow to inf. Capping exp's
        *output* instead would make the backward pass multiply a zero clamp
        gradient by an infinite exp gradient, which is nan.
        """
        max_exponent = math.log(torch.finfo(raw_design.dtype).max) - 1.0
        D_raw = raw_design[..., 0:1]
        R_raw = raw_design[..., 1:2]

        D1 = torch.exp((D_raw - self.log_k_mean).clamp(max=max_exponent))
        R0 = DELAYED_REWARD * torch.sigmoid(R_raw)
        return torch.cat((D1, R0), dim=-1)

    def _inverse(self, design: Tensor) -> Tensor:
        """Recovers raw designs from designs; the inverse of _call."""
        D1 = design[..., 0:1]
        R0 = design[..., 1:2]
        D_raw = torch.log(D1.clamp(min=_D1_FLOOR)) + self.log_k_mean
        R_raw = torch.logit(R0 / DELAYED_REWARD, eps=1e-6)
        return torch.cat((D_raw, R_raw), dim=-1)

    def log_abs_det_jacobian(self, raw_design: Tensor, design: Tensor) -> Tensor:
        """Log-det of the exp / scaled-sigmoid reparametrisation.

        The map is diagonal, so this is the sum of the two per-coordinate
        terms: ``d D_1/d x_d = exp(x_d - log_k_mean)`` and
        ``d R_0/d x_r = 100 * sigmoid(x_r) * sigmoid(-x_r)``.

        This describes the mathematical transform; :meth:`_call` caps the
        exponent against overflow, and beyond that cap the realised derivative
        is 0 rather than what this reports.
        """
        D_raw = raw_design[..., 0]
        R_raw = raw_design[..., 1]
        log_det_D1 = D_raw - self.log_k_mean
        log_det_R0 = (
            math.log(DELAYED_REWARD)
            + torch.nn.functional.logsigmoid(R_raw)
            + torch.nn.functional.logsigmoid(-R_raw)
        )
        return log_det_D1 + log_det_R0


class TDDesignPrior(TorchDistribution):
    """Standard normal raw designs pushed through :class:`TDDesignTransform`.

    Args:
        log_k_mean (float, optional):
            The transform's :math:`D_1` shift (default: 0.0).
    """

    def __init__(self, log_k_mean: float = 0.0):
        """Builds the pushforward."""
        raw = Independent(Normal(torch.zeros(2), 1.0), 1)
        transform = TDDesignTransform(log_k_mean)
        super().__init__(TransformedDistribution(raw, [transform]))
        self.log_k_mean = transform.log_k_mean

    @property
    def transform(self) -> TDDesignTransform:
        """The map from raw designs."""
        return self.dist.transforms[0]


class TDPrior(Distribution):
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
            (default: 1.5).
        alpha_scale (float, optional):
            Scale of the half-normal prior over :math:`\alpha` (default: 2).
    """

    def __init__(self, k_loc=-4.25, k_scale=1.5, alpha_scale=2.0) -> None:
        """Initializes TD prior parameters.

        The scales are standard deviations, matching the official DAD code
        (log_k ~ Normal(-4.25, 1.5), alpha ~ HalfNormal(2)).
        """
        super().__init__(event_shape=(2,))

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

    @property
    def support(self) -> constraints.Constraint:
        """``k > 0`` and ``alpha >= 0``."""
        return blocks([constraints.positive, constraints.nonnegative], [1, 1])

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, 2]``.
        """
        shape = (*as_sample_shape(sample_shape), 1)
        k_samples = self.k_lognormal.sample(shape)
        alpha_samples = self.alpha_halfnormal.sample(shape)
        return torch.cat([k_samples, alpha_samples], dim=-1)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 2]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[...]``.
        """
        k = theta[..., 0]
        alpha = theta[..., 1]

        logp_k = self.k_lognormal.log_prob(k)
        logp_alpha = self.alpha_halfnormal.log_prob(alpha)
        logp = logp_k + logp_alpha
        return logp


class TemporalDiscounting(Task):
    r"""Hyperbolic temporal-discounting task.

    A participant chooses between a delayed reward of 100 available after
    :math:`D_1` days, and an immediate reward :math:`R_0`. The subjective
    value of the delayed reward decays hyperbolically:

    .. math::
        V_1(\theta, x) = \frac{100}{\left(1 + (k D_1)^s\right)^b}

    .. math::
        p(y=1 \mid \theta, x) = \epsilon + (1 - 2\epsilon)\,
            \Phi\!\left(\frac{V_1(\theta, x) - R_0}{\alpha}\right)

    Convention: design :math:`x = (D_1, R_0)`; :math:`\theta = (k, \alpha)`.

    The two exponents sit on opposite sides of the parenthesis and behave
    very differently. :math:`s` (``exponent``, Rachlin) is *inside*, and
    :math:`b` (``outer_exponent``, Green-Myerson) is *outside*. Both default
    to 1, recovering the standard hyperbolic model. Useful for misspecification
    experiments.

    Args:
        epsilon (float, optional):
            Lapse-rate bias towards random responding (default: 0.01).
        exponent (float, optional):
            Inner exponent :math:`s` in the discounting curve; ``s=1`` gives
            the standard hyperbolic (HTD) model, other values give the Rachlin
            model (default: 1.0).
        outer_exponent (float, optional):
            Outer exponent :math:`b`; ``b=1`` gives the standard hyperbolic
            (HTD) model, other values give the Green-Myerson model
            (default: 1.0).
        temperature (float, optional):
            Relaxation temperature. If 0, outcomes are sampled from a
            (non-reparametrized) Bernoulli; otherwise a RelaxedBernoulli is
            used. Selects both the sampler and the likelihood, so a positive
            value changes the model being scored, not just how it is drawn
            from: report EIG at ``temperature=0`` to stay comparable with the
            discrete model (default: 0.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default TDPrior is used.
        design_prior (Distribution, optional):
            Distribution of random designs; its support is the design space.
            If None, the default :class:`TDDesignPrior` is used.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Only the RelaxedBernoulli branch (``temperature > 0``) is pathwise."""
        return self.temperature > 0

    def __init__(
        self,
        epsilon: float = 0.01,
        exponent: float = 1.0,
        outer_exponent: float = 1.0,
        temperature: float = 0.0,
        prior=None,
        design_prior: Distribution | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the TD task."""
        if prior is None:
            prior = TDPrior()

        if design_prior is None:
            log_k_mean = float(prior.k_loc + 0.5 * prior.k_scale**2)
            design_prior = TDDesignPrior(log_k_mean=log_k_mean)

        # If temperature=0, we do not have reparametrised sampling
        # Otherwise, uses a relaxed categorical at sampling time.
        # Set before super().__init__, which reads reparameterizable_outcomes.
        self.temperature = temperature

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[2],
            prior=prior,
            discrete_y=True,
            design_prior=design_prior,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        # Acts as a bias towards y=1
        self.epsilon = epsilon

        # Aka "s" - s=1 corresponds to the HTD model
        # Other values correspond to the Rachlin model
        self.exponent = exponent

        # Aka "b" - b=1 corresponds to the HTD model
        # Other values correspond to the Green-Myerson model
        self.outer_exponent = outer_exponent

        # Numerical clamping
        self.clamp_val = 1e-6

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"TemporalDiscounting(exponent={self.exponent}, "
            f"outer_exponent={self.outer_exponent})"
        )

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
        # Floor alpha: prior samples can round to exactly zero, making the
        # division below produce nan gradients.
        alpha = theta[:, 1].unsqueeze(-1).clamp(min=self.clamp_val)

        D1 = design[:, 0].unsqueeze(-1)
        R0 = design[:, 1].unsqueeze(-1)

        V1 = DELAYED_REWARD / (1 + (k * D1).pow(self.exponent)).pow(self.outer_exponent)

        # Clamp for numerical stability: normal_cdf is saturated beyond +-15
        # (phi(15) ~ 5e-50), but the unclamped backward can produce inf/nan
        # gradients via 1/alpha when alpha is tiny.
        cdf_in = ((V1 - R0) / alpha).clamp(-15.0, 15.0)
        p_y1 = self.epsilon + (1.0 - 2 * self.epsilon) * normal_cdf(cdf_in)
        return p_y1.clamp(self.clamp_val, 1.0 - self.clamp_val)

    def _relaxed_bernoulli(self, probs: Tensor) -> RelaxedBernoulli:
        """Builds the relaxed likelihood, with temperature as a tensor.

        ``RelaxedBernoulli.log_prob`` calls ``.log()`` on temperature, so a
        float works for sampling but not for scoring.
        """
        temperature = torch.as_tensor(
            self.temperature, dtype=probs.dtype, device=probs.device
        )
        return RelaxedBernoulli(temperature=temperature, probs=probs)

    def _outcome_logp(
        self,
        y: Tensor,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Computes log probability of observations under the model.

        Note:
            Scores under whichever likelihood ``temperature`` selects, matching
            :meth:`_sample_outcome`: the hard Bernoulli pmf at ``temperature=0``,
            and the RelaxedBernoulli density above it. Scoring a relaxed draw
            under the hard pmf would return ``-inf`` for every observation,
            since a relaxed draw is never exactly 0 or 1. Keeping the two in
            step is also what leaves sPCE a valid bound, on the relaxed model's
            mutual information, which tends to the discrete one as
            ``temperature -> 0``.

        Args:
            y (Tensor):
                Observations of shape ``[b, 1]``, taking values in ``{0, 1}``
                (or in ``(0, 1)`` when ``temperature > 0``).
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        p_y1 = self.make_prob(theta, design)

        if self.temperature > 0:
            # RelaxedBernoulli has no mass at the endpoints and rsample can
            # round to them in float32, so keep y strictly interior.
            y_safe = y.clamp(self.clamp_val, 1.0 - self.clamp_val)
            return self._relaxed_bernoulli(p_y1).log_prob(y_safe).squeeze(-1)

        p_y1 = p_y1.squeeze(-1)
        log_p_y1 = torch.log(p_y1)
        log_p_y0 = torch.log1p(-p_y1)

        y = y.squeeze(-1)

        is_zero = y == 0
        is_one = y == 1

        logp = torch.full_like(p_y1, -torch.inf)
        logp = torch.where(is_one, log_p_y1, logp)
        logp = torch.where(is_zero, log_p_y0, logp)

        return logp

    def _sample_outcome(
        self,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Samples an outcome from the Bernoulli likelihood.

        Uses a reparametrized RelaxedBernoulli when ``temperature > 0``, and a
        (non-reparametrized) Bernoulli otherwise.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 2]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 1]``, taking values in ``{0, 1}``
                (or a relaxed value in ``(0, 1)`` when ``temperature > 0``).
        """
        p_y1 = self.make_prob(theta, design)  # [b, 1]
        if self.temperature > 0:
            return self._relaxed_bernoulli(p_y1).rsample()
        return Bernoulli(p_y1).sample()
