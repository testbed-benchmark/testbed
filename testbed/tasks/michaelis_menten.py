"""Michaelis-Menten task.

References:
    1. https://openreview.net/forum?id=uyNJLdCvFG
"""

from collections.abc import Iterable

import torch
from torch import Tensor

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


class MichaelisMentenPrior(Distribution):
    r"""Uniform prior over Michaelis-Menten parameters.

    The default prior is:

    .. math::
        V_{\max}, V_{\mathrm{half}} \sim \mathrm{Unif}(20, 200)

    Convention: :math:`\theta = (V_{\max}, V_{\mathrm{half}})`.

    Args:
        vmax_lb (float, optional):
            Lower bound of the uniform prior over :math:`V_{\max}` (default: 20).
        vmax_ub (float, optional):
            Upper bound of the uniform prior over :math:`V_{\max}` (default: 200).
        vhalf_lb (float, optional):
            Lower bound of the uniform prior over :math:`V_{\mathrm{half}}` (default: 20).
        vhalf_ub (float, optional):
            Upper bound of the uniform prior over :math:`V_{\mathrm{half}}` (default: 200).
    """

    def __init__(
        self,
        vmax_lb: float = 20,
        vmax_ub: float = 200,
        vhalf_lb: float = 20,
        vhalf_ub: float = 200,
    ):
        """Initializes Michaelis-Menten prior bounds."""
        super().__init__(event_shape=(2,))

        # lower/upper bounds
        self.register_buffer("lbs", torch.tensor([vmax_lb, vhalf_lb]))
        self.register_buffer("ubs", torch.tensor([vmax_ub, vhalf_ub]))

        # log normalizing constant
        self.register_buffer("log_z", -torch.log(self.ubs - self.lbs).sum())

    def sample(self, sample_shape: int | Iterable[int] = ()) -> Tensor:
        """Draws samples of parameters from the prior.

        Args:
            sample_shape (int or iterable):
                Shape of the batch of draws (default: ()).

        Returns:
            Tensor:
                Samples, shape ``[*sample_shape, 2]``.
        """
        noise = torch.rand(self._extended_shape(sample_shape), device=self.lbs.device)
        samples = (self.ubs - self.lbs) * noise + self.lbs
        return samples

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 2]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[...,]``.
        """
        in_support = ((theta >= self.lbs) & (theta <= self.ubs)).all(dim=-1)
        logp = torch.where(
            in_support, self.log_z, torch.tensor(-torch.inf, device=theta.device)
        )

        return logp


class MichaelisMenten(Task):
    r"""Michaelis-Menten chemical kinetics task.

    Likelihood:

    .. math::
        \mu(\theta, x) = \frac{V_{\max} x^s}{V_{\mathrm{half}}^s + x^s}

    .. math::
        p(y \mid \theta, x) = \mathcal{N}(\mu(\theta, x), \sigma^2)

    Convention: :math:`\theta = (V_{\max}, V_{\mathrm{half}})`; the design
    :math:`x` is the substrate concentration.

    Args:
        noise_scale (float, optional):
            Standard deviation of observation noise (:math:`\sigma`) (default: 2.5).
        exponent (float, optional):
            Hill exponent :math:`s`; ``s=1`` gives the standard Michaelis-Menten
            model, other values give the Hill model (default: 1.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default MichaelisMentenPrior is used.
        design_space (DesignSpace, optional):
            Controls the design parametrization. If None, a
            :class:`~testbed.core.design_spaces.BoxDesignSpace` over
            ``[0, 400]`` is used -- concentrations are non-negative, and the
            upper bound covers twice the largest :math:`V_{\mathrm{half}}` the
            default prior admits, where the response has saturated.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Gaussian noise added to a differentiable mean."""
        return True

    def __init__(
        self,
        noise_scale: float = 2.5,
        exponent: float = 1.0,
        prior=None,
        design_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the Michaelis-Menten task."""
        # Default prior
        if prior is None:
            prior = MichaelisMentenPrior()

        if design_space is None:
            design_space = BoxDesignSpace(low=0.0, high=400.0, event_shape=[1])

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[1],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        # Standard deviation of observation noise (assumed Gaussian)
        self.noise_scale = noise_scale

        # Aka "s" -- s=1 is the standard Michaelis-Menten model
        # Other values give rise to the so-called Hill model
        self.exponent = exponent

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"MichaelisMenten(noise_scale={self.noise_scale}, exponent={self.exponent})"
        )

    def make_mean(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the conditional mean E[y | theta, design].

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``, with columns
                ``(V_max, V_half)``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.

        Returns:
            Tensor:
                Mean of shape ``[b, 1]``.
        """
        vmax = theta[:, 0].unsqueeze(-1)
        vhalf = theta[:, 1].unsqueeze(-1)

        num = vmax * design.pow(self.exponent)
        denom = vhalf.pow(self.exponent) + design.pow(self.exponent)
        return num / denom

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
                Observations of shape ``[b, 1]``.
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(
        self,
        theta: Tensor,
        design: Tensor,
        y_history: Tensor = None,
        design_history: Tensor = None,
    ) -> Tensor:
        """Samples from the likelihood p(y | theta, design) using reparametrization.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 2]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.
            y_history (Tensor, optional):
                Ignored in this memoryless task.
            design_history (Tensor, optional):
                Ignored in this memoryless task.

        Returns:
            Tensor:
                Sampled outcomes of shape ``[b, 1]``.
        """
        loc = self.make_mean(theta, design)  # [b, 1]
        return torch.randn_like(loc) * self.noise_scale + loc
