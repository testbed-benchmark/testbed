"""Pharmacokinetic task.

References:
    1. https://arxiv.org/abs/2111.02329
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, TransformedDistribution
from torch.distributions.transforms import ExpTransform

from ..core.design_spaces import BoxDesignSpace, DesignSpace
from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base import Task


class PKPrior(Distribution):
    r"""Log-normal prior for theta = (kalpha, ke, v).

    The default prior is:

    .. math::
        p(\theta) = \mathcal{N}\left([\log 1, \log 0.1, \log 20], 0.05 I_3 \right)

    Args:
        theta_loc (Tensor, optional):
            Prior mean (default: None). Shape ``[3,]``.
        theta_covmat (Tensor, optional):
            Prior covariance (default: None). Shape ``[3, 3]``.
    """

    def __init__(
        self,
        theta_loc=None,
        theta_covmat=None,
    ) -> None:
        """Initializes prior parameters."""
        super().__init__()

        if theta_loc is None:
            m = torch.tensor([1.0, 0.1, 20])
            theta_loc = torch.log(m)
        if theta_covmat is None:
            theta_covmat = 0.05 * torch.eye(3)

        # All persistent tensors registered as buffers
        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.dim = 3

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
                Samples, shape ``[n_samples, 3]``
        """
        dist = self.theta_lognormal
        samples = dist.sample((n_samples,))
        return samples.reshape(-1, 3)

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates prior log probabilities.

        Args:
            theta (Tensor):
                Points at which to evaluate the log-probability. Shape ``[..., 3]``.

        Returns:
            Tensor:
                Log probs, shape ``[...,]``
        """
        dist = self.theta_lognormal
        return dist.log_prob(theta)


class PharmacoKinetic(Task):
    r"""Pharmacokinetic task.

    A one-compartment model with first-order absorption. The design is the time
    ``x`` at which blood is sampled, and theta = (ka, ke, V) are the absorption
    rate, elimination rate and volume of distribution. The concentration is

    .. math::
        \mu(\theta, x) = \frac{D_v}{V} \frac{k_a}{k_a - k_e}
                         \left(e^{-k_e x} - e^{-k_a x}\right)

    observed under combined multiplicative and additive noise, so that the
    observation standard deviation scales with the concentration:

    .. math::
        p(y \mid \theta, x) = \mathcal{N}\left(\mu(\theta, x),\;
            (\sigma_{\mathrm{mult}}\,\mu(\theta, x))^2 + \sigma_{\mathrm{add}}^2\right)

    Args:
        mult_noise_scale (float, optional):
            Multiplicative noise scale (:math:`\sigma_{\mathrm{mult}}`) (default: 0.1).
        add_noise_scale (float, optional):
            Additive noise scale (:math:`\sigma_{\mathrm{add}}`) (default: ``sqrt(0.1)``).
        decay_const (float, optional):
            Dose constant :math:`D_v` (default: 400.0).
        prior (Distribution, optional):
            Prior over parameters. If None, the default PKPrior is used.
        design_space (DesignSpace, optional):
            Sampling times. If None, defaults to ``[0, 24]`` hours.
    """

    @property
    def reparameterizable_outcomes(self) -> bool:
        """Gaussian noise added to a differentiable mean."""
        return True

    def __init__(
        self,
        mult_noise_scale: float = 0.1,
        add_noise_scale: float = 0.1**0.5,
        decay_const: float = 400.0,
        prior=None,
        design_space: DesignSpace | None = None,
        reparameterize_outcomes: bool | None = None,
    ) -> None:
        """Initializes the task."""
        if prior is None:
            prior = PKPrior()

        if design_space is None:
            design_space = BoxDesignSpace(low=0.0, high=24.0, event_shape=[1])

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[3],
            event_shape_design=[1],
            prior=prior,
            design_space=design_space,
            reparameterize_outcomes=reparameterize_outcomes,
        )

        self.mult_noise_scale = mult_noise_scale
        self.add_noise_scale = add_noise_scale
        self.decay_const = decay_const

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return (
            f"PharmacoKinetic(mult_noise_scale={self.mult_noise_scale}, "
            f"add_noise_scale={self.add_noise_scale}, "
            f"decay_const={self.decay_const})"
        )

    def obs_std(self, concentration: Tensor) -> Tensor:
        """Observation noise, which scales with the concentration.

        Args:
            concentration (Tensor):
                Concentrations of shape ``[b, 1]``.

        Returns:
            Tensor:
                Standard deviations of shape ``[b, 1]``.
        """
        mult = self.mult_noise_scale * concentration
        return torch.sqrt(mult**2 + self.add_noise_scale**2)

    def concentration(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the concentration given theta, design.

        Args:
            theta (Tensor):
                Parameter tensor of shape ``[b, 3]``.
            design (Tensor):
                Design tensor of shape ``[b, 1]``.

        Returns:
            Tensor:
                Concentration of shape ``[b, 1]``.
        """
        ka = theta[..., 0:1]
        ke = theta[..., 1:2]
        v = theta[..., 2:3]

        term1 = self.decay_const * ka / (v * (ka - ke))
        term2 = torch.exp(-ke * design) - torch.exp(-ka * design)
        return term1 * term2

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
                Parameter tensor of shape ``[b, n_sources, 3]``.
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
        loc = self.concentration(theta, design)
        logp = normal_log_prob(y, loc, self.obs_std(loc))
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
                Parameter tensor of shape ``[b, 3]``.
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
        loc = self.concentration(theta, design)  # [b, 1]
        return loc + self.obs_std(loc) * torch.randn_like(loc)
