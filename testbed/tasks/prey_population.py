"""Prey-population task from Blau et al. (2022)."""

import torch
from torch import Tensor
from torch.distributions import Binomial, LogNormal

from ..core.distributions import Distribution
from ..envs.design_spaces import CategoricalDesignSpace
from .base import Task


class PreyPopulationPrior(Distribution):
    r"""Independent log-normal prior over attack rate and handling time.

    The prior follows Appendix B.3 of Blau et al. (2022):

    .. math::
        \log a, \log T_h \sim \mathcal{N}(-1.4, 1.35^2).

    Args:
        loc: Mean of both latent log-parameters.
        scale: Standard deviation of both latent log-parameters.

    Attributes:
        loc: Mean of each latent log-parameter.
        scale: Standard deviation of each latent log-parameter.
    """

    def __init__(self, loc: float = -1.4, scale: float = 1.35) -> None:
        """Initializes the prior parameters.

        Args:
            loc: Mean of both latent log-parameters.
            scale: Standard deviation of both latent log-parameters.
        """
        super().__init__()
        self.register_buffer("loc", torch.tensor(loc))
        self.register_buffer("scale", torch.tensor(scale))

    @property
    def lognormal(self) -> torch.distributions.Distribution:
        """Returns the shared log-normal marginal distribution.

        Returns:
            LogNormal distribution for either parameter.
        """
        return LogNormal(self.loc, self.scale)

    def sample(self, n_samples: int = 1) -> Tensor:
        """Draws samples of attack rate and handling time.

        Args:
            n_samples: Number of independent samples to draw.

        Returns:
            Parameter samples with shape ``[n_samples, 2]`` and columns
            ``(a, T_h)``.
        """
        return self.lognormal.sample((n_samples, 2))

    def log_prob(self, theta: Tensor) -> Tensor:
        """Evaluates the joint log probability of attack rate and handling time.

        Args:
            theta: Parameter values with shape ``[..., 2]`` and columns
                ``(a, T_h)``.

        Returns:
            Joint log probabilities with shape ``[...]``.
        """
        return self.lognormal.log_prob(theta).sum(dim=-1)


class PreyPopulation(Task):
    r"""Infer predator attack rate and handling time from prey consumption.

    An initial prey population :math:`d` survives for ``duration`` hours under
    Holling's type-III functional response:

    .. math::
        \frac{dN}{d\tau} = -\frac{aN^2}{1 + aT_hN^2}.

    The number consumed is binomial with probability
    :math:`p_T = (d - N_T) / d`. The final population is computed from the
    positive closed-form solution of the initial value problem.

    Args:
        duration: Length of each experiment in hours (default: 24).
        prior: Prior over ``(a, T_h)``. Defaults to :class:`PreyPopulationPrior`.

    Attributes:
        duration: Length of each experiment in hours.
        design_space: Categorical space of initial populations from 1 through
            300 prey.
    """

    def __init__(
        self,
        duration: float = 24.0,
        prior: Distribution | None = None,
    ) -> None:
        """Initializes the prey-population task.

        Args:
            duration: Length of each experiment in hours.
            prior: Prior over ``(a, T_h)``. If omitted, uses
                :class:`PreyPopulationPrior`.

        Raises:
            ValueError: If the duration is non-positive.
        """
        if duration <= 0:
            raise ValueError("duration must be positive.")
        if prior is None:
            prior = PreyPopulationPrior()

        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[2],
            event_shape_design=[1],
            prior=prior,
            discrete_y=True,
            design_space=CategoricalDesignSpace(
                torch.arange(1, 301, dtype=torch.get_default_dtype()).unsqueeze(-1)
            ),
        )
        self.duration = duration

    def __repr__(self) -> str:
        """Returns a compact representation of the experiment duration."""
        return f"PreyPopulation(duration={self.duration})"

    def population_after(self, theta: Tensor, design: Tensor) -> Tensor:
        """Solves for the final prey population.

        Args:
            theta: Parameters of shape ``[b, 2]`` with columns ``(a, T_h)``.
            design: Initial populations of shape ``[b, 1]``.

        Returns:
            Final populations of shape ``[b, 1]``.
        """
        attack_rate, handling_time = theta[:, 0:1], theta[:, 1:2]
        quadratic = attack_rate * handling_time
        linear = attack_rate * self.duration - quadratic * design + design.reciprocal()
        discriminant = linear.square() + 4 * quadratic
        root = discriminant.sqrt()
        return torch.where(
            linear >= 0,
            2 / (linear + root),
            (root - linear) / (2 * quadratic),
        )

    def make_consumption_probability(self, theta: Tensor, design: Tensor) -> Tensor:
        """Computes the probability that each initial prey animal is consumed.

        Args:
            theta: Parameters of shape ``[b, 2]`` with columns ``(a, T_h)``.
            design: Initial populations of shape ``[b, 1]``.

        Returns:
            Consumption probabilities of shape ``[b, 1]``.
        """
        final_population = self.population_after(theta, design)
        return ((design - final_population) / design).clamp(0.0, 1.0)

    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        """Evaluates the binomial observation log probability."""
        probability = self.make_consumption_probability(theta, design).squeeze(-1)
        return Binomial(total_count=design.squeeze(-1), probs=probability).log_prob(
            y.squeeze(-1)
        )

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        """Samples the number of prey consumed after the experiment."""
        probability = self.make_consumption_probability(theta, design).squeeze(-1)
        return (
            Binomial(total_count=design.squeeze(-1), probs=probability)
            .sample()
            .unsqueeze(-1)
        )
