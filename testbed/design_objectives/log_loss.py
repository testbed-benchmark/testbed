"""Implements a log-loss objective for training action policies."""

from torch import Tensor, nn

from ..core.distributions import Distribution
from ..tasks.base import Task
from .loss import _check_reduction, _check_same_shape


class LogLoss(nn.Module):
    r"""Negative log-likelihood of the true outcome under an estimated theta.

    .. math::
        L(\mathrm{input}, \mathrm{target}) =
            \mathbb{E}_{p(x)\, p(y \mid x, \mathrm{target})}
            [-\log p(y \mid x, \mathrm{input})]

    Estimated via a Monte Carlo expectation: for each batch element, designs
    ``x`` are drawn from the design prior, an outcome ``y`` is sampled from
    the task likelihood at the true ``target`` theta, and its log-probability
    is evaluated under the estimated ``input`` theta.

    The design prior is ``task.design_space.sample`` by default, so a task
    carries its own ``p(x)``. Pass ``design_distr`` to override it with an
    explicit distribution.

    Args:
        task (Task):
            Task defining the likelihood ``p(y | x, theta)`` and, by default,
            the design space designs are drawn from.
        n_outer_samples (int):
            Number of Monte Carlo (design, outcome) samples per batch element.
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).
        design_distr (Distribution, optional):
            Distribution over designs ``x`` used to estimate the outer
            expectation (default: None). If None, designs are drawn from
            ``task.design_space``.
    """

    def __init__(
        self,
        task: Task,
        n_outer_samples: int,
        reduction: str = "mean",
        design_distr: Distribution | None = None,
    ) -> None:
        """Initializes the loss."""
        super().__init__()

        self.design_distr = design_distr
        self.task = task
        self.n_outer_samples = int(n_outer_samples)
        self.reduction = _check_reduction(reduction)

    def _sample_designs(self, b: int) -> Tensor:
        """Draws ``n_outer_samples`` designs per batch element.

        Args:
            b (int):
                Batch size.

        Returns:
            Tensor:
                Designs, shape ``[b, n_outer_samples, *event_shape_design]``.
        """
        task = self.task
        if self.design_distr is None:
            return task.design_space.sample(
                (b, self.n_outer_samples), device=task.device
            )

        return self.design_distr.sample((b, self.n_outer_samples))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            input (Tensor):
                Estimated theta, shape ``[b, *event_shape_theta]``.
            target (Tensor):
                True theta, shape ``[b, *event_shape_theta]``.

        Returns:
            Tensor:
                If ``reduction`` is ``"none"``, shape ``[b, n_outer_samples]``.
                Otherwise, a scalar.

        Raises:
            ValueError: If ``input`` and ``target`` have different shapes.
        """
        _check_same_shape(input, target)
        task = self.task
        b = input.shape[0]

        # [b, n_outer_samples, *event_shape_design]
        design_samples = self._sample_designs(b)

        # y ~ p(y | x, target); target is fixed per batch element, shared
        # across the n_outer_samples designs.
        target = target.unsqueeze(1)  # [b, 1, *event_shape_theta]
        y_samples = task.sample_outcome(
            target, design_samples
        )  # [b, n_outer_samples, *event_shape_y]

        input = input.unsqueeze(1)  # [b, 1, *event_shape_theta]
        log_probs = task.outcome_logp(
            y_samples, input, design_samples
        )  # [b, n_outer_samples]

        if self.reduction == "sum":
            return -log_probs.sum()
        if self.reduction == "mean":
            return -log_probs.mean()
        return -log_probs
