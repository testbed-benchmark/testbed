"""Base design-objective interfaces."""

from abc import ABC, abstractmethod

from torch import Tensor

from ..core.rollout import Rollout
from ..tasks.base import Task


class Objective(ABC):
    """Shared base for design objectives."""

    def __init__(self, task: Task, fused: bool = False):
        """Initializes the objective.

        Args:
            task: Task whose expected information gain is estimated.
            fused: If True, score a trajectory in one compiled call rather
                than looping over the T steps. See :meth:`Task.history_logp`.
        """
        self.task = task
        self.fused = bool(fused)


class DesignObjective(Objective):
    """Estimates the gain of a single design given the history so far.

    Used by :class:`SequentialDesigner`, which optimizes one design at a time
    by gradient ascent. Examples include :class:`NMC` and :class:`PCE`.
    """

    def __call__(
        self,
        design: Tensor,
        y_history: Tensor | None = None,
        design_history: Tensor | None = None,
    ) -> Tensor:
        """Estimates the gain of a design. Alias for :meth:`estimate_gain`."""
        return self.estimate_gain(
            design, y_history=y_history, design_history=design_history
        )

    @abstractmethod
    def estimate_gain(
        self,
        design: Tensor,
        y_history: Tensor | None = None,
        design_history: Tensor | None = None,
    ) -> Tensor:
        """Estimates the expected information gain of a design.

        Args:
            design: Candidate design of shape ``[*event_shape_design]``.
            y_history: Outcomes observed so far, or None at the first step.
                ``task.current_prior`` is assumed to already condition on them.
            design_history: Designs observed so far, aligned with
                ``y_history``, or None at the first step.

        Returns:
            Scalar gain estimate.
        """
        ...


class RolloutObjective(Objective):
    """Estimates a gain, or a training surrogate for it, from a full rollout.

    Used to train amortized policies for :class:`PolicyDesigner`. Examples
    include :class:`SNMC`, :class:`SPCE` and :class:`Reinforce`.
    """

    def __call__(self, rollout: Rollout, **kwargs) -> Tensor:
        """Estimates the gain of a rollout. Alias for :meth:`estimate_gain`."""
        return self.estimate_gain(rollout, **kwargs)

    @abstractmethod
    def estimate_gain(self, rollout: Rollout) -> Tensor:
        """Estimates the expected information gain of a batched rollout.

        Args:
            rollout: Batch of ``b`` simulated rollouts.

        Returns:
            Scalar gain estimate, typically ``per_sample_gain(rollout).mean()``.
        """
        ...

    @abstractmethod
    def per_sample_gain(self, rollout: Rollout) -> Tensor:
        """Estimates the gain of each rollout in a batch.

        Args:
            rollout: Batch of ``b`` simulated rollouts.

        Returns:
            Per-sample gain estimates of shape ``[b]``.
        """
        ...
