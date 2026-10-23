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
                than looping over the T steps. See
                :meth:`~testbed.tasks.base.Task.history_logp`.
        """
        self.task = task
        self.fused = bool(fused)


class DesignObjective(Objective):
    """Estimates the gain of a single design given the history so far.

    Used by :class:`~testbed.designers.sequential.SequentialDesigner`, which
    optimizes one design at a time by gradient ascent. Examples include
    :class:`~testbed.design_objectives.snmc.NMC` and
    :class:`~testbed.design_objectives.spce.PCE`.
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
            y_history: Outcomes observed so far, of shape
                ``[1, t, *event_shape_y]``. ``task.current_prior`` is assumed
                to already condition on them. At the first step ``t == 0``
                (see :meth:`~testbed.tasks.base.Task.null_history`). None
                means no history and is only meant for direct calls.
            design_history: Designs observed so far, of shape
                ``[1, t, *event_shape_design]``, aligned with ``y_history``.
                None means no history.

        Returns:
            Scalar gain estimate.
        """
        ...


class RolloutObjective(Objective):
    """Estimates a gain, or a training surrogate for it, from a full rollout.

    Used to train amortized policies for
    :class:`~testbed.designers.amortized.PolicyDesigner` and static designs
    (see :mod:`testbed.methods.static`). Examples include
    :class:`~testbed.design_objectives.snmc.SNMC`,
    :class:`~testbed.design_objectives.spce.SPCE` and
    :class:`~testbed.design_objectives.reinforce.Reinforce`.

    Subclasses may add their own keyword arguments to :meth:`estimate_gain`
    and :meth:`per_sample_gain`, e.g. ``y_context`` and ``design_context`` on
    SPCE and SNMC.
    """

    def __call__(self, rollout: Rollout, **kwargs) -> Tensor:
        """Estimates the gain of a rollout. Alias for :meth:`estimate_gain`.

        Args:
            rollout: Batch of ``b`` simulated rollouts.
            **kwargs: Subclass-specific keyword arguments, forwarded to
                :meth:`estimate_gain`.

        Returns:
            Scalar gain estimate.
        """
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
