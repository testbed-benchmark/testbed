"""Base design-objective interfaces."""

from abc import ABC, abstractmethod

from torch import Tensor

from ..core.rollout import Rollout
from ..tasks.base import Task


class Objective(ABC):
    """Shared base for design objectives."""

    def __init__(self, task: Task, fused: bool = False):
        """Initializes.

        fused scores a trajectory in one compiled call rather than looping over
        the T steps. See ``Task.history_logp``.
        """
        self.task = task
        self.fused = bool(fused)


class DesignObjective(Objective):
    """Estimates a gain for a single design (no history).

    Used by `SequentialDesigner`, which optimizes one design at a time via
    gradient ascent (e.g. `NMC`, `PCE`).
    """

    def __call__(
        self,
        design: Tensor,
        y_history: Tensor | None = None,
        design_history: Tensor | None = None,
    ) -> Tensor:
        """Estimates gain for one design."""
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
        """Estimates expected information gain for a design.

        design: ``[*event_shape_design]``.
        y_history, design_history: the history observed so far, which
        ``task.current_prior`` already conditions on (None at the first step).
        returns: scalar gain estimate.
        """
        ...


class RolloutObjective(Objective):
    """Estimates a gain/surrogate from a full rollout.

    Used by amortized (`PolicyDesigner`) training (e.g. `SNMC`, `SPCE`,
    `Reinforce`).
    """

    def __call__(self, rollout: Rollout, **kwargs) -> Tensor:
        """Estimates gain for a simulated rollout."""
        return self.estimate_gain(rollout, **kwargs)

    @abstractmethod
    def estimate_gain(self, rollout: Rollout) -> Tensor:
        """Estimates expected information gain (or a training surrogate) from a rollout.

        returns: scalar, typically ``per_sample_gain(rollout).mean()``.
        """
        ...

    @abstractmethod
    def per_sample_gain(self, rollout: Rollout) -> Tensor:
        """Estimates per-sample gains (or surrogates) from a batched rollout.

        returns: [b]
        """
        ...
