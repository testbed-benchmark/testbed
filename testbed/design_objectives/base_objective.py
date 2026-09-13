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
        self.fused = bool(fused) and task.memoryless


class DesignObjective(Objective):
    """Estimates a gain for a single design (no history).

    Used by `SequentialDesigner`, which optimizes one design at a time via
    gradient ascent (e.g. `NMC`, `PCE`).
    """

    @abstractmethod
    def estimate_gain(self, design: Tensor) -> Tensor:
        """Estimates expected information gain for a design."""
        ...


class RolloutObjective(Objective):
    """Estimates a gain/surrogate from a full rollout.

    Used by amortized (`PolicyDesigner`) training (e.g. `SNMC`, `SPCE`,
    `Reinforce`).
    """

    @abstractmethod
    def estimate_gain(self, rollout: Rollout) -> Tensor:
        """Estimates expected information gain (or a training surrogate) from a rollout."""
        ...
