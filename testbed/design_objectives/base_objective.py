"""Base design-objective interface."""

from abc import ABC, abstractmethod

from torch import Tensor

from ..tasks.base import Task


class Objective(ABC):
    """Base class for design objectives."""

    def __init__(self, task: Task):
        """Stores the task."""
        self.task = task

    @abstractmethod
    def estimate_gain(self, design: Tensor):
        """Estimates expected information gain."""
        ...
