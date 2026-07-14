"""Base design-objective interface."""

from abc import ABC, abstractmethod

from torch import Tensor

from ..tasks.base import Task


class Objective(ABC):
    """Base class for design objectives."""

    def __init__(
        self,
        task: Task,
        is_expected_log_ratio: bool = False,
        requires_policy_log_probs: bool = False,
    ):
        """Initializes."""
        self.task = task

        # True for expected-log-ratio objectives (SPCE / SNMC): with a deterministic
        # or reparameterized policy, Reinforce adds the DAD Eq. 16 outcome score.
        self.is_expected_log_ratio = is_expected_log_ratio

        # Does this objective require log-probs from the policy? (e.g., REINFORCE)
        self.requires_policy_log_probs = requires_policy_log_probs

    @abstractmethod
    def estimate_gain(self, design: Tensor):
        """Estimates expected information gain."""
        ...
