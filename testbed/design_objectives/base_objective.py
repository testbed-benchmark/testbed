from abc import ABC, abstractmethod

from torch import Tensor

from ..tasks.base_task import Task

class Objective(ABC):

    def __init__(self, task: Task):
        self.task = task

    @abstractmethod
    def estimate_gain(self, design: Tensor): ...