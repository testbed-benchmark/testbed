"""Implements loss functions used to train ActionBED policies."""

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from testbed.core.util import sort_by_norm


class LogMSELoss(_Loss):
    """Implements the LogMSELoss proposed in the ActionBED paper."""

    __constants__ = ["reduction", "epsilon"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        epsilon: float = 1e-8,
    ) -> None:
        """Initializes."""
        super().__init__(size_average, reduce, reduction)
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass."""
        mse = F.mse_loss(input, target, reduction=self.reduction)
        return mse + torch.log(mse + self.epsilon)


class SortedMSELoss(_Loss):
    """Implements the OrderedMSE Loss proposed in the ActionBED paper.

    See https://arxiv.org/abs/2606.23662 Equation 56.
    Note that they only propose this for the "Log-MSE" loss, but this version
    is for the standard MSE loss, up to sorting.
    """

    __constants__ = ["reduction", "epsilon"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        epsilon: float = 1e-8,
    ) -> None:
        """Initializes."""
        super().__init__(size_average, reduce, reduction)
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass.

        Assumes input and target both have shape
        ``[batch, num_objects, *event_shape_object]``.
        Each object is sorted according to its norm prior to computing the MSE.
        """
        sorted_input = sort_by_norm(input)
        sorted_target = sort_by_norm(target)

        mse = F.mse_loss(sorted_input, sorted_target, reduction=self.reduction)
        return mse


class SortedLogMSELoss(_Loss):
    """Implements the OrderedMSE Loss proposed in the ActionBED paper.

    See https://arxiv.org/abs/2606.23662 Equation 56.
    """

    __constants__ = ["reduction", "epsilon"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        epsilon: float = 1e-8,
    ) -> None:
        """Initializes."""
        super().__init__(size_average, reduce, reduction)
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass.

        Assumes input and target both have shape
        ``[batch, num_objects, *event_shape_object]``.
        Each object is sorted according to its norm prior to computing the MSE.
        """
        sorted_input = sort_by_norm(input)
        sorted_target = sort_by_norm(target)

        mse = F.mse_loss(sorted_input, sorted_target, reduction=self.reduction)
        return mse + torch.log(mse + self.epsilon)
