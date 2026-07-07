"""Implements loss functions used to train ActionBED policies."""

import itertools
from functools import lru_cache

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
        reduction: str = "mean",
        epsilon: float = 1e-6,
    ) -> None:
        """Initializes."""
        super().__init__(reduction=reduction)
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

    __constants__ = ["reduction"]

    def __init__(
        self,
        reduction: str = "mean",
        sort_input: bool = False,
    ) -> None:
        """Initializes."""
        super().__init__(reduction=reduction)

        # Should we sort the inputs?
        # When these come from a NN, this introduces non-differentiability
        self.sort_input = sort_input

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass.

        Assumes input, target are both of shape [batch, num_objects, *event_shape_object].
        Each object is sorted according to its norm prior to computing the MSE.
        """
        if self.sort_input:
            input = sort_by_norm(input)
        target = sort_by_norm(target)

        mse = F.mse_loss(input, target, reduction=self.reduction)
        return mse


class SortedLogMSELoss(_Loss):
    """Implements the OrderedMSE Loss proposed in the ActionBED paper.

    See https://arxiv.org/abs/2606.23662 Equation 56.
    """

    __constants__ = ["reduction"]

    def __init__(
        self, reduction: str = "mean", epsilon: float = 1e-6, sort_input: bool = False
    ) -> None:
        """Initializes."""
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        super().__init__(reduction=reduction)

        # Should we sort the inputs?
        # When these come from a NN, this introduces non-differentiability
        self.sort_input = sort_input

        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Runs the forward pass.

        Assumes input, target are both of shape [batch, num_objects, *event_shape_object].
        Each object is sorted according to its norm prior to computing the MSE.

        [Warning] This computes the element-wise log to match the ActionBED paper.
        """
        if self.sort_input:
            input = sort_by_norm(input)
        target = sort_by_norm(target)

        mse = F.mse_loss(input, target, reduction="none")
        log_mse = mse + torch.log(mse + self.epsilon)

        if self.reduction == "sum":
            log_mse = log_mse.sum()
        elif self.reduction == "mean":
            log_mse = log_mse.mean()

        return log_mse


@lru_cache(maxsize=None)
def _permutation_index(n: int) -> Tensor:
    """All permutations of ``range(n)`` as a ``[n!, n]`` LongTensor (cached, CPU)."""
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)


class PermutationInvariantMSELoss(_Loss):
    r"""Permutation-invariant MSE loss (ActionBED, Appendix F.3, eq. 47).

    Inputs are tensors of shape ``[*batch, n_objects, dim_object]``. Writing
    :math:`K` = ``n_objects``, the per-sample loss is

        min_{sigma in S_K} * sum_k || input_k - target_{sigma(k)} ||_2^2

    i.e. the squared error under the assignment of estimate-objects to
    target-objects that minimises it, so the ordering of the object axis is
    irrelevant. The optimum is found by brute-force enumeration of all ``K!``
    permutation. Only practical for small ``K`` (roughly <= 8).

    Note that ``reduction='mean'`` normalizes by B*K*D.
    """

    __constants__ = ["reduction"]

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes."""
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss."""
        if input.shape != target.shape:
            raise ValueError(
                f"input and target must have the same shape, got "
                f"{tuple(input.shape)} and {tuple(target.shape)}."
            )
        if input.dim() < 2:
            raise ValueError(
                f"expected shape [*batch, n_objects, dim_object]; got a "
                f"{input.dim()}-D tensor."
            )

        K, D = input.shape[-2:]

        # Pairwise squared L2 distance between every estimate object i and
        # target object j:  cost[..., i, j] = || input_i - target_j ||_2^2
        diff = input.unsqueeze(-2) - target.unsqueeze(-3)  # [*batch, K, K, d]
        cost = diff.pow(2).sum(dim=-1)  # [*batch, K, K]

        perms = _permutation_index(K).to(input.device)  # [K!, K]
        rows = torch.arange(K, device=input.device)  # [K]

        perm_cost = cost[..., rows, perms].sum(dim=-1)  # [*batch, K!]

        # minimise over sigma
        pi_mse = perm_cost.min(dim=-1).values  # [*batch]

        if self.reduction == "none":
            return pi_mse
        if self.reduction == "mean":
            pi_mse = pi_mse / (D * K)
            return pi_mse.mean()
        if self.reduction == "sum":
            return pi_mse.sum()
