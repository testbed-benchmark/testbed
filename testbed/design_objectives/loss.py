"""Implements various loss functions used to train action and design policies.

References:
    1. https://arxiv.org/abs/2606.23662
"""

import itertools
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from testbed.core.util import sort_by_norm


def _check_same_shape(input: Tensor, target: Tensor) -> None:
    """Checks that ``input`` and ``target`` have identical shapes.

    Args:
        input (Tensor):
            Estimated values.
        target (Tensor):
            Target values.

    Raises:
        ValueError: If ``input`` and ``target`` have different shapes.
    """
    if input.shape != target.shape:
        raise ValueError(
            f"input and target must have the same shape, got "
            f"{tuple(input.shape)} and {tuple(target.shape)}."
        )


def _check_reduction(reduction: str) -> str:
    """Validates a ``reduction`` argument.

    Args:
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"``.

    Returns:
        str:
            ``reduction``, unchanged.

    Raises:
        ValueError: If ``reduction`` is not one of the supported values.
    """
    if reduction not in ("sum", "mean", "none"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    return reduction


class LotDecisionLoss(nn.Module):
    """Asymmetric accept/reject loss for lot inspection.

    The input is an acceptance logit and the target is a latent lot parameter
    vector whose first component indicates whether the lot is good. Its forward
    value is the terminal decision loss used by the destructive lot inspection
    PPO example. A cost-sensitive binary-cross-entropy surrogate supplies the
    gradient with respect to the acceptance logit.

    Args:
        false_accept_cost: Loss for accepting a bad lot.
        false_reject_cost: Loss for rejecting a good lot.
        reduction: One of ``"sum"``, ``"mean"``, or ``"none"``.
    """

    def __init__(
        self,
        false_accept_cost: float = 1.0,
        false_reject_cost: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initializes the loss."""
        super().__init__()
        if false_accept_cost < 0.0 or false_reject_cost < 0.0:
            raise ValueError("decision loss costs must be non-negative.")
        self.false_accept_cost = float(false_accept_cost)
        self.false_reject_cost = float(false_reject_cost)
        self.reduction = _check_reduction(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes expected terminal decision loss for each lot."""
        if input.shape != (*target.shape[:-1], 1):
            raise ValueError(
                "input must have one acceptance logit per target; got "
                f"{tuple(input.shape)} and {tuple(target.shape)}."
            )

        good = target[..., 0].bool()
        acceptance_logit = input.squeeze(-1)
        accepted = acceptance_logit >= 0.0
        decision_loss = self.false_accept_cost * (accepted & ~good).to(target.dtype)
        decision_loss += self.false_reject_cost * (~accepted & good).to(target.dtype)

        surrogate = (
            self.false_accept_cost
            * (~good).to(target.dtype)
            * F.softplus(acceptance_logit)
        )
        surrogate += (
            self.false_reject_cost
            * good.to(target.dtype)
            * F.softplus(-acceptance_logit)
        )
        loss = surrogate + (decision_loss - surrogate).detach()

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss


def _elementwise_log_mse(
    input: Tensor, target: Tensor, epsilon: float, reduction: str
) -> Tensor:
    r"""Computes an element-wise sum of mse and log-mse, then reduces.

    .. math::
        \mathrm{mse} = (\mathrm{input} - \mathrm{target})^2, \qquad
        \ell = \mathrm{mse} + \log(\mathrm{mse} + \epsilon)

    The log must be applied element-wise (rather than to an already-reduced
    mse) to match the ActionBED paper [1]; shared by :class:`LogMSELoss` and
    :class:`SortedLogMSELoss` so the two formulas can't drift apart.

    Args:
        input (Tensor):
            Estimated values.
        target (Tensor):
            Target values, the same shape as ``input``.
        epsilon (float):
            Small constant added before taking the log, for numerical
            stability.
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"``.

    Returns:
        Tensor:
            If ``reduction`` is ``"none"``, the same shape as ``input``.
            Otherwise, a scalar.
    """
    mse = F.mse_loss(input, target, reduction="none")
    log_mse = mse + torch.log(mse + epsilon)

    if reduction == "sum":
        return log_mse.sum()
    if reduction == "mean":
        return log_mse.mean()
    return log_mse


def _maybe_sort_by_norm(
    input: Tensor, target: Tensor, sort_input: bool
) -> tuple[Tensor, Tensor]:
    """Sorts target (and input, if requested) by object norm.

    The target is always sorted into a canonical order; the input is only
    sorted when ``sort_input`` is set, since doing so introduces
    non-differentiability if the input comes from a neural network.

    Args:
        input (Tensor):
            Estimated objects, of shape ``[batch, n_objects, *event_shape_object]``
            or ``[n_objects, *event_shape_object]`` (unbatched).
        target (Tensor):
            Target objects, the same shape as ``input``.
        sort_input (bool):
            Whether to also sort ``input`` by norm.

    Returns:
        tuple[Tensor, Tensor]:
            The ``(input, target)`` pair, with the target (and, if
            requested, the input) sorted by norm along the objects axis.
    """
    if sort_input:
        input = sort_by_norm(input)
    return input, sort_by_norm(target)


class LogMSELoss(nn.Module):
    r"""Sum of a standard MSE loss and an MSE loss on a log-scale.

    Proposed in the ActionBED paper [1]. Computes, element-wise:

    .. math::
        \mathrm{mse} = (\mathrm{input} - \mathrm{target})^2, \qquad
        \ell = \mathrm{mse} + \log(\mathrm{mse} + \epsilon)

    Args:
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).
        epsilon (float):
            Small constant added before taking the log, for numerical
            stability (default: ``1e-6``).

    Warning:
        The log above is computed element-wise (rather than on an
        already-reduced mse), to match the ActionBED paper [1].
    """

    def __init__(
        self,
        reduction: str = "mean",
        epsilon: float = 1e-6,
    ) -> None:
        """Initializes the loss."""
        super().__init__()
        self.reduction = _check_reduction(reduction)
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            input (Tensor):
                Estimated values shape ``[..., dim]``.
            target (Tensor):
                Target values, the same shape as ``input``.

        Returns:
            Tensor:
                If ``reduction`` is ``"none"``, the same shape as ``input``.
                Otherwise, a scalar.

        Raises:
            ValueError: If ``input`` and ``target`` have different shapes.
        """
        _check_same_shape(input, target)
        return _elementwise_log_mse(input, target, self.epsilon, self.reduction)


class SortedMSELoss(nn.Module):
    """An MSELoss where targets are sorted by their norm before comparison.

    See the ActionBED paper [1], Equation 56. Note that the paper only
    proposes this for the "Log-MSE" loss, but this version is for the
    standard MSE loss, up to sorting.

    Args:
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).
        sort_input (bool):
            Whether to also sort ``input`` by norm before comparing to the
            (always-sorted) target. Sorting the input introduces
            non-differentiability if it comes from a neural network
            (default: ``False``).
    """

    def __init__(
        self,
        reduction: str = "mean",
        sort_input: bool = False,
    ) -> None:
        """Initializes the loss."""
        super().__init__()
        self.reduction = _check_reduction(reduction)
        self.sort_input = sort_input

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Each object is sorted according to its norm prior to computing the MSE.

        Args:
            input (Tensor):
                Estimated objects, of shape ``[batch, n_objects, *event_shape_object]``
                or ``[n_objects, *event_shape_object]`` (unbatched).
            target (Tensor):
                Target objects, the same shape as ``input``.

        Returns:
            Tensor:
                If ``reduction`` is ``"none"``, the same shape as ``input``.
                Otherwise, a scalar.

        Raises:
            ValueError: If ``input`` and ``target`` have different shapes.
        """
        _check_same_shape(input, target)
        input, target = _maybe_sort_by_norm(input, target, self.sort_input)
        return F.mse_loss(input, target, reduction=self.reduction)


class SortedLogMSELoss(nn.Module):
    """A LogMSELoss where targets are sorted by their norm before comparison.

    See the ActionBED paper [1], Equation 56. Combines the element-wise log
    of :class:`LogMSELoss` with the norm-based sorting of
    :class:`SortedMSELoss`.

    Args:
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).
        epsilon (float):
            Small constant added before taking the log, for numerical
            stability (default: ``1e-6``).
        sort_input (bool):
            Whether to also sort ``input`` by norm before comparing to the
            (always-sorted) target. Sorting the input introduces
            non-differentiability if it comes from a neural network
            (default: ``False``).

    Warning:
        The log is computed element-wise (rather than on an already-reduced
        mse), to match the ActionBED paper [1].
    """

    def __init__(
        self, reduction: str = "mean", epsilon: float = 1e-6, sort_input: bool = False
    ) -> None:
        """Initializes the loss."""
        super().__init__()
        self.reduction = _check_reduction(reduction)
        self.sort_input = sort_input
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Each object is sorted according to its norm prior to computing the MSE.

        Args:
            input (Tensor):
                Estimated objects, of shape ``[batch, n_objects, *event_shape_object]``
                or ``[n_objects, *event_shape_object]`` (unbatched).
            target (Tensor):
                Target objects, the same shape as ``input``.

        Returns:
            Tensor:
                If ``reduction`` is ``"none"``, the same shape as ``input``.
                Otherwise, a scalar.

        Raises:
            ValueError: If ``input`` and ``target`` have different shapes.
        """
        _check_same_shape(input, target)
        input, target = _maybe_sort_by_norm(input, target, self.sort_input)
        return _elementwise_log_mse(input, target, self.epsilon, self.reduction)


@lru_cache(maxsize=None)
def _permutation_index(n: int) -> Tensor:
    """All permutations of ``range(n)``, cached (CPU).

    Args:
        n (int):
            Number of objects to permute.

    Returns:
        Tensor:
            LongTensor of shape ``[n!, n]``, one permutation per row.
    """
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long)


class PermutationInvariantMSELoss(nn.Module):
    r"""A permutation-invariant version of the MSE loss, defined by the optimal matching between objects.

    See the ActionBED paper [1], Appendix F.3, eq. 47.

    Inputs are tensors of shape ``[*batch, n_objects, dim_object]``. Writing
    :math:`K` = ``n_objects``, the per-sample loss is

    .. math::
        \min_{\sigma \in S_K} \sum_{k=1}^K
            \lVert \mathrm{input}_k - \mathrm{target}_{\sigma(k)} \rVert_2^2

    i.e. the squared error under the assignment of estimate-objects to
    target-objects that minimises it, so the ordering of the object axis is
    irrelevant. The optimum is found by brute-force enumeration of all ``K!``
    permutations. Only practical for small ``K`` (roughly <= 8).

    Args:
        reduction (str):
            One of ``"sum"``, ``"mean"``, or ``"none"`` (default: ``"mean"``).

    Note:
        ``reduction='mean'`` normalizes by ``B*K*D``, not just ``B``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        """Initializes the loss."""
        super().__init__()
        self.reduction = _check_reduction(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            input (Tensor):
                Estimated objects, of shape ``[*batch, n_objects, dim_object]``.
            target (Tensor):
                Target objects, the same shape as ``input``.

        Returns:
            Tensor:
                If ``reduction`` is ``"none"``, shape ``[*batch]``.
                Otherwise, a scalar.

        Raises:
            ValueError: If ``input`` and ``target`` have different shapes, or
                if they have fewer than 2 dimensions.
        """
        _check_same_shape(input, target)
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

        # minimise over permutations
        pi_mse = perm_cost.min(dim=-1).values  # [*batch]

        if self.reduction == "none":
            return pi_mse
        if self.reduction == "mean":
            pi_mse = pi_mse / (D * K)
            return pi_mse.mean()
        if self.reduction == "sum":
            return pi_mse.sum()
