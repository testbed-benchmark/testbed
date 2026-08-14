"""Shared validation and flattening for experiment histories."""

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PreparedHistory:
    """History tensors in an actor-friendly flattened batch representation.

    ``batch_shape`` is retained so callers can restore their public result
    shape after evaluating an encoder or actor on the flattened histories.
    """

    batch_shape: torch.Size
    y_flat: Tensor
    design_flat: Tensor
    mask_flat: Tensor | None


def prepare_history(
    y_history,
    design_history,
    *,
    event_shape_design: Sequence[int],
    event_shape_y: Sequence[int] | None = None,
    mask: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PreparedHistory:
    """Validate and flatten histories with arbitrary leading batch dimensions.

    Args:
        y_history: Outcomes shaped ``[*batch_shape, T, *event_shape_y]``.
            If ``event_shape_y`` is omitted, the outcome event is the complete
            non-batch suffix after ``T``.
        design_history: Designs shaped
            ``[*batch_shape, T, *event_shape_design]``.
        event_shape_design: Exact shape of one design event.
        event_shape_y: Optional exact shape of one outcome event.
        mask: Optional boolean validity mask shaped ``[*batch_shape, T]``.
        device: Optional target device for history tensors and the mask.
        dtype: Optional target floating-point dtype for history tensors.

    Returns:
        A :class:`PreparedHistory` whose histories have shape
        ``[prod(batch_shape), T, flattened_event_size]``.

    Raises:
        ValueError: If a history event, batch shape, history length, or mask is
            incompatible with the declared history contract.
    """
    design_event_shape = torch.Size(event_shape_design)
    y_event_shape = None if event_shape_y is None else torch.Size(event_shape_y)
    y_history = torch.as_tensor(y_history, device=device, dtype=dtype)
    design_history = torch.as_tensor(design_history, device=device, dtype=dtype)

    design_t_dim = design_history.ndim - len(design_event_shape) - 1
    if design_t_dim < 0:
        raise ValueError(
            "design_history must have shape "
            f"[*batch_shape, T, {tuple(design_event_shape)}]."
        )
    batch_shape = torch.Size(design_history.shape[:design_t_dim])
    history_length = design_history.shape[design_t_dim]
    expected_design_suffix = (history_length, *design_event_shape)
    if tuple(design_history.shape[design_t_dim:]) != expected_design_suffix:
        raise ValueError(
            "design_history must have shape "
            f"[*batch_shape, T, {tuple(design_event_shape)}]."
        )

    if y_event_shape is None:
        y_t_dim = len(batch_shape)
        if y_history.ndim < y_t_dim + 2:
            raise ValueError(
                "y_history must have shape [*batch_shape, T, *event_shape_y]."
            )
    else:
        y_t_dim = y_history.ndim - len(y_event_shape) - 1
        if y_t_dim < 0 or tuple(y_history.shape[y_t_dim:]) != (
            history_length,
            *y_event_shape,
        ):
            raise ValueError(
                "y_history must have shape "
                f"[*batch_shape, T, {tuple(y_event_shape)}]."
            )

    if (
        y_t_dim != len(batch_shape)
        or torch.Size(y_history.shape[:y_t_dim]) != batch_shape
        or y_history.shape[y_t_dim] != history_length
    ):
        raise ValueError(
            "y_history must have the same batch shape and history length as "
            "design_history."
        )

    batch_size = prod(batch_shape) if batch_shape else 1
    y_dim = (
        y_event_shape.numel()
        if y_event_shape is not None
        else prod(y_history.shape[y_t_dim + 1 :])
    )
    y_flat = y_history.reshape(batch_size, history_length, y_dim)
    design_flat = design_history.reshape(
        batch_size, history_length, design_event_shape.numel()
    )

    mask_flat = None
    if mask is not None:
        mask = torch.as_tensor(mask, device=device)
        expected_mask_shape = (*batch_shape, history_length)
        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(
                "mask must have shape "
                f"[*batch_shape, T]; expected {expected_mask_shape}."
            )
        if mask.dtype is not torch.bool:
            raise ValueError("mask must be a boolean tensor.")
        mask_flat = mask.reshape(batch_size, history_length)

    return PreparedHistory(batch_shape, y_flat, design_flat, mask_flat)
