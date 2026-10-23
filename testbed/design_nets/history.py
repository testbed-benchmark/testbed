"""Shared validation and flattening for experiment histories."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import prod

import torch
from torch import Tensor

from .modules import affine_rescale


@dataclass(frozen=True)
class PreparedHistory:
    """History tensors in an actor-friendly flattened batch representation.

    ``batch_shape`` is retained so callers can restore their public result
    shape after evaluating an encoder or actor on the flattened histories.

    Attributes:
        batch_shape: Original leading batch dimensions.
        y_flat: Outcomes with shape ``[prod(batch_shape), T, dim_y]``.
        design_flat: Designs with shape
            ``[prod(batch_shape), T, dim_design]``.
        mask_flat: Optional validity mask with shape ``[prod(batch_shape), T]``.
    """

    batch_shape: torch.Size
    y_flat: Tensor
    design_flat: Tensor
    mask_flat: Tensor | None


def prepare_history(
    y_history,
    design_history,
    *,
    event_shape_y: Sequence[int],
    event_shape_design: Sequence[int],
    mask: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PreparedHistory:
    """Validate and flatten histories with arbitrary leading batch dimensions.

    Args:
        y_history: Outcomes shaped ``[*batch_shape, T, *event_shape_y]``.
        design_history: Designs shaped
            ``[*batch_shape, T, *event_shape_design]``.
        event_shape_y: Exact shape of one outcome event.
        event_shape_design: Exact shape of one design event.
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
    y_event_shape = torch.Size(event_shape_y)
    design_event_shape = torch.Size(event_shape_design)
    y_history = torch.as_tensor(y_history, device=device, dtype=dtype)
    design_history = torch.as_tensor(design_history, device=device, dtype=dtype)

    design_t_dim = design_history.ndim - len(design_event_shape) - 1
    if design_t_dim < 0 or design_history.shape[design_t_dim + 1 :] != (
        design_event_shape
    ):
        raise ValueError(
            "design_history must have shape "
            f"[*batch_shape, T, {tuple(design_event_shape)}]."
        )
    batch_shape = design_history.shape[:design_t_dim]
    history_length = design_history.shape[design_t_dim]
    if y_history.shape != (*batch_shape, history_length, *y_event_shape):
        raise ValueError(
            f"y_history must have shape [*batch_shape, T, {tuple(y_event_shape)}] "
            "with the same batch shape and history length as design_history."
        )

    batch_size = prod(batch_shape) if batch_shape else 1
    y_flat = y_history.reshape(batch_size, history_length, y_event_shape.numel())
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


def rescale_designs(
    designs: Tensor, low: Tensor, high: Tensor, target_low: float, target_high: float
) -> Tensor:
    """Maps flattened designs from ``[low, high]`` into ``[target_low, target_high]``.

    Args:
        designs: Designs with shape ``[..., dim_design]``.
        low: Lower bounds, broadcastable against ``designs``.
        high: Upper bounds, shaped like ``low``.
        target_low: Lower end of the target range.
        target_high: Upper end of the target range.

    Returns:
        Rescaled designs shaped like ``designs``. Coordinates whose bounds have
        zero width map to zero.
    """
    width = high - low
    scaled = target_low + (designs - low) * (target_high - target_low) / width
    return torch.where(width == 0, torch.zeros_like(scaled), scaled)


def normalize_history(
    history: PreparedHistory,
    *,
    design_bounds: tuple[Tensor, Tensor] | None = None,
    design_target: tuple[float, float] = (0.0, 1.0),
    outcome_bounds: tuple[Tensor, Tensor] | None = None,
) -> PreparedHistory:
    """Rescales a prepared history's designs and outcomes.

    Args:
        history: History from :func:`prepare_history`.
        design_bounds: Optional ``(low, high)`` design bounds, each
            broadcastable against ``[..., dim_design]``. Designs are left
            unchanged if omitted.
        design_target: Range that ``design_bounds`` map onto.
        outcome_bounds: Optional ``(low, high)`` outcome bounds, each
            broadcastable against ``[..., dim_y]``, mapped onto ``[0, 1]``.
            Outcomes are left unchanged if omitted.

    Returns:
        A :class:`PreparedHistory` with the same shapes and mask.
    """
    y_flat, design_flat = history.y_flat, history.design_flat
    if design_bounds is not None:
        design_flat = rescale_designs(design_flat, *design_bounds, *design_target)
    if outcome_bounds is not None:
        y_flat = affine_rescale(y_flat, *outcome_bounds, 0.0, 1.0)
    return replace(history, y_flat=y_flat, design_flat=design_flat)
