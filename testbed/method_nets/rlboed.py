"""Q-networks used by RL-BOED's SAC critics."""

from math import prod

import torch
from torch import Tensor, nn

from testbed.design_nets.encoder import DeepSetEncoder, HistoryEncoder
from testbed.design_nets.history import PreparedHistory, prepare_history
from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale


def _flat_bounds(bound: Tensor) -> Tensor:
    """Flattens per-coordinate bounds, leaving a scalar bound to broadcast."""
    return bound.reshape(-1) if bound.numel() > 1 else bound


class _RLBOEDCritic(nn.Module):
    """Shared history handling for the RL-BOED critics.

    Histories follow the shared design-net convention: outcomes shaped
    ``[*batch_shape, T, *event_shape_y]``, designs shaped
    ``[*batch_shape, T, *event_shape_design]`` and an optional boolean mask
    shaped ``[*batch_shape, T]``.
    """

    def __init__(
        self,
        *,
        event_shape_y,
        event_shape_design,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h,
        dim_emb: int,
        encoder: HistoryEncoder | None,
    ):
        super().__init__()
        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_design = torch.Size(event_shape_design)
        self.dim_y = prod(self.event_shape_y)
        self.dim_design = prod(self.event_shape_design)
        self.encoder = encoder or DeepSetEncoder(
            (self.dim_y,), (self.dim_design,), dim_h=dim_h, dim_emb=dim_emb
        )
        if (self.encoder.dim_y, self.encoder.dim_design) != (
            self.dim_y,
            self.dim_design,
        ):
            raise ValueError(
                "encoder event sizes must match event_shape_y and event_shape_design."
            )
        self.dim_emb = self.encoder.dim_emb
        # Buffers keep the shapes they were given so existing checkpoints load.
        for name, value, size in (
            ("design_low", design_low, self.dim_design),
            ("design_high", design_high, self.dim_design),
            ("outcome_low", outcome_low, self.dim_y),
            ("outcome_high", outcome_high, self.dim_y),
        ):
            value = torch.as_tensor(value)
            if value.numel() not in (1, size):
                raise ValueError(
                    f"{name} must be a scalar or have one entry per coordinate."
                )
            self.register_buffer(name, value)

    def _normalize_designs(self, designs: Tensor, low: float, high: float) -> Tensor:
        """Maps flattened physical designs into ``[low, high]`` coordinates."""
        design_low = _flat_bounds(self.design_low)
        width = _flat_bounds(self.design_high) - design_low
        scaled = low + (designs - design_low) * (high - low) / width
        return torch.where(width == 0, torch.zeros_like(scaled), scaled)

    def _prepare(
        self, y_history, design_history, mask: Tensor | None
    ) -> PreparedHistory:
        """Validates, flattens, and normalizes a history for the encoder."""
        prepared = prepare_history(
            y_history,
            design_history,
            event_shape_y=self.event_shape_y,
            event_shape_design=self.event_shape_design,
            mask=mask,
            device=self.design_low.device,
            dtype=self.encoder.null_enc.dtype,
        )
        y_flat = affine_rescale(
            prepared.y_flat,
            _flat_bounds(self.outcome_low),
            _flat_bounds(self.outcome_high),
            0.0,
            1.0,
        )
        design_flat = self._normalize_designs(prepared.design_flat, 0.0, 1.0)
        return PreparedHistory(
            prepared.batch_shape, y_flat, design_flat, prepared.mask_flat
        )

    def _encode(self, history: PreparedHistory) -> Tensor:
        """Encodes a prepared history to ``[prod(batch_shape), dim_emb]``."""
        return self.encoder.encode_flat(
            history.y_flat, history.design_flat, mask=history.mask_flat
        )


class RLBOEDQNetwork(_RLBOEDCritic):
    """Continuous critic with a configurable history encoder.

    Args:
        event_shape_y: Shape of one outcome.
        event_shape_design: Shape of one design.
        design_low: Lower references for design normalization, either a scalar
            or one entry per design coordinate.
        design_high: Upper references for design normalization, shaped like
            ``design_low``.
        outcome_low: Lower outcome bounds, a scalar or one per coordinate.
        outcome_high: Upper outcome bounds, shaped like ``outcome_low``.
        dim_h: Hidden layer widths; an integer specifies one layer.
        dim_emb: History-summary dimension for the default encoder.
        encoder: Optional history encoder over flattened events. Defaults to
            ``DeepSetEncoder``.
        head: Optional readout of concatenated history encoding and design.
    """

    def __init__(
        self,
        *,
        event_shape_y,
        event_shape_design,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h=(128, 128),
        dim_emb: int = 64,
        encoder: HistoryEncoder | None = None,
        head: nn.Module | None = None,
    ):
        """Initializes the continuous-action critic."""
        super().__init__(
            event_shape_y=event_shape_y,
            event_shape_design=event_shape_design,
            design_low=design_low,
            design_high=design_high,
            outcome_low=outcome_low,
            outcome_high=outcome_high,
            dim_h=dim_h,
            dim_emb=dim_emb,
            encoder=encoder,
        )
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.emitter_net = head
        if self.emitter_net is None:
            self.emitter_net = MLP(
                [self.dim_emb + self.dim_design] + hidden_widths + [1],
                activation=nn.ReLU,
            )
            _initialize_linear_layers(self.emitter_net)

    def forward(
        self,
        y_history,
        design_history,
        design: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Estimates Q from an encoded history summary and candidate design.

        Args:
            y_history: Outcomes shaped ``[*batch_shape, T, *event_shape_y]``.
            design_history: Physical designs shaped
                ``[*batch_shape, T, *event_shape_design]``.
            design: Candidate physical designs shaped
                ``[*batch_shape, *event_shape_design]``.
            mask: Optional boolean validity mask shaped ``[*batch_shape, T]``.

        Returns:
            Tensor: Q-values with shape ``[*batch_shape, 1]``.
        """
        history = self._prepare(y_history, design_history, mask)
        design = torch.as_tensor(design, device=history.design_flat.device)
        if design.shape != (*history.batch_shape, *self.event_shape_design):
            raise ValueError(
                "design must have shape [*batch_shape, *event_shape_design]."
            )
        design = self._normalize_designs(
            design.reshape(-1, self.dim_design).to(history.design_flat.dtype),
            -1.0,
            1.0,
        )
        q = self.emitter_net(torch.cat([self._encode(history), design], dim=-1))
        return q.reshape(*history.batch_shape, 1)


class RLBOEDDiscreteQNetwork(_RLBOEDCritic):
    """Configurable history critic returning Q-values for categorical designs.

    Args:
        event_shape_y: Shape of one outcome.
        event_shape_design: Shape of one design.
        n_designs: Number of categorical designs.
        design_low: Lower bounds for design normalization, a scalar or one
            entry per design coordinate.
        design_high: Upper bounds for design normalization, shaped like
            ``design_low``.
        outcome_low: Lower outcome bounds, a scalar or one per coordinate.
        outcome_high: Upper outcome bounds, shaped like ``outcome_low``.
        dim_h: Hidden layer widths; an integer specifies one layer.
        dim_emb: History-summary dimension.
        encoder: Optional history encoder over flattened events. Defaults to
            ``DeepSetEncoder``.
        head: Optional readout from history encoding to one Q-value per design.
    """

    def __init__(
        self,
        *,
        event_shape_y,
        event_shape_design,
        n_designs: int,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h=(128, 128),
        dim_emb: int = 64,
        encoder: HistoryEncoder | None = None,
        head: nn.Module | None = None,
    ):
        """Initializes the discrete critic."""
        super().__init__(
            event_shape_y=event_shape_y,
            event_shape_design=event_shape_design,
            design_low=design_low,
            design_high=design_high,
            outcome_low=outcome_low,
            outcome_high=outcome_high,
            dim_h=dim_h,
            dim_emb=dim_emb,
            encoder=encoder,
        )
        self.n_designs = int(n_designs)
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.emitter_net = head
        if self.emitter_net is None:
            self.emitter_net = MLP(
                [self.dim_emb] + hidden_widths + [self.n_designs], activation=nn.ReLU
            )
            _initialize_linear_layers(self.emitter_net)

    def forward(
        self, y_history, design_history, *, mask: Tensor | None = None
    ) -> Tensor:
        """Estimates all design Q-values from an encoded history.

        Args:
            y_history: Outcomes shaped ``[*batch_shape, T, *event_shape_y]``.
            design_history: Physical designs shaped
                ``[*batch_shape, T, *event_shape_design]``.
            mask: Optional boolean validity mask shaped ``[*batch_shape, T]``.

        Returns:
            Q-values with shape ``[*batch_shape, n_designs]``.
        """
        history = self._prepare(y_history, design_history, mask)
        q = self.emitter_net(self._encode(history))
        return q.reshape(*history.batch_shape, self.n_designs)
