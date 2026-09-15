"""Policy and value networks used by RL-BOED."""

import torch
from torch import Tensor, nn

from testbed.design_nets.encoder import DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale
from testbed.envs.history import History


class RLBOEDQNetwork(nn.Module):
    """Continuous critic with a configurable history encoder.

    Args:
        dim_y: Flattened outcome dimension.
        dim_design: Flattened design dimension.
        design_low: Per-coordinate lower references for design normalization.
        design_high: Per-coordinate upper references for design normalization.
        outcome_low: Per-coordinate lower outcome bounds.
        outcome_high: Per-coordinate upper outcome bounds.
        dim_h: Hidden layer widths; an integer specifies one layer.
        dim_emb: History-summary dimension for the default encoder.
        encoder: Optional history encoder. Defaults to ``DeepSetEncoder``.
        head: Optional readout of concatenated history encoding and design.
    """

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h=(128, 128),
        dim_emb: int = 64,
        encoder: HistoryEncoder | None = None,
        head: nn.Module | None = None,
    ):
        """Initializes the continuous-action critic.

        Args:
            dim_y: Flattened outcome dimension.
            dim_design: Flattened design dimension.
            design_low: Per-coordinate lower references for design normalization.
            design_high: Per-coordinate upper references for design normalization.
            outcome_low: Per-coordinate lower outcome bounds.
            outcome_high: Per-coordinate upper outcome bounds.
            dim_h: Hidden layer widths; an integer specifies one layer.
            dim_emb: History-summary dimension for the default encoder.
            encoder: Optional history encoder. Defaults to ``DeepSetEncoder``.
            head: Optional readout of concatenated history encoding and design.
        """
        super().__init__()
        self.dim_y = dim_y
        self.dim_design = dim_design
        self.encoder = encoder or DeepSetEncoder(
            (self.dim_y,), (self.dim_design,), dim_h=dim_h, dim_emb=dim_emb
        )
        self.dim_emb = self.encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.emitter_net = head
        if self.emitter_net is None:
            self.emitter_net = MLP(
                [self.dim_emb + dim_design] + hidden_widths + [1],
                activation=nn.ReLU,
            )
            _initialize_linear_layers(self.emitter_net)
        self.register_buffer("design_low", torch.as_tensor(design_low))
        self.register_buffer("design_high", torch.as_tensor(design_high))
        self.register_buffer("outcome_low", torch.as_tensor(outcome_low))
        self.register_buffer("outcome_high", torch.as_tensor(outcome_high))

    def forward(self, observation: History, design: Tensor) -> Tensor:
        """Estimates Q from an encoded history summary and candidate design.

        Args:
            observation: History containing designs, outcomes, and an optional
                padding mask.
            design: Candidate physical designs with shape
                ``[batch, *event_shape_design]``.

        Returns:
            Tensor: Q-values with shape ``[batch, 1]``.
        """
        hist_designs = observation.designs
        outcomes = observation.outcomes
        mask = observation.mask
        batch_size = hist_designs.shape[0]
        history_length = hist_designs.shape[1]
        hist_designs = hist_designs.reshape(batch_size, history_length, self.dim_design)
        outcomes = outcomes.reshape(batch_size, history_length, self.dim_y)
        design = design.reshape(batch_size, self.dim_design)
        hist_designs = affine_rescale(
            hist_designs,
            self.design_low,
            self.design_high,
            0.0,
            1.0,
        )
        outcomes = affine_rescale(
            outcomes,
            self.outcome_low,
            self.outcome_high,
            0.0,
            1.0,
        )
        design = affine_rescale(
            design,
            self.design_low,
            self.design_high,
            -1.0,
            1.0,
        )

        history_summary = self.encoder(outcomes, hist_designs, mask=mask)

        return self.emitter_net(torch.cat([history_summary, design], dim=-1))


class RLBOEDDiscreteQNetwork(nn.Module):
    """Configurable history critic returning Q-values for categorical designs.

    Attributes:
        dim_y: Flattened outcome dimension.
        dim_design: Flattened design dimension.
        n_designs: Number of categorical designs.
        dim_emb: History-summary dimension.
        design_low: Per-coordinate lower bounds for design normalization.
        design_high: Per-coordinate upper bounds for design normalization.
        outcome_low: Lower bound for outcome normalization.
        outcome_high: Upper bound for outcome normalization.
        encoder: Module that summarizes the masked experiment history.
        emitter_net: Module that maps a history embedding to Q-values.
    """

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
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
        """Initializes the discrete critic.

        Args:
            dim_y: Flattened outcome dimension.
            dim_design: Flattened design dimension.
            n_designs: Number of categorical designs.
            design_low: Per-coordinate lower bounds for design normalization.
            design_high: Per-coordinate upper bounds for design normalization.
            outcome_low: Lower bound for outcome normalization.
            outcome_high: Upper bound for outcome normalization.
            dim_h: Hidden layer widths; an integer specifies one layer.
            dim_emb: History-summary dimension.
            encoder: Optional history encoder. Defaults to ``DeepSetEncoder``.
            head: Optional readout from history encoding to one Q-value per design.
        """
        super().__init__()
        self.dim_y = int(dim_y)
        self.dim_design = int(dim_design)
        self.n_designs = int(n_designs)
        self.encoder = encoder or DeepSetEncoder(
            (self.dim_y,), (self.dim_design,), dim_h=dim_h, dim_emb=dim_emb
        )
        self.dim_emb = self.encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.emitter_net = head
        if self.emitter_net is None:
            self.emitter_net = MLP(
                [self.dim_emb] + hidden_widths + [self.n_designs], activation=nn.ReLU
            )
            _initialize_linear_layers(self.emitter_net)
        self.register_buffer("design_low", torch.as_tensor(design_low))
        self.register_buffer("design_high", torch.as_tensor(design_high))
        self.register_buffer("outcome_low", torch.as_tensor(outcome_low))
        self.register_buffer("outcome_high", torch.as_tensor(outcome_high))

    def forward(self, observation: History) -> Tensor:
        """Estimates all design Q-values from an encoded history.

        Args:
            observation: History containing designs, outcomes, and an optional
                padding mask.

        Returns:
            Q-values with shape ``[batch, n_designs]``.
        """
        designs = observation.designs
        outcomes = observation.outcomes
        batch_size, history_length = designs.shape[:2]
        designs = designs.reshape(batch_size, history_length, self.dim_design)
        outcomes = outcomes.reshape(batch_size, history_length, self.dim_y)
        width = self.design_high - self.design_low
        scaled_designs = (designs - self.design_low) / width
        designs = torch.where(
            width == 0, torch.zeros_like(scaled_designs), scaled_designs
        )
        outcomes = affine_rescale(
            outcomes, self.outcome_low, self.outcome_high, 0.0, 1.0
        )
        return self.emitter_net(self.encoder(outcomes, designs, mask=observation.mask))
