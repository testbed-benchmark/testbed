"""Policy and value networks used by RL-BOED."""

import torch
import torch.nn as nn
from torch import Tensor

from testbed.design_nets.modules import (
    MLP,
    _initialize_linear_layers,
    _initialize_ppo_linear_layers,
    affine_rescale,
)


class RLBOEDQNetwork(nn.Module):
    """DeepSets critic with masked sum pooling over experiment history."""

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h: int = 128,
        dim_emb: int = 64,
    ):
        """Initializes the history encoder and action-conditioned emitter."""
        super().__init__()
        self.dim_y = dim_y
        self.dim_design = dim_design
        self.dim_emb = dim_emb
        self.encoder_net = MLP(
            [dim_y + dim_design, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.emitter_net = MLP(
            [dim_emb + dim_design, dim_h, dim_h, 1],
            activation=nn.ReLU,
        )
        _initialize_linear_layers(self)
        self.register_buffer("design_low", torch.as_tensor(design_low))
        self.register_buffer("design_high", torch.as_tensor(design_high))
        self.register_buffer("outcome_low", torch.as_tensor(outcome_low))
        self.register_buffer("outcome_high", torch.as_tensor(outcome_high))

    def forward(self, observation: dict[str, Tensor], action: Tensor) -> Tensor:
        """Estimates Q from a permutation-invariant history summary and action."""
        designs = observation["designs"]
        outcomes = observation["outcomes"]
        mask = observation.get("mask")
        batch_size = designs.shape[0]
        history_length = designs.shape[1]
        designs = designs.reshape(batch_size, history_length, self.dim_design)
        outcomes = outcomes.reshape(batch_size, history_length, self.dim_y)
        action = action.reshape(batch_size, self.dim_design)
        designs = affine_rescale(
            designs,
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
        action = affine_rescale(
            action,
            self.design_low,
            self.design_high,
            -1.0,
            1.0,
        )

        if history_length == 0:
            history_summary = designs.new_zeros(batch_size, self.dim_emb)
        else:
            encoded_history = self.encoder_net(torch.cat([outcomes, designs], dim=-1))
            if mask is not None:
                encoded_history = encoded_history * mask.to(
                    dtype=encoded_history.dtype
                ).unsqueeze(-1)
            history_summary = encoded_history.sum(dim=1)

        return self.emitter_net(torch.cat([history_summary, action], dim=-1))


class RLBOEDValueNetwork(nn.Module):
    """DeepSets state-value network with masked sum pooling over history."""

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        design_low,
        design_high,
        outcome_low,
        outcome_high,
        dim_h: int = 128,
        dim_emb: int = 64,
    ):
        """Initializes the history encoder and value emitter."""
        super().__init__()
        self.dim_y = dim_y
        self.dim_design = dim_design
        self.dim_emb = dim_emb
        self.encoder_net = MLP(
            [dim_y + dim_design, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.emitter_net = MLP([dim_emb, dim_h, dim_h, 1], activation=nn.ReLU)
        _initialize_ppo_linear_layers(
            self,
            output_scales={self.emitter_net.net[-1]: 1.0},
        )
        self.register_buffer("design_low", torch.as_tensor(design_low))
        self.register_buffer("design_high", torch.as_tensor(design_high))
        self.register_buffer("outcome_low", torch.as_tensor(outcome_low))
        self.register_buffer("outcome_high", torch.as_tensor(outcome_high))

    def forward(self, observation: dict[str, Tensor]) -> Tensor:
        """Estimates V from a permutation-invariant history summary."""
        designs = observation["designs"]
        outcomes = observation["outcomes"]
        mask = observation.get("mask")
        batch_size = designs.shape[0]
        history_length = designs.shape[1]
        designs = designs.reshape(batch_size, history_length, self.dim_design)
        outcomes = outcomes.reshape(batch_size, history_length, self.dim_y)

        design_denom = torch.where(
            self.design_high > self.design_low,
            self.design_high - self.design_low,
            torch.ones_like(self.design_high),
        )
        designs = (designs - self.design_low) / design_denom
        outcomes = affine_rescale(
            outcomes,
            self.outcome_low,
            self.outcome_high,
            0.0,
            1.0,
        )

        if history_length == 0:
            history_summary = designs.new_zeros(batch_size, self.dim_emb)
        else:
            encoded_history = self.encoder_net(torch.cat([outcomes, designs], dim=-1))
            if mask is not None:
                encoded_history = encoded_history * mask.to(
                    dtype=encoded_history.dtype
                ).unsqueeze(-1)
            history_summary = encoded_history.sum(dim=1)

        return self.emitter_net(history_summary)
