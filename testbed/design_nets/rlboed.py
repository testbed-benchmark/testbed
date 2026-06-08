"""Policy and value networks used by RL-BOED."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from testbed.design_nets.modules import MLP
from testbed.design_nets.stochastic import GaussianPolicyActor


def affine_rescale(
    value: Tensor,
    source_low,
    source_high,
    target_low,
    target_high,
) -> Tensor:
    """Affinely rescales a tensor between bounded coordinate systems."""
    source_low = torch.as_tensor(source_low, device=value.device, dtype=value.dtype)
    source_high = torch.as_tensor(source_high, device=value.device, dtype=value.dtype)
    target_low = torch.as_tensor(target_low, device=value.device, dtype=value.dtype)
    target_high = torch.as_tensor(target_high, device=value.device, dtype=value.dtype)
    return target_low + (value - source_low) * (
        (target_high - target_low) / (source_high - source_low)
    )


class RLBOEDGaussianActor(GaussianPolicyActor):
    """RL-BOED DeepSets architecture with Gaussian policy output heads."""

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        dim_h: int = 128,
        dim_emb: int = 64,
        init_std: float = math.sqrt(1.0 / 3.0),
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
    ):
        """Initializes the history encoder and stochastic emitter."""
        super().__init__()
        self.dim_emb = dim_emb
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder_net = MLP(
            [dim_y + dim_design, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.emitter_net = MLP(
            [dim_emb, dim_h, dim_h],
            activation=nn.ReLU,
        )
        self.mean_head = nn.Linear(dim_h, dim_design)
        self.log_std_head = nn.Linear(dim_h, dim_design)

        nn.init.constant_(self.log_std_head.bias, math.log(init_std))

    def forward(
        self,
        y_history: Tensor,
        design_history: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Maps padded histories to Gaussian mean and log standard deviation."""
        batch_size, history_length, _ = design_history.shape

        if history_length == 0:
            enc = design_history.new_zeros(batch_size, self.dim_emb)
        else:
            enc_input = torch.cat([y_history, design_history], dim=-1)
            encoded_history = self.encoder_net(enc_input)
            if mask is not None:
                encoded_history = encoded_history * mask.to(
                    dtype=encoded_history.dtype
                ).unsqueeze(-1)
            enc = encoded_history.sum(dim=1)

        hidden = self.emitter_net(enc)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(
            min=self.log_std_min,
            max=self.log_std_max,
        )
        return mean, log_std


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
            -1.0,
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
