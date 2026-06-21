"""Policy and value networks used by RL-BOED."""

import torch
import torch.nn as nn
from torch import Tensor

from testbed.design_nets.modules import MLP, _initialize_linear_layers, affine_rescale
from testbed.envs.history import History


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
        """Initializes the history encoder and design-conditioned emitter."""
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

    def forward(self, observation: History, design: Tensor) -> Tensor:
        """Estimates Q from a permutation-invariant history summary and design."""
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

        encoded_history = self.encoder_net(torch.cat([outcomes, hist_designs], dim=-1))
        if mask is not None:
            encoded_history = encoded_history * mask.to(
                dtype=encoded_history.dtype
            ).unsqueeze(-1)
        history_summary = encoded_history.sum(dim=1)

        return self.emitter_net(torch.cat([history_summary, design], dim=-1))
