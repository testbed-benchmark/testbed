"""Neural-network building blocks."""

from math import sqrt

import torch
import torch.nn as nn
from torch import Tensor


def _initialize_linear_layers(module: nn.Module) -> None:
    """Matches the Xavier/zero-bias initialization used by Garage."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def _initialize_ppo_linear_layers(
    module: nn.Module,
    *,
    output_scales: dict[nn.Linear, float] | None = None,
) -> None:
    """Matches the orthogonal initialization used by CleanRL PPO."""
    output_scales = output_scales or {}
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, output_scales.get(layer, sqrt(2.0)))
            nn.init.zeros_(layer.bias)


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


class MLP(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, dims, activation=nn.ReLU, norm=None, dropout=0.0, activate_last=False
    ):
        """Implements a basic MLP.

        dims: list of layer dimensions, e.g., [3, 256, 256, 16]
        activation: activation class (nn.ReLU, nn.GELU, ...)
        norm: normalization class (nn.LayerNorm, nn.BatchNorm1d) or None
        dropout: dropout probability
        activate_last: whether to apply activation after the final layer
        """
        super().__init__()

        layers = []
        n_layers = len(dims) - 1

        for i in range(n_layers):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = i == n_layers - 1
            if not is_last or activate_last:
                if norm is not None:
                    layers.append(norm(out_dim))

                if activation is not None:
                    layers.append(activation())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Applies the MLP."""
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Implements a standard transformer-style encoder block with pre-norms.

    x = x + MHA(LN(x))
    x = x + MLP(LN(x)).

    Heuristics:
    - dim / heads = 32
    """

    def __init__(self, dim, num_heads, batch_first=True, expand=4, dropout=0.0):
        """Initializes attention and feed-forward layers."""
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=batch_first, dropout=dropout
        )
        self.mlp = MLP([dim, expand * dim, dim], activation=nn.GELU)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Applies the encoder block."""
        # x: [b, T, dim]
        # out: [b, T, dim]

        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.dropout(attn_out)

        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))

        return x


class AttentionPool(nn.Module):
    """Implements a learnable, attention-based pooling mechanism."""

    def __init__(self, dim, num_heads, query_init_scale=0.02):
        """Initializes pooling parameters."""
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * query_init_scale)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """Pools a sequence into one embedding."""
        # x: [b, T, dim]
        q = self.query.expand(x.size(0), -1, -1)  # [b, 1, dim]
        out, _ = self.attn(q, self.ln(x), self.ln(x))
        return out.squeeze(1)  # [b, dim]
