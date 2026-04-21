from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Simple DAD (location finding, temporal discounting)
# Copied directly from: https://github.com/marcelhedman/stepdad/blob/master/src/stepdad/policy/dad.py#L73
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Maps a (design, outcome) pair to a fixed-dim embedding."""

    def __init__(self, design_dim: int, obs_dim: int, hidden_dim: int, encoding_dim: int) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        self.net = nn.Sequential(
            nn.Linear(design_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
        )

    def forward(self, xi: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            xi: [B, design_dim]
            y:  [B, obs_dim]
        Returns:
            [B, encoding_dim]
        """
        return self.net(torch.cat([xi, y], dim=-1))


class _Emitter(nn.Module):
    """Maps an aggregated encoding to the next design."""

    def __init__(self, encoding_dim: int, design_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, design_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Args:
            embedding: [B, encoding_dim]
        Returns:
            [B, design_dim]
        """
        return self.net(embedding)


class MarcelNet(nn.Module):
    """Permutation-invariant DAD policy via sum-aggregation.


    Args:
        design_dim:   dimensionality of each design.
        obs_dim:      dimensionality of each observation.
        hidden_dim:   hidden width of the encoder first layer (default 256).
        encoding_dim: width of the encoder output / emitter input (default 16).
    """

    def __init__(
        self,
        design_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        encoding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim
        self.encoder = _Encoder(design_dim, obs_dim, hidden_dim, encoding_dim)
        self.emitter = _Emitter(encoding_dim, design_dim)
        self.register_parameter(
            "empty_value", nn.Parameter(torch.zeros(encoding_dim))
        )

    # TODO clean up
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, designs: Tensor, outcomes: Tensor) -> Tensor:
        """Propose the next design given the history.

        Args:
            designs:  [B, t, design_dim]
            outcomes: [B, t, obs_dim]
        Returns:
            [B, design_dim]
        """
        B = designs.shape[0]
        t = designs.shape[1]

        if t == 0:
            # No history — use learned empty value broadcast over batch
            enc = self.empty_value.unsqueeze(0).expand(B, -1)
        else:
            enc = sum(
                self.encoder(designs[:, i, :], outcomes[:, i, :])
                for i in range(t)
            )

        return self.emitter(enc)