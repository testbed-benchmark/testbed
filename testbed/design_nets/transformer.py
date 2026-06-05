"""Transformer-based continuous DAD policy networks."""

import torch
import torch.nn as nn

from testbed.design_nets.base_design_net import DesignNet
from testbed.design_nets.modules import MLP, AttentionPool, TransformerEncoderBlock


class TransformerDADNet(DesignNet):
    """Continuous transformer policy with configurable history pooling.

    - ``pooling="mean"``: mean pooling over history tokens.
    - ``pooling="weighted"``: learned scalar weights over history tokens.
    - ``pooling="attention"``: learned attention pooling.
    - ``pooling="cls"``: prepend a learned class token and read it out.
    """

    _POOLING_OPTIONS = {"mean", "weighted", "attention", "cls"}

    def __init__(
        self,
        dim_y: int = 1,
        dim_design: int = 2,
        dim_h: int = 256,
        dim_emb: int = 128,
        num_heads: int = 4,
        num_blocks: int = 3,
        pooling: str = "mean",
    ):
        """Initializes network layers."""
        super().__init__()

        if pooling not in self._POOLING_OPTIONS:
            valid = ", ".join(sorted(self._POOLING_OPTIONS))
            raise ValueError(f"pooling must be one of: {valid}.")

        self.pooling = pooling
        self.input_encoder = MLP(
            [dim_y + dim_design, dim_h, dim_emb], activation=nn.GELU
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(dim_emb, num_heads) for _ in range(num_blocks)]
        )
        self.decoder = MLP([dim_emb, dim_emb, dim_design], activation=nn.GELU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable.
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

        if pooling == "weighted":
            self.pool_weights = nn.Parameter(torch.randn(dim_emb))
        elif pooling == "attention":
            self.pool = AttentionPool(dim_emb, num_heads)
        elif pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim_emb) * 0.02)

    def forward(self, y_history, design_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        design_history: [b, T, dim_design].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = self.null_enc[None, :].expand(b, -1)
        else:
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.input_encoder(enc_input)  # [b, T, dim_emb]

            if self.pooling == "cls":
                cls = self.cls_token.expand(b, -1, -1)  # [b, 1, dim_emb]
                enc = torch.cat([cls, enc], dim=1)  # [b, T+1, dim_emb]

            for block in self.transformer_blocks:
                enc = block(enc)

            enc = self._pool(enc)

        out = self.decoder(enc)  # [b, dim_design]
        return out

    def _pool(self, enc):
        """Pools encoded history tokens into one embedding."""
        if self.pooling == "mean":
            return enc.mean(dim=1)

        if self.pooling == "weighted":
            weights = torch.einsum("btd,d->bt", enc, self.pool_weights)
            weights = torch.softmax(weights, dim=1)
            return torch.einsum("bt,btd->bd", weights, enc)

        if self.pooling == "attention":
            return self.pool(enc)

        if self.pooling == "cls":
            return enc[:, 0, :]

        raise RuntimeError(f"Unknown pooling mode: {self.pooling}.")
