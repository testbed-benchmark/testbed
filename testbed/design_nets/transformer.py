"""Attention-based DAD policy networks."""

import torch
import torch.nn as nn

from testbed.design_nets import DesignNet
from testbed.design_nets.modules import MLP, AttentionPool, TransformerEncoderBlock


class AttentiveDAD(DesignNet):
    """Transformer policy with mean pooling.

    Possible improvements:
    [1] Go beyond mean-pooling
    [2] Different embedders for y/design.
    """

    def __init__(
        self, dim_y=1, dim_design=2, dim_h=256, dim_emb=128, num_heads=4, num_blocks=3
    ):
        """Initializes network layers."""
        super().__init__()

        # Could possibly try out random Fourier features for our encoder
        self.input_encoder = MLP(
            [dim_y + dim_design, dim_h, dim_emb], activation=nn.GELU
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(dim_emb, num_heads) for _ in range(num_blocks)]
        )

        self.decoder = MLP([dim_emb, dim_emb, dim_design], activation=nn.GELU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

    def forward(self, y_history, design_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        design_history: [b, T, dim_design].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = self.null_enc[None, :].expand(b, -1)
        else:
            # [1] Encode (y, design) inputs via an MLP
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.input_encoder(enc_input)  # [B, T, dim_emb]

            # [2] Attention blocks
            for block in self.transformer_blocks:
                enc = block(enc)

            # [3] Pooling
            # TODO try out other variants
            enc = enc.mean(dim=1)

        # [4] Readout MLP
        out = self.decoder(enc)  # [b, dim_design]
        return out


class AttentiveDAD_v2(DesignNet):
    """Transformer policy with learned weighted pooling.

    Changes:
    - Uses a learned weighting aggregation mechanism.
    """

    def __init__(
        self, dim_y=1, dim_design=2, dim_h=256, dim_emb=128, num_heads=4, num_blocks=3
    ):
        """Initializes network layers."""
        super().__init__()

        # Could possibly try out random Fourier features for our encoder
        self.input_encoder = MLP(
            [dim_y + dim_design, dim_h, dim_emb], activation=nn.GELU
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(dim_emb, num_heads) for _ in range(num_blocks)]
        )

        self.decoder = MLP([dim_emb, dim_emb, dim_design], activation=nn.GELU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

        # Initialize weighting for pooling step randomly
        self.pool_weights = nn.Parameter(torch.randn(dim_emb))

    def forward(self, y_history, design_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        design_history: [b, T, dim_design].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = self.null_enc[None, :].expand(b, -1)
        else:
            # [1] Encode (y, design) inputs via an MLP
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.input_encoder(enc_input)  # [B, T, dim_emb]

            # [2] Attention blocks
            for block in self.transformer_blocks:
                enc = block(enc)

            # [3] Pooling
            weights = torch.einsum("btd,d->bt", enc, self.pool_weights)  # [b, T]
            weights = torch.softmax(weights, dim=1)
            enc = torch.einsum("bt,btd->bd", weights, enc)

        # [4] Readout MLP
        out = self.decoder(enc)  # [b, dim_design]
        return out


class AttentiveDAD_v3(DesignNet):
    """Transformer policy with attention pooling.

    Changes:
    - Uses attention pooling to aggregate the embeddings prior to readout.
    """

    def __init__(
        self, dim_y=1, dim_design=2, dim_h=256, dim_emb=128, num_heads=4, num_blocks=3
    ):
        """Initializes network layers."""
        super().__init__()

        # Could possibly try out random Fourier features for our encoder
        self.input_encoder = MLP(
            [dim_y + dim_design, dim_h, dim_emb], activation=nn.GELU
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(dim_emb, num_heads) for _ in range(num_blocks)]
        )

        self.decoder = MLP([dim_emb, dim_emb, dim_design], activation=nn.GELU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

        self.pool = AttentionPool(dim_emb, num_heads)

    def forward(self, y_history, design_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        design_history: [b, T, dim_design].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = self.null_enc[None, :].expand(b, -1)
        else:
            # [1] Encode (y, design) inputs via an MLP
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.input_encoder(enc_input)  # [B, T, dim_emb]

            # [2] Attention blocks
            for block in self.transformer_blocks:
                enc = block(enc)

            # [3] Pooling
            enc = self.pool(enc)

        # [4] Readout MLP
        out = self.decoder(enc)  # [b, dim_design]
        return out


class AttentiveDAD_v4(DesignNet):
    """Transformer policy with a class token.

    Changes:
    - Uses attention pooling to aggregate the embeddings prior to readout.
    """

    def __init__(
        self, dim_y=1, dim_design=2, dim_h=256, dim_emb=128, num_heads=4, num_blocks=3
    ):
        """Initializes network layers."""
        super().__init__()

        # Could possibly try out random Fourier features for our encoder
        self.input_encoder = MLP(
            [dim_y + dim_design, dim_h, dim_emb], activation=nn.GELU
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(dim_emb, num_heads) for _ in range(num_blocks)]
        )

        self.decoder = MLP([dim_emb, dim_emb, dim_design], activation=nn.GELU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

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
            # [1] Encode (y, design) inputs via an MLP
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.input_encoder(enc_input)  # [B, T, dim_emb]

            # [2] cat class token
            cls = self.cls_token.expand(b, -1, -1)  # [b, 1, dim_emb]
            enc = torch.cat([cls, enc], dim=1)  # [b, T+1, dim_emb]

            # [3] Attention blocks
            for block in self.transformer_blocks:
                enc = block(enc)

            # [4] Readout cls embedding -- no pooling
            enc = enc[:, 0, :]

        # [5] Readout MLP
        out = self.decoder(enc)  # [b, dim_design]
        return out
