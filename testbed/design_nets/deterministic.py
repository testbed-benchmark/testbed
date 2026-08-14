"""Deterministic policy networks."""

from math import prod

import torch
from torch import nn

from testbed.design_nets.base import DesignNet
from testbed.design_nets.encoder import AttentiveEncoder, DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP


class DeterministicPolicyNet(DesignNet):
    """Arbitrary encoder followed by an MLP readout layer.

    The MLP readout produces a flat output which is reshaped to event_shape_out.

    Args:
        encoder: HistoryEncoder object.
        event_shape_out: Tuple or torch.Size describing network output shapes.
        dim_h: Per-layer hidden widths, e.g. [128, 16]. An int specifies a
            single hidden layer. Defaults to the encoder's dim_emb.
        activation: Activation class for the decoder, e.g. nn.ReLU or nn.GELU.
    """

    def __init__(
        self,
        encoder: HistoryEncoder,
        event_shape_out,
        dim_h=None,
        activation=nn.GELU,
    ):
        """Initializes."""
        super().__init__()

        self.encoder = encoder

        self.event_shape_out = torch.Size(event_shape_out)

        dim_h = dim_h or encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)

        layer_widths = [encoder.dim_emb] + hidden_widths + [prod(self.event_shape_out)]
        self.decoder = MLP(layer_widths, activation=activation)

    def forward(self, y_history, design_history, mask=None):
        """Forward pass of the policy net. Combines the encoder with the decoder, which is assumed to be an MLP."""
        enc = self.encoder(y_history, design_history, mask=mask)  # [b, dim_emb]
        out = self.decoder(enc)  # [b, prod(out)]
        return out.reshape(*enc.shape[:-1], *self.event_shape_out)


class AttentivePolicyNet(DeterministicPolicyNet):
    """Preset: AttentiveEncoder (transformer + CLS pooling) + MLP readout.

    This is a TNP-style architecture and is closely related to (but not exactly the same) as the
    architecture used in the ActionBED (Rossa et al., 2026) paper.
    """

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_out,
        encoder_dim_h=(64,),
        dim_emb=64,
        num_heads=4,
        num_blocks=2,
        decoder_dim_h=None,
        decoder_activation=nn.GELU,
    ):
        """Initializes."""
        encoder = AttentiveEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            dim_emb=dim_emb,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

        super().__init__(
            encoder,
            event_shape_out,
            dim_h=decoder_dim_h,
            activation=decoder_activation,
        )


class DeepSetPolicyNet(DeterministicPolicyNet):
    """Preset: DeepSetEncoder (per-step MLP + sum pooling) + MLP readout.

    This architecture is closely related to that appearing in the original DAD (Foster et al., 2021) paper.
    The main difference is that the null encoding is learnable here, but was fixed to zero in the original paper.
    """

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_out,
        encoder_dim_h=(128, 128),
        dim_emb=64,
        decoder_dim_h=None,
        decoder_activation=nn.GELU,
    ):
        """Initializes."""
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            dim_emb=dim_emb,
        )
        super().__init__(
            encoder,
            event_shape_out,
            dim_h=decoder_dim_h,
            activation=decoder_activation,
        )


class MLPPolicyNet(DesignNet):
    """Preset: An MLP-based policy with no explicit history aggregation.

    Works by flattening all inputs. Note that this won't allow for variable-length sequences.
    Thus, it's mostly useful for action policies which only act at a fixed time point.
    """

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        event_shape_out,
        dims,
        activation=nn.ReLU,
    ):
        """Initializes."""
        super().__init__()

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_design = torch.Size(event_shape_design)

        self.dim_y = prod(self.event_shape_y)
        self.dim_design = prod(self.event_shape_design)

        self.event_shape_out = torch.Size(event_shape_out)

        self.mlp = MLP(dims, activation=activation)

    def forward(self, y_history, design_history, mask=None):
        """Forward pass of the policy net."""
        if mask is not None:
            raise NotImplementedError("Masks are not supported by MLPPolicyNet.")

        b, T = y_history.shape[0], y_history.shape[1]
        tokens = torch.cat(
            [
                y_history.reshape(b, T, self.dim_y),
                design_history.reshape(b, T, self.dim_design),
            ],
            dim=-1,
        )  # [b, T, dim_in]

        tokens = tokens.reshape(b, -1)  # [b, T * dim_in]

        out = self.mlp(tokens)
        out = out.reshape(b, *self.event_shape_out)
        return out
