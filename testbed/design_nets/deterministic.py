"""Deterministic policy networks."""

from math import prod

import torch
from torch import nn

from testbed.design_nets.base_design_net import DesignNet
from testbed.design_nets.encoder import AttentiveEncoder, DeepSetEncoder, HistoryEncoder
from testbed.design_nets.modules import MLP


class DeterministicPolicyNet(DesignNet):
    """Arbitrary encoder followed by an MLP readout layer.

    The MLP readout produces a flat output which is reshaped to event_shape_out.

    Args:
    encoder: HistoryEncoder object
    event_shape_out: tuple or torch.Size of shape for outputs of the network
    decoder_layers: number of hidden layers in decoder
    dim_h: width of hidden layers, or a list of per-layer widths (overrides
        decoder_layers); defaults to dim_emb from the encoder
    activation: activation class (nn.ReLU, nn.GELU, ...) for the decoder
    """

    def __init__(
        self,
        encoder: HistoryEncoder,
        event_shape_out,
        decoder_layers=1,
        dim_h=None,
        activation=nn.GELU,
    ):
        """Initializes."""
        super().__init__()

        self.encoder = encoder

        self.event_shape_out = torch.Size(event_shape_out)

        dim_h = dim_h or encoder.dim_emb
        hidden_widths = (
            decoder_layers * [dim_h] if isinstance(dim_h, int) else list(dim_h)
        )

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
        encoder_dim_h=256,
        dim_emb=128,
        num_heads=4,
        num_blocks=3,
        decoder_layers=1,
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
            decoder_layers=decoder_layers,
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
        encoder_dim_h=128,
        encoder_layers=2,
        dim_emb=64,
        decoder_layers=1,
        decoder_dim_h=None,
        decoder_activation=nn.GELU,
    ):
        """Initializes."""
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            hidden_layers=encoder_layers,
            dim_emb=dim_emb,
        )
        super().__init__(
            encoder,
            event_shape_out,
            decoder_layers=decoder_layers,
            dim_h=decoder_dim_h,
            activation=decoder_activation,
        )
