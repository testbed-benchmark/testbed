"""Deterministic policy networks."""

from math import prod

import torch
from torch import Tensor, nn

from testbed.design_nets.base import DesignNet, DesignProposal
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
        """Initializes a deterministic policy.

        Args:
            encoder: History encoder that produces a ``dim_emb`` summary.
            event_shape_out: Shape of one design proposal.
            dim_h: Decoder hidden widths. An integer selects one layer; defaults
                to the encoder embedding width.
            activation: Activation class used by the decoder.
        """
        super().__init__()

        self.encoder = encoder

        self.event_shape_out = torch.Size(event_shape_out)

        dim_h = dim_h or encoder.dim_emb
        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)

        layer_widths = [encoder.dim_emb] + hidden_widths + [prod(self.event_shape_out)]
        self.decoder = MLP(layer_widths, activation=activation)

    def forward(self, y_history, design_history, mask=None):
        """Returns a deterministic raw design for each supplied history.

        Args:
            y_history: Outcome histories accepted by ``encoder``.
            design_history: Design histories accepted by ``encoder``.
            mask: Optional validity mask for padded history positions.

        Returns:
            Tensor: Raw designs with shape
            ``[*batch_shape, *event_shape_out]``.
        """
        enc = self.encoder(y_history, design_history, mask=mask)  # [b, dim_emb]
        out = self.decoder(enc)  # [b, prod(out)]
        return out.reshape(*enc.shape[:-1], *self.event_shape_out)

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ):
        """Returns the deterministic design in the common proposal format.

        Args:
            y_history: Outcome history accepted by :meth:`forward`.
            design_history: Design history accepted by :meth:`forward`.
            mask: Optional validity mask for padded history positions.
            reparameterize_policy: Ignored because this policy is deterministic.

        Returns:
            DesignProposal: A proposal without a log probability.
        """
        del reparameterize_policy
        raw_design = self(y_history, design_history, mask=mask)
        return DesignProposal(raw_design=raw_design)


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
        """Initializes the attentive policy preset.

        Args:
            event_shape_y: Shape of one outcome event.
            event_shape_design: Shape of one design-history event.
            event_shape_out: Shape of one proposed design.
            encoder_dim_h: Hidden widths in the token encoder.
            dim_emb: History embedding width.
            num_heads: Number of attention heads.
            num_blocks: Number of Transformer blocks.
            decoder_dim_h: Optional decoder hidden widths.
            decoder_activation: Decoder activation class.
        """
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
        encoder_activation=nn.ReLU,
        decoder_dim_h=None,
        decoder_activation=nn.GELU,
    ):
        """Initializes the DeepSets policy preset.

        Args:
            event_shape_y: Shape of one outcome event.
            event_shape_design: Shape of one design-history event.
            event_shape_out: Shape of one proposed design.
            encoder_dim_h: Hidden widths in the per-token encoder.
            dim_emb: History embedding width.
            encoder_activation: Encoder activation class.
            decoder_dim_h: Optional decoder hidden widths.
            decoder_activation: Decoder activation class.
        """
        encoder = DeepSetEncoder(
            event_shape_y,
            event_shape_design,
            dim_h=encoder_dim_h,
            dim_emb=dim_emb,
            activation=encoder_activation,
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
        """Initializes the fixed-length MLP policy.

        Args:
            event_shape_y: Shape of one outcome event.
            event_shape_design: Shape of one design-history event.
            event_shape_out: Shape of one proposed design.
            dims: Widths of the flattened-input MLP, including input and output.
            activation: Activation class used by the MLP.
        """
        super().__init__()

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_design = torch.Size(event_shape_design)

        self.dim_y = prod(self.event_shape_y)
        self.dim_design = prod(self.event_shape_design)

        self.event_shape_out = torch.Size(event_shape_out)

        self.mlp = MLP(dims, activation=activation)

    def forward(self, y_history, design_history, mask=None):
        """Returns a raw design from a fixed-length unmasked history.

        Args:
            y_history: Outcome history with leading batch and time dimensions.
            design_history: Design history with matching batch and time dimensions.
            mask: Unsupported padding mask; must be ``None``.

        Returns:
            Tensor: Raw designs with shape ``[batch, *event_shape_out]``.

        Raises:
            NotImplementedError: If ``mask`` is provided.
        """
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

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ):
        """Returns the deterministic design in the common proposal format.

        Args:
            y_history: Outcome history accepted by :meth:`forward`.
            design_history: Design history accepted by :meth:`forward`.
            mask: Optional validity mask; unsupported by this policy.
            reparameterize_policy: Ignored because this policy is deterministic.

        Returns:
            DesignProposal: A proposal without a log probability.
        """
        del reparameterize_policy
        raw_design = self(y_history, design_history, mask=mask)
        return DesignProposal(raw_design=raw_design)
