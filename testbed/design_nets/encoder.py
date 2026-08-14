"""History encoding backbones used to build various nets."""

from abc import ABC, abstractmethod
from math import prod

import torch
from torch import nn

from testbed.design_nets.history import PreparedHistory, prepare_history
from testbed.design_nets.modules import MLP, TransformerEncoderBlock


class HistoryEncoder(nn.Module, ABC):
    """Encodes an experiment history ``{(x_i, y_i)}`` into a fixed-size vector.

    Shape:
        ``y_history``: ``[b, T, *event_shape_y]``
        ``design_history``: ``[b, T, *event_shape_design]``
        ``mask``: ``[b, T]`` or None
        ``encoding``: ``[b, dim_emb]``

    Subclasses implement ``encode_tokens``.
    T=0 is passed through so each architecture supplies its
    own "null history" representation.

    Concatenates (x, y) before passing to the encoder.
    Flattens x, y if they have multi-dimensional event shapes.
    """

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        dim_emb,
    ):
        """Initialize the encoder.

        Args:
            event_shape_y (tuple | torch.Size): Shape of the observation event.
            event_shape_design (tuple | torch.Size): Shape of the design event.
            dim_emb (int): Embedding dimension.
        """
        super().__init__()

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_design = torch.Size(event_shape_design)

        self.dim_y = prod(self.event_shape_y)
        self.dim_design = prod(self.event_shape_design)

        self.dim_in = self.dim_y + self.dim_design
        self.dim_emb = int(dim_emb)

        # Embedding for an empty or fully masked history; initialized to zero.
        self.null_enc = nn.Parameter(torch.zeros(self.dim_emb))

    @property
    def device(self) -> torch.device:
        """Returns the device of the HistoryEncoder."""
        return next(self.parameters()).device

    @abstractmethod
    def encode_tokens(self, tokens, mask):
        """[b, T, dim_in] (T may be 0) -> [b, dim_emb]."""
        ...

    def empty_encoding(self, b):
        """Encoding for the empty history (T=0) case."""
        return self.null_enc.unsqueeze(0).expand(b, -1)  # [b, dim_emb]

    def forward(self, y_history, design_history, mask=None):
        """Runs the forward pass of the encoder."""
        prepared = self._prepare_history(
            y_history,
            design_history,
            event_shape_y=self.event_shape_y,
            event_shape_design=self.event_shape_design,
            mask=mask,
        )
        return self._encode_prepared(prepared)

    def encode_flat(self, y_history, design_history, mask=None):
        """Encode histories already flattened to ``[batch, T, dim_event]``.

        Policy actors use this method after shared policy preparation. External
        callers should use :meth:`forward`, which accepts arbitrary event and
        batch shapes.
        """
        prepared = self._prepare_history(
            y_history,
            design_history,
            event_shape_y=(self.dim_y,),
            event_shape_design=(self.dim_design,),
            mask=mask,
        )
        return self._encode_prepared(prepared)

    def _prepare_history(
        self,
        y_history,
        design_history,
        *,
        event_shape_y,
        event_shape_design,
        mask,
    ) -> PreparedHistory:
        """Validates and flattens a history in this encoder's dtype and device."""
        reference = self.null_enc
        return prepare_history(
            y_history,
            design_history,
            event_shape_y=event_shape_y,
            event_shape_design=event_shape_design,
            mask=mask,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _encode_prepared(self, history: PreparedHistory):
        """Encode validated, flattened histories and restore ``batch_shape``."""
        tokens = torch.cat(
            [
                history.y_flat,
                history.design_flat,
            ],
            dim=-1,
        )  # [b, T, dim_in]
        encoding = self.encode_tokens(tokens, history.mask_flat)
        return encoding.reshape(*history.batch_shape, self.dim_emb)


class AttentiveEncoder(HistoryEncoder):
    """Transformer + CLS-token pooling."""

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        dim_h=(64,),
        dim_emb=64,
        num_heads=4,
        num_blocks=2,
        dropout: float = 0.0,
        use_time_embedding: bool = True,
        max_T: int = 128,
        use_type_embedding: bool = False,
    ):
        """Initializes."""
        super().__init__(
            event_shape_y,
            event_shape_design,
            dim_emb,
        )

        # Learnable query, used as the readout
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_emb))

        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        self.input_encoder = MLP(
            [self.dim_in] + hidden_widths + [dim_emb], activation=nn.GELU
        )

        # Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(dim_emb, num_heads, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        # Optional time embeddings
        self.use_time_embedding = use_time_embedding
        if self.use_time_embedding:
            self.time_emb = nn.Embedding(max_T + 1, dim_emb)
            self.max_T = max_T

        # Optional embeddings for history/cls types
        self.use_type_embedding = use_type_embedding
        if self.use_type_embedding:
            self.type_emb = nn.Embedding(2, dim_emb)

        # Normalization layers
        self.enc_norm = nn.LayerNorm(dim_emb)
        self.enc_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim_emb)

    def encode_tokens(self, tokens, mask=None):
        """Runs the forward pass of the encoder.

        Note that masking is not yet supported for the AttentiveEncoder.
        """
        if mask is not None:
            raise NotImplementedError(
                "Masking not yet supported with an attention-based encoder."
            )

        b, T, _ = tokens.shape

        # Input projection
        enc = self.input_encoder(tokens)  # [b, T, dim_emb]

        # Class token
        cls = self.cls_token.expand(b, 1, -1)  # [b, 1, dim_emb]

        # Add time embeddings
        if self.use_time_embedding:
            if T > 0:
                time_idx = (
                    torch.arange(T, device=tokens.device).unsqueeze(0).expand(b, T)
                )
                enc = enc + self.time_emb(time_idx)

            # Always use T for the class token
            cls_time_idx = torch.full((b, 1), T, dtype=torch.long, device=tokens.device)
            cls = cls + self.time_emb(cls_time_idx)

        # Add type embeddings
        if self.use_type_embedding:
            hist_idx = torch.zeros(b, T, dtype=torch.long, device=tokens.device)
            enc = enc + self.type_emb(hist_idx)

            cls_idx = torch.ones(b, 1, dtype=torch.long, device=tokens.device)
            cls = cls + self.type_emb(cls_idx)

        # Stack and normalize
        enc = torch.cat([enc, cls], dim=1)  # [b, T+1, dim_emb]
        enc = self.enc_norm(enc)
        enc = self.enc_dropout(enc)

        # Attention blocks
        for block in self.blocks:
            enc = block(enc)

        # cls readout
        cls_out = enc[:, -1, :]
        cls_out = self.output_norm(cls_out)
        return cls_out


class DeepSetEncoder(HistoryEncoder):
    """Per-step MLP + masked sum pooling (DeepSets style). Permutation invariant."""

    def __init__(
        self,
        event_shape_y,
        event_shape_design,
        dim_h=(128, 128),
        dim_emb=64,
    ):
        """dim_h: per-layer hidden widths as a list; an int is a single layer."""
        super().__init__(
            event_shape_y,
            event_shape_design,
            dim_emb,
        )

        hidden_widths = [dim_h] if isinstance(dim_h, int) else list(dim_h)
        layer_widths = [self.dim_in] + hidden_widths + [dim_emb]
        self.encoder_net = MLP(layer_widths, activation=nn.ReLU)

    def encode_tokens(self, tokens, mask=None):
        """Runs the forward pass of the encoder."""
        b, T, _ = tokens.shape

        if T == 0:
            return self.empty_encoding(b)

        enc = self.encoder_net(tokens)  # [b, T, dim_emb]

        if mask is not None:
            m = mask.to(enc.dtype).unsqueeze(-1)  # [b, T, 1]
            enc = enc * m
            pooled = enc.sum(dim=1)  # [b, dim_emb]

            # Need to handle the case that the tokens are fully masked
            # Here, we return self.null_enc instead of just zeroes
            empty_row = mask.any(dim=1).logical_not()  # [b] — True if all padding
            pooled = torch.where(
                empty_row.unsqueeze(-1), self.empty_encoding(b), pooled
            )

            return pooled

        return enc.sum(dim=1)  # [b, dim_emb]
