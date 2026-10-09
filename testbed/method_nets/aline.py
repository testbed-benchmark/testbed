"""Neural architecture for ALINE.

The implementation follows Huang et al. (2025), specialized to full-parameter
Bayesian experimental design.
"""

from dataclasses import dataclass
from math import prod

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, MixtureSameFamily, Normal

from testbed.design_nets.history import prepare_history


class _AlineEncoderLayer(nn.TransformerEncoderLayer):
    """A transformer encoder layer that also works before any observations exist.

    ALINE's target tokens represent the parameters we want to estimate. They
    may attend only to observations, so before the first observation their
    attention mask blocks every position (the entire row is ``-inf``).
    PyTorch's default ``TransformerEncoderLayer.forward()`` can select an
    optimized path during evaluation with gradients disabled that produces
    NaNs for these fully masked rows.

    We override ``forward()`` to run the usual attention, residual additions,
    feedforward network, and normalization steps explicitly, bypassing that
    optimized encoder path. With ALINE's floating-point mask, the attention
    block then uses scaled dot-product attention (SDPA), which returns zero
    for fully masked rows. The residual connection preserves the target
    token's own representation even when there are no observations to read.

    This implementation assumes ``norm_first=False`` (the default): each
    normalization follows its residual addition.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = self.norm1(
            src
            + self._sa_block(src, src_mask, src_key_padding_mask, is_causal=is_causal)
        )
        return self.norm2(x + self._ff_block(x))


@dataclass
class AlineOutput:
    """Outputs from a forward pass through ALINE.

    Attributes:
        acquisition_logits: Candidate scores, shape ``[*batch_shape, N]``.
        mixture_logits: Component logits, shape
            ``[*batch_shape, dim_theta, n_components]``.
        mixture_means: Component means, shaped like ``mixture_logits``.
        mixture_scales: Component scales, shaped like ``mixture_logits``.
    """

    acquisition_logits: Tensor
    mixture_logits: Tensor
    mixture_means: Tensor
    mixture_scales: Tensor

    @property
    def posterior(self) -> MixtureSameFamily:
        """Returns the factorized marginal posterior approximation."""
        return MixtureSameFamily(
            Categorical(logits=self.mixture_logits),
            Normal(self.mixture_means, self.mixture_scales),
        )


class AlineModel(nn.Module):
    """Joint acquisition and amortized-inference network for ALINE."""

    def __init__(
        self,
        *,
        event_shape_y,
        event_shape_theta,
        event_shape_design,
        embedding_dim: int = 32,
        feedforward_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
        n_components: int = 10,
        min_scale: float = 1e-4,
        design_center=0.0,
        design_scale=1.0,
        outcome_center=0.0,
        outcome_scale=1.0,
        canonicalize_theta: bool = False,
    ) -> None:
        """Initializes the ALINE architecture.

        Args:
            event_shape_y: Shape of a single observation, excluding batch and
                time dimensions.
            event_shape_theta: Shape of a single parameter sample, excluding
                the batch dimension.
            event_shape_design: Shape of a single design, excluding batch and
                time or candidate dimensions.
            embedding_dim: Embedding dimension, divisible by ``n_heads``.
            feedforward_dim: Hidden dimension of the feedforward networks.
            n_heads: Number of attention heads per transformer layer.
            n_layers: Number of transformer encoder layers.
            dropout: Transformer dropout probability in [0, 1].
            n_components: Number of Gaussian mixture components per
                scalar parameter's marginal posterior.
            min_scale: Lower bound on Gaussian component standard
                deviations.
            design_center: Finite center for design normalization, broadcastable
                to ``event_shape_design``.
            design_scale: Finite positive scale for design normalization,
                broadcastable to ``event_shape_design``.
            outcome_center: Finite center for observation normalization,
                broadcastable to ``event_shape_y``.
            outcome_scale: Finite positive scale for observation normalization,
                broadcastable to ``event_shape_y``.
            canonicalize_theta: Whether to use a consistent ordering for
                exchangeable parameter objects, such as location-finding
                sources. Requires an object axis in ``event_shape_theta``,
                with ``len(event_shape_theta) >= 2``.
        """
        super().__init__()
        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_theta = torch.Size(event_shape_theta)
        self.event_shape_design = torch.Size(event_shape_design)
        self.dim_y = prod(self.event_shape_y)
        self.dim_theta = prod(self.event_shape_theta)
        self.dim_design = prod(self.event_shape_design)
        self.min_scale = float(min_scale)
        self.canonicalize_theta = bool(canonicalize_theta)
        if self.canonicalize_theta and len(self.event_shape_theta) < 2:
            raise ValueError(
                "canonicalize_theta requires an exchangeable object axis, i.e. "
                f"len(event_shape_theta) >= 2; got {tuple(self.event_shape_theta)}."
            )

        if embedding_dim <= 0 or feedforward_dim <= 0:
            raise ValueError("Embedding and feedforward dimensions must be positive.")
        if n_heads <= 0 or embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads.")
        if n_layers <= 0 or n_components <= 0:
            raise ValueError("n_layers and n_components must be positive.")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive.")

        design_center, design_scale = self._as_float_pair(design_center, design_scale)
        design_center = design_center.broadcast_to(self.event_shape_design)
        design_scale = design_scale.broadcast_to(self.event_shape_design)
        if (
            not torch.all(torch.isfinite(design_center))
            or not torch.all(torch.isfinite(design_scale))
            or torch.any(design_scale <= 0)
        ):
            raise ValueError("design_scale must be finite and positive.")
        self.register_buffer("design_center", design_center.reshape(-1).clone())
        self.register_buffer("design_scale", design_scale.reshape(-1).clone())

        outcome_center, outcome_scale = self._as_float_pair(
            outcome_center, outcome_scale
        )
        outcome_center = outcome_center.broadcast_to(self.event_shape_y)
        outcome_scale = outcome_scale.broadcast_to(self.event_shape_y)
        if (
            not torch.all(torch.isfinite(outcome_center))
            or not torch.all(torch.isfinite(outcome_scale))
            or torch.any(outcome_scale <= 0)
        ):
            raise ValueError("outcome_scale must be positive.")

        self.register_buffer("outcome_center", outcome_center.reshape(-1).clone())
        self.register_buffer("outcome_scale", outcome_scale.reshape(-1).clone())

        self.design_embedder = self._make_mlp(
            self.dim_design, feedforward_dim, embedding_dim
        )
        self.outcome_embedder = self._make_mlp(
            self.dim_y, feedforward_dim, embedding_dim
        )
        self.target_tokens = nn.Parameter(torch.randn(self.dim_theta, embedding_dim))

        layer = _AlineEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.acquisition_head = self._make_mlp(embedding_dim, feedforward_dim, 1)
        self.inference_heads = nn.ModuleList(
            self._make_mlp(embedding_dim, feedforward_dim, 3)
            for _ in range(n_components)
        )

    @staticmethod
    def _as_float_pair(first, second) -> tuple[Tensor, Tensor]:
        """Puts a (bound, bound) or (center, scale) pair on one float dtype.

        Casting either one to the other's dtype would truncate a fractional
        value against an integer partner -- ``(0, 0.5)`` bounds would collapse
        to ``(0, 0)``. Integers are legitimate input here, so promote the pair
        and fall back to the default dtype when neither side is floating point.
        """
        first = torch.as_tensor(first)
        second = torch.as_tensor(second, device=first.device)
        dtype = torch.promote_types(first.dtype, second.dtype)
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.get_default_dtype()
        return first.to(dtype=dtype), second.to(dtype=dtype)

    @staticmethod
    def _make_mlp(dim_in: int, dim_hidden: int, dim_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def _normalize_designs(self, designs: Tensor) -> Tensor:
        """Normalizes flattened designs shaped ``[b, n, dim_design]``."""
        return (designs - self.design_center) / self.design_scale

    def _normalize_outcomes(self, outcomes: Tensor) -> Tensor:
        """Normalizes flattened outcomes shaped ``[b, n, dim_y]``."""
        return (outcomes - self.outcome_center) / self.outcome_scale

    def _attention_mask(
        self, n_context: int, n_query: int, *, device: torch.device
    ) -> Tensor:
        n_total = n_context + n_query + self.dim_theta
        mask = torch.full((n_total, n_total), float("-inf"), device=device)
        query_start = n_context
        target_start = n_context + n_query

        if n_context:
            mask[:, :n_context] = 0.0

        mask[query_start:target_start, target_start:] = 0.0

        return mask

    def forward(
        self,
        y_history,
        design_history,
        candidates,
        *,
        mask: Tensor | None = None,
        available_mask: Tensor | None = None,
    ) -> AlineOutput:
        """Scores candidate designs and estimates parameter marginals.

        Args:
            y_history: Context outcomes shaped
                ``[*batch_shape, T, *event_shape_y]``.
            design_history: Context designs shaped
                ``[*batch_shape, T, *event_shape_design]``.
            candidates: Candidate designs shaped
                ``[*batch_shape, N, *event_shape_design]`` with ``N >= 1``.
            mask: Optional boolean validity mask for padded context positions,
                shaped ``[*batch_shape, T]``.
            available_mask: Optional boolean mask of selectable candidates,
                shaped ``[*batch_shape, N]``. Unavailable candidates receive
                ``-inf`` acquisition logits.

        Returns:
            AlineOutput: Outputs with leading dimensions ``batch_shape``.
        """
        history = prepare_history(
            y_history,
            design_history,
            event_shape_y=self.event_shape_y,
            event_shape_design=self.event_shape_design,
            mask=mask,
            device=self.design_center.device,
            dtype=self.design_center.dtype,
        )
        batch_shape = history.batch_shape
        batch_size, n_context = history.design_flat.shape[:2]
        candidates = torch.as_tensor(
            candidates, device=self.design_center.device, dtype=self.design_center.dtype
        )
        n_query = candidates.shape[len(batch_shape)] if candidates.ndim else 0
        if candidates.shape != (*batch_shape, n_query, *self.event_shape_design):
            raise ValueError(
                "candidates must have shape [*batch_shape, N, *event_shape_design] "
                "with the same batch_shape as the history."
            )
        if n_query == 0:
            raise ValueError("ALINE requires at least one candidate design.")
        candidates = candidates.reshape(batch_size, n_query, self.dim_design)

        context_embeddings = self.design_embedder(
            self._normalize_designs(history.design_flat)
        ) + self.outcome_embedder(self._normalize_outcomes(history.y_flat))
        query_embeddings = self.design_embedder(self._normalize_designs(candidates))
        target_embeddings = self.target_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat(
            [context_embeddings, query_embeddings, target_embeddings], dim=1
        )

        attention_mask = self._attention_mask(
            n_context, n_query, device=embeddings.device
        )
        padding_mask = None
        if history.mask_flat is not None:
            # Float, like attention_mask, so PyTorch merges the two without
            # a mismatched-mask-type warning.
            padding_mask = torch.zeros(
                batch_size, embeddings.shape[1], device=embeddings.device
            )
            padding_mask[:, :n_context] = padding_mask[:, :n_context].masked_fill(
                ~history.mask_flat, float("-inf")
            )
        encoded = self.transformer(
            embeddings, mask=attention_mask, src_key_padding_mask=padding_mask
        )
        query_encoded = encoded[:, n_context : n_context + n_query]
        target_encoded = encoded[:, n_context + n_query :]

        acquisition_logits = self.acquisition_head(query_encoded).squeeze(-1)
        acquisition_logits = acquisition_logits.reshape(*batch_shape, n_query)
        if available_mask is not None:
            available_mask = torch.as_tensor(available_mask, device=embeddings.device)
            if available_mask.dtype is not torch.bool:
                raise ValueError("available_mask must be a boolean tensor.")
            if available_mask.shape != acquisition_logits.shape:
                raise ValueError("available_mask must have shape [*batch_shape, N].")
            acquisition_logits = acquisition_logits.masked_fill(
                ~available_mask, float("-inf")
            )

        raw_components = torch.stack(
            [head(target_encoded) for head in self.inference_heads], dim=-2
        ).reshape(*batch_shape, self.dim_theta, len(self.inference_heads), 3)
        mixture_means = raw_components[..., 0]
        mixture_scales = F.softplus(raw_components[..., 1]) + self.min_scale
        mixture_logits = raw_components[..., 2]
        return AlineOutput(
            acquisition_logits=acquisition_logits,
            mixture_logits=mixture_logits,
            mixture_means=mixture_means,
            mixture_scales=mixture_scales,
        )
