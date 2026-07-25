"""Neural architecture for ALINE.

The implementation follows Huang et al. (2025), while focusing on parameter
targets for Bayesian experimental design.
"""

from dataclasses import dataclass
from math import prod

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, MixtureSameFamily, Normal


@dataclass
class AlineOutput:
    """Outputs from a forward pass through ALINE."""

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
        design_low,
        design_high,
        embedding_dim: int = 32,
        feedforward_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
        n_components: int = 10,
        min_scale: float = 1e-4,
        outcome_center=0.0,
        outcome_scale=1.0,
    ) -> None:
        """Initializes the ALINE architecture."""
        super().__init__()
        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_theta = torch.Size(event_shape_theta)
        self.event_shape_design = torch.Size(event_shape_design)
        self.dim_y = prod(self.event_shape_y)
        self.dim_theta = prod(self.event_shape_theta)
        self.dim_design = prod(self.event_shape_design)
        self.min_scale = float(min_scale)

        if embedding_dim <= 0 or feedforward_dim <= 0:
            raise ValueError("Embedding and feedforward dimensions must be positive.")
        if n_heads <= 0 or embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads.")
        if n_layers <= 0 or n_components <= 0:
            raise ValueError("n_layers and n_components must be positive.")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive.")

        design_low = torch.as_tensor(design_low).broadcast_to(self.event_shape_design)
        design_high = torch.as_tensor(
            design_high, device=design_low.device, dtype=design_low.dtype
        ).broadcast_to(self.event_shape_design)
        if (
            not torch.all(torch.isfinite(design_low))
            or not torch.all(torch.isfinite(design_high))
            or torch.any(design_low >= design_high)
        ):
            raise ValueError("Each design lower bound must be below its upper bound.")

        outcome_center = torch.as_tensor(outcome_center).broadcast_to(
            self.event_shape_y
        )
        outcome_scale = torch.as_tensor(
            outcome_scale, device=outcome_center.device, dtype=outcome_center.dtype
        ).broadcast_to(self.event_shape_y)
        if (
            not torch.all(torch.isfinite(outcome_center))
            or not torch.all(torch.isfinite(outcome_scale))
            or torch.any(outcome_scale <= 0)
        ):
            raise ValueError("outcome_scale must be positive.")

        self.register_buffer("design_low", design_low.reshape(-1))
        self.register_buffer("design_high", design_high.reshape(-1))
        self.register_buffer("outcome_center", outcome_center.reshape(-1))
        self.register_buffer("outcome_scale", outcome_scale.reshape(-1))

        self.design_embedder = self._make_mlp(
            self.dim_design, feedforward_dim, embedding_dim
        )
        self.outcome_embedder = self._make_mlp(
            self.dim_y, feedforward_dim, embedding_dim
        )
        self.target_tokens = nn.Parameter(torch.randn(self.dim_theta, embedding_dim))

        layer = nn.TransformerEncoderLayer(
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
    def _make_mlp(dim_in: int, dim_hidden: int, dim_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def _normalize_designs(self, designs: Tensor) -> Tensor:
        if torch.Size(designs.shape[2:]) != self.event_shape_design:
            raise ValueError("Design tensors must end in event_shape_design.")
        designs = designs.reshape(*designs.shape[:2], self.dim_design)
        return (
            2.0 * (designs - self.design_low) / (self.design_high - self.design_low)
            - 1.0
        )

    def _normalize_outcomes(self, outcomes: Tensor) -> Tensor:
        if torch.Size(outcomes.shape[2:]) != self.event_shape_y:
            raise ValueError("Outcome tensors must end in event_shape_y.")
        outcomes = outcomes.reshape(*outcomes.shape[:2], self.dim_y)
        return (outcomes - self.outcome_center) / self.outcome_scale

    def _attention_mask(
        self,
        n_context: int,
        n_query: int,
        target_mask: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        n_total = n_context + n_query + self.dim_theta
        mask = torch.full((n_total, n_total), float("-inf"), device=device)
        query_start = n_context
        target_start = n_context + n_query

        if n_context:
            mask[:, :n_context] = 0.0

        query_indices = torch.arange(query_start, target_start, device=device)
        mask[query_indices, query_indices] = 0.0
        selected_targets = torch.where(target_mask)[0] + target_start
        mask[query_indices[:, None], selected_targets] = 0.0

        target_indices = torch.arange(target_start, n_total, device=device)
        mask[target_indices, target_indices] = 0.0

        return mask

    def forward(
        self,
        context_y: Tensor,
        context_designs: Tensor,
        candidates: Tensor,
        target_mask: Tensor,
        available_mask: Tensor | None = None,
    ) -> AlineOutput:
        """Scores candidate designs and estimates parameter marginals."""
        if context_y.shape[:2] != context_designs.shape[:2]:
            raise ValueError(
                "Context outcomes and designs must share batch/time shape."
            )
        if context_designs.shape[0] != candidates.shape[0]:
            raise ValueError("Context and candidate batch sizes must match.")
        if candidates.shape[1] == 0:
            raise ValueError("ALINE requires at least one candidate design.")

        target_mask = torch.as_tensor(
            target_mask, device=candidates.device, dtype=torch.bool
        )
        if target_mask.shape != (self.dim_theta,) or not torch.any(target_mask):
            raise ValueError(
                f"target_mask must be a nonempty bool tensor of shape "
                f"({self.dim_theta},)."
            )

        batch_size, n_context = context_designs.shape[:2]
        n_query = candidates.shape[1]
        context_embeddings = self.design_embedder(
            self._normalize_designs(context_designs)
        ) + self.outcome_embedder(self._normalize_outcomes(context_y))
        query_embeddings = self.design_embedder(self._normalize_designs(candidates))
        target_embeddings = self.target_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = torch.cat(
            [context_embeddings, query_embeddings, target_embeddings], dim=1
        )

        attention_mask = self._attention_mask(
            n_context, n_query, target_mask, device=embeddings.device
        )
        encoded = self.transformer(embeddings, mask=attention_mask)
        query_encoded = encoded[:, n_context : n_context + n_query]
        target_encoded = encoded[:, n_context + n_query :]

        acquisition_logits = self.acquisition_head(query_encoded).squeeze(-1)
        if available_mask is not None:
            available_mask = torch.as_tensor(
                available_mask, device=candidates.device, dtype=torch.bool
            )
            if available_mask.shape != acquisition_logits.shape:
                raise ValueError("available_mask must match the candidate batch shape.")
            acquisition_logits = acquisition_logits.masked_fill(
                ~available_mask, float("-inf")
            )

        raw_components = torch.stack(
            [head(target_encoded) for head in self.inference_heads], dim=-2
        )
        mixture_means = raw_components[..., 0]
        mixture_scales = F.softplus(raw_components[..., 1]) + self.min_scale
        mixture_logits = raw_components[..., 2]
        return AlineOutput(
            acquisition_logits=acquisition_logits,
            mixture_logits=mixture_logits,
            mixture_means=mixture_means,
            mixture_scales=mixture_scales,
        )
