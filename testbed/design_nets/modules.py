"""Neural-network building blocks."""

import torch
from torch import Tensor, nn
from torch.distributions import constraints
from torch.distributions.transforms import Transform


def _initialize_linear_layers(module: nn.Module) -> None:
    """Matches the Xavier/zero-bias initialization used by Garage."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
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
        attn_out, _ = self.attn(h, h, h, need_weights=False)
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


class BlockTransform(Transform):
    r"""Apply transforms to contiguous blocks of the last dimension.

    Each block ``i`` reads ``raw_sizes[i]`` unconstrained coordinates, applies
    ``transforms[i]``, and writes ``theta_sizes[i]`` constrained coordinates.
    Blocks remain in order, and any leading dimensions are treated as batch
    dimensions.

    Input and output widths are dim_raw = sum(raw_sizes) dim_theta = sum(theta_sizes)

    The separate size lists are necessary because a transform may change width. For
    example, stick-breaking maps ``R^2`` to a 3-component simplex.

    In the CES case::
        raw_sizes = [1, 2, 1]
        theta_sizes = [1, 3, 1]

        input: [z_rho | z_a1 z_a2 | z_u] # width 4
        output: [rho | a_1 a_2 a_3 | u] # width 5


    PyTorch's ``CatTransform`` and ``biject_to(constraints.cat(...))`` assume a
    single set of block lengths, so they cannot correctly represent this kind of
    width-changing concatenation. ``BlockTransform`` keeps input and output block
    widths separate to avoid that ambiguity.

    Args:
        transforms: One ``Transform`` per block.
        raw_sizes: Number of input coordinates consumed by each block.
        theta_sizes: Number of output coordinates produced by each block.
    """

    domain = constraints.independent(constraints.real, 1)
    # Under-declared on purpose: the honest codomain is a cat of interval /
    # simplex / positive constraints, but biject_to on that is the silent bug
    # above, so declaring it would invite the shortcut this class replaces.
    # Nothing consumes codomain here; log_prob and sample call the maps directly.
    codomain = constraints.independent(constraints.real, 1)
    bijective = True

    def __init__(self, transforms, raw_sizes, theta_sizes, cache_size: int = 0):
        """Initializes the block transform."""
        super().__init__(cache_size=cache_size)

        self.transforms = list(transforms)
        self.raw_sizes = [int(s) for s in raw_sizes]
        self.theta_sizes = [int(s) for s in theta_sizes]

        if not (len(self.transforms) == len(self.raw_sizes) == len(self.theta_sizes)):
            raise ValueError(
                "transforms, raw_sizes and theta_sizes must have equal length; got "
                f"{len(self.transforms)}, {len(self.raw_sizes)}, {len(self.theta_sizes)}."
            )
        if any(s <= 0 for s in self.raw_sizes + self.theta_sizes):
            raise ValueError("Block sizes must be positive.")

        self.dim_raw = sum(self.raw_sizes)
        self.dim_theta = sum(self.theta_sizes)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        blocks = ", ".join(
            f"{type(t).__name__}({r}->{d})"
            for t, r, d in zip(self.transforms, self.raw_sizes, self.theta_sizes)
        )
        return f"{type(self).__name__}([{blocks}])"

    def _split(self, x: Tensor, sizes, name: str):
        """Splits the last axis into per-block slices, checking its width."""
        if x.shape[-1] != sum(sizes):
            raise ValueError(
                f"{name} must have trailing size {sum(sizes)}; got {tuple(x.shape)}."
            )
        return torch.split(x, sizes, dim=-1)

    def _call(self, raw: Tensor) -> Tensor:
        """Maps unconstrained ``[..., dim_raw]`` to constrained ``[..., dim_theta]``."""
        blocks = self._split(raw, self.raw_sizes, "raw")
        return torch.cat(
            [t(block) for t, block in zip(self.transforms, blocks)], dim=-1
        )

    def _inverse(self, theta: Tensor) -> Tensor:
        """Maps constrained ``[..., dim_theta]`` to unconstrained ``[..., dim_raw]``."""
        blocks = self._split(theta, self.theta_sizes, "theta")
        return torch.cat(
            [t.inv(block) for t, block in zip(self.transforms, blocks)], dim=-1
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Log ``|det dtheta/draw|`` at ``x``, summed over blocks; shape ``[*batch]``.

        Summing block terms is valid because the Jacobian is
        block-diagonal: each block's output depends only on its own input, so the
        determinant factorises. A transform whose output read another block's
        input would make this wrong.
        """
        raw_blocks = self._split(x, self.raw_sizes, "x")
        theta_blocks = self._split(y, self.theta_sizes, "y")

        batch_ndim = x.ndim - 1
        total = x.new_zeros(x.shape[:batch_ndim])
        for transform, raw_block, theta_block in zip(
            self.transforms, raw_blocks, theta_blocks
        ):
            term = transform.log_abs_det_jacobian(raw_block, theta_block)
            # Elementwise blocks (sigmoid, exp) return [*batch, size]; blocks with
            # a vector domain (stick-breaking) have already consumed their event
            # dim and return [*batch]. Reduce whatever is left over the batch.
            while term.ndim > batch_ndim:
                term = term.sum(-1)
            total = total + term
        return total
