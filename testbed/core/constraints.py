"""Constraints and transforms that ``torch.distributions`` lacks.

Constraints on theta and on designs are the support of a distribution (the
prior, or the design prior), and raw coordinates come from a transform onto
that support. Torch supplies nearly all of this; this module adds the pieces
it does not:

- :func:`blocks`, for heterogeneous components concatenated along the last dim,
  whose ``biject_to`` is a :class:`BlockTransform`.
- :func:`finite_set`, for a finite set of points such as a pool of designs.
- :func:`default_design_transform`, the raw-to-design map of a design support.
"""

import torch
from torch import Tensor
from torch.distributions import biject_to, constraints
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    IndependentTransform,
    TanhTransform,
    Transform,
    identity_transform,
)


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
    widths separate to avoid that ambiguity; ``biject_to`` builds one from a
    :func:`blocks` constraint.

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

    def forward_shape(self, shape):
        """Shape of the output for an input of ``shape``."""
        return (*shape[:-1], self.dim_theta)

    def inverse_shape(self, shape):
        """Shape of the input for an output of ``shape``."""
        return (*shape[:-1], self.dim_raw)

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


class ClampedTanhTransform(TanhTransform):
    """``TanhTransform`` whose inverse clamps to the open interval first.

    Torch's inverse sends a value on the boundary to an infinite raw value; this
    one sends it to a large finite one, so boundary designs stay usable.
    """

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverts tanh, clamping to ``[-1 + eps, 1 - eps]``."""
        eps = torch.finfo(y.dtype).eps
        return torch.atanh(y.clamp(-1 + eps, 1 - eps))


class _Blocks(constraints.Constraint):
    """Constraints on contiguous blocks of the last dim.

    Each part constrains its block elementwise (``event_dim == 0``, e.g.
    ``positive``) or as a whole (``event_dim == 1``, e.g. ``simplex``).

    Args:
        parts: One constraint per block.
        sizes: Width of each block in the constrained coordinates.
    """

    event_dim = 1

    def __init__(self, parts, sizes):
        """Initializes the per-block constraints."""
        self.parts = list(parts)
        self.sizes = [int(s) for s in sizes]
        if len(self.parts) != len(self.sizes):
            raise ValueError("parts and sizes must have equal length.")
        if any(c.event_dim > 1 for c in self.parts):
            raise ValueError("Each part must have event_dim 0 or 1.")
        super().__init__()

    @property
    def is_discrete(self) -> bool:
        """Whether any block is discrete."""
        return any(c.is_discrete for c in self.parts)

    def check(self, value: Tensor) -> Tensor:
        """Checks every block, returning shape ``[*batch]``."""
        checks = [
            c.check(block) if c.event_dim else c.check(block).all(-1)
            for c, block in zip(self.parts, value.split(self.sizes, -1))
        ]
        return torch.stack(checks).all(0)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"{type(self).__name__}({self.parts}, sizes={self.sizes})"


@biject_to.register(_Blocks)
def _biject_to_blocks(constraint: _Blocks) -> BlockTransform:
    """Maps raw coordinates onto every block with that block's own bijection."""
    transforms = [biject_to(c) for c in constraint.parts]
    raw_sizes = [
        t.inverse_shape((s,))[-1] for t, s in zip(transforms, constraint.sizes)
    ]
    return BlockTransform(transforms, raw_sizes, constraint.sizes)


def _matches(values: Tensor, value: Tensor) -> Tensor:
    """Returns ``[*batch, n]`` indicators of which of ``values`` each point equals.

    Args:
        values: The set, shape ``[n, *event_shape]``.
        value: Points of shape ``[*batch, *event_shape]``.
    """
    values = values.to(device=value.device, dtype=value.dtype)
    event_numel = values[0].numel()
    flat_value = value.reshape(-1, event_numel)
    flat_values = values.reshape(values.shape[0], event_numel)
    matches = (flat_value.unsqueeze(1) == flat_values.unsqueeze(0)).all(dim=-1)
    batch_shape = value.shape[: value.ndim - (values.ndim - 1)]
    return matches.reshape(*batch_shape, values.shape[0])


class _FiniteSet(constraints.Constraint):
    """Membership in a finite set of points, by exact equality.

    Args:
        values: The points, shape ``[n, *event_shape]``.
    """

    is_discrete = True

    def __init__(self, values: Tensor):
        """Initializes the set."""
        self.values = values
        self.event_dim = values.ndim - 1
        super().__init__()

    def check(self, value: Tensor) -> Tensor:
        """Checks each point is in the set, returning shape ``[*batch]``."""
        return _matches(self.values, value).any(dim=-1)

    def __repr__(self) -> str:
        """Returns a compact representation."""
        return f"{type(self).__name__}(n={self.values.shape[0]})"


blocks = _Blocks
finite_set = _FiniteSet


def _base_constraint(support: constraints.Constraint) -> constraints.Constraint:
    """Unwraps ``independent`` to the elementwise constraint inside."""
    while isinstance(support, constraints.independent):
        support = support.base_constraint
    return support


def interval_bounds(support: constraints.Constraint) -> tuple[Tensor, Tensor] | None:
    """Returns ``(low, high)`` if ``support`` is a box, else None."""
    base = _base_constraint(support)
    if isinstance(base, constraints.interval):
        return base.lower_bound, base.upper_bound
    return None


def is_unconstrained(support: constraints.Constraint) -> bool:
    """Whether ``support`` is all of real space."""
    return isinstance(_base_constraint(support), type(constraints.real))


def in_support(support: constraints.Constraint, value: Tensor) -> bool:
    """Whether every point in a batch lies in ``support``.

    An unconstrained support accepts any value without checking it, so that
    validating designs costs nothing (and does not sync the GPU) when there is
    nothing to validate. An empty batch, e.g. a ``T=0`` history, is vacuously in
    the support.
    """
    if is_unconstrained(support) or value.numel() == 0:
        return True
    return bool(torch.all(support.check(value)).item())


def default_design_transform(support: constraints.Constraint) -> Transform:
    r"""Maps raw designs onto ``support``.

    This is ``biject_to(support)``, except that:

    - A box maps through ``bias + scale * tanh(raw)`` rather than ``biject_to``'s
      sigmoid, with ``bias = (high + low) / 2`` and ``scale = (high - low) / 2``,
      and a clamped inverse so that designs on the boundary have finite raw values.
    - A discrete support has no smooth map onto it, so it uses the identity.

    The transform reduces over the support's event dims, so its
    ``log_abs_det_jacobian`` maps ``[*batch, *event_shape]`` to ``[*batch]``.

    Raises:
        ValueError: If a box has an infinite or non-positive width.
    """
    if support.is_discrete:
        return IndependentTransform(identity_transform, support.event_dim)
    bounds = interval_bounds(support)
    if bounds is None:
        return biject_to(support)
    low, high = bounds
    scale = (high - low) / 2.0
    if not bool(torch.all(torch.isfinite(scale) & (scale > 0)).item()):
        raise ValueError(
            "A box design space's tanh reparametrisation requires finite bounds "
            f"with low < high; got low={low}, high={high}."
        )
    squash = ComposeTransform(
        [ClampedTanhTransform(), AffineTransform((high + low) / 2.0, scale)]
    )
    return IndependentTransform(squash, support.event_dim)
