"""Tensor shape and probability helpers."""

import torch
from torch import Tensor


def matches_event_shape(tensor: Tensor, event_shape) -> bool:
    """Checks whether a tensor's trailing dimensions are exactly ``event_shape``.

    The leading dimensions are unconstrained -- they are the batch shape. An
    empty ``event_shape`` matches any tensor; a tensor with fewer dimensions
    than ``event_shape`` matches nothing.

    This is the predicate form of the check :func:`broadcast_and_flatten`
    raises on, for callers that want to branch rather than fail (e.g.
    ``DesignSpace.contains``).
    """
    event_shape = torch.Size(event_shape)
    if tensor.ndim < len(event_shape):
        return False
    return tensor.shape[tensor.ndim - len(event_shape) :] == event_shape


def squeeze_history(tensor: Tensor, event_shape: torch.Size) -> Tensor:
    """Ensures tensor has shape ``[T, *event_shape]`` by squeezing leading size-1 dims.

    Raises ValueError if extra leading dims exist but are not all size 1 (batched
    multi-run data that can't be unambiguously reduced to a single sequence).
    """
    expected_ndim = 1 + len(event_shape)
    if tensor.ndim == expected_ndim:
        return tensor
    if tensor.ndim < expected_ndim:
        raise ValueError(
            f"Tensor must have shape [T, *event_shape] (ndim={expected_ndim}), "
            f"got shape {tuple(tensor.shape)}."
        )
    n_extra = tensor.ndim - expected_ndim
    leading = tensor.shape[:n_extra]
    if any(s != 1 for s in leading):
        raise ValueError(
            f"Tensor has leading batch dimensions {tuple(leading)} that are not all "
            f"size 1 and cannot be automatically squeezed. "
            f"Got shape {tuple(tensor.shape)}, expected ndim={expected_ndim}. "
            "Select a single run (e.g. tensor[0]) before passing."
        )
    return tensor.reshape(tensor.shape[n_extra:])


def broadcast_and_flatten(*tensors_with_event_shapes):
    """Broadcasts batch dims and flatten them into one dimension.

    Uses broadcast_to (zero-copy view) so no data is copied unless the
    computation genuinely requires it (i.e. a 0-stride expanded dim must be
    reshaped into a different total shape).

    Returns:
        Pair of ``[b_flat, *event_shape_i]`` tensors and broadcast batch shape.
    """
    batch_shapes = []
    for t, e in tensors_with_event_shapes:
        e = torch.Size(e)
        if not matches_event_shape(t, e):
            raise ValueError(
                f"Expected tensor with trailing (event) shape {tuple(e)}, "
                f"got shape {tuple(t.shape)}."
            )
        batch_shapes.append(t.shape[: t.ndim - len(e)])

    target_batch = torch.broadcast_shapes(*batch_shapes)
    b_flat = 1
    for s in target_batch:
        b_flat *= s

    result = []
    for t, e in tensors_with_event_shapes:
        full_target = target_batch + torch.Size(e)
        result.append(t.broadcast_to(full_target).reshape(b_flat, *e))

    return result, target_batch


def expand_along_batch(*tensors):
    """Expands tensors along the first batch dimension."""
    # Expands tensors along the first (batch) dimension so that they agree in this dim.
    # Requires tensors to have either the same batch dimension or a batch dimension of one.

    if not tensors:
        return []

    batch_sizes = [t.shape[0] for t in tensors]
    target_b = max(batch_sizes)

    for b in batch_sizes:
        # Every batch size must be either 1 or the target batch size.
        if b != 1 and b != target_b:
            raise ValueError(
                f"Incompatible batch dimensions. Batch sizes must either be 1 or "
                f"agree on a single value. Got: {batch_sizes}"
            )

    return tuple(
        t.expand(target_b, *t.shape[1:]) if t.shape[0] == 1 else t for t in tensors
    )


def batch_trace(x):
    """Computes traces for a batch of square matrices."""
    # x: [b, m, m]
    # takes the traces of each m x m matrix in x
    return torch.einsum("b m m -> b", x)


def normal_log_prob(y, loc, scale):
    """Computes univariate Gaussian log probabilities."""
    # Computes the log-pdf of a univariate Gaussian N(loc, scale^2)

    scale = torch.as_tensor(scale, device=y.device)

    var = scale**2
    term1 = ((y - loc) ** 2) / var
    term2 = torch.log(2 * torch.pi * var)
    return -0.5 * (term1 + term2)


def normal_cdf(x):
    """Computes the standard normal CDF."""
    return 0.5 * (1 + torch.erf(x / 2**0.5))


def sort_by_key(x: Tensor, key: Tensor) -> Tensor:
    """Sorts the ``n_objects`` axis of ``x`` by ascending ``key``.

    Args:
        x: ``[batch, n_objects, *object_shape]``.
        key: ``[batch, n_objects]``, the per-object sort key.

    Returns:
        Tensor with the same shape as ``x``, objects reordered by ``key``.
    """
    b, n = x.shape[:2]
    idx = torch.argsort(key, dim=1)
    idx = idx.view(b, n, *([1] * (x.ndim - 2))).expand_as(x)
    return torch.gather(x, dim=1, index=idx)


def sort_by_norm(x: Tensor) -> Tensor:
    """Sorts tensors according to their norm.

    Args:
        x: Either
            ``[batch, n_objects, *object_shape]``, or
            ``[n_objects, *object_shape]``.

    Returns:
        Tensor with the same shape as ``x``, with objects sorted by
        increasing Euclidean norm.
    """
    unbatched = x.ndim == 2
    if unbatched:
        x = x.unsqueeze(0)

    norms = torch.linalg.norm(x.flatten(start_dim=2), dim=-1)
    x = sort_by_key(x, norms)

    if unbatched:
        x = x.squeeze(0)
    return x


def sort_by_first_coordinate(x: Tensor) -> Tensor:
    """Sorts tensors according to their first coordinate.

    This is the canonicalization used by ScoreBED (Phillips, Kerrigan,
    Rainforth, UAI 2026), to remove the label-switching symmetry
    of multi-source location-finding-style tasks (whose likelihood is
    invariant to the ordering of an exchangeable ``n_objects`` axis, e.g.
    LocationFinding's ``n_sources``): sorting collapses every permutation of
    a given configuration onto the same canonical representative, so the
    posterior only needs to fit one point per equivalence class.

    Args:
        x: Either
            ``[batch, n_objects, *object_shape]``, or
            ``[n_objects, *object_shape]``, with ``object_shape`` non-empty.

    Returns:
        Tensor with the same shape as ``x``, with objects sorted by
        increasing first coordinate.
    """
    unbatched = x.ndim == 2
    if unbatched:
        x = x.unsqueeze(0)

    # for each entry, what is its first coordinate?
    first_coord = x.flatten(start_dim=2)[..., 0]
    x = sort_by_key(x, first_coord)

    if unbatched:
        x = x.squeeze(0)
    return x


def shape_as_str(x: Tensor | None) -> str:
    """Returns a tensor's shape as a string; for nice printing."""
    if x is None:
        return "None"
    return str(tuple(x.shape))
