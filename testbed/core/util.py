import torch


def broadcast_and_flatten(*tensors_with_event_shapes):
    """
    For each (tensor, event_shape) pair, broadcast the batch dims (all dims
    except the last len(event_shape)) together using torch.broadcast_shapes,
    then flatten them to a single b_flat dimension.

    Uses broadcast_to (zero-copy view) so no data is copied unless the
    computation genuinely requires it (i.e. a 0-stride expanded dim must be
    reshaped into a different total shape).

    Returns: (list of [b_flat, *event_shape_i] tensors, broadcast batch_shape)
    """
    batch_shapes = []
    for t, e in tensors_with_event_shapes:
        batch_shapes.append(t.shape[:t.ndim - len(e)])

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
    # Expands tensors along the first (batch) dimension so that they agree in this dim.
    # Requires tensors to have either the same batch dimension or a batch dimension of one.
    
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
        
    return tuple(t.expand(target_b, *t.shape[1:]) if t.shape[0] == 1 else t for t in tensors)

def batch_trace(x):
    # x: [b, m, m]
    # takes the traces of each m x m matrix in x
    return torch.einsum('b m m -> b', x)

def normal_log_prob(y, loc, scale):
    # Computes the log-pdf of a univariate Gaussian N(loc, scale^2)

    scale = torch.as_tensor(scale, device=y.device)
    
    var = scale ** 2
    term1 = ((y - loc) ** 2) / var
    term2 = torch.log(2 * torch.pi * var)
    return -0.5 * (term1 + term2)

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / 2**0.5))
