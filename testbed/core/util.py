
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