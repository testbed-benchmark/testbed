"""Independent Gaussian conjugate inference in float64."""

import jax.numpy as jnp


def posterior(n, sum_y, config):
    variance = 1.0 / (config.tau**-2 + n * config.sigma**-2)
    mean = variance * (config.mu * config.tau**-2 + sum_y * config.sigma**-2)
    return mean, jnp.sqrt(variance)


def update(n, sum_y, candidate, observation):
    rows = jnp.arange(n.shape[0])
    return n.at[rows, candidate].add(1), sum_y.at[rows, candidate].add(observation)


def posterior_from_counts(prefix, counts, config):
    counts = jnp.broadcast_to(counts, prefix.shape[:2])
    sums = jnp.take_along_axis(prefix, counts[..., None], axis=-1)[..., 0]
    return posterior(counts, sums, config)
