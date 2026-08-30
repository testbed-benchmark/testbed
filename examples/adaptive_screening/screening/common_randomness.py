"""Candidate-specific potential observations, stable across episode batching."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Episodes(NamedTuple):
    theta: object
    noise: object
    uniform: object
    permutation: object

    def subset(self, start, stop):
        return Episodes(*(x[start:stop] for x in self))


@partial(jax.jit, static_argnames=("config", "count"))
def generate(config, seed, count, start=0):
    """Each global episode ID owns four independent PRNG streams."""

    def one(index):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), index), 4)
        return Episodes(
            config.mu
            + config.tau * jax.random.normal(keys[0], (config.d,), dtype=jnp.float64),
            config.sigma
            * jax.random.normal(keys[1], (config.d, config.T), dtype=jnp.float64),
            jax.random.uniform(keys[2], (config.T,), dtype=jnp.float64),
            jax.random.permutation(keys[3], config.d),
        )

    return jax.vmap(one)(jnp.arange(count, dtype=jnp.uint32) + start)


def pregenerate(config, dataset, batch_size=1000):
    """Materialize the complete fixed dataset before any policy uses it."""
    chunks = [
        generate(config, dataset.seed, min(batch_size, dataset.episodes - i), i)
        for i in range(0, dataset.episodes, batch_size)
    ]
    return Episodes(
        *(np.concatenate([np.asarray(c[j]) for c in chunks]) for j in range(4))
    )


def permuted(episodes):
    """Map physical labels into each episode's shared deterministic tie order."""
    rows = jnp.arange(episodes.theta.shape[0])[:, None]
    return Episodes(
        jnp.asarray(episodes.theta)[rows, episodes.permutation],
        jnp.asarray(episodes.noise)[rows, episodes.permutation],
        jnp.asarray(episodes.uniform),
        jnp.asarray(episodes.permutation),
    )


def observation_prefix(episodes):
    ordered = permuted(episodes)
    observations = ordered.theta[..., None] + ordered.noise
    return jnp.pad(jnp.cumsum(observations, axis=-1), ((0, 0), (0, 0), (1, 0)))
