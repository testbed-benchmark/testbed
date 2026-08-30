"""Vectorized JAX simulator and an exact terminal-only static/two-stage path."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .common_randomness import permuted
from .policies import cyclic_counts, leader_choice, ucb_choice
from .posterior import posterior, posterior_from_counts, update


class Trajectory(NamedTuple):
    counts: object
    sum_y: object
    mean: object
    sd: object
    actions: object
    observations: object
    first_leader: object


@partial(jax.jit, static_argnames=("config", "policy"))
def rollout(episodes, config, policy):
    """Actions are zero-based positions in the shared episode permutation."""
    policy.validate(config)
    data = permuted(episodes)
    size = data.theta.shape[0]
    rows = jnp.arange(size)
    n = jnp.zeros((size, config.d), dtype=jnp.int32)
    q = jnp.zeros_like(n, dtype=jnp.float64)
    first = jnp.full_like(n, config.T + 1)
    leader = jnp.zeros(size, dtype=jnp.int32)

    def step(carry, t):
        n, q, first, leader = carry
        mean, sd = posterior(n, q, config)
        if policy.kind == "ucb":
            action = ucb_choice(mean, sd, policy.beta)
        elif policy.kind == "random":
            action = jnp.floor(data.uniform[:, t] * config.d).astype(jnp.int32)
        elif policy.kind in ("eig", "nonadaptive"):
            k = config.d if policy.kind == "eig" else policy.k
            action = jnp.full((size,), t % k, dtype=jnp.int32)
        else:
            proposed = leader_choice(
                mean, sd, policy.k, policy.kappa, policy.h == "margin"
            )
            leader = jnp.where(t == policy.S, proposed, leader)
            action = jnp.where(t < policy.S, t % policy.k, leader)
        y = data.theta[rows, action] + data.noise[rows, action, n[rows, action]]
        n, q = update(n, q, action, y)
        new_mean, _ = posterior(n, q, config)
        best = jnp.argmax(new_mean, axis=-1)
        first = first.at[rows, best].set(jnp.minimum(first[rows, best], t + 1))
        return (n, q, first, leader), (action, y)

    (n, q, first, _), (actions, ys) = jax.lax.scan(
        step, (n, q, first, leader), jnp.arange(config.T, dtype=jnp.int32)
    )
    mean, sd = posterior(n, q, config)
    return Trajectory(n, q, mean, sd, actions.T, ys.T, first)


def static_terminal(prefix, config, k):
    return posterior_from_counts(prefix, cyclic_counts(config.T, k, config.d), config)


def explore_terminal(prefix, config, k, S, margin, kappa):
    """Identical potential observations to rollout, without scanning fixed stages."""
    n = jnp.broadcast_to(cyclic_counts(S, k, config.d), prefix.shape[:2])
    mean, sd = posterior_from_counts(prefix, n, config)
    leader = leader_choice(mean, sd, k, kappa, margin)
    n = n.at[jnp.arange(n.shape[0]), leader].add(config.T - S)
    return posterior_from_counts(prefix, n, config)
