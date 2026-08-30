"""Deployment actions: 0 denotes default; 1..d index permuted candidates."""

import jax.numpy as jnp


def bayes_action(mean):
    return jnp.where(jnp.max(mean, axis=-1) > 0, jnp.argmax(mean, axis=-1) + 1, 0)


def fixed_action(mean, sd, kappa):
    eligible = mean - kappa * sd > 0
    winner = jnp.argmax(jnp.where(eligible, mean, -jnp.inf), axis=-1) + 1
    return jnp.where(jnp.any(eligible, axis=-1), winner, 0)


def selected_value(values, action):
    chosen = jnp.take_along_axis(
        values, jnp.maximum(action - 1, 0)[..., None], axis=-1
    )[..., 0]
    return jnp.where(action > 0, chosen, 0.0)


def terminal_return(mean, sd, kappa):
    return selected_value(mean, fixed_action(mean, sd, kappa))
