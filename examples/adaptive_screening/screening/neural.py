"""Permutation-equivariant neural policy and portable checkpoints."""

from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import msgpack
import numpy as np

from .common_randomness import permuted
from .environment import Trajectory
from .posterior import posterior, update


def _dense(key, fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    weight = jax.random.uniform(
        key, (fan_in, fan_out), minval=-limit, maxval=limit, dtype=jnp.float32
    )
    return {"weight": weight, "bias": jnp.zeros((fan_out,), dtype=jnp.float32)}


def init_policy(key):
    """Initialize the architecture from Section 12 of the specification."""
    keys = iter(jax.random.split(key, 5))
    return {
        "encoder": (
            _dense(next(keys), 6, 64),
            _dense(next(keys), 64, 64),
        ),
        "scorer": (
            _dense(next(keys), 64 + 64 + 64 + 10, 128),
            _dense(next(keys), 128, 64),
            _dense(next(keys), 64, 1),
        ),
    }


def _apply_dense(layer, values):
    return values @ layer["weight"] + layer["bias"]


def _top_two(values):
    return jax.lax.top_k(values, 2)[0]


def policy_logits(params, mean, sd, counts, step, budget, kappa_train):
    """Return candidate logits; all leading batch dimensions are supported."""
    local = jnp.stack(
        (
            mean,
            sd,
            jnp.log(sd + 1e-8),
            counts / budget,
            mean / (sd + 1e-6),
            mean - kappa_train * sd,
        ),
        axis=-1,
    )
    encoded = jax.nn.silu(_apply_dense(params["encoder"][0], local))
    encoded = jax.nn.silu(_apply_dense(params["encoder"][1], encoded))
    mean_pool = jnp.mean(encoded, axis=-2)
    max_pool = jnp.max(encoded, axis=-2)
    margins = mean - kappa_train * sd
    top_mean = _top_two(mean)
    top_margin = _top_two(margins)
    total = jnp.sum(counts, axis=-1)
    proportions = counts / jnp.maximum(total[..., None], 1)
    entropy = -jnp.sum(
        jnp.where(
            proportions > 0,
            proportions * jnp.log(jnp.maximum(proportions, 1e-30)),
            0.0,
        ),
        axis=-1,
    ) / jnp.log(mean.shape[-1])
    entropy = jnp.where(total > 0, entropy, 0.0)
    step = jnp.broadcast_to(jnp.asarray(step), mean.shape[:-1])
    globals_ = jnp.stack(
        (
            step / budget,
            (budget - step) / budget,
            top_mean[..., 0],
            top_mean[..., 1],
            top_margin[..., 0],
            top_margin[..., 1],
            jnp.mean(sd, axis=-1),
            jnp.min(sd, axis=-1),
            jnp.mean(counts > 0, axis=-1),
            entropy,
        ),
        axis=-1,
    )
    candidates = mean.shape[-1]
    pooled = jnp.concatenate((mean_pool, max_pool, globals_), axis=-1)
    pooled = jnp.broadcast_to(pooled[..., None, :], (*mean.shape, pooled.shape[-1]))
    scored = jnp.concatenate((encoded, pooled), axis=-1)
    scored = jax.nn.silu(_apply_dense(params["scorer"][0], scored))
    scored = jax.nn.silu(_apply_dense(params["scorer"][1], scored))
    logits = _apply_dense(params["scorer"][2], scored)[..., 0]
    return jnp.reshape(logits, (*mean.shape[:-1], candidates))


def parameter_count(params):
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def _categorical_from_uniform(logits, uniform):
    probabilities = jax.nn.softmax(logits, axis=-1)
    action = jnp.sum(uniform[..., None] > jnp.cumsum(probabilities, axis=-1), axis=-1)
    return jnp.minimum(action, logits.shape[-1] - 1).astype(jnp.int32)


class NeuralRollout(NamedTuple):
    trajectory: Trajectory
    entropy: object
    first_action: object


@partial(jax.jit, static_argnames=("config", "greedy"))
def rollout_policy(params, episodes, config, kappa_train, greedy=False):
    """Evaluate a learned design on fixed common-random-number episodes."""
    data = permuted(episodes)
    size = data.theta.shape[0]
    rows = jnp.arange(size)
    n = jnp.zeros((size, config.d), dtype=jnp.int32)
    q = jnp.zeros((size, config.d), dtype=data.theta.dtype)
    first = jnp.full_like(n, config.T + 1)

    def step(carry, t):
        n, q, first = carry
        mean, sd = posterior(n, q, config)
        logits = policy_logits(params, mean, sd, n, t, config.T, kappa_train)
        if greedy:
            action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            action = _categorical_from_uniform(logits, data.uniform[:, t])
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
        y = data.theta[rows, action] + data.noise[rows, action, n[rows, action]]
        n, q = update(n, q, action, y)
        new_mean, _ = posterior(n, q, config)
        best = jnp.argmax(new_mean, axis=-1)
        first = first.at[rows, best].set(jnp.minimum(first[rows, best], t + 1))
        return (n, q, first), (action, y, entropy)

    (n, q, first), (actions, observations, entropy) = jax.lax.scan(
        step, (n, q, first), jnp.arange(config.T, dtype=jnp.int32)
    )
    mean, sd = posterior(n, q, config)
    trajectory = Trajectory(
        n,
        q,
        mean,
        sd,
        actions.T,
        observations.T,
        first,
    )
    return NeuralRollout(trajectory, jnp.mean(entropy, axis=0), actions[0])


def save_checkpoint(path, params, metadata):
    """Write a genuine msgpack checkpoint without framework-specific state."""
    leaves = []
    for value in jax.tree_util.tree_leaves(params):
        array = np.asarray(value)
        leaves.append(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "data": array.tobytes(order="C"),
            }
        )
    payload = msgpack.packb(
        {"format_version": 1, "metadata": metadata, "leaves": leaves},
        use_bin_type=True,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def load_checkpoint(path):
    record = msgpack.unpackb(Path(path).read_bytes(), raw=False)
    if record.get("format_version") != 1:
        raise ValueError("Unsupported neural checkpoint format")
    template = init_policy(jax.random.key(0))
    template_leaves, structure = jax.tree_util.tree_flatten(template)
    if len(template_leaves) != len(record["leaves"]):
        raise ValueError("Checkpoint does not match the neural architecture")
    leaves = []
    for expected, encoded in zip(template_leaves, record["leaves"], strict=True):
        value = np.frombuffer(encoded["data"], dtype=np.dtype(encoded["dtype"])).copy()
        value = value.reshape(encoded["shape"])
        if value.shape != expected.shape:
            raise ValueError("Checkpoint tensor has an invalid shape")
        leaves.append(jnp.asarray(value))
    return jax.tree_util.tree_unflatten(structure, leaves), record["metadata"]
