"""UCB imitation and REINFORCE training for the learned design policy."""

import math
import subprocess
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import yaml

from .action_rules import fixed_action, selected_value
from .common_randomness import Episodes, permuted
from .config import PROJECT
from .neural import (
    init_policy,
    parameter_count,
    policy_logits,
    rollout_policy,
    save_checkpoint,
)
from .posterior import posterior, update


@dataclass(frozen=True)
class BehaviorCloningSettings:
    learning_rate: float = 0.001
    updates: int = 1000
    trajectory_batch_size: int = 512


@dataclass(frozen=True)
class ReinforceSettings:
    learning_rate_initial: float = 0.0003
    learning_rate_final: float = 0.00003
    warmup_steps: int = 250
    total_updates: int = 6000
    trajectory_batch_size: int = 1024
    weight_decay: float = 0.00001
    global_gradient_clip: float = 1.0
    entropy_coefficient_initial: float = 0.01
    entropy_coefficient_final: float = 0.001
    entropy_decay_steps: int = 3000
    validation_every_updates: int = 200
    validation_episodes: int = 10000
    early_stopping_patience_updates: int = 1600
    minimum_validation_improvement: float = 0.0005


@dataclass(frozen=True)
class TrainingSettings:
    behavior_cloning: BehaviorCloningSettings = BehaviorCloningSettings()
    reinforce: ReinforceSettings = ReinforceSettings()
    validation_seed: int = 424242

    def to_dict(self):
        return asdict(self)


def _categorical(logits, uniform):
    probabilities = jax.nn.softmax(logits, axis=-1)
    action = jnp.sum(uniform[..., None] > jnp.cumsum(probabilities, axis=-1), axis=-1)
    return jnp.minimum(action, logits.shape[-1] - 1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("config", "count"))
def sample_training_episodes(config, key, count):
    """Generate fresh float32 episodes, including an explicit random permutation."""
    keys = jax.random.split(key, (count, 4))
    theta = jax.vmap(
        lambda k: (
            config.mu
            + config.tau * jax.random.normal(k, (config.d,), dtype=jnp.float32)
        )
    )(keys[:, 0])
    noise = jax.vmap(
        lambda k: (
            config.sigma * jax.random.normal(k, (config.d, config.T), dtype=jnp.float32)
        )
    )(keys[:, 1])
    uniform = jax.vmap(lambda k: jax.random.uniform(k, (config.T,), dtype=jnp.float32))(
        keys[:, 2]
    )
    permutation = jax.vmap(lambda k: jax.random.permutation(k, config.d))(keys[:, 3])
    return Episodes(theta, noise, uniform, permutation)


@partial(jax.jit, static_argnames=("config",))
def ucb_imitation_states(episodes, config, beta):
    """Roll out the frozen UCB teacher and retain every pre-action state."""
    data = permuted(episodes)
    size = data.theta.shape[0]
    rows = jnp.arange(size)
    n = jnp.zeros((size, config.d), dtype=jnp.int32)
    q = jnp.zeros((size, config.d), dtype=jnp.float32)

    def step(carry, t):
        n, q = carry
        mean, sd = posterior(n, q, config)
        action = jnp.argmax(mean + beta * sd, axis=-1).astype(jnp.int32)
        y = data.theta[rows, action] + data.noise[rows, action, n[rows, action]]
        new = update(n, q, action, y)
        return new, (mean, sd, n, action)

    _, states = jax.lax.scan(step, (n, q), jnp.arange(config.T, dtype=jnp.int32))
    mean, sd, counts, action = states
    return mean.swapaxes(0, 1), sd.swapaxes(0, 1), counts.swapaxes(0, 1), action.T


def _bc_loss(params, states, config, kappa_train):
    mean, sd, counts, labels = states
    steps = jnp.broadcast_to(jnp.arange(config.T), labels.shape)
    logits = policy_logits(params, mean, sd, counts, steps, config.T, kappa_train)
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(log_probabilities, labels[..., None], axis=-1)[..., 0]
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return -jnp.mean(chosen), accuracy


def _make_bc_step(config, kappa_train, optimizer):
    @jax.jit
    def step(params, state, examples):
        (loss, accuracy), gradients = jax.value_and_grad(_bc_loss, has_aux=True)(
            params, examples, config, kappa_train
        )
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, loss, accuracy

    return step


def _policy_gradient_loss(params, episodes, config, kappa_train, entropy_coefficient):
    data = permuted(episodes)
    size = data.theta.shape[0]
    rows = jnp.arange(size)
    n = jnp.zeros((size, config.d), dtype=jnp.int32)
    q = jnp.zeros((size, config.d), dtype=jnp.float32)

    def step(carry, t):
        n, q = carry
        mean, sd = posterior(n, q, config)
        logits = policy_logits(params, mean, sd, n, t, config.T, kappa_train)
        action = _categorical(logits, data.uniform[:, t])
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        probabilities = jnp.exp(log_probabilities)
        chosen = log_probabilities[rows, action]
        entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
        y = data.theta[rows, action] + data.noise[rows, action, n[rows, action]]
        return update(n, q, action, y), (chosen, entropy)

    (n, q), (log_probabilities, entropies) = jax.lax.scan(
        step, (n, q), jnp.arange(config.T, dtype=jnp.int32)
    )
    mean, sd = posterior(n, q, config)
    action = fixed_action(mean, sd, kappa_train)
    rewards = jax.lax.stop_gradient(selected_value(mean, action))
    baseline = (jnp.sum(rewards) - rewards) / (size - 1)
    advantages = rewards - baseline
    advantages = jax.lax.stop_gradient(
        advantages / (jnp.std(advantages, ddof=1) + 1e-6)
    )
    actor = -jnp.mean(advantages * jnp.sum(log_probabilities, axis=0))
    entropy = jnp.mean(entropies)
    return actor - entropy_coefficient * entropy, (
        jnp.mean(rewards),
        entropy,
        jnp.std(rewards, ddof=1),
    )


def _entropy_coefficient(update_number, settings):
    fraction = min(update_number / settings.entropy_decay_steps, 1.0)
    return settings.entropy_coefficient_initial + fraction * (
        settings.entropy_coefficient_final - settings.entropy_coefficient_initial
    )


def _make_reinforce_step(config, kappa_train, optimizer):
    @jax.jit
    def step(params, state, episodes, entropy_coefficient):
        (loss, auxiliary), gradients = jax.value_and_grad(
            _policy_gradient_loss, has_aux=True
        )(params, episodes, config, kappa_train, entropy_coefficient)
        gradient_norm = optax.tree.norm(gradients)
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
        finite = jnp.logical_and(
            jnp.isfinite(gradient_norm),
            jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(x))
                        for x in jax.tree_util.tree_leaves(params)
                    ]
                )
            ),
        )
        return params, state, loss, auxiliary, gradient_norm, finite

    return step


def validate_policy(params, episodes, config, kappa_train, batch_size=1000):
    rewards, entropies, first_actions = [], [], []
    for start in range(0, len(episodes.theta), batch_size):
        part = episodes.subset(start, min(start + batch_size, len(episodes.theta)))
        result = rollout_policy(params, part, config, kappa_train, greedy=False)
        trajectory = result.trajectory
        action = fixed_action(trajectory.mean, trajectory.sd, kappa_train)
        rewards.append(np.asarray(selected_value(trajectory.mean, action)))
        entropies.append(np.asarray(result.entropy))
        first_actions.append(np.asarray(result.first_action))
    rewards = np.concatenate(rewards)
    first_actions = np.concatenate(first_actions)
    return {
        "J": float(rewards.mean()),
        "J_se": float(rewards.std(ddof=1) / math.sqrt(len(rewards))),
        "entropy": float(np.concatenate(entropies).mean()),
        "first_action_max_fraction": float(
            np.bincount(first_actions, minlength=config.d).max() / len(first_actions)
        ),
        "episodes": int(len(rewards)),
    }


def _write_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _checkpoint_metadata(base, phase, update_number, validation):
    return {
        **base,
        "phase": phase,
        "update": int(update_number),
        "validation": validation,
    }


def train_one(
    config,
    gate,
    kappa_train,
    seed,
    beta_teacher,
    output,
    settings=TrainingSettings(),
):
    """Train one independently seeded policy and write all required artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    base_metadata = {
        "environment_hash": gate["environment_hash"],
        "pilot_result_hash": gate["pilot_result_hash"],
        "code_commit": commit,
        "kappa_train": float(kappa_train),
        "seed": int(seed),
        "parameter_count": parameter_count(init_policy(jax.random.key(0))),
        "teacher_beta": float(beta_teacher),
        "training_settings": settings.to_dict(),
    }
    resolved = {
        "environment": config.to_dict(),
        **base_metadata,
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True)
    )
    (output / "code_commit.txt").write_text(commit + "\n")

    main_key = jax.random.key(seed)
    init_key, bc_key, reinforce_key, validation_key = jax.random.split(main_key, 4)
    params = init_policy(init_key)
    training_rows = []

    bc = settings.behavior_cloning
    bc_optimizer = optax.adam(bc.learning_rate)
    bc_state = bc_optimizer.init(params)
    bc_step = _make_bc_step(config, kappa_train, bc_optimizer)
    for update_number in range(1, bc.updates + 1):
        episodes = sample_training_episodes(
            config, jax.random.fold_in(bc_key, update_number), bc.trajectory_batch_size
        )
        examples = ucb_imitation_states(episodes, config, beta_teacher)
        params, bc_state, loss, accuracy = bc_step(params, bc_state, examples)
        if update_number == 1 or update_number % 25 == 0:
            training_rows.append(
                {
                    "phase": "behavior_cloning",
                    "update": update_number,
                    "loss": float(loss),
                    "teacher_accuracy": float(accuracy),
                    "return": np.nan,
                    "entropy": np.nan,
                    "gradient_norm": np.nan,
                    "finite": True,
                }
            )

    validation_episodes = sample_training_episodes(
        config, validation_key, settings.reinforce.validation_episodes
    )
    bc_validation = validate_policy(params, validation_episodes, config, kappa_train)
    save_checkpoint(
        output / "checkpoint_bc.msgpack",
        params,
        _checkpoint_metadata(
            base_metadata, "behavior_cloning", bc.updates, bc_validation
        ),
    )

    reinforce = settings.reinforce
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=reinforce.learning_rate_initial,
        warmup_steps=reinforce.warmup_steps,
        decay_steps=reinforce.total_updates,
        end_value=reinforce.learning_rate_final,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(reinforce.global_gradient_clip),
        optax.adamw(schedule, weight_decay=reinforce.weight_decay),
    )
    optimizer_state = optimizer.init(params)
    reinforce_step = _make_reinforce_step(config, kappa_train, optimizer)
    validation_rows = [
        {
            "phase": "behavior_cloning",
            "update": 0,
            **bc_validation,
            "is_best": True,
        }
    ]
    best_validation = bc_validation["J"]
    best_params = params
    best_update = 0
    last_significant_update = 0
    stopped_early = False
    last_auxiliary = (jnp.nan, jnp.nan, jnp.nan)
    last_gradient_norm = jnp.nan
    last_finite = jnp.array(True)
    completed_updates = 0

    for update_number in range(1, reinforce.total_updates + 1):
        episodes = sample_training_episodes(
            config,
            jax.random.fold_in(reinforce_key, update_number),
            reinforce.trajectory_batch_size,
        )
        entropy_coefficient = _entropy_coefficient(update_number, reinforce)
        (
            params,
            optimizer_state,
            loss,
            last_auxiliary,
            last_gradient_norm,
            last_finite,
        ) = reinforce_step(
            params, optimizer_state, episodes, jnp.float32(entropy_coefficient)
        )
        completed_updates = update_number
        if update_number == 1 or update_number % 25 == 0:
            reward, entropy, reward_sd = last_auxiliary
            training_rows.append(
                {
                    "phase": "reinforce",
                    "update": update_number,
                    "loss": float(loss),
                    "teacher_accuracy": np.nan,
                    "return": float(reward),
                    "return_sd": float(reward_sd),
                    "entropy": float(entropy),
                    "entropy_coefficient": entropy_coefficient,
                    "gradient_norm": float(last_gradient_norm),
                    "finite": bool(last_finite),
                }
            )
        if not bool(last_finite):
            _write_csv(output / "training_log.csv", training_rows)
            raise FloatingPointError("Nonfinite gradient or parameter during training")

        if update_number % reinforce.validation_every_updates != 0:
            continue
        validation = validate_policy(params, validation_episodes, config, kappa_train)
        improved = validation["J"] > best_validation
        significant = (
            validation["J"]
            >= best_validation + reinforce.minimum_validation_improvement
        )
        if improved:
            best_validation = validation["J"]
            best_params = params
            best_update = update_number
            save_checkpoint(
                output / "checkpoint_best.msgpack",
                best_params,
                _checkpoint_metadata(
                    base_metadata, "reinforce", best_update, validation
                ),
            )
        if significant:
            last_significant_update = update_number
        validation_rows.append(
            {
                "phase": "reinforce",
                "update": update_number,
                **validation,
                "is_best": improved,
            }
        )
        save_checkpoint(
            output / "checkpoint_last.msgpack",
            params,
            _checkpoint_metadata(base_metadata, "reinforce", update_number, validation),
        )
        _write_csv(output / "training_log.csv", training_rows)
        _write_csv(output / "validation_log.csv", validation_rows)
        if (
            update_number - last_significant_update
            >= reinforce.early_stopping_patience_updates
        ):
            stopped_early = True
            break

    if best_update == 0:
        save_checkpoint(
            output / "checkpoint_best.msgpack",
            best_params,
            _checkpoint_metadata(base_metadata, "behavior_cloning", 0, bc_validation),
        )
    final_validation = validate_policy(params, validation_episodes, config, kappa_train)
    save_checkpoint(
        output / "checkpoint_last.msgpack",
        params,
        _checkpoint_metadata(
            base_metadata, "reinforce", completed_updates, final_validation
        ),
    )
    _write_csv(output / "training_log.csv", training_rows)
    _write_csv(output / "validation_log.csv", validation_rows)
    summary = {
        **base_metadata,
        "behavior_cloning_validation": bc_validation,
        "best_validation_J": best_validation,
        "best_update": best_update,
        "completed_updates": completed_updates,
        "stopped_early": stopped_early,
        "all_finite": bool(last_finite),
        "final_gradient_norm": float(last_gradient_norm),
        "final_training_return": float(last_auxiliary[0]),
    }
    from .io import write_json

    write_json(output / "training_summary.json", summary)
    return summary
