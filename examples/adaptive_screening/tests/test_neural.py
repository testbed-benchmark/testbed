import jax
import jax.numpy as jnp
import numpy as np

from screening.common_randomness import generate
from screening.neural import (
    init_policy,
    load_checkpoint,
    parameter_count,
    policy_logits,
    rollout_policy,
    save_checkpoint,
)
from screening.train import (
    BehaviorCloningSettings,
    ReinforceSettings,
    TrainingSettings,
    sample_training_episodes,
    train_one,
)


def test_neural_policy_is_permutation_equivariant_and_dimension_free():
    params = init_policy(jax.random.key(7))
    rng = np.random.default_rng(8)
    mean = jnp.asarray(rng.normal(size=(5, 16)))
    sd = jnp.asarray(rng.uniform(0.1, 1.0, size=(5, 16)))
    counts = jnp.asarray(rng.integers(0, 12, size=(5, 16)))
    permutation = rng.permutation(16)
    logits = policy_logits(params, mean, sd, counts, 11, 24, 2.5)
    permuted_logits = policy_logits(
        params,
        mean[:, permutation],
        sd[:, permutation],
        counts[:, permutation],
        11,
        24,
        2.5,
    )
    np.testing.assert_allclose(permuted_logits, logits[:, permutation], atol=1e-6)
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(permuted_logits[:, inverse], logits, atol=1e-6)
    assert parameter_count(params) == parameter_count(init_policy(jax.random.key(9)))


def test_neural_gradients_checkpoint_and_rollout_are_finite(small_config, tmp_path):
    params = init_policy(jax.random.key(10))
    mean = jnp.vstack((jnp.zeros(4), jnp.array([3.0, 1.0, -2.0, 0.2])))
    sd = jnp.vstack((jnp.ones(4), jnp.array([0.1, 0.2, 0.4, 0.7])))
    counts = jnp.vstack((jnp.zeros(4, dtype=jnp.int32), jnp.array([6, 5, 1, 0])))
    gradients = jax.grad(
        lambda p: jnp.sum(policy_logits(p, mean, sd, counts, 4, 6, 2.5) ** 2)
    )(params)
    assert all(
        np.isfinite(value).all() for value in jax.tree_util.tree_leaves(gradients)
    )
    path = tmp_path / "checkpoint.msgpack"
    save_checkpoint(path, params, {"seed": 10, "kappa_train": 2.5})
    restored, metadata = load_checkpoint(path)
    assert metadata == {"seed": 10, "kappa_train": 2.5}
    for left, right in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    result = rollout_policy(restored, generate(small_config, 12, 32), small_config, 2.5)
    assert all(np.isfinite(value).all() for value in result.trajectory)
    assert np.isfinite(result.entropy).all()


def test_fresh_training_episodes_and_tiny_end_to_end_training(small_config, tmp_path):
    first = sample_training_episodes(small_config, jax.random.key(1), 16)
    repeat = sample_training_episodes(small_config, jax.random.key(1), 16)
    different = sample_training_episodes(small_config, jax.random.key(2), 16)
    for left, right in zip(first, repeat, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first.theta, different.theta)
    settings = TrainingSettings(
        behavior_cloning=BehaviorCloningSettings(updates=2, trajectory_batch_size=16),
        reinforce=ReinforceSettings(
            warmup_steps=1,
            total_updates=2,
            trajectory_batch_size=16,
            validation_every_updates=1,
            validation_episodes=32,
            early_stopping_patience_updates=10,
        ),
        validation_seed=424242,
    )
    output = tmp_path / "training"
    summary = train_one(
        small_config,
        {"environment_hash": "test", "pilot_result_hash": "test"},
        0.0,
        101,
        2.0,
        output,
        settings,
    )
    assert summary["all_finite"]
    assert {
        "checkpoint_bc.msgpack",
        "checkpoint_best.msgpack",
        "checkpoint_last.msgpack",
        "training_log.csv",
        "validation_log.csv",
        "resolved_config.yaml",
        "code_commit.txt",
        "training_summary.json",
    } <= {path.name for path in output.iterdir()}
