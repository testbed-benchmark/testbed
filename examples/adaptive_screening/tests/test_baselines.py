import jax.numpy as jnp
import numpy as np
import pytest

from screening.common_randomness import generate, observation_prefix, permuted
from screening.config import Config
from screening.environment import explore_terminal, rollout, static_terminal
from screening.metrics import mean_interval, trajectory_metrics
from screening.policies import Policy, cyclic_counts, leader_choice, ucb_choice
from screening.posterior import posterior


def test_common_randomness_and_batch_invariance(small_config):
    whole = generate(small_config, 13, 17)
    again = generate(small_config, 13, 17)
    a, b = generate(small_config, 13, 8), generate(small_config, 13, 9, 8)
    for x, y, first, last in zip(whole, again, a, b, strict=True):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, np.concatenate([first, last]))
    assert not np.array_equal(whole.theta, generate(small_config, 14, 17).theta)
    np.testing.assert_array_equal(
        np.sort(whole.permutation, axis=-1), np.tile(np.arange(4), (17, 1))
    )
    ordered = permuted(whole)
    policy = Policy("ucb", beta=1.2)
    result = rollout(whole, small_config, policy)
    repeat = rollout(again, small_config, policy)
    np.testing.assert_array_equal(result.actions, repeat.actions)
    counts = np.zeros_like(result.counts)
    for t in range(small_config.T):
        choices = np.asarray(result.actions[:, t])
        rows = np.arange(17)
        expected = (
            np.asarray(ordered.theta)[rows, choices]
            + np.asarray(ordered.noise)[rows, choices, counts[rows, choices]]
        )
        np.testing.assert_array_equal(result.observations[:, t], expected)
        counts[rows, choices] += 1


def test_permutation_ties_and_random_uniform_coupling(small_config):
    data = generate(small_config, 25, 256)
    ucb = rollout(data, small_config, Policy("ucb", beta=0))
    np.testing.assert_array_equal(ucb.actions[:, 0], 0)
    # The same tie maps to an independently permuted physical candidate per episode.
    assert len(np.unique(np.asarray(data.permutation)[:, 0])) == small_config.d
    random = rollout(data, small_config, Policy("random"))
    np.testing.assert_array_equal(
        random.actions, np.floor(np.asarray(data.uniform) * small_config.d)
    )
    np.testing.assert_array_equal(
        ucb_choice(jnp.array([[1.0, 1.0, 0.0]]), jnp.ones((1, 3)), 2), [0]
    )


def test_balanced_eig_and_subsets():
    config = Config()
    data = generate(config, 24, 256)
    eig = rollout(data, config, Policy("eig"))
    assert np.max(np.asarray(eig.counts).max(-1) - np.asarray(eig.counts).min(-1)) <= 1
    for k in (1, 7, 16):
        result = rollout(data, config, Policy("nonadaptive", k=k))
        assert (np.asarray(result.actions) < k).all()
        np.testing.assert_array_equal(
            result.counts,
            np.broadcast_to(cyclic_counts(config.T, k, config.d), result.counts.shape),
        )
        mean, sd = static_terminal(observation_prefix(data), config, k)
        np.testing.assert_allclose(result.mean, mean, atol=1e-14)
        np.testing.assert_allclose(result.sd, sd, atol=1e-14)


@pytest.mark.parametrize("h", ["mean", "margin"])
def test_explore_certify_leader_frozen_and_terminal_equivalence(small_config, h):
    data = generate(small_config, 19, 256)
    prefix = observation_prefix(data)
    for k, S in ((1, 1), (3, 4), (4, 5)):
        policy = Policy("explore_certify", k=k, S=S, h=h, kappa=2.5)
        result = rollout(data, small_config, policy)
        np.testing.assert_array_equal(
            result.actions[:, S:],
            np.repeat(
                np.asarray(result.actions[:, S : S + 1]), small_config.T - S, axis=1
            ),
        )
        mean, sd = explore_terminal(prefix, small_config, k, S, h == "margin", 2.5)
        np.testing.assert_allclose(result.mean, mean, atol=1e-14)
        np.testing.assert_allclose(result.sd, sd, atol=1e-14)
        n = np.zeros((256, 4), dtype=int)
        sums = np.zeros((256, 4))
        rows = np.arange(256)
        for t in range(S):
            n[rows, result.actions[:, t]] += 1
            sums[rows, result.actions[:, t]] += result.observations[:, t]
        m, s = posterior(n, sums, small_config)
        np.testing.assert_array_equal(
            result.actions[:, S], leader_choice(m, s, k, 2.5, h == "margin")
        )


def test_ucb_selects_largest_index_every_round(small_config):
    data = generate(small_config, 156, 256)
    result = rollout(data, small_config, Policy("ucb", beta=1.4))
    n = np.zeros((256, 4), dtype=int)
    sums = np.zeros((256, 4))
    for t in range(6):
        mean, sd = posterior(n, sums, small_config)
        choices = np.asarray(result.actions[:, t])
        np.testing.assert_array_equal(choices, np.argmax(mean + 1.4 * sd, axis=-1))
        n[np.arange(256), choices] += 1
        sums[np.arange(256), choices] += result.observations[:, t]


def test_adaptive_rb_return_matches_realized_utility():
    config = Config()
    data = generate(config, 793, 50000)
    result = rollout(data, config, Policy("ucb", beta=1.4))
    metrics = trajectory_metrics(result, permuted(data).theta, 2.5, config)
    delta = mean_interval(metrics["return"] - metrics["selected_utility"])
    assert abs(delta["mean"]) < 5 * delta["se"]
