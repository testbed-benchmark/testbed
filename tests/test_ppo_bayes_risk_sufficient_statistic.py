"""Tests for sufficient-statistic Bayes-risk PPO on lot inspection."""

import importlib.util
from pathlib import Path

import pytest
import torch

EXAMPLE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "ppo_bayes_risk_destructive_lot_inspection_sufficient_statistic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ppo_bayes_risk_destructive_lot_inspection_sufficient_statistic",
    EXAMPLE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
ppo_counts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppo_counts)


def test_count_sufficient_statistic_counts_by_zone():
    designs = torch.tensor([[0, 1, 1, 2]])
    outcomes = torch.tensor([[[1.0], [0.0], [1.0], [1.0]]])
    mask = torch.tensor([[True, True, True, False]])

    counts = ppo_counts.count_sufficient_statistic(
        designs,
        outcomes,
        mask,
        num_zones=3,
        horizon=4,
    )

    expected = torch.tensor([[0.25, 0.50, 0.00, 0.25, 0.25, 0.00]])
    torch.testing.assert_close(counts, expected)


def test_count_policy_samples_from_compact_observation():
    policy = ppo_counts.CountCategoricalPolicyNet(
        num_zones=3,
        horizon=4,
        hidden_dim=8,
    )
    obs = {"counts": torch.zeros(5, 6)}

    sample = policy.sample_from_observation(obs)
    log_prob, entropy = policy.evaluate_actions_from_observation(obs, sample.action)

    assert sample.action.shape == (5,)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert torch.all((sample.action >= 0) & (sample.action < 3))


def test_sufficient_statistic_example_runs_tiny_training():
    pytest.importorskip("scipy")
    args = ppo_counts.Args(
        save=False,
        cuda=False,
        verbose=False,
        num_zones=3,
        horizon=2,
        hidden_dim=8,
        num_training_iterations=1,
        num_envs=2,
        ppo_num_minibatches=2,
        ppo_update_epochs=1,
        log_frequency=1,
        eval_episodes=4,
    )

    result, checkpoint = ppo_counts.train(args)

    assert checkpoint is None
    assert result.checkpoint["algorithm"] == "ppo_bayes_risk_sufficient_statistic"
    assert isinstance(result.designer.policy_net, ppo_counts.CountCategoricalPolicyNet)
