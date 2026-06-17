"""Tests for Bayes-risk PPO on destructive lot inspection."""

import importlib.util
from pathlib import Path

import pytest
import torch

from testbed.design_nets import CategoricalPolicyNet
from testbed.envs.history import History

EXAMPLE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "ppo_bayes_risk_destructive_lot_inspection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ppo_bayes_risk_destructive_lot_inspection",
    EXAMPLE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
ppo_bayes_risk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppo_bayes_risk)


def make_history(designs, outcomes):
    designs = torch.as_tensor(designs, dtype=torch.get_default_dtype())
    outcomes = torch.as_tensor(outcomes, dtype=torch.get_default_dtype()).unsqueeze(-1)
    mask = torch.ones_like(designs, dtype=torch.bool)
    return History(
        max_episode_length=designs.shape[-1],
        designs=designs,
        outcomes=outcomes,
        mask=mask,
    )


def test_downstream_acceptance_rule_rejects_uninspected_zone():
    pytest.importorskip("scipy")
    history = make_history([[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]])

    accepted = ppo_bayes_risk.downstream_accepts_lot(
        history,
        num_zones=2,
        tau=0.7,
        alpha=0.05,
    )

    torch.testing.assert_close(accepted, torch.tensor([False]))


def test_downstream_acceptance_rule_accepts_when_all_zones_pass():
    pytest.importorskip("scipy")
    history = make_history([[0, 0, 0, 1, 1, 1]], [[0, 0, 0, 0, 0, 0]])

    accepted = ppo_bayes_risk.downstream_accepts_lot(
        history,
        num_zones=2,
        tau=0.7,
        alpha=0.05,
    )

    torch.testing.assert_close(accepted, torch.tensor([True]))


def test_lot_loss_uses_asymmetric_false_accept_and_false_reject_costs():
    accepted = torch.tensor([True, True, False, False])
    theta = torch.tensor(
        [
            [0.05, 0.08],
            [0.20, 0.08],
            [0.05, 0.08],
            [0.20, 0.08],
        ]
    )

    loss = ppo_bayes_risk.lot_loss(
        accepted,
        theta,
        tau=0.10,
        false_accept_cost=10.0,
        false_reject_cost=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor([0.0, 10.0, 1.0, 0.0]))


def test_ppo_bayes_risk_example_runs_tiny_training():
    pytest.importorskip("scipy")
    args = ppo_bayes_risk.Args(
        save=False,
        cuda=False,
        verbose=False,
        num_zones=3,
        horizon=2,
        num_training_iterations=1,
        num_envs=2,
        ppo_num_minibatches=2,
        ppo_update_epochs=1,
        log_frequency=1,
        eval_episodes=4,
    )

    result, checkpoint = ppo_bayes_risk.train(args)

    assert checkpoint is None
    assert result.checkpoint["algorithm"] == "ppo_bayes_risk"
    assert result.checkpoint["tau"].item() == pytest.approx(args.tau)
    assert result.checkpoint["false_accept_cost"].item() == pytest.approx(
        args.false_accept_cost
    )
    assert isinstance(result.designer.policy_net, CategoricalPolicyNet)
