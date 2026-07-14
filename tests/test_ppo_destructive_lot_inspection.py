"""Tests for PPO RL-BOED on discrete lot inspection."""

import importlib.util
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import torch

from testbed.design_nets import CategoricalPolicyNet, RLBOEDValueNetwork
from testbed.designers import PolicyDesigner
from testbed.envs import CategoricalDesignSpace
from testbed.envs.history import History
from testbed.methods import RLBOED, FitResult, RLBOEDConfig
from testbed.tasks import DestructiveLotInspection

EXAMPLE_PATH = (
    Path(__file__).parents[1] / "examples" / "ppo_destructive_lot_inspection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ppo_destructive_lot_inspection", EXAMPLE_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
ppo_destructive_lot_inspection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppo_destructive_lot_inspection)


def assert_singular_values_match_scale(weight, scale):
    singular_values = torch.linalg.svdvals(weight.detach())
    torch.testing.assert_close(
        singular_values,
        torch.full_like(singular_values, scale),
        rtol=1e-5,
        atol=1e-6,
    )


def tiny_config(**overrides):
    kwargs = {
        "outcome_low": 0.0,
        "outcome_high": 1.0,
        "n_contrastive": 1,
        "num_training_iterations": 2,
        "num_envs": 2,
        "gamma": 0.99,
        "ppo_num_minibatches": 2,
        "ppo_update_epochs": 1,
        "log_frequency": 1,
        "verbose": False,
    }
    kwargs.update(overrides)
    return RLBOEDConfig(**kwargs)


def test_ppo_networks_use_cleanrl_orthogonal_initialization():
    policy = CategoricalPolicyNet(CategoricalDesignSpace(3), dim_y=1)
    value_net = RLBOEDValueNetwork(
        dim_y=1,
        dim_design=1,
        design_low=torch.tensor([0]),
        design_high=torch.tensor([2]),
        outcome_low=0.0,
        outcome_high=1.0,
    )

    assert_singular_values_match_scale(
        policy.actor.encoder_net.net[0].weight,
        sqrt(2.0),
    )
    assert_singular_values_match_scale(policy.actor.logits_head.weight, 0.01)
    assert_singular_values_match_scale(value_net.encoder_net.net[0].weight, sqrt(2.0))
    assert_singular_values_match_scale(value_net.emitter_net.net[-1].weight, 1.0)
    torch.testing.assert_close(
        policy.actor.logits_head.bias,
        torch.zeros_like(policy.actor.logits_head.bias),
    )
    torch.testing.assert_close(
        value_net.emitter_net.net[-1].bias,
        torch.zeros_like(value_net.emitter_net.net[-1].bias),
    )


def test_rlboed_fit_dispatches_discrete_design_spaces_to_ppo(tmp_path):
    task = DestructiveLotInspection(num_zones=3)
    config = tiny_config()

    result = RLBOED(config).fit(task, horizon=2, design_space=task.design_space)

    assert isinstance(result, FitResult)
    assert isinstance(result.designer, PolicyDesigner)
    assert isinstance(result.designer.policy_net, CategoricalPolicyNet)
    assert result.checkpoint["algorithm"] == "ppo"
    assert result.checkpoint["env_steps"] == 8
    assert result.checkpoint["global_step"] == 8
    assert "designer" in result.checkpoint
    assert "policy" in result.checkpoint
    assert "value_net" in result.checkpoint
    assert "optimizer" in result.checkpoint

    theta = task.sample_prior(2)
    y_history, design_history = result.designer.rollout(theta)
    assert y_history.shape == (2, 2, 1)
    assert design_history.shape == (2, 2)
    assert task.design_space.contains(design_history)

    checkpoint_path = tmp_path / "ppo_rlboed.pt"
    result.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["algorithm"] == "ppo"


def test_ppo_supports_separate_actor_and_critic_optimizers():
    task = DestructiveLotInspection(num_zones=3)
    config = tiny_config(
        ppo_actor_learning_rate=1e-2,
        ppo_critic_learning_rate=2.5e-4,
    )

    result = RLBOED(config).fit(task, horizon=2, design_space=task.design_space)

    assert result.checkpoint["optimizer"] is None
    assert result.checkpoint["actor_optimizer"] is not None
    assert result.checkpoint["critic_optimizer"] is not None
    step_metrics = next(metrics for metrics in result.metrics if "approx_kl" in metrics)
    assert torch.isfinite(torch.tensor(step_metrics["actor_grad_norm"]))
    assert torch.isfinite(torch.tensor(step_metrics["critic_grad_norm"]))


def test_conservative_rule_uses_decision_maker_prior_and_evidence():
    args = ppo_destructive_lot_inspection.Args(
        cuda=False,
        num_zones=3,
        batch_size=4,
        dm_good_probability=0.4,
        dm_suspicious_zones="0",
        dm_rho_grid_size=8,
    )
    task = ppo_destructive_lot_inspection.make_task(args, torch.device("cpu"))
    rule = ppo_destructive_lot_inspection.make_decision_rule(args, task)
    history = History(
        max_episode_length=4,
        designs=torch.zeros(2, 4),
        outcomes=torch.tensor(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[4.0], [4.0], [4.0], [4.0]],
            ]
        ),
        mask=torch.ones(2, 4, dtype=torch.bool),
    )

    posterior_good = rule.posterior_good_probability(history)

    assert posterior_good[0] > args.dm_good_probability
    assert posterior_good[1] < args.dm_good_probability
    assert rule.threshold == 0.8


def test_default_conservative_rule_can_accept_favorable_history():
    args = ppo_destructive_lot_inspection.Args(cuda=False)
    task = ppo_destructive_lot_inspection.make_task(args, torch.device("cpu"))
    rule = ppo_destructive_lot_inspection.make_decision_rule(args, task)
    history = History(
        max_episode_length=args.horizon,
        designs=(torch.arange(args.horizon) % args.num_zones).unsqueeze(0),
        outcomes=torch.zeros(1, args.horizon, 1),
        mask=torch.ones(1, args.horizon, dtype=torch.bool),
    )

    assert rule(history).item()


def test_fixed_rule_evaluation_samples_stochastic_policy():
    class RecordingPolicy:
        def __init__(self):
            self.deterministic = None

        def eval(self):
            pass

        def sample_from_observation(self, observation, *, deterministic):
            self.deterministic = deterministic
            return SimpleNamespace(action=torch.zeros(observation["mask"].shape[0]))

    args = ppo_destructive_lot_inspection.Args(
        cuda=False,
        num_zones=3,
        horizon=2,
        batch_size=2,
        dm_suspicious_zones="0",
        dm_rho_grid_size=4,
        eval_episodes=32,
    )
    policy = RecordingPolicy()

    ppo_destructive_lot_inspection.evaluate_policy_mean_risk(args, policy)

    assert policy.deterministic is False


def test_ppo_destructive_lot_example_trains_for_fixed_rule():
    args = ppo_destructive_lot_inspection.Args(
        save=False,
        cuda=False,
        verbose=False,
        num_zones=3,
        batch_size=2,
        horizon=2,
        dm_suspicious_zones="0",
        dm_rho_grid_size=4,
        num_training_iterations=1,
        num_envs=2,
        ppo_num_minibatches=2,
        ppo_update_epochs=1,
        log_frequency=1,
    )

    result, checkpoint = ppo_destructive_lot_inspection.train(args)

    assert checkpoint is None
    assert isinstance(result.designer.policy_net, CategoricalPolicyNet)
    assert result.checkpoint["algorithm"] == "ppo_fixed_conservative_dm"
    torch.testing.assert_close(
        result.checkpoint["dm_acceptance_threshold"], torch.tensor(0.8)
    )
