"""Tests for PPO on destructive lot inspection."""

import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf

from testbed.design_nets import CategoricalPolicyNet, RLBOEDValueNetwork
from testbed.designers import PolicyDesigner
from testbed.envs import CategoricalDesignSpace
from testbed.envs.history import History
from testbed.methods.ppo import Trainer, build_env, make_networks, validate_config
from testbed.tasks import DestructiveLotInspection

EXAMPLE_PATH = (
    Path(__file__).parents[1] / "examples" / "ppo_destructive_lot_inspection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ppo_destructive_lot_inspection", EXAMPLE_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
ppo_example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ppo_example)


def make_config():
    """Builds a small CPU configuration for unit tests."""
    return OmegaConf.create(
        {
            "seed": 1,
            "torch_deterministic": True,
            "T": 2,
            "reward": {
                "n_contrastive": 2,
                "outcome_low": 0.0,
                "outcome_high": 4.0,
            },
            "ppo": {
                "num_training_iterations": 1,
                "num_envs": 4,
                "gamma": 0.99,
                "learning_rate": 2.5e-4,
                "actor_learning_rate": None,
                "critic_learning_rate": None,
                "anneal_lr": False,
                "gae_lambda": 0.95,
                "num_minibatches": 2,
                "update_epochs": 1,
                "norm_adv": True,
                "clip_coef": 0.2,
                "clip_vloss": True,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": None,
                "log_frequency": 1,
            },
        }
    )


def test_categorical_policy_samples_and_evaluates_designs():
    policy = CategoricalPolicyNet(
        CategoricalDesignSpace(3), dim_y=1, outcome_bounds=(0.0, 4.0)
    )
    observation = History.empty((5,), 2, (), (1,))

    sample = policy.sample_from_observation(observation)
    log_prob, entropy = policy.evaluate_designs_from_observation(
        observation, sample.design
    )

    assert sample.design.shape == (5,)
    assert log_prob.shape == (5,)
    assert entropy.shape == (5,)
    assert CategoricalDesignSpace(3).contains(sample.design)


def test_ppo_networks_use_discrete_task_shapes():
    task = DestructiveLotInspection(num_zones=3)
    cfg = make_config()

    policy, value = make_networks(task, task.design_space, cfg)

    assert isinstance(policy, CategoricalPolicyNet)
    assert isinstance(value, RLBOEDValueNetwork)
    assert policy.nvec == (3,)


def test_ppo_trainer_runs_one_update():
    task = DestructiveLotInspection(num_zones=3)
    cfg = make_config()
    validate_config(task, cfg, task.design_space)
    policy, value = make_networks(task, task.design_space, cfg)
    env = build_env(cfg, task, task.design_space)
    designer = PolicyDesigner(task, cfg.T, policy, deterministic=False)
    trainer = Trainer(task, designer, env, value, cfg)

    metrics = trainer.train_step(0)

    assert metrics[0]["env_steps"] == 8.0
    assert torch.isfinite(torch.tensor(metrics[0]["policy_loss"]))
    assert torch.isfinite(torch.tensor(metrics[0]["value_loss"]))


def test_reproduction_script_preserves_experiment_defaults():
    args = ppo_example.Args()

    assert args.seed == 1
    assert args.horizon == 30
    assert args.batch_size == 8
    assert args.num_training_iterations == 2_000
    assert args.num_envs == 256
    assert args.gamma == 1.0
    assert args.ppo_learning_rate == 0.01
    assert args.ppo_critic_learning_rate == 2.5e-4
    assert args.ppo_anneal_lr is False
    assert args.ppo_gae_lambda == 1.0
    assert args.ppo_ent_coef == 0.0


def test_reproduction_script_runs_fixed_rule_experiment():
    args = ppo_example.Args(
        cuda=False,
        save=False,
        num_zones=3,
        horizon=2,
        batch_size=2,
        dm_suspicious_zones="0",
        dm_rho_grid_size=4,
        eval_episodes=4,
        num_training_iterations=1,
        num_envs=4,
        ppo_num_minibatches=2,
    )

    result, checkpoint_path = ppo_example.train(args)
    evaluation = ppo_example.evaluate_policy_mean_risk(args, result.designer.policy_net)

    assert checkpoint_path is None
    assert result.checkpoint["algorithm"] == "ppo_fixed_conservative_dm"
    assert result.checkpoint["config"] == ppo_example.asdict(args)
    assert torch.isfinite(torch.tensor(evaluation["mean_risk"]))
