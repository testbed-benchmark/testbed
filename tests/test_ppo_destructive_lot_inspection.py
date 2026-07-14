"""Tests for PPO on destructive lot inspection."""

import torch
from omegaconf import OmegaConf

from testbed.design_nets import CategoricalPolicyNet, RLBOEDValueNetwork
from testbed.designers import PolicyDesigner
from testbed.envs import CategoricalDesignSpace
from testbed.envs.history import History
from testbed.methods.ppo import Trainer, build_env, make_networks, validate_config
from testbed.tasks import DestructiveLotInspection


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
