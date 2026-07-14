"""RL-BOED training with Proximal Policy Optimization."""

from __future__ import annotations

import random
import time

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from testbed.core.setup import setup
from testbed.design_nets import CategoricalPolicyNet, RLBOEDValueNetwork
from testbed.designers import PolicyDesigner
from testbed.envs import BEDEnv, RLBOEDDenseReward
from testbed.envs.design_spaces import (
    CategoricalDesignSpace,
    DesignSpace,
    IntegerDesignSpace,
    ProductDesignSpace,
)
from testbed.envs.history import History
from testbed.tasks.base import Task


def _set_seed(seed: int, torch_deterministic: bool) -> None:
    """Sets Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def _is_supported_design_space(design_space: DesignSpace) -> bool:
    """Checks whether a design space can use a factorized categorical policy."""
    if isinstance(design_space, (CategoricalDesignSpace, IntegerDesignSpace)):
        return True
    if isinstance(design_space, ProductDesignSpace):
        return all(_is_supported_design_space(space) for space in design_space.spaces)
    return False


def validate_config(task: Task, cfg, design_space: DesignSpace) -> None:
    """Rejects unsupported tasks and invalid settings."""
    ppo = cfg.ppo
    positive_ints = {
        "ppo.num_training_iterations": ppo.num_training_iterations,
        "ppo.num_envs": ppo.num_envs,
        "ppo.num_minibatches": ppo.num_minibatches,
        "ppo.update_epochs": ppo.update_epochs,
        "ppo.log_frequency": ppo.log_frequency,
        "T": cfg.T,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if task.implicit_likelihood:
        raise ValueError("PPO requires a task with an explicit likelihood.")
    if not _is_supported_design_space(design_space):
        raise TypeError(
            "PPO requires CategoricalDesignSpace, IntegerDesignSpace, "
            "or an all-discrete ProductDesignSpace."
        )
    if ppo.num_envs * cfg.T % ppo.num_minibatches != 0:
        raise ValueError("ppo.num_envs * T must be divisible by ppo.num_minibatches.")
    if cfg.reward.n_contrastive < 0:
        raise ValueError("reward.n_contrastive must be non-negative.")
    if cfg.reward.outcome_low >= cfg.reward.outcome_high:
        raise ValueError("reward.outcome_low must be less than reward.outcome_high.")
    if not 0.0 <= ppo.gae_lambda <= 1.0:
        raise ValueError("ppo.gae_lambda must be between 0 and 1.")
    if ppo.learning_rate <= 0:
        raise ValueError("ppo.learning_rate must be positive.")
    if ppo.actor_learning_rate is not None and ppo.actor_learning_rate <= 0:
        raise ValueError("ppo.actor_learning_rate must be positive when set.")
    if ppo.critic_learning_rate is not None and ppo.critic_learning_rate <= 0:
        raise ValueError("ppo.critic_learning_rate must be positive when set.")
    if ppo.clip_coef <= 0:
        raise ValueError("ppo.clip_coef must be positive.")
    if ppo.ent_coef < 0 or ppo.vf_coef < 0:
        raise ValueError("ppo.ent_coef and ppo.vf_coef must be non-negative.")
    if ppo.max_grad_norm <= 0:
        raise ValueError("ppo.max_grad_norm must be positive.")
    if ppo.target_kl is not None and ppo.target_kl <= 0:
        raise ValueError("ppo.target_kl must be positive when set.")


def make_networks(task: Task, design_space: DesignSpace, cfg):
    """Builds the categorical policy and state-value network."""
    policy = CategoricalPolicyNet(
        design_space,
        dim_y=task.event_shape_y.numel(),
        outcome_bounds=(cfg.reward.outcome_low, cfg.reward.outcome_high),
    ).to(task.device)
    value = RLBOEDValueNetwork(
        dim_y=task.event_shape_y.numel(),
        dim_design=task.event_shape_design.numel(),
        design_low=policy.design_low,
        design_high=policy.design_high,
        outcome_low=cfg.reward.outcome_low,
        outcome_high=cfg.reward.outcome_high,
    ).to(task.device)
    return policy, value


def build_env(cfg, task, design_space, *, n_contrastive=None):
    """Builds the vectorized BED environment used by PPO."""
    if n_contrastive is None:
        n_contrastive = cfg.reward.n_contrastive
    return BEDEnv(
        task,
        horizon=cfg.T,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(n_contrastive),
        batch_shape=(cfg.ppo.num_envs,),
    )


def _clone_history(history: History) -> History:
    """Copies a history before the environment mutates it."""
    return History(
        max_episode_length=history.max_episode_length,
        designs=history.designs.detach().clone(),
        outcomes=history.outcomes.detach().clone(),
        mask=history.mask.detach().clone(),
    )


def _flatten_histories(histories: list[History]) -> History:
    """Flattens time and environment axes into one batch axis."""
    designs = torch.stack([history.designs for history in histories])
    outcomes = torch.stack([history.outcomes for history in histories])
    mask = torch.stack([history.mask for history in histories])
    batch_size = designs.shape[0] * designs.shape[1]
    return History(
        max_episode_length=histories[0].max_episode_length,
        designs=designs.reshape(batch_size, *designs.shape[2:]),
        outcomes=outcomes.reshape(batch_size, *outcomes.shape[2:]),
        mask=mask.reshape(batch_size, *mask.shape[2:]),
    )


def _index_history(history: History, indices: torch.Tensor) -> History:
    """Selects a minibatch from a flattened history."""
    return History(
        max_episode_length=history.max_episode_length,
        designs=history.designs[indices],
        outcomes=history.outcomes[indices],
        mask=history.mask[indices],
    )


class Trainer:
    """Training class for RL-BOED PPO."""

    def __init__(self, task, designer, env, value_net, cfg, manager=None):
        """Initializes the trainer."""
        if not isinstance(designer, PolicyDesigner):
            raise TypeError("PPO Trainer requires a PolicyDesigner.")
        self.task = task
        self.designer = designer
        self.policy = designer.policy_net
        self.env = env
        self.value_net = value_net
        self.cfg = cfg
        self.manager = manager
        self.metrics: list[dict[str, float]] = []
        self.env_steps = 0
        self.completed_episodes = 0
        self._start_time = time.time()

        ppo = cfg.ppo
        self.actor_parameters = list(self.policy.parameters())
        self.critic_parameters = list(value_net.parameters())
        self.separate_optimizers = (
            ppo.actor_learning_rate is not None or ppo.critic_learning_rate is not None
        )
        if self.separate_optimizers:
            self.actor_optimizer = optim.Adam(
                self.actor_parameters,
                lr=ppo.actor_learning_rate or ppo.learning_rate,
                eps=1e-5,
            )
            self.critic_optimizer = optim.Adam(
                self.critic_parameters,
                lr=ppo.critic_learning_rate or ppo.learning_rate,
                eps=1e-5,
            )
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(
                self.actor_parameters + self.critic_parameters,
                lr=ppo.learning_rate,
                eps=1e-5,
            )
            self.actor_optimizer = None
            self.critic_optimizer = None

    def train(self):
        """Runs PPO training and saves the final checkpoint."""
        self.policy.train()
        self.value_net.train()
        self._start_time = time.time()
        for step in range(self.cfg.ppo.num_training_iterations):
            metrics = self.train_step(step)
            self.metrics.extend(metrics)
            if self.manager and self.manager.metric_logger:
                for item in metrics:
                    self.manager.metric_logger.log(item)
            if self.manager and step % self.cfg.ppo.log_frequency == 0:
                latest = metrics[-1]
                self.manager.event_logger.info(
                    f"iteration={step}, env_steps={self.env_steps}, "
                    f"return={latest['episodic_return']:.4f}, "
                    f"policy_loss={latest['policy_loss']:.4f}, "
                    f"value_loss={latest['value_loss']:.4f}"
                )

        if self.manager is not None:
            torch.save(
                self.build_checkpoint(step=step), self.manager.path("final_step.pt")
            )
            if self.manager.metric_logger is not None:
                self.manager.metric_logger.close()

    def train_step(self, training_iteration: int) -> list[dict[str, float]]:
        """Collects one rollout and applies PPO minibatch updates."""
        cfg = self.cfg
        ppo = cfg.ppo
        horizon = self.env.horizon
        batch_size = ppo.num_envs * horizon
        minibatch_size = batch_size // ppo.num_minibatches

        if ppo.anneal_lr:
            fraction = 1.0 - training_iteration / ppo.num_training_iterations
            if self.separate_optimizers:
                self.actor_optimizer.param_groups[0]["lr"] = fraction * (
                    ppo.actor_learning_rate or ppo.learning_rate
                )
                self.critic_optimizer.param_groups[0]["lr"] = fraction * (
                    ppo.critic_learning_rate or ppo.learning_rate
                )
            else:
                self.optimizer.param_groups[0]["lr"] = fraction * ppo.learning_rate

        observation, _ = self.env.reset(
            seed=cfg.seed if training_iteration == 0 else None
        )
        next_done = torch.zeros(ppo.num_envs, device=self.task.device)
        observations = []
        designs = []
        log_probs = []
        rewards = torch.zeros(horizon, ppo.num_envs, device=self.task.device)
        dones = torch.zeros_like(rewards)
        values = torch.zeros_like(rewards)
        episodic_returns = torch.zeros(ppo.num_envs, device=self.task.device)

        for timestep in range(horizon):
            observations.append(_clone_history(observation))
            dones[timestep] = next_done
            with torch.no_grad():
                sample = self.policy.sample_from_observation(observation)
                values[timestep] = self.value_net(observation).flatten()
            designs.append(sample.design.detach())
            log_probs.append(sample.log_prob.detach())
            observation, reward, terminated, truncated, _ = self.env.step(sample.design)
            next_done = (terminated | truncated).float()
            rewards[timestep] = reward
            episodic_returns += reward
            self.env_steps += ppo.num_envs

        with torch.no_grad():
            next_value = self.value_net(observation).flatten()
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for timestep in reversed(range(horizon)):
                if timestep == horizon - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[timestep + 1]
                    next_values = values[timestep + 1]
                delta = (
                    rewards[timestep]
                    + ppo.gamma * next_values * next_nonterminal
                    - values[timestep]
                )
                last_gae = (
                    delta + ppo.gamma * ppo.gae_lambda * next_nonterminal * last_gae
                )
                advantages[timestep] = last_gae
            returns = advantages + values

        flat_observation = _flatten_histories(observations)
        flat_designs = torch.stack(designs).reshape(
            batch_size, *self.task.event_shape_design
        )
        old_log_probs = torch.stack(log_probs).reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        old_values = values.reshape(-1)
        indices = np.arange(batch_size)
        clip_fractions = []

        for _ in range(ppo.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                minibatch = torch.as_tensor(
                    indices[start : start + minibatch_size], device=self.task.device
                )
                minibatch_observation = _index_history(flat_observation, minibatch)
                new_log_prob, entropy = self.policy.evaluate_designs_from_observation(
                    minibatch_observation, flat_designs[minibatch]
                )
                new_value = self.value_net(minibatch_observation).view(-1)
                log_ratio = new_log_prob - old_log_probs[minibatch]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > ppo.clip_coef).float().mean().item()
                    )

                minibatch_advantages = flat_advantages[minibatch]
                if ppo.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std(unbiased=False) + 1e-8)
                policy_loss = torch.max(
                    -minibatch_advantages * ratio,
                    -minibatch_advantages
                    * torch.clamp(ratio, 1 - ppo.clip_coef, 1 + ppo.clip_coef),
                ).mean()

                if ppo.clip_vloss:
                    value_loss_unclipped = (new_value - flat_returns[minibatch]) ** 2
                    value_clipped = old_values[minibatch] + torch.clamp(
                        new_value - old_values[minibatch],
                        -ppo.clip_coef,
                        ppo.clip_coef,
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            value_loss_unclipped,
                            (value_clipped - flat_returns[minibatch]) ** 2,
                        ).mean()
                    )
                else:
                    value_loss = (
                        0.5 * ((new_value - flat_returns[minibatch]) ** 2).mean()
                    )
                entropy_loss = entropy.mean()

                if self.separate_optimizers:
                    self.actor_optimizer.zero_grad()
                    (policy_loss - ppo.ent_coef * entropy_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters, ppo.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    (ppo.vf_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_parameters, ppo.max_grad_norm
                    )
                    self.critic_optimizer.step()
                else:
                    loss = (
                        policy_loss
                        - ppo.ent_coef * entropy_loss
                        + ppo.vf_coef * value_loss
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters + self.critic_parameters,
                        ppo.max_grad_norm,
                    )
                    self.optimizer.step()
            if ppo.target_kl is not None and approx_kl > ppo.target_kl:
                break

        self.completed_episodes += ppo.num_envs
        target = flat_returns.detach().cpu().numpy()
        prediction = old_values.detach().cpu().numpy()
        variance = np.var(target)
        explained_variance = (
            np.nan if variance == 0 else 1 - np.var(target - prediction) / variance
        )
        metrics = {
            "step": float(training_iteration),
            "env_steps": float(self.env_steps),
            "episodes": float(self.completed_episodes),
            "sps": float(
                int(self.env_steps / max(time.time() - self._start_time, 1e-12))
            ),
            "episodic_return": episodic_returns.mean().item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": float(np.mean(clip_fractions)),
            "explained_variance": float(explained_variance),
        }
        return [metrics]

    def build_checkpoint(self, step=None):
        """Builds a checkpoint containing inference and training state."""
        checkpoint = {
            "algorithm": "ppo",
            "config": OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=False
            ),
            "horizon": self.env.horizon,
            "designer": self.policy.state_dict(),
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": None
            if self.optimizer is None
            else self.optimizer.state_dict(),
            "actor_optimizer": (
                None
                if self.actor_optimizer is None
                else self.actor_optimizer.state_dict()
            ),
            "critic_optimizer": (
                None
                if self.critic_optimizer is None
                else self.critic_optimizer.state_dict()
            ),
            "env_steps": self.env_steps,
            "training_iterations": self.cfg.ppo.num_training_iterations,
        }
        if step is not None:
            checkpoint["step"] = step
        return checkpoint


@hydra.main(version_base="1.3", config_path="../recipes", config_name="ppo")
def main(cfg):
    """Entry point for RL-BOED PPO training."""
    manager, task = setup(cfg)
    design_space = task.design_space
    validate_config(task, cfg, design_space)
    _set_seed(cfg.seed, cfg.torch_deterministic)
    policy, value_net = make_networks(task, design_space, cfg)
    env = build_env(cfg, task, design_space)
    designer = PolicyDesigner(task, cfg.T, policy, deterministic=False)
    trainer = Trainer(task, designer, env, value_net, cfg, manager=manager)
    trainer.train()


if __name__ == "__main__":
    main()
