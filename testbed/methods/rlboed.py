"""RL-BOED training with Soft Actor-Critic."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from testbed.design_nets import (
    RLBOEDGaussianActor,
    RLBOEDQNetwork,
    TanhGaussianPolicyNet,
)
from testbed.designers import PolicyDesigner
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    PaddedHistoryObservation,
    RLBOEDDenseReward,
)
from testbed.envs.design_spaces import DesignSpace
from testbed.tasks.base_task import Task

from .base import FitResult


@dataclass
class RLBOEDConfig:
    """RL-BOED and SAC hyperparameters."""

    outcome_low: float
    outcome_high: float
    seed: int = 1
    torch_deterministic: bool = True

    n_contrastive: int = 100_000

    total_timesteps: int = 20_000
    num_envs: int = 100
    buffer_size: int = 2_000_000
    gamma: float = 0.9
    tau: float = 1e-3
    batch_size: int = 4096
    learning_starts: int = 1_000
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    log_frequency: int = 100


class RLBOED:
    """Trains a policy using RL-BOED's dense reward and SAC."""

    def __init__(self, config: RLBOEDConfig):
        """Initializes the method configuration."""
        self.config = config

    def fit(
        self,
        task: Task,
        *,
        horizon: int,
        design_space: DesignSpace,
    ) -> FitResult:
        """Trains and returns a deployable policy designer."""
        cfg = self.config
        self._validate(task, horizon, design_space)
        self._set_seed()

        device = task.device
        design_low = design_space.low.to(device)
        design_high = design_space.high.to(device)
        env = BEDEnv(
            task,
            horizon=horizon,
            design_space=design_space,
            reward_fn=RLBOEDDenseReward(cfg.n_contrastive),
            observation_fn=PaddedHistoryObservation(horizon),
            batch_shape=(cfg.num_envs,),
        )

        dim_y = task.event_shape_y.numel()
        dim_design = task.event_shape_design.numel()
        actor_module = RLBOEDGaussianActor(
            dim_y=dim_y,
            dim_design=dim_design,
        )
        actor = TanhGaussianPolicyNet(
            design_space,
            actor=actor_module,
            normalize_design_history=True,
            outcome_bounds=(cfg.outcome_low, cfg.outcome_high),
        ).to(device)

        critic_kwargs = {
            "dim_y": dim_y,
            "dim_design": dim_design,
            "design_low": design_low,
            "design_high": design_high,
            "outcome_low": cfg.outcome_low,
            "outcome_high": cfg.outcome_high,
        }
        qf1 = RLBOEDQNetwork(**critic_kwargs).to(device)
        qf2 = RLBOEDQNetwork(**critic_kwargs).to(device)
        qf1_target = RLBOEDQNetwork(**critic_kwargs).to(device)
        qf2_target = RLBOEDQNetwork(**critic_kwargs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        q_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()),
            lr=cfg.q_lr,
        )
        actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.policy_lr)

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(cfg.buffer_size, device="cpu"),
            batch_size=cfg.batch_size,
            pin_memory=device.type == "cuda",
        )

        metrics: list[dict[str, float]] = []
        start_time = time.time()
        obs, _ = env.reset(seed=cfg.seed)
        episodic_returns = torch.zeros(cfg.num_envs, device=device)
        completed_episodes = 0
        actor_loss = torch.tensor(float("nan"))

        for global_step in range(cfg.total_timesteps):
            if global_step < cfg.learning_starts:
                designs = design_space.sample(
                    (cfg.num_envs,),
                    device=device,
                )
            else:
                with torch.no_grad():
                    policy_sample = actor.sample_from_observation(obs)
                    designs = policy_sample.action
            replay_obs = {key: value.detach().clone() for key, value in obs.items()}
            next_obs, rewards, terminations, truncations, _ = env.step(designs)
            dones = terminations | truncations
            replay_buffer.extend(
                TensorDict(
                    {
                        "observation": replay_obs,
                        "next_observation": next_obs,
                        "action": designs,
                        "reward": rewards,
                        "done": dones,
                    },
                    batch_size=[cfg.num_envs],
                )
                .detach()
                .to("cpu")
            )

            episodic_returns += rewards
            if dones.any():
                completed_episodes += cfg.num_envs
                episode_metrics = {
                    "step": float(global_step),
                    "episodes": float(completed_episodes),
                    "episodic_return": episodic_returns.mean().item(),
                }
                metrics.append(episode_metrics)
                print(
                    f"global_step={global_step}, "
                    f"episodes={completed_episodes}, "
                    f"episodic_return={episode_metrics['episodic_return']:.4f}"
                )
                episodic_returns.zero_()
                next_obs, _ = env.reset()
            obs = next_obs

            if (
                global_step >= cfg.learning_starts
                and len(replay_buffer) >= cfg.batch_size
            ):
                data = replay_buffer.sample().to(device, non_blocking=True)
                data_obs = data["observation"]
                data_next_obs = data["next_observation"]
                data_actions = data["action"]
                data_rewards = data["reward"]
                data_dones = data["done"]

                with torch.no_grad():
                    next_policy_sample = actor.sample_from_observation(data_next_obs)
                    next_actions = next_policy_sample.action
                    next_log_pi = next_policy_sample.log_prob.unsqueeze(-1)
                    qf1_next_target = qf1_target(data_next_obs, next_actions)
                    qf2_next_target = qf2_target(data_next_obs, next_actions)
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - cfg.alpha * next_log_pi
                    )
                    next_q_value = data_rewards + (
                        1.0 - data_dones.float()
                    ) * cfg.gamma * min_qf_next_target.view(-1)

                qf1_a_values = qf1(data_obs, data_actions).view(-1)
                qf2_a_values = qf2(data_obs, data_actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % cfg.policy_frequency == 0:
                    for _ in range(cfg.policy_frequency):
                        policy_sample = actor.sample_from_observation(data_obs)
                        pi = policy_sample.action
                        log_pi = policy_sample.log_prob.unsqueeze(-1)
                        min_qf_pi = torch.min(
                            qf1(data_obs, pi),
                            qf2(data_obs, pi),
                        )
                        actor_loss = (cfg.alpha * log_pi - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                if global_step % cfg.target_network_frequency == 0:
                    self._soft_update(qf1, qf1_target)
                    self._soft_update(qf2, qf2_target)

                if global_step % cfg.log_frequency == 0:
                    step_metrics = {
                        "step": float(global_step),
                        "sps": float(
                            int((global_step + 1) / (time.time() - start_time))
                        ),
                        "qf_loss": qf_loss.item() / 2.0,
                        "actor_loss": actor_loss.item(),
                        "alpha": cfg.alpha,
                    }
                    metrics.append(step_metrics)
                    message = (
                        f"SPS={int(step_metrics['sps'])}, "
                        f"qf_loss={step_metrics['qf_loss']:.4f}, "
                        f"actor_loss={step_metrics['actor_loss']:.4f}, "
                        f"alpha={cfg.alpha:.4f}"
                    )
                    print(message)

        actor.eval()
        designer = PolicyDesigner(task, horizon, actor)
        checkpoint = {
            "config": asdict(cfg),
            "horizon": horizon,
            "design_low": design_low.detach().cpu(),
            "design_high": design_high.detach().cpu(),
            "outcome_low": torch.as_tensor(cfg.outcome_low),
            "outcome_high": torch.as_tensor(cfg.outcome_high),
            # The trained actor is the deployable designer, so its weights are
            # stored under both keys: "designer" for inference-only loading and
            # "actor" for resuming SAC training alongside the critic/optimizers.
            "designer": actor.state_dict(),
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "qf1_target": qf1_target.state_dict(),
            "qf2_target": qf2_target.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "alpha": cfg.alpha,
            "global_step": cfg.total_timesteps,
        }
        return FitResult(
            designer=designer,
            metrics=metrics,
            checkpoint=checkpoint,
        )

    def _soft_update(self, source, target) -> None:
        """Polyak-updates a target Q network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data
                + (1.0 - self.config.tau) * target_param.data
            )

    def _set_seed(self) -> None:
        """Sets random seeds to match the reference script."""
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

    def _validate(
        self,
        task: Task,
        horizon: int,
        design_space: DesignSpace,
    ) -> None:
        """Rejects unsupported tasks and invalid settings."""
        cfg = self.config
        positive_ints = {
            "total_timesteps": cfg.total_timesteps,
            "num_envs": cfg.num_envs,
            "buffer_size": cfg.buffer_size,
            "batch_size": cfg.batch_size,
            "horizon": horizon,
            "log_frequency": cfg.log_frequency,
            "policy_frequency": cfg.policy_frequency,
            "target_network_frequency": cfg.target_network_frequency,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if cfg.n_contrastive < 0:
            raise ValueError("n_contrastive must be non-negative.")
        if cfg.outcome_low >= cfg.outcome_high:
            raise ValueError("outcome_low must be less than outcome_high.")
        if cfg.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if task.likelihood_type != "explicit":
            raise ValueError("RLBOED requires a task with an explicit likelihood.")

        if not isinstance(design_space, BoxDesignSpace):
            raise TypeError("This RLBOED SAC implementation requires a BoxDesignSpace.")
