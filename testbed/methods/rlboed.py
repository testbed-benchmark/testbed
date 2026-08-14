"""RL-BOED training with Soft Actor-Critic."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from math import ceil, log

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import optim

from testbed.core.design_spaces import (
    BoxDesignSpace,
    DefaultDesignSpace,
    DesignSpace,
    FiniteDesignSpace,
)
from testbed.core.setup import setup
from testbed.design_nets import (
    CategoricalPolicyNet,
    DeepSetCategoricalActor,
    DeepSetGaussianActor,
    GaussianPolicyNet,
    RLBOEDDiscreteQNetwork,
    RLBOEDQNetwork,
    StochasticDesignNet,
)
from testbed.designers import PolicyDesigner, build_policy_designer
from testbed.envs import BEDEnv
from testbed.envs.history import History
from testbed.envs.rewards import RLBOEDDenseReward
from testbed.tasks.base import Task


@dataclass
class ReplayBatch:
    """A batch of transitions sampled from the replay buffer."""

    observation: dict[str, torch.Tensor]
    next_observation: dict[str, torch.Tensor]
    design: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


def _to_device(
    batch: ReplayBatch, device: torch.device, *, non_blocking: bool = False
) -> ReplayBatch:
    def move(x: torch.Tensor) -> torch.Tensor:
        return x.to(device, non_blocking=non_blocking)

    return ReplayBatch(
        observation={k: move(v) for k, v in batch.observation.items()},
        next_observation={k: move(v) for k, v in batch.next_observation.items()},
        design=move(batch.design),
        reward=move(batch.reward),
        done=move(batch.done),
    )


class _CompactEpisodeReplayBuffer:
    """Stores fixed-horizon episodes and reconstructs transitions on sample.

    This replay buffer makes it efficient to store sequences of fixed-length
    observations and designs, which are the inputs to the policy and Q networks.
    """

    def __init__(
        self,
        *,
        buffer_size: int,
        batch_size: int,
        horizon: int,
        event_shape_design: torch.Size,
        event_shape_y: torch.Size,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ):
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.horizon = int(horizon)
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

        self.capacity_episodes = ceil(self.buffer_size / self.horizon)
        self.event_shape_design = torch.Size(event_shape_design)
        self.event_shape_y = torch.Size(event_shape_y)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory) and self.device.type == "cpu"

        self.designs = torch.zeros(
            self.capacity_episodes,
            self.horizon,
            *self.event_shape_design,
            device=self.device,
        )
        self.outcomes = torch.zeros(
            self.capacity_episodes,
            self.horizon,
            *self.event_shape_y,
            device=self.device,
        )
        self.policy_designs = torch.zeros(
            self.capacity_episodes,
            self.horizon,
            *self.event_shape_design,
            device=self.device,
        )
        self.rewards = torch.zeros(
            self.capacity_episodes,
            self.horizon,
            device=self.device,
        )
        self.dones = torch.zeros(
            self.capacity_episodes,
            self.horizon,
            device=self.device,
            dtype=torch.bool,
        )

        self._next_episode = 0
        self._episodes_written = 0

    def __len__(self) -> int:
        return min(self._episodes_written * self.horizon, self.buffer_size)

    def extend_episode_batch(
        self,
        *,
        designs: torch.Tensor,
        outcomes: torch.Tensor,
        policy_designs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Stores a batch of complete episodes."""
        designs = designs.detach().to(self.device)
        outcomes = outcomes.detach().to(self.device)
        policy_designs = policy_designs.detach().to(self.device)
        rewards = rewards.detach().to(self.device)
        dones = dones.detach().to(self.device)

        batch_size = designs.shape[0]
        expected_designs = (batch_size, self.horizon, *self.event_shape_design)
        expected_outcomes = (batch_size, self.horizon, *self.event_shape_y)
        if designs.shape != expected_designs:
            raise ValueError(
                f"designs must have shape {expected_designs}; "
                f"got {tuple(designs.shape)}."
            )
        if outcomes.shape != expected_outcomes:
            raise ValueError(
                f"outcomes must have shape {expected_outcomes}; "
                f"got {tuple(outcomes.shape)}."
            )
        if policy_designs.shape != expected_designs:
            raise ValueError(
                f"policy_designs must have shape {expected_designs}; "
                f"got {tuple(policy_designs.shape)}."
            )
        if rewards.shape != (batch_size, self.horizon):
            raise ValueError(
                "rewards must have shape "
                f"{(batch_size, self.horizon)}; got {tuple(rewards.shape)}."
            )
        if dones.shape != (batch_size, self.horizon):
            raise ValueError(
                "dones must have shape "
                f"{(batch_size, self.horizon)}; got {tuple(dones.shape)}."
            )
        if batch_size > self.capacity_episodes:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer capacity ({self.capacity_episodes} episodes). "
                "Increase buffer_size or reduce num_envs."
            )

        slots = (
            self._next_episode + torch.arange(batch_size, device=self.device)
        ) % self.capacity_episodes
        self.designs[slots] = designs
        self.outcomes[slots] = outcomes
        self.policy_designs[slots] = policy_designs
        self.rewards[slots] = rewards
        self.dones[slots] = dones
        self._next_episode = (self._next_episode + batch_size) % self.capacity_episodes
        self._episodes_written += batch_size

    def sample(self) -> ReplayBatch:
        """Samples transitions and reconstructs padded-history observations."""
        n_transitions = len(self)
        if n_transitions < self.batch_size:
            raise RuntimeError("Cannot sample before the buffer has enough data.")

        offsets = torch.randint(n_transitions, (self.batch_size,), device=self.device)
        newest_flat_exclusive = self._episodes_written * self.horizon
        flat_indices = newest_flat_exclusive - n_transitions + offsets
        episode_numbers = flat_indices.div(self.horizon, rounding_mode="floor")
        timesteps = flat_indices.remainder(self.horizon)
        slots = episode_numbers.remainder(self.capacity_episodes)

        episode_designs = self.designs[slots]
        episode_outcomes = self.outcomes[slots]
        observation = self._build_observation(
            episode_designs, episode_outcomes, timesteps
        )
        next_observation = self._build_observation(
            episode_designs, episode_outcomes, timesteps + 1
        )
        data = ReplayBatch(
            observation=observation,
            next_observation=next_observation,
            design=self.policy_designs[slots, timesteps],
            reward=self.rewards[slots, timesteps],
            done=self.dones[slots, timesteps],
        )
        if self.pin_memory:
            data = ReplayBatch(
                observation={k: v.pin_memory() for k, v in data.observation.items()},
                next_observation={
                    k: v.pin_memory() for k, v in data.next_observation.items()
                },
                design=data.design.pin_memory(),
                reward=data.reward.pin_memory(),
                done=data.done.pin_memory(),
            )
        return data

    def _build_observation(
        self,
        episode_designs: torch.Tensor,
        episode_outcomes: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Reconstructs padded observations for sampled episode prefixes."""
        batch_size = episode_designs.shape[0]
        designs = torch.zeros(
            batch_size,
            self.horizon,
            *self.event_shape_design,
            device=self.device,
            dtype=self.designs.dtype,
        )
        outcomes = torch.zeros(
            batch_size,
            self.horizon,
            *self.event_shape_y,
            device=self.device,
            dtype=self.outcomes.dtype,
        )
        mask = torch.arange(self.horizon, device=self.device).unsqueeze(0) < (
            history_lengths.unsqueeze(1)
        )
        if mask.any():
            designs[mask] = episode_designs[mask]
            outcomes[mask] = episode_outcomes[mask]
        return {
            "designs": designs,
            "outcomes": outcomes,
            "mask": mask,
        }


def _resolve_buffer_device(buffer_device: str) -> torch.device:
    """Returns the replay storage device requested by the config."""
    requested = buffer_device.strip().lower()
    try:
        resolved = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(
            "buffer_device must be 'cpu', 'cuda', or 'cuda:<index>'."
        ) from error

    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("buffer_device must be 'cpu', 'cuda', or 'cuda:<index>'.")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("buffer_device='cuda' requires CUDA to be available.")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"buffer_device='{resolved}' is outside the available "
                f"CUDA device count ({torch.cuda.device_count()})."
            )
    return resolved


def _set_seed(seed: int, torch_deterministic: bool) -> None:
    """Sets random seeds to match the reference script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Polyak-updates a target network toward a source network."""
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def validate_config(task: Task, cfg, design_space: DesignSpace) -> None:
    """Rejects unsupported tasks and invalid settings."""
    sac = cfg.sac
    positive_ints = {
        "sac.num_training_iterations": sac.num_training_iterations,
        "sac.num_envs": sac.num_envs,
        "sac.buffer_size": sac.buffer_size,
        "sac.batch_size": sac.batch_size,
        "sac.gradient_steps_per_iteration": sac.gradient_steps_per_iteration,
        "T": cfg.T,
        "sac.log_frequency": sac.log_frequency,
        "sac.policy_frequency": sac.policy_frequency,
        "sac.target_network_frequency": sac.target_network_frequency,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if cfg.reward.n_contrastive < 0:
        raise ValueError("reward.n_contrastive must be non-negative.")
    if sac.buffer_size < sac.learning_starts:
        raise ValueError("sac.buffer_size must be at least sac.learning_starts.")
    if cfg.reward.outcome_low >= cfg.reward.outcome_high:
        raise ValueError("reward.outcome_low must be less than reward.outcome_high.")
    if sac.alpha is not None and sac.alpha < 0:
        raise ValueError("sac.alpha must be non-negative when fixed.")
    if sac.initial_alpha <= 0:
        raise ValueError("sac.initial_alpha must be positive.")
    if sac.learning_starts < 0:
        raise ValueError("sac.learning_starts must be non-negative.")
    if sac.entropy_anneal_rate < 0:
        raise ValueError("sac.entropy_anneal_rate must be non-negative.")
    if sac.policy_init_std <= 0:
        raise ValueError("sac.policy_init_std must be positive.")
    if task.implicit_likelihood:
        raise ValueError("RLBOED requires a task with an explicit likelihood.")
    if not isinstance(design_space, (BoxDesignSpace, FiniteDesignSpace)):
        raise TypeError(
            "This RLBOED SAC implementation requires a BoxDesignSpace or "
            "FiniteDesignSpace."
        )


@dataclass
class _SACNetworks:
    actor: StochasticDesignNet
    design_space: DesignSpace
    qf1: RLBOEDQNetwork | RLBOEDDiscreteQNetwork
    qf2: RLBOEDQNetwork | RLBOEDDiscreteQNetwork
    qf1_target: RLBOEDQNetwork | RLBOEDDiscreteQNetwork
    qf2_target: RLBOEDQNetwork | RLBOEDDiscreteQNetwork
    q_optimizer: optim.Adam
    actor_optimizer: optim.Adam
    log_alpha: torch.Tensor
    alpha_optimizer: optim.Adam | None


def make_networks(
    task: Task,
    design_space: DesignSpace,
    cfg,
    device: torch.device,
) -> _SACNetworks:
    """Builds the SAC actor, twin critics, target networks, optimizers, and alpha."""
    sac = cfg.sac
    event_shape_y = task.event_shape_y
    event_shape_design = task.event_shape_design
    dim_y = event_shape_y.numel()
    dim_design = event_shape_design.numel()

    if isinstance(design_space, BoxDesignSpace):
        actor_module = DeepSetGaussianActor(
            event_shape_y=event_shape_y,
            event_shape_design=event_shape_design,
            init_std=sac.policy_init_std,
        )
        actor = GaussianPolicyNet(
            design_space,
            actor=actor_module,
            normalize_design_history=True,
            design_history_bounds=(0.0, 1.0),
            outcome_bounds=(cfg.reward.outcome_low, cfg.reward.outcome_high),
        ).to(device)
        critic_kwargs = {
            "dim_y": dim_y,
            "dim_design": dim_design,
            "design_low": design_space.low.to(device),
            "design_high": design_space.high.to(device),
            "outcome_low": cfg.reward.outcome_low,
            "outcome_high": cfg.reward.outcome_high,
        }
        qf1 = RLBOEDQNetwork(**critic_kwargs).to(device)
        qf2 = RLBOEDQNetwork(**critic_kwargs).to(device)
    else:
        values = design_space.values.reshape(design_space.n_designs, dim_design)
        actor_module = DeepSetCategoricalActor(
            event_shape_y=event_shape_y,
            event_shape_design=event_shape_design,
            n_designs=design_space.n_designs,
        )
        actor = CategoricalPolicyNet(
            design_space,
            actor=actor_module,
            normalize_design_history=True,
            design_history_bounds=(0.0, 1.0),
            outcome_bounds=(cfg.reward.outcome_low, cfg.reward.outcome_high),
        ).to(device)
        critic_kwargs = {
            "dim_y": dim_y,
            "dim_design": dim_design,
            "n_designs": design_space.n_designs,
            "design_low": values.amin(dim=0).to(device),
            "design_high": values.amax(dim=0).to(device),
            "outcome_low": cfg.reward.outcome_low,
            "outcome_high": cfg.reward.outcome_high,
        }
        qf1 = RLBOEDDiscreteQNetwork(**critic_kwargs).to(device)
        qf2 = RLBOEDDiscreteQNetwork(**critic_kwargs).to(device)
    qf1_target = copy.deepcopy(qf1)
    qf2_target = copy.deepcopy(qf2)

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=sac.q_lr,
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=sac.policy_lr)

    if sac.alpha is None:
        log_alpha = torch.tensor(
            log(sac.initial_alpha),
            device=device,
            requires_grad=True,
        )
        alpha_optimizer: optim.Adam | None = optim.Adam([log_alpha], lr=sac.policy_lr)
    else:
        fixed_log_alpha = float("-inf") if sac.alpha == 0 else log(sac.alpha)
        log_alpha = torch.tensor(fixed_log_alpha, device=device)
        alpha_optimizer = None

    return _SACNetworks(
        actor=actor,
        design_space=design_space,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
    )


@dataclass
class _EpisodeBatch:
    designs: torch.Tensor
    outcomes: torch.Tensor
    policy_designs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    episodic_return: torch.Tensor


def _propose_design(
    actor: StochasticDesignNet,
    design_space: DesignSpace,
    obs: History,
    *,
    reparameterize_policy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a raw policy proposal and maps it into the environment space."""
    proposal = actor.propose(
        obs.outcomes,
        obs.designs,
        mask=obs.mask,
        reparameterize_policy=reparameterize_policy,
    )
    if proposal.log_prob is None:
        raise RuntimeError("A stochastic RLBOED actor must return a log probability.")
    return design_space.to_design(proposal.raw_design, proposal.log_prob)


def collect_episode_batch(
    env: BEDEnv,
    actor: StochasticDesignNet,
    task: Task,
    cfg,
    *,
    seed: int | None = None,
) -> _EpisodeBatch:
    """Rolls out the actor for one episode batch and returns the transitions."""
    horizon = env.horizon
    device = task.device

    obs, _ = env.reset(seed=seed)

    episodic_returns = torch.zeros(cfg.sac.num_envs, device=device)
    episode_policy_designs = []
    episode_rewards = []
    episode_dones = []

    for _ in range(horizon):
        with torch.no_grad():
            designs, _ = _propose_design(actor, env.design_space, obs)
        next_obs, rewards, terminations, truncations, _ = env.step(designs)

        dones = terminations | truncations
        episode_policy_designs.append(designs.detach())
        episode_rewards.append(rewards.detach())
        episode_dones.append(dones.detach())
        episodic_returns += rewards
        obs = next_obs

    return _EpisodeBatch(
        designs=env.history.designs,
        outcomes=env.history.outcomes,
        policy_designs=torch.stack(episode_policy_designs, dim=1),
        rewards=torch.stack(episode_rewards, dim=1),
        dones=torch.stack(episode_dones, dim=1),
        episodic_return=episodic_returns.mean(),
    )


@dataclass
class _SACLosses:
    qf_loss: torch.Tensor
    actor_loss: torch.Tensor | None
    alpha_loss: torch.Tensor | None
    optimization_step: int


def sac_update(
    replay_buffer: _CompactEpisodeReplayBuffer,
    nets: _SACNetworks,
    target_entropy: float,
    device: torch.device,
    horizon: int,
    cfg,
    optimization_step: int,
) -> _SACLosses:
    """Performs one SAC gradient step and returns the losses."""
    data = _to_device(replay_buffer.sample(), device, non_blocking=True)
    data_obs = History(
        max_episode_length=horizon,
        designs=data.observation["designs"],
        outcomes=data.observation["outcomes"],
        mask=data.observation["mask"],
    )
    data_next_obs = History(
        max_episode_length=horizon,
        designs=data.next_observation["designs"],
        outcomes=data.next_observation["outcomes"],
        mask=data.next_observation["mask"],
    )
    data_designs = data.design
    data_rewards = data.reward
    data_dones = data.done
    alpha = nets.log_alpha.exp().detach()

    is_categorical = isinstance(nets.actor, CategoricalPolicyNet)
    with torch.no_grad():
        if is_categorical:
            next_probs, next_log_probs = (
                nets.actor.design_probabilities_from_observation(data_next_obs)
            )
            min_qf_next_target = torch.min(
                nets.qf1_target(data_next_obs), nets.qf2_target(data_next_obs)
            )
            next_value = (
                next_probs * (min_qf_next_target - alpha * next_log_probs)
            ).sum(dim=-1)
        else:
            next_designs, next_log_prob = _propose_design(
                nets.actor,
                nets.design_space,
                data_next_obs,
                reparameterize_policy=True,
            )
            next_log_pi = next_log_prob.unsqueeze(-1)
            qf1_next_target = nets.qf1_target(data_next_obs, next_designs)
            qf2_next_target = nets.qf2_target(data_next_obs, next_designs)
            next_value = (
                torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
            ).view(-1)
        next_q_value = (
            data_rewards + (1.0 - data_dones.float()) * cfg.sac.gamma * next_value
        )

    if is_categorical:
        design_indices = (
            nets.actor.design_space.encode(data_designs).long().unsqueeze(-1)
        )
        qf1_a_values = nets.qf1(data_obs).gather(1, design_indices).squeeze(-1)
        qf2_a_values = nets.qf2(data_obs).gather(1, design_indices).squeeze(-1)
    else:
        qf1_a_values = nets.qf1(data_obs, data_designs).view(-1)
        qf2_a_values = nets.qf2(data_obs, data_designs).view(-1)
    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    nets.q_optimizer.zero_grad()
    qf_loss.backward()
    nets.q_optimizer.step()

    actor_loss = None
    alpha_loss = None

    if optimization_step % cfg.sac.policy_frequency == 0:
        if is_categorical:
            probs, log_probs = nets.actor.design_probabilities_from_observation(
                data_obs
            )
            min_qf_pi = torch.min(nets.qf1(data_obs), nets.qf2(data_obs))
            actor_loss = (probs * (alpha * log_probs - min_qf_pi)).sum(dim=-1).mean()
        else:
            pi, policy_log_prob = _propose_design(
                nets.actor,
                nets.design_space,
                data_obs,
                reparameterize_policy=True,
            )
            log_pi = policy_log_prob.unsqueeze(-1)
            min_qf_pi = torch.min(nets.qf1(data_obs, pi), nets.qf2(data_obs, pi))
            actor_loss = (alpha * log_pi - min_qf_pi).mean()

        nets.actor_optimizer.zero_grad()
        actor_loss.backward()
        nets.actor_optimizer.step()

        if nets.alpha_optimizer is not None:
            if is_categorical:
                entropy_error = (
                    (probs * (log_probs + target_entropy)).sum(dim=-1).detach()
                )
                alpha_loss = -(nets.log_alpha.exp() * entropy_error).mean()
            else:
                alpha_loss = -(
                    nets.log_alpha.exp() * (log_pi.detach() + target_entropy)
                ).mean()
            nets.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nets.alpha_optimizer.step()

    if optimization_step % cfg.sac.target_network_frequency == 0:
        _soft_update(nets.qf1, nets.qf1_target, cfg.sac.tau)
        _soft_update(nets.qf2, nets.qf2_target, cfg.sac.tau)

    return _SACLosses(
        qf_loss=qf_loss,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        optimization_step=optimization_step + 1,
    )


# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for RL-BOED (SAC) training."""

    def __init__(self, task, designer, env, replay_buffer, nets, cfg, manager=None):
        """Initializes the trainer."""
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.env = env
        self.replay_buffer = replay_buffer
        self.nets = nets
        self.cfg = cfg
        self.manager = manager

        self.device = task.device
        if cfg.sac.target_entropy is None:
            self.target_entropy = (
                0.98 * log(self.nets.actor.design_space.n_designs)
                if isinstance(self.nets.actor, CategoricalPolicyNet)
                else -float(task.event_shape_design.numel())
            )
        else:
            self.target_entropy = float(cfg.sac.target_entropy)

        # logging / checkpointing
        self.metrics: list[dict[str, float]] = []
        self.best_metric = -torch.inf

        # training counters
        self.optimization_step = 0
        self.env_steps = 0
        self.completed_episodes = 0
        self._first_collect = True
        self._start_time = time.time()

        # most recent losses, surfaced in the periodic log
        self._qf_loss = torch.tensor(float("nan"))
        self._actor_loss = torch.tensor(float("nan"))
        self._alpha_loss = torch.tensor(float("nan"))

    def train(self):
        """Full training loop."""
        self.designer.design_net.train()
        self._start_time = time.time()

        for step in range(self.cfg.sac.num_training_iterations):
            metrics = self.train_step()
            metrics["step"] = step
            self.metrics.append(metrics)

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step
                self.metrics.append(eval_metrics)

                if self.manager:
                    self.manager.event_logger.info(
                        f"[Eval @ step {step}/{self.cfg.sac.num_training_iterations}]: "
                        f"spce={eval_metrics['eval/spce']:.4f} "
                        f"± {eval_metrics['eval/spce_se']:.4f}"
                    )
                    if self.manager.metric_logger is not None:
                        self.manager.metric_logger.log(eval_metrics)

            # periodic loss logging
            if step % self.cfg.sac.log_frequency == 0:
                step_metrics = self._loss_metrics(step)
                self.metrics.append(step_metrics)

                if self.manager and self.manager.metric_logger:
                    self.manager.metric_logger.log(step_metrics)
                if self.manager:
                    self.manager.event_logger.info(
                        f"iteration={step}, env_steps={self.env_steps}, "
                        f"return={metrics['episodic_return']:.4f}, "
                        f"qf_loss={step_metrics['qf_loss']:.4f}, "
                        f"actor_loss={step_metrics['actor_loss']:.4f}, "
                        f"alpha={step_metrics['alpha']:.4f}"
                    )

            self._maybe_save_best(step, metrics)

        if self.manager is not None:
            save_path = self.manager.path("final_step.pt")
            torch.save(self.build_checkpoint(step=step), save_path)
            if self.manager.metric_logger is not None:
                self.manager.metric_logger.close()

    def train_step(self):
        """Takes a single training iteration (collect + gradient steps)."""
        cfg = self.cfg
        sac = cfg.sac
        horizon = self.env.horizon
        transitions_per_episode_batch = sac.num_envs * horizon

        # During warm-up, collect enough episode batches to reach learning_starts.
        episode_batches = 1
        if len(self.replay_buffer) < sac.learning_starts:
            episode_batches = ceil(
                (sac.learning_starts - len(self.replay_buffer))
                / transitions_per_episode_batch
            )

        iteration_returns = []
        for _ in range(episode_batches):
            seed = cfg.seed if self._first_collect else None
            self._first_collect = False
            batch = collect_episode_batch(
                self.env, self.nets.actor, self.task, cfg, seed=seed
            )
            self.replay_buffer.extend_episode_batch(
                designs=batch.designs,
                outcomes=batch.outcomes,
                policy_designs=batch.policy_designs,
                rewards=batch.rewards,
                dones=batch.dones,
            )
            self.completed_episodes += sac.num_envs
            self.env_steps += sac.num_envs * horizon
            iteration_returns.append(batch.episodic_return)

        if (
            len(self.replay_buffer) >= sac.learning_starts
            and len(self.replay_buffer) >= sac.batch_size
        ):
            for _ in range(sac.gradient_steps_per_iteration):
                losses = sac_update(
                    self.replay_buffer,
                    self.nets,
                    self.target_entropy,
                    self.device,
                    horizon,
                    cfg,
                    self.optimization_step,
                )
                self._qf_loss = losses.qf_loss
                if losses.actor_loss is not None:
                    self._actor_loss = losses.actor_loss
                if losses.alpha_loss is not None:
                    self._alpha_loss = losses.alpha_loss
                self.optimization_step = losses.optimization_step

        if sac.alpha is None:
            self.target_entropy -= sac.entropy_anneal_rate

        return {
            "env_steps": float(self.env_steps),
            "episodes": float(self.completed_episodes),
            "episodic_return": torch.stack(iteration_returns).mean().item(),
        }

    @torch.no_grad()
    def evaluate(self):
        """Estimates the policy-mean SPCE of the deterministic policy."""
        was_training = self.designer.design_net.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        self.designer.design_net.eval()

        eval_env = build_env(
            self.cfg,
            self.task,
            self.env.design_space,
            n_contrastive=self.cfg.eval_n_contrastive,
        )
        obs, _ = eval_env.reset(seed=self.cfg.seed + 1)
        returns = torch.zeros(self.cfg.sac.num_envs, device=self.device)
        for _ in range(self.env.horizon):
            raw_designs = self.designer.design_net(
                obs.outcomes, obs.designs, mask=obs.mask
            )
            designs = eval_env.design_space.to_design(raw_designs)
            obs, rewards, _, _, _ = eval_env.step(designs)
            returns += rewards

        self.designer.design_net.train(was_training)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        mean = returns.mean().item()
        standard_error = (
            returns.std(unbiased=True).div(self.cfg.sac.num_envs**0.5).item()
        )
        return {"eval/spce": mean, "eval/spce_se": standard_error}

    def _loss_metrics(self, step):
        """Builds the periodic loss metrics dict."""
        elapsed = max(time.time() - self._start_time, 1e-12)
        return {
            "step": step,
            "env_steps": float(self.env_steps),
            "sps": float(int(self.env_steps / elapsed)),
            "qf_loss": self._qf_loss.item() / 2.0,
            "actor_loss": self._actor_loss.item(),
            "alpha_loss": self._alpha_loss.item(),
            "alpha": self.nets.log_alpha.exp().item(),
            "target_entropy": self.target_entropy,
        }

    def _maybe_save_best(self, step, metrics, min_step=0):
        # Saves a checkpoint of the model if it's the highest episodic return seen.
        # NB: our objective is maximizing the episodic return (an MI bound).

        # min_step controls how long we wait before saving; useful for avoiding a
        # ton of i/o at the start.
        if step < min_step:
            return

        if self.manager is None:
            return

        metric_val = metrics["episodic_return"]
        is_better = metric_val > self.best_metric

        if is_better:
            self.best_metric = metric_val

            chkpt_path = self.manager.path("best.pt")
            torch.save(
                self.build_checkpoint(step=step, metric=metric_val),
                chkpt_path,
            )

            self.manager.event_logger.info(
                f"Saving new best checkpoint : "
                f"episodic_return={metric_val} @ step={step}"
            )

    def build_checkpoint(self, step=None, metric=None):
        """Builds the checkpoint dict holding weights and training state."""
        design_space = self.env.design_space
        # The trained actor is the deployable designer, so its weights are stored
        # under both keys: "designer" for inference-only loading and "actor" for
        # resuming SAC training alongside the critic/optimizers.
        actor_state_dict = self.nets.actor.state_dict()
        checkpoint = {
            "config": OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=False
            ),
            "horizon": self.env.horizon,
            "design_low": (
                design_space.low.detach().cpu()
                if isinstance(design_space, BoxDesignSpace)
                else design_space.values.amin(dim=0).detach().cpu()
            ),
            "design_high": (
                design_space.high.detach().cpu()
                if isinstance(design_space, BoxDesignSpace)
                else design_space.values.amax(dim=0).detach().cpu()
            ),
            "outcome_low": torch.as_tensor(self.cfg.reward.outcome_low),
            "outcome_high": torch.as_tensor(self.cfg.reward.outcome_high),
            "designer": actor_state_dict,
            "actor": actor_state_dict,
            "qf1": self.nets.qf1.state_dict(),
            "qf2": self.nets.qf2.state_dict(),
            "qf1_target": self.nets.qf1_target.state_dict(),
            "qf2_target": self.nets.qf2_target.state_dict(),
            "actor_optimizer": self.nets.actor_optimizer.state_dict(),
            "q_optimizer": self.nets.q_optimizer.state_dict(),
            "alpha": self.nets.log_alpha.exp().detach().cpu(),
            "log_alpha": self.nets.log_alpha.detach().cpu(),
            "alpha_optimizer": (
                None
                if self.nets.alpha_optimizer is None
                else self.nets.alpha_optimizer.state_dict()
            ),
            "target_entropy": self.target_entropy,
            "optimization_step": self.optimization_step,
            "env_steps": self.env_steps,
            "training_iterations": self.cfg.sac.num_training_iterations,
            "global_step": self.cfg.sac.num_training_iterations,
        }
        if isinstance(design_space, FiniteDesignSpace):
            checkpoint["design_values"] = design_space.values.detach().cpu()
        if step is not None:
            checkpoint["step"] = step
        if metric is not None:
            checkpoint["metric"] = metric
        return checkpoint


# ============================================================================
# Helper + factory functions
# ============================================================================


def build_design_space(cfg, task):
    """Returns the task's declared space or configures a continuous box.

    Every task now carries a design space, with ``DefaultDesignSpace`` standing
    in for "declares no real space of its own", so that -- rather than ``None``
    -- is what falls back to the configured box.
    """
    if not isinstance(task.design_space, DefaultDesignSpace):
        return task.design_space
    return BoxDesignSpace(
        cfg.design_space.low,
        cfg.design_space.high,
        event_shape=task.event_shape_design,
    )


def build_env(cfg, task, design_space, *, n_contrastive=None):
    """Instantiates the BED environment with the RL-BOED dense reward.

    ``n_contrastive`` overrides ``cfg.reward.n_contrastive`` for the reward's
    contrastive-sample count; evaluation passes ``cfg.eval_n_contrastive`` to
    score the policy with more contrastive samples than training.
    """
    if n_contrastive is None:
        n_contrastive = cfg.reward.n_contrastive
    return BEDEnv(
        task,
        horizon=cfg.T,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(n_contrastive),
        batch_shape=(cfg.sac.num_envs,),
    )


def build_replay_buffer(cfg, task):
    """Instantiates the compact episode replay buffer."""
    buffer_device = _resolve_buffer_device(cfg.sac.buffer_device)
    return _CompactEpisodeReplayBuffer(
        buffer_size=cfg.sac.buffer_size,
        batch_size=cfg.sac.batch_size,
        horizon=cfg.T,
        event_shape_design=task.event_shape_design,
        event_shape_y=task.event_shape_y,
        device=buffer_device,
        pin_memory=task.device.type == "cuda" and buffer_device.type == "cpu",
    )


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="rlboed")
def main(cfg):
    """Entry point for RL-BOED (SAC) training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/rlboed.py [overrides...]``.
    """
    manager, task = setup(cfg)

    design_space = build_design_space(cfg, task)
    validate_config(task, cfg, design_space)

    # Seed immediately before network construction (mirrors the original
    # fit_rlboed ordering) so weight initialization is reproducible.
    _set_seed(cfg.seed, cfg.torch_deterministic)
    nets = make_networks(task, design_space, cfg, task.device)

    env = build_env(cfg, task, design_space)
    replay_buffer = build_replay_buffer(cfg, task)
    designer = build_policy_designer(
        task,
        cfg.T,
        nets.actor,
        design_space=design_space,
    )

    trainer = Trainer(
        task=task,
        designer=designer,
        env=env,
        replay_buffer=replay_buffer,
        nets=nets,
        cfg=cfg,
        manager=manager,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    manager.event_logger.info(
        f"policy-mean SPCE with {cfg.eval_n_contrastive} contrastive samples: "
        f"{eval_metrics['eval/spce']:.4f} ± {eval_metrics['eval/spce_se']:.4f}"
    )


if __name__ == "__main__":
    main()
