"""RL-BOED training with Soft Actor-Critic."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from math import ceil, log

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from testbed.design_nets import (
    DeepSetGaussianActor,
    RLBOEDQNetwork,
    TanhGaussianPolicyNet,
)
from testbed.designers import PolicyDesigner
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
)
from testbed.envs.design_spaces import DesignSpace
from testbed.envs.history import History
from testbed.envs.rewards import RLBOEDDenseReward
from testbed.tasks.base_task import Task

from .base import FitResult


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


@dataclass
class RLBOEDConfig:
    """RL-BOED and SAC hyperparameters."""

    outcome_low: float
    outcome_high: float
    seed: int = 373693
    torch_deterministic: bool = True

    n_contrastive: int = 100_000

    num_training_iterations: int = 2_001
    num_envs: int = 100
    buffer_size: int = 10_000_000
    buffer_device: str = "cpu"
    gamma: float = 0.9
    tau: float = 1e-3
    batch_size: int = 4096
    learning_starts: int = 100_000
    gradient_steps_per_iteration: int = 64
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    policy_init_std: float = 3**-0.5
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float | None = None
    initial_alpha: float = 1.0
    target_entropy: float | None = None
    entropy_anneal_rate: float = 1 / 1.4e4
    log_frequency: int = 100
    verbose: bool = True


def _resolve_buffer_device(config: RLBOEDConfig) -> torch.device:
    """Returns the replay storage device requested by the config."""
    requested = config.buffer_device.strip().lower()
    try:
        buffer_device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(
            "buffer_device must be 'cpu', 'cuda', or 'cuda:<index>'."
        ) from error

    if buffer_device.type not in {"cpu", "cuda"}:
        raise ValueError("buffer_device must be 'cpu', 'cuda', or 'cuda:<index>'.")
    if buffer_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("buffer_device='cuda' requires CUDA to be available.")
        if (
            buffer_device.index is not None
            and buffer_device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"buffer_device='{buffer_device}' is outside the available "
                f"CUDA device count ({torch.cuda.device_count()})."
            )
    return buffer_device


def _set_seed(config: RLBOEDConfig) -> None:
    """Sets random seeds to match the reference script."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic


def _soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Polyak-updates a target network toward a source network."""
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def validate_config(
    task: Task,
    horizon: int,
    design_space: DesignSpace,
    config: RLBOEDConfig,
) -> None:
    """Rejects unsupported tasks and invalid settings."""
    positive_ints = {
        "num_training_iterations": config.num_training_iterations,
        "num_envs": config.num_envs,
        "buffer_size": config.buffer_size,
        "batch_size": config.batch_size,
        "gradient_steps_per_iteration": config.gradient_steps_per_iteration,
        "horizon": horizon,
        "log_frequency": config.log_frequency,
        "policy_frequency": config.policy_frequency,
        "target_network_frequency": config.target_network_frequency,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.n_contrastive < 0:
        raise ValueError("n_contrastive must be non-negative.")
    if config.buffer_size < config.learning_starts:
        raise ValueError("buffer_size must be at least learning_starts.")
    if config.outcome_low >= config.outcome_high:
        raise ValueError("outcome_low must be less than outcome_high.")
    if config.alpha is not None and config.alpha < 0:
        raise ValueError("alpha must be non-negative when fixed.")
    if config.initial_alpha <= 0:
        raise ValueError("initial_alpha must be positive.")
    if config.learning_starts < 0:
        raise ValueError("learning_starts must be non-negative.")
    if config.entropy_anneal_rate < 0:
        raise ValueError("entropy_anneal_rate must be non-negative.")
    if config.policy_init_std <= 0:
        raise ValueError("policy_init_std must be positive.")
    if task.likelihood_type != "explicit":
        raise ValueError("RLBOED requires a task with an explicit likelihood.")
    if not isinstance(design_space, BoxDesignSpace):
        raise TypeError("This RLBOED SAC implementation requires a BoxDesignSpace.")


@dataclass
class _SACNetworks:
    actor: TanhGaussianPolicyNet
    qf1: RLBOEDQNetwork
    qf2: RLBOEDQNetwork
    qf1_target: RLBOEDQNetwork
    qf2_target: RLBOEDQNetwork
    q_optimizer: optim.Adam
    actor_optimizer: optim.Adam
    log_alpha: torch.Tensor
    alpha_optimizer: optim.Adam | None


def make_networks(
    task: Task,
    design_space: DesignSpace,
    config: RLBOEDConfig,
    device: torch.device,
) -> _SACNetworks:
    """Builds the SAC actor, twin critics, target networks, optimizers, and alpha."""
    design_low = design_space.low.to(device)
    design_high = design_space.high.to(device)
    dim_y = task.event_shape_y.numel()
    dim_design = task.event_shape_design.numel()

    actor_module = DeepSetGaussianActor(
        dim_y=dim_y,
        dim_design=dim_design,
        init_std=config.policy_init_std,
    )
    actor = TanhGaussianPolicyNet(
        design_space,
        actor=actor_module,
        normalize_design_history=True,
        design_history_bounds=(0.0, 1.0),
        outcome_bounds=(config.outcome_low, config.outcome_high),
    ).to(device)

    critic_kwargs = {
        "dim_y": dim_y,
        "dim_design": dim_design,
        "design_low": design_low,
        "design_high": design_high,
        "outcome_low": config.outcome_low,
        "outcome_high": config.outcome_high,
    }
    qf1 = RLBOEDQNetwork(**critic_kwargs).to(device)
    qf2 = RLBOEDQNetwork(**critic_kwargs).to(device)
    qf1_target = copy.deepcopy(qf1)
    qf2_target = copy.deepcopy(qf2)

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=config.q_lr,
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.policy_lr)

    if config.alpha is None:
        log_alpha = torch.tensor(
            log(config.initial_alpha),
            device=device,
            requires_grad=True,
        )
        alpha_optimizer: optim.Adam | None = optim.Adam(
            [log_alpha], lr=config.policy_lr
        )
    else:
        fixed_log_alpha = float("-inf") if config.alpha == 0 else log(config.alpha)
        log_alpha = torch.tensor(fixed_log_alpha, device=device)
        alpha_optimizer = None

    return _SACNetworks(
        actor=actor,
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


def collect_episode_batch(
    env: BEDEnv,
    actor: TanhGaussianPolicyNet,
    task: Task,
    config: RLBOEDConfig,
    *,
    seed: int | None = None,
) -> _EpisodeBatch:
    """Rolls out the actor for one episode batch and returns the transitions."""
    horizon = env.horizon
    device = task.device

    obs, _ = env.reset(seed=seed)

    episodic_returns = torch.zeros(config.num_envs, device=device)
    episode_policy_designs = []
    episode_rewards = []
    episode_dones = []

    for _ in range(horizon):
        with torch.no_grad():
            policy_sample = actor.sample_from_observation(obs)
            designs = policy_sample.design
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
    config: RLBOEDConfig,
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

    with torch.no_grad():
        next_policy_sample = nets.actor.sample_from_observation(data_next_obs)
        next_designs = next_policy_sample.design
        next_log_pi = next_policy_sample.log_prob.unsqueeze(-1)
        qf1_next_target = nets.qf1_target(data_next_obs, next_designs)
        qf2_next_target = nets.qf2_target(data_next_obs, next_designs)
        min_qf_next_target = (
            torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
        )
        next_q_value = data_rewards + (
            1.0 - data_dones.float()
        ) * config.gamma * min_qf_next_target.view(-1)

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

    if optimization_step % config.policy_frequency == 0:
        policy_sample = nets.actor.sample_from_observation(data_obs)
        pi = policy_sample.design
        log_pi = policy_sample.log_prob.unsqueeze(-1)
        min_qf_pi = torch.min(nets.qf1(data_obs, pi), nets.qf2(data_obs, pi))
        actor_loss = (alpha * log_pi - min_qf_pi).mean()

        nets.actor_optimizer.zero_grad()
        actor_loss.backward()
        nets.actor_optimizer.step()

        if nets.alpha_optimizer is not None:
            alpha_loss = -(
                nets.log_alpha.exp() * (log_pi.detach() + target_entropy)
            ).mean()
            nets.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nets.alpha_optimizer.step()

    if optimization_step % config.target_network_frequency == 0:
        _soft_update(nets.qf1, nets.qf1_target, config.tau)
        _soft_update(nets.qf2, nets.qf2_target, config.tau)

    return _SACLosses(
        qf_loss=qf_loss,
        actor_loss=actor_loss,
        alpha_loss=alpha_loss,
        optimization_step=optimization_step + 1,
    )


def fit_rlboed(
    task: Task,
    horizon: int,
    design_space: DesignSpace,
    config: RLBOEDConfig,
) -> FitResult:
    """Trains and returns a deployable policy designer using RL-BOED with SAC."""
    validate_config(task, horizon, design_space, config)
    _set_seed(config)

    device = task.device
    buffer_device = _resolve_buffer_device(config)
    nets = make_networks(task, design_space, config, device)

    env = BEDEnv(
        task,
        horizon=horizon,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(config.n_contrastive),
        batch_shape=(config.num_envs,),
    )

    target_entropy = (
        -float(task.event_shape_design.numel())
        if config.target_entropy is None
        else float(config.target_entropy)
    )

    replay_buffer = _CompactEpisodeReplayBuffer(
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        horizon=horizon,
        event_shape_design=task.event_shape_design,
        event_shape_y=task.event_shape_y,
        device=buffer_device,
        pin_memory=device.type == "cuda" and buffer_device.type == "cpu",
    )

    metrics: list[dict[str, float]] = []
    start_time = time.time()
    completed_episodes = 0
    env_steps = 0
    optimization_step = 0
    qf_loss = torch.tensor(float("nan"))
    actor_loss = torch.tensor(float("nan"))
    alpha_loss = torch.tensor(float("nan"))

    transitions_per_episode_batch = config.num_envs * horizon
    for training_iteration in range(config.num_training_iterations):
        episode_batches = 1
        if len(replay_buffer) < config.learning_starts:
            episode_batches = ceil(
                (config.learning_starts - len(replay_buffer))
                / transitions_per_episode_batch
            )

        iteration_returns = []
        for episode_batch in range(episode_batches):
            seed = (
                config.seed if training_iteration == 0 and episode_batch == 0 else None
            )
            batch = collect_episode_batch(env, nets.actor, task, config, seed=seed)
            replay_buffer.extend_episode_batch(
                designs=batch.designs,
                outcomes=batch.outcomes,
                policy_designs=batch.policy_designs,
                rewards=batch.rewards,
                dones=batch.dones,
            )
            completed_episodes += config.num_envs
            env_steps += config.num_envs * horizon
            iteration_returns.append(batch.episodic_return)

        episode_metrics = {
            "step": float(training_iteration),
            "env_steps": float(env_steps),
            "episodes": float(completed_episodes),
            "episodic_return": torch.stack(iteration_returns).mean().item(),
        }
        metrics.append(episode_metrics)

        if (
            len(replay_buffer) >= config.learning_starts
            and len(replay_buffer) >= config.batch_size
        ):
            for _ in range(config.gradient_steps_per_iteration):
                losses = sac_update(
                    replay_buffer,
                    nets,
                    target_entropy,
                    device,
                    horizon,
                    config,
                    optimization_step,
                )
                qf_loss = losses.qf_loss
                if losses.actor_loss is not None:
                    actor_loss = losses.actor_loss
                if losses.alpha_loss is not None:
                    alpha_loss = losses.alpha_loss
                optimization_step = losses.optimization_step

        if config.alpha is None:
            target_entropy -= config.entropy_anneal_rate

        if training_iteration % config.log_frequency == 0:
            alpha = nets.log_alpha.exp().item()
            step_metrics = {
                "step": float(training_iteration),
                "env_steps": float(env_steps),
                "sps": float(int(env_steps / (time.time() - start_time))),
                "qf_loss": qf_loss.item() / 2.0,
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": alpha,
                "target_entropy": target_entropy,
            }
            metrics.append(step_metrics)
            if config.verbose:
                print(
                    f"iteration={training_iteration}, "
                    f"env_steps={env_steps}, "
                    f"return={episode_metrics['episodic_return']:.4f}, "
                    f"qf_loss={step_metrics['qf_loss']:.4f}, "
                    f"actor_loss={step_metrics['actor_loss']:.4f}, "
                    f"alpha={alpha:.4f}"
                )

    nets.actor.eval()
    designer = PolicyDesigner(task, horizon, nets.actor, deterministic=False)
    design_low = design_space.low.to(device)
    design_high = design_space.high.to(device)
    actor_state_dict = nets.actor.state_dict()
    checkpoint = {
        "config": asdict(config),
        "horizon": horizon,
        "design_low": design_low.detach().cpu(),
        "design_high": design_high.detach().cpu(),
        "outcome_low": torch.as_tensor(config.outcome_low),
        "outcome_high": torch.as_tensor(config.outcome_high),
        # The trained actor is the deployable designer, so its weights are
        # stored under both keys: "designer" for inference-only loading and
        # "actor" for resuming SAC training alongside the critic/optimizers.
        "designer": actor_state_dict,
        "actor": actor_state_dict,
        "qf1": nets.qf1.state_dict(),
        "qf2": nets.qf2.state_dict(),
        "qf1_target": nets.qf1_target.state_dict(),
        "qf2_target": nets.qf2_target.state_dict(),
        "actor_optimizer": nets.actor_optimizer.state_dict(),
        "q_optimizer": nets.q_optimizer.state_dict(),
        "alpha": nets.log_alpha.exp().detach().cpu(),
        "log_alpha": nets.log_alpha.detach().cpu(),
        "alpha_optimizer": (
            None if nets.alpha_optimizer is None else nets.alpha_optimizer.state_dict()
        ),
        "target_entropy": target_entropy,
        "optimization_step": optimization_step,
        "env_steps": env_steps,
        "training_iterations": config.num_training_iterations,
        "global_step": config.num_training_iterations,
    }
    return FitResult(
        designer=designer,
        metrics=metrics,
        checkpoint=checkpoint,
    )
