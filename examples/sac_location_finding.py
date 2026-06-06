"""Train a location-finding policy with Soft Actor-Critic.

This follows CleanRL's continuous-action SAC implementation while using TestBED's
torch-native, Gymnasium-style ``BEDEnv`` API. The dense reward is the factorized
RL-BOED reward from Foster et al., "Optimizing Sequential Experimental Design
with Deep Reinforcement Learning".

Example smoke run:
    uv run python examples/sac_location_finding.py \
        --total-timesteps 100 --learning-starts 20 --n-contrastive 16 \
        --buffer-size 1000 --no-cuda --no-save
"""

import argparse
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import LazyTensorStorage, ReplayBuffer

from testbed.design_nets import GaussianPolicyActor, TanhGaussianPolicyNet
from testbed.design_nets.modules import MLP
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    PaddedHistoryObservation,
    RLBOEDDenseReward,
)
from testbed.tasks import LocationFinding


@dataclass
class Args:
    """Command-line arguments."""

    exp_name: str = Path(__file__).stem
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    save: bool = True
    output_dir: str = "runs"

    # Source-location experiment settings from the RL-BOED paper.
    n_sources: int = 2
    dim_theta: int = 2
    horizon: int = 30
    n_contrastive: int = 100_000
    design_low: float = -4.0
    design_high: float = 4.0
    outcome_low: float = -3.0
    outcome_high: float = 10.0

    # SAC settings from the paper and its Adaptive_Source_SAC implementation.
    total_timesteps: int = 20_000
    num_envs: int = 100
    buffer_size: int = 10_000_000
    gamma: float = 0.9
    tau: float = 1e-3
    batch_size: int = 4096
    # The reference starts training after 1e5 transitions. Each vector step
    # contributes num_envs transitions, giving 1e5 / 100 = 1,000 steps.
    learning_starts: int = 1_000
    policy_lr: float = 1e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    entropy_anneal_rate: float = 1.0 / 14_000
    log_frequency: int = 100


def parse_args() -> Args:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default=Args.exp_name)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument(
        "--torch-deterministic",
        action=argparse.BooleanOptionalAction,
        default=Args.torch_deterministic,
    )
    parser.add_argument(
        "--cuda", action=argparse.BooleanOptionalAction, default=Args.cuda
    )
    parser.add_argument(
        "--save", action=argparse.BooleanOptionalAction, default=Args.save
    )
    parser.add_argument("--output-dir", default=Args.output_dir)

    parser.add_argument("--n-sources", type=int, default=Args.n_sources)
    parser.add_argument("--dim-theta", type=int, default=Args.dim_theta)
    parser.add_argument("--horizon", type=int, default=Args.horizon)
    parser.add_argument("--n-contrastive", type=int, default=Args.n_contrastive)
    parser.add_argument("--design-low", type=float, default=Args.design_low)
    parser.add_argument("--design-high", type=float, default=Args.design_high)
    parser.add_argument("--outcome-low", type=float, default=Args.outcome_low)
    parser.add_argument("--outcome-high", type=float, default=Args.outcome_high)

    parser.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument("--buffer-size", type=int, default=Args.buffer_size)
    parser.add_argument("--gamma", type=float, default=Args.gamma)
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument("--batch-size", type=int, default=Args.batch_size)
    parser.add_argument("--learning-starts", type=int, default=Args.learning_starts)
    parser.add_argument("--policy-lr", type=float, default=Args.policy_lr)
    parser.add_argument("--q-lr", type=float, default=Args.q_lr)
    parser.add_argument("--policy-frequency", type=int, default=Args.policy_frequency)
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=Args.target_network_frequency,
    )
    parser.add_argument("--alpha", type=float, default=Args.alpha)
    parser.add_argument(
        "--autotune",
        action=argparse.BooleanOptionalAction,
        default=Args.autotune,
    )
    parser.add_argument(
        "--entropy-anneal-rate",
        type=float,
        default=Args.entropy_anneal_rate,
    )
    parser.add_argument("--log-frequency", type=int, default=Args.log_frequency)
    return Args(**vars(parser.parse_args()))


class DADGaussianActor(GaussianPolicyActor):
    """RL-BOED DeepSets architecture with Gaussian policy output heads."""

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        dim_h: int = 128,
        dim_emb: int = 64,
        init_std: float = math.sqrt(1.0 / 3.0),
        log_std_min: float = -20.0,
        log_std_max: float = 0.0,
    ):
        """Initializes the history encoder and stochastic emitter."""
        super().__init__()
        self.dim_emb = dim_emb
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.encoder_net = MLP(
            [dim_y + dim_design, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.emitter_net = MLP(
            [dim_emb, dim_h, dim_h],
            activation=nn.ReLU,
        )
        self.mean_head = nn.Linear(dim_h, dim_design)
        self.log_std_head = nn.Linear(dim_h, dim_design)

        nn.init.constant_(self.log_std_head.bias, math.log(init_std))

    def forward(
        self,
        y_history: Tensor,
        design_history: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Maps padded histories to Gaussian mean and log standard deviation."""
        batch_size, history_length, _ = design_history.shape

        if history_length == 0:
            enc = design_history.new_zeros(batch_size, self.dim_emb)
        else:
            enc_input = torch.cat([y_history, design_history], dim=-1)
            encoded_history = self.encoder_net(enc_input)
            if mask is not None:
                encoded_history = encoded_history * mask.to(
                    dtype=encoded_history.dtype
                ).unsqueeze(-1)
            enc = encoded_history.sum(dim=1)

        hidden = self.emitter_net(enc)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(
            min=self.log_std_min,
            max=self.log_std_max,
        )
        return mean, log_std


class SoftQNetwork(nn.Module):
    """DeepSets critic with masked sum pooling over experiment history."""

    def __init__(
        self,
        *,
        dim_y: int,
        dim_design: int,
        dim_h: int = 128,
        dim_emb: int = 64,
    ):
        """Initializes the history encoder and action-conditioned emitter."""
        super().__init__()
        self.dim_emb = dim_emb
        self.encoder_net = MLP(
            [dim_y + dim_design, dim_h, dim_h, dim_emb],
            activation=nn.ReLU,
        )
        self.emitter_net = MLP(
            [dim_emb + dim_design, dim_h, dim_h, 1],
            activation=nn.ReLU,
        )

    def forward(self, observation: dict[str, Tensor], action: Tensor) -> Tensor:
        """Estimates Q from a permutation-invariant history summary and action."""
        designs = observation["designs"]
        outcomes = observation["outcomes"]
        mask = observation.get("mask")
        batch_size, history_length, _ = designs.shape

        if history_length == 0:
            history_summary = designs.new_zeros(batch_size, self.dim_emb)
        else:
            encoded_history = self.encoder_net(torch.cat([outcomes, designs], dim=-1))
            if mask is not None:
                encoded_history = encoded_history * mask.to(
                    dtype=encoded_history.dtype
                ).unsqueeze(-1)
            history_summary = encoded_history.sum(dim=1)

        return self.emitter_net(torch.cat([history_summary, action], dim=-1))


def affine_rescale(
    value: Tensor,
    source_low,
    source_high,
    target_low,
    target_high,
) -> Tensor:
    """Affinely rescales a tensor between bounded coordinate systems."""
    source_low = torch.as_tensor(source_low, device=value.device, dtype=value.dtype)
    source_high = torch.as_tensor(source_high, device=value.device, dtype=value.dtype)
    target_low = torch.as_tensor(target_low, device=value.device, dtype=value.dtype)
    target_high = torch.as_tensor(
        target_high, device=value.device, dtype=value.dtype
    )
    return target_low + (value - source_low) * (
        (target_high - target_low) / (source_high - source_low)
    )


def normalize_observation(
    observation: dict[str, Tensor],
    *,
    design_low: float,
    design_high: float,
    outcome_low: float,
    outcome_high: float,
) -> dict[str, Tensor]:
    """Normalizes physical history values to the released policy coordinates."""
    normalized = dict(observation)
    normalized["designs"] = affine_rescale(
        observation["designs"],
        design_low,
        design_high,
        0.0,
        1.0,
    )
    normalized["outcomes"] = affine_rescale(
        observation["outcomes"],
        outcome_low,
        outcome_high,
        0.0,
        1.0,
    )
    return normalized


def action_to_design(
    action: Tensor,
    *,
    design_low: float,
    design_high: float,
) -> Tensor:
    """Maps a normalized SAC action in [-1, 1] to a physical design."""
    return affine_rescale(action, -1.0, 1.0, design_low, design_high)


def annealed_target_entropy(
    initial_target_entropy: float,
    entropy_anneal_rate: float,
    iteration: int,
) -> float:
    """Returns target entropy after the given number of completed iterations."""
    return initial_target_entropy - entropy_anneal_rate * iteration


def validate_args(args: Args) -> None:
    """Rejects invalid settings before allocating large tensors."""
    positive_ints = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "log_frequency": args.log_frequency,
        "policy_frequency": args.policy_frequency,
        "target_network_frequency": args.target_network_frequency,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.n_contrastive < 0:
        raise ValueError("n_contrastive must be non-negative.")
    if args.design_low >= args.design_high:
        raise ValueError("design_low must be less than design_high.")
    if args.outcome_low >= args.outcome_high:
        raise ValueError("outcome_low must be less than outcome_high.")
    if args.entropy_anneal_rate < 0:
        raise ValueError("entropy_anneal_rate must be non-negative.")


def train(args: Args) -> Path | None:
    """Trains and optionally saves a SAC policy."""
    validate_args(args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = LocationFinding(args.n_sources, args.dim_theta).to(device)
    design_space = BoxDesignSpace(
        args.design_low,
        args.design_high,
        event_shape=task.event_shape_design,
    )
    policy_action_space = BoxDesignSpace(
        -1.0,
        1.0,
        event_shape=task.event_shape_design,
    )
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(args.n_contrastive),
        observation_fn=PaddedHistoryObservation(args.horizon),
        batch_shape=(args.num_envs,),
    )
    normalize_obs = partial(
        normalize_observation,
        design_low=args.design_low,
        design_high=args.design_high,
        outcome_low=args.outcome_low,
        outcome_high=args.outcome_high,
    )

    dim_y = task.event_shape_y.numel()
    dim_design = task.event_shape_design.numel()
    actor_module = DADGaussianActor(dim_y=dim_y, dim_design=dim_design)
    actor = TanhGaussianPolicyNet(policy_action_space, actor=actor_module).to(device)

    qf1 = SoftQNetwork(dim_y=dim_y, dim_design=dim_design).to(device)
    qf2 = SoftQNetwork(dim_y=dim_y, dim_design=dim_design).to(device)
    qf1_target = SoftQNetwork(dim_y=dim_y, dim_design=dim_design).to(device)
    qf2_target = SoftQNetwork(dim_y=dim_y, dim_design=dim_design).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        initial_target_entropy = -float(dim_design)
        target_entropy = initial_target_entropy
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        target_entropy = 0.0
        log_alpha = None
        alpha = args.alpha
        alpha_optimizer = None

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device="cpu"),
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )

    start_time = time.time()
    raw_obs, _ = env.reset(seed=args.seed)
    obs = normalize_obs(raw_obs)
    episodic_returns = torch.zeros(args.num_envs, device=device)
    completed_episodes = 0
    actor_loss = torch.tensor(float("nan"))
    alpha_loss = torch.tensor(float("nan"))

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = policy_action_space.sample((args.num_envs,), device=device)
        else:
            with torch.no_grad():
                actions = actor.sample_from_observation(obs).action

        designs = action_to_design(
            actions,
            design_low=args.design_low,
            design_high=args.design_high,
        )
        replay_obs = {key: value.detach().clone() for key, value in obs.items()}
        raw_next_obs, rewards, terminations, truncations, _ = env.step(designs)
        next_obs = normalize_obs(raw_next_obs)
        dones = terminations | truncations
        replay_buffer.extend(
            TensorDict(
                {
                    "observation": replay_obs,
                    "next_observation": next_obs,
                    "action": actions,
                    "reward": rewards,
                    "done": dones,
                },
                batch_size=[args.num_envs],
            )
            .detach()
            .to("cpu")
        )

        episodic_returns += rewards
        if dones.any():
            completed_episodes += args.num_envs
            mean_return = episodic_returns.mean().item()
            print(
                f"global_step={global_step}, "
                f"episodes={completed_episodes}, "
                f"episodic_return={mean_return:.4f}"
            )
            episodic_returns.zero_()
            raw_next_obs, _ = env.reset()
            next_obs = normalize_obs(raw_next_obs)
        obs = next_obs

        if (
            global_step >= args.learning_starts
            and len(replay_buffer) >= args.batch_size
        ):
            data = replay_buffer.sample().to(device, non_blocking=True)
            data_obs = data["observation"]
            data_next_obs = data["next_observation"]
            data_actions = data["action"]
            data_rewards = data["reward"]
            data_dones = data["done"]

            with torch.no_grad():
                next_policy_sample = actor.sample_from_observation(
                    data_next_obs
                )
                next_actions = next_policy_sample.action
                next_log_pi = next_policy_sample.log_prob.unsqueeze(-1)
                qf1_next_target = qf1_target(data_next_obs, next_actions)
                qf2_next_target = qf2_target(data_next_obs, next_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_pi
                )
                next_q_value = data_rewards + (
                    1.0 - data_dones.float()
                ) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data_obs, data_actions).view(-1)
            qf2_a_values = qf2(data_obs, data_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    policy_sample = actor.sample_from_observation(data_obs)
                    pi = policy_sample.action
                    log_pi = policy_sample.log_prob.unsqueeze(-1)
                    qf1_pi = qf1(data_obs, pi)
                    qf2_pi = qf2(data_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        assert log_alpha is not None
                        assert alpha_optimizer is not None
                        with torch.no_grad():
                            log_pi = actor.sample_from_observation(
                                data_obs
                            ).log_prob.unsqueeze(-1)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

            if global_step % args.log_frequency == 0:
                sps = int((global_step + 1) / (time.time() - start_time))
                message = (
                    f"SPS={sps}, qf_loss={qf_loss.item() / 2.0:.4f}, "
                    f"actor_loss={actor_loss.item():.4f}, alpha={alpha:.4f}"
                )
                if args.autotune:
                    message += f", alpha_loss={alpha_loss.item():.4f}"
                    message += f", target_entropy={target_entropy:.6f}"
                print(message)

        if args.autotune:
            target_entropy = annealed_target_entropy(
                initial_target_entropy,
                args.entropy_anneal_rate,
                global_step + 1,
            )

    if not args.save:
        return None

    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "sac_location_finding.pt"
    torch.save(
        {
            "args": asdict(args),
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


if __name__ == "__main__":
    checkpoint = train(parse_args())
    if checkpoint is not None:
        print(f"saved checkpoint to {os.fspath(checkpoint)}")
