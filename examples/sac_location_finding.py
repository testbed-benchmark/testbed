"""Train a policy for location finding with the RL-BOED algorithm."""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from testbed.design_nets import RLBOEDGaussianActor, RLBOEDQNetwork
from testbed.envs import BoxDesignSpace
from testbed.methods import RLBOED, RLBOEDConfig
from testbed.tasks import LocationFinding

# Compatibility aliases for code that imported these from the original example.
DADGaussianActor = RLBOEDGaussianActor
SoftQNetwork = RLBOEDQNetwork


@dataclass
class Args:
    exp_name: str = Path(__file__).stem
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    save: bool = True
    output_dir: str = "runs"

    n_sources: int = 2
    dim_theta: int = 2
    horizon: int = 30
    n_contrastive: int = 100_000
    design_low: float = -4.0
    design_high: float = 4.0
    outcome_low: float = -3.0
    outcome_high: float = 10.0

    total_timesteps: int = 20_000
    num_envs: int = 100
    buffer_size: int = 10_000_000
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


def parse_args() -> Args:
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
    parser.add_argument("--log-frequency", type=int, default=Args.log_frequency)
    return Args(**vars(parser.parse_args()))


def train(args: Args) -> Path | None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = LocationFinding(args.n_sources, args.dim_theta).to(device)
    design_space = BoxDesignSpace(
        args.design_low, args.design_high, event_shape=task.event_shape_design
    )
    config = RLBOEDConfig(
        outcome_low=args.outcome_low,
        outcome_high=args.outcome_high,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        n_contrastive=args.n_contrastive,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        log_frequency=args.log_frequency,
    )
    result = RLBOED(config).fit(
        task,
        horizon=args.horizon,
        design_space=design_space,
    )

    if not args.save:
        return None

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.output_dir) / run_name / "sac_location_finding.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    checkpoint = train(parse_args())
    if checkpoint is not None:
        print(f"saved checkpoint to {os.fspath(checkpoint)}")
