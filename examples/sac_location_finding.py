"""Train a policy for location finding with the RL-BOED algorithm."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    RLBOEDDenseReward,
)
from testbed.methods import RLBOEDConfig, fit_rlboed
from testbed.tasks import LocationFinding

EVAL_N_CONTRASTIVE = 1_000_000


@dataclass
class Args:
    exp_name: str = Path(__file__).stem
    seed: int = 373693
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

    parser.add_argument(
        "--num-training-iterations",
        type=int,
        default=Args.num_training_iterations,
        help="Number of SAC training iterations. The paper used 20,001.",
    )
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument("--buffer-size", type=int, default=Args.buffer_size)
    parser.add_argument("--buffer-device", default=Args.buffer_device)
    parser.add_argument("--gamma", type=float, default=Args.gamma)
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument("--batch-size", type=int, default=Args.batch_size)
    parser.add_argument("--learning-starts", type=int, default=Args.learning_starts)
    parser.add_argument(
        "--gradient-steps-per-iteration",
        type=int,
        default=Args.gradient_steps_per_iteration,
    )
    parser.add_argument("--policy-lr", type=float, default=Args.policy_lr)
    parser.add_argument("--q-lr", type=float, default=Args.q_lr)
    parser.add_argument("--policy-init-std", type=float, default=Args.policy_init_std)
    parser.add_argument("--policy-frequency", type=int, default=Args.policy_frequency)
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=Args.target_network_frequency,
    )
    parser.add_argument("--alpha", type=float, default=Args.alpha)
    parser.add_argument("--initial-alpha", type=float, default=Args.initial_alpha)
    parser.add_argument("--target-entropy", type=float, default=Args.target_entropy)
    parser.add_argument(
        "--entropy-anneal-rate",
        type=float,
        default=Args.entropy_anneal_rate,
    )
    parser.add_argument("--log-frequency", type=int, default=Args.log_frequency)
    return Args(**vars(parser.parse_args()))


def make_task_and_design_space(args: Args, device: torch.device):
    task = LocationFinding(args.n_sources, args.dim_theta).to(device)
    design_space = BoxDesignSpace(
        args.design_low, args.design_high, event_shape=task.event_shape_design
    )
    return task, design_space


def train(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task, design_space = make_task_and_design_space(args, device)
    config = RLBOEDConfig(
        outcome_low=args.outcome_low,
        outcome_high=args.outcome_high,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        n_contrastive=args.n_contrastive,
        num_training_iterations=args.num_training_iterations,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        buffer_device=args.buffer_device,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        gradient_steps_per_iteration=args.gradient_steps_per_iteration,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        policy_init_std=args.policy_init_std,
        policy_frequency=args.policy_frequency,
        target_network_frequency=args.target_network_frequency,
        alpha=args.alpha,
        initial_alpha=args.initial_alpha,
        target_entropy=args.target_entropy,
        entropy_anneal_rate=args.entropy_anneal_rate,
        log_frequency=args.log_frequency,
    )
    result = fit_rlboed(task, args.horizon, design_space, config)

    if not args.save:
        return result, None

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.output_dir) / run_name / "sac_location_finding.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(checkpoint_path)
    return result, checkpoint_path


@torch.no_grad()
def evaluate_policy_mean(args: Args, policy_net) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task, design_space = make_task_and_design_space(args, device)
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(EVAL_N_CONTRASTIVE),
        batch_shape=(args.num_envs,),
    )
    policy_net.eval()
    observation, _ = env.reset(seed=args.seed + 1)
    returns = torch.zeros(args.num_envs, device=device)
    for _ in range(args.horizon):
        designs = policy_net.sample_from_observation(
            observation, deterministic=True
        ).design
        observation, rewards, _, _, _ = env.step(designs)
        returns += rewards

    mean = returns.mean().item()
    standard_error = returns.std(unbiased=True).div(args.num_envs**0.5).item()
    return mean, standard_error


if __name__ == "__main__":
    args = parse_args()
    result, checkpoint = train(args)
    if checkpoint is not None:
        print(f"saved checkpoint to {checkpoint}")
    mean, standard_error = evaluate_policy_mean(args, result.designer.policy_net)
    print(
        f"policy-mean SPCE estimate with {EVAL_N_CONTRASTIVE} "
        f"contrastive samples: {mean:.4f} ± {standard_error:.4f}"
    )
