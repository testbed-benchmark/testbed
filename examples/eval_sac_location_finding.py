"""Evaluate a saved RL-BOED policy for location finding."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from testbed.design_nets import DeepSetGaussianActor, TanhGaussianPolicyNet
from testbed.designers import PolicyDesigner
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    PaddedHistoryObservation,
    RLBOEDDenseReward,
)
from testbed.tasks import LocationFinding

EVAL_N_CONTRASTIVE = 1_000_000


@dataclass
class Args:
    checkpoint: str = ""
    n_sources: int = 2
    seed: int = 1
    cuda: bool = True
    num_envs: int = 100
    deterministic: bool = True


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a .pt checkpoint saved by sac_location_finding.py")
    parser.add_argument(
        "--n-sources",
        type=int,
        default=Args.n_sources,
        help="Number of sources; must match the training run (not stored in checkpoint).",
    )
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=Args.cuda)
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=Args.deterministic,
        help="Use policy mean (True) or sample stochastically (False).",
    )
    return Args(**vars(parser.parse_args()))


def load_policy(checkpoint_path: str | Path, n_sources: int, device: torch.device) -> tuple[PolicyDesigner, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    design_low = checkpoint["design_low"].to(device)
    design_high = checkpoint["design_high"].to(device)
    outcome_low = checkpoint["outcome_low"].item()
    outcome_high = checkpoint["outcome_high"].item()
    horizon = checkpoint["horizon"]

    # dim_theta is recoverable from the saved design bounds shape.
    dim_theta = design_low.shape[0]
    task = LocationFinding(n_sources, dim_theta).to(device)
    design_space = BoxDesignSpace(design_low, design_high, event_shape=task.event_shape_design)

    actor_module = DeepSetGaussianActor(
        dim_y=task.event_shape_y.numel(),
        dim_design=task.event_shape_design.numel(),
        init_std=cfg.get("policy_init_std", 3**-0.5),
    )
    actor = TanhGaussianPolicyNet(
        design_space,
        actor=actor_module,
        normalize_design_history=True,
        design_history_bounds=(0.0, 1.0),
        outcome_bounds=(outcome_low, outcome_high),
        physical_action_log_prob=False,
    ).to(device)
    actor.load_state_dict(checkpoint["designer"])
    actor.eval()

    policy_net = actor
    return policy_net, {
        "task": task,
        "design_space": design_space,
        "horizon": horizon,
        "outcome_low": outcome_low,
        "outcome_high": outcome_high,
    }


@torch.no_grad()
def evaluate_policy_mean(
    policy_net,
    task,
    design_space,
    horizon: int,
    num_envs: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> tuple[float, float]:
    env = BEDEnv(
        task,
        horizon=horizon,
        design_space=design_space,
        reward_fn=RLBOEDDenseReward(EVAL_N_CONTRASTIVE),
        observation_fn=PaddedHistoryObservation(horizon),
        batch_shape=(num_envs,),
    )
    observation, _ = env.reset(seed=seed)
    returns = torch.zeros(num_envs, device=device)
    for _ in range(horizon):
        designs = policy_net.sample_from_observation(observation, deterministic=deterministic).action
        observation, rewards, _, _, _ = env.step(designs)
        returns += rewards

    mean = returns.mean().item()
    standard_error = returns.std(unbiased=True).div(num_envs**0.5).item()
    return mean, standard_error


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    policy_net, env_kwargs = load_policy(args.checkpoint, args.n_sources, device)

    mean, se = evaluate_policy_mean(
        policy_net,
        task=env_kwargs["task"],
        design_space=env_kwargs["design_space"],
        horizon=env_kwargs["horizon"],
        num_envs=args.num_envs,
        seed=args.seed,
        device=device,
        deterministic=args.deterministic,
    )
    mode = "deterministic" if args.deterministic else "stochastic"
    print(
        f"{mode} SPCE estimate ({args.num_envs} envs, "
        f"{EVAL_N_CONTRASTIVE} contrastive samples): {mean:.4f} ± {se:.4f}"
    )
