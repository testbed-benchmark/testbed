"""Train a PPO RL-BOED policy for destructive lot inspection."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from testbed.envs import BEDEnv, PaddedHistoryObservation, RLBOEDDenseReward
from testbed.methods import RLBOED, RLBOEDConfig
from testbed.tasks import DestructiveLotInspection

EVAL_N_CONTRASTIVE = 10_000


@dataclass
class Args:
    exp_name: str = Path(__file__).stem
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    save: bool = True
    output_dir: str = "runs"

    num_zones: int = 4
    prior_a: float = 1.0
    prior_b: float = 19.0
    horizon: int = 6
    n_contrastive: int = 256

    num_training_iterations: int = 200
    num_envs: int = 64
    gamma: float = 0.99
    ppo_learning_rate: float = 2.5e-4
    ppo_anneal_lr: bool = True
    ppo_gae_lambda: float = 0.95
    ppo_num_minibatches: int = 4
    ppo_update_epochs: int = 4
    ppo_norm_adv: bool = True
    ppo_clip_coef: float = 0.2
    ppo_clip_vloss: bool = True
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_target_kl: float | None = None
    log_frequency: int = 10


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

    parser.add_argument("--num-zones", type=int, default=Args.num_zones)
    parser.add_argument("--prior-a", type=float, default=Args.prior_a)
    parser.add_argument("--prior-b", type=float, default=Args.prior_b)
    parser.add_argument("--horizon", type=int, default=Args.horizon)
    parser.add_argument("--n-contrastive", type=int, default=Args.n_contrastive)

    parser.add_argument(
        "--num-training-iterations",
        type=int,
        default=Args.num_training_iterations,
    )
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument("--gamma", type=float, default=Args.gamma)
    parser.add_argument(
        "--ppo-learning-rate",
        type=float,
        default=Args.ppo_learning_rate,
    )
    parser.add_argument(
        "--ppo-anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=Args.ppo_anneal_lr,
    )
    parser.add_argument("--ppo-gae-lambda", type=float, default=Args.ppo_gae_lambda)
    parser.add_argument(
        "--ppo-num-minibatches",
        type=int,
        default=Args.ppo_num_minibatches,
    )
    parser.add_argument("--ppo-update-epochs", type=int, default=Args.ppo_update_epochs)
    parser.add_argument(
        "--ppo-norm-adv",
        action=argparse.BooleanOptionalAction,
        default=Args.ppo_norm_adv,
    )
    parser.add_argument("--ppo-clip-coef", type=float, default=Args.ppo_clip_coef)
    parser.add_argument(
        "--ppo-clip-vloss",
        action=argparse.BooleanOptionalAction,
        default=Args.ppo_clip_vloss,
    )
    parser.add_argument("--ppo-ent-coef", type=float, default=Args.ppo_ent_coef)
    parser.add_argument("--ppo-vf-coef", type=float, default=Args.ppo_vf_coef)
    parser.add_argument(
        "--ppo-max-grad-norm",
        type=float,
        default=Args.ppo_max_grad_norm,
    )
    parser.add_argument("--ppo-target-kl", type=float, default=Args.ppo_target_kl)
    parser.add_argument("--log-frequency", type=int, default=Args.log_frequency)
    return Args(**vars(parser.parse_args()))


def make_task(args: Args, device: torch.device):
    return DestructiveLotInspection(
        num_zones=args.num_zones,
        prior_a=args.prior_a,
        prior_b=args.prior_b,
    ).to(device)


def train(args: Args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    config = RLBOEDConfig(
        outcome_low=0.0,
        outcome_high=1.0,
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        n_contrastive=args.n_contrastive,
        num_training_iterations=args.num_training_iterations,
        num_envs=args.num_envs,
        gamma=args.gamma,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_anneal_lr=args.ppo_anneal_lr,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_num_minibatches=args.ppo_num_minibatches,
        ppo_update_epochs=args.ppo_update_epochs,
        ppo_norm_adv=args.ppo_norm_adv,
        ppo_clip_coef=args.ppo_clip_coef,
        ppo_clip_vloss=args.ppo_clip_vloss,
        ppo_ent_coef=args.ppo_ent_coef,
        ppo_vf_coef=args.ppo_vf_coef,
        ppo_max_grad_norm=args.ppo_max_grad_norm,
        ppo_target_kl=args.ppo_target_kl,
        log_frequency=args.log_frequency,
    )
    result = RLBOED(config).fit(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
    )

    if not args.save:
        return result, None

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.output_dir) / run_name / "ppo_destructive_lot.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(checkpoint_path)
    return result, checkpoint_path


@torch.no_grad()
def evaluate_policy_mean(args: Args, policy_net) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=RLBOEDDenseReward(EVAL_N_CONTRASTIVE),
        observation_fn=PaddedHistoryObservation(args.horizon),
        batch_shape=(args.num_envs,),
    )
    policy_net.eval()
    observation, _ = env.reset(seed=args.seed + 1)
    returns = torch.zeros(args.num_envs, device=device)
    for _ in range(args.horizon):
        designs = policy_net.sample_from_observation(
            observation,
            deterministic=True,
        ).action
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
        f"contrastive samples: {mean:.4f} +/- {standard_error:.4f}"
    )
