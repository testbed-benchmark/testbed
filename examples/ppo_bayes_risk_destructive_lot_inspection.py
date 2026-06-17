"""Train a PPO policy minimizing Bayes risk for destructive lot inspection."""

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import Tensor

from testbed.design_nets import CategoricalPolicyNet, RLBOEDValueNetwork
from testbed.designers import PolicyDesigner
from testbed.envs import BEDEnv, PaddedHistoryObservation, RewardFunction
from testbed.envs.history import History
from testbed.methods import FitResult
from testbed.tasks import DestructiveLotInspection

EVAL_EPISODES = 10_000


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
    tau: float = 0.10
    alpha: float = 0.05
    false_accept_cost: float = 10.0
    false_reject_cost: float = 1.0
    eval_episodes: int = EVAL_EPISODES

    num_training_iterations: int = 200
    num_envs: int = 64
    gamma: float = 1.0
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
    verbose: bool = True


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
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument("--alpha", type=float, default=Args.alpha)
    parser.add_argument(
        "--false-accept-cost",
        type=float,
        default=Args.false_accept_cost,
    )
    parser.add_argument(
        "--false-reject-cost",
        type=float,
        default=Args.false_reject_cost,
    )
    parser.add_argument("--eval-episodes", type=int, default=Args.eval_episodes)

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
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=Args.verbose,
    )
    return Args(**vars(parser.parse_args()))


def make_task(args: Args, device: torch.device):
    return DestructiveLotInspection(
        num_zones=args.num_zones,
        prior_a=args.prior_a,
        prior_b=args.prior_b,
    ).to(device)


def _beta_ppf(q, a, b):
    try:
        from scipy.stats import beta
    except ImportError as error:
        raise ImportError(
            "This example requires scipy for Clopper-Pearson bounds. "
            "Install it with `pip install scipy`."
        ) from error
    return beta.ppf(q, a, b)


def clopper_pearson_upper(
    n: Tensor,
    s: Tensor,
    alpha: float,
) -> Tensor:
    """Computes one-sided exact binomial upper confidence bounds."""
    n_cpu = n.detach().cpu().numpy()
    s_cpu = s.detach().cpu().numpy()
    upper = np.ones_like(n_cpu, dtype=np.float64)
    valid = (n_cpu > 0) & (s_cpu < n_cpu)
    upper[valid] = _beta_ppf(
        1.0 - alpha,
        s_cpu[valid] + 1.0,
        n_cpu[valid] - s_cpu[valid],
    )
    return torch.as_tensor(upper, device=n.device, dtype=torch.get_default_dtype())


def zone_counts(history: History, num_zones: int) -> tuple[Tensor, Tensor]:
    """Counts inspected and defective units per zone."""
    designs = history.designs.long()
    outcomes = history.outcomes.squeeze(-1)
    mask = history.mask
    n_by_zone = []
    s_by_zone = []
    for zone in range(num_zones):
        selected = (designs == zone) & mask
        n_by_zone.append(selected.sum(dim=-1))
        s_by_zone.append((outcomes * selected.to(outcomes.dtype)).sum(dim=-1))
    n = torch.stack(n_by_zone, dim=-1)
    s = torch.stack(s_by_zone, dim=-1)
    return n, s


def downstream_accepts_lot(
    history: History,
    *,
    num_zones: int,
    tau: float,
    alpha: float,
) -> Tensor:
    n, s = zone_counts(history, num_zones)
    upper = clopper_pearson_upper(n, s, alpha)
    return (upper <= tau).all(dim=-1)


def lot_loss(
    accepted: Tensor,
    theta: Tensor,
    *,
    tau: float,
    false_accept_cost: float,
    false_reject_cost: float,
) -> Tensor:
    lot_good = (theta <= tau).all(dim=-1)
    false_accept = accepted & ~lot_good
    false_reject = ~accepted & lot_good
    loss = torch.zeros(theta.shape[:-1], device=theta.device)
    false_accept_loss = torch.full_like(loss, false_accept_cost)
    false_reject_loss = torch.full_like(loss, false_reject_cost)
    loss = torch.where(false_accept, false_accept_loss, loss)
    loss = torch.where(false_reject, false_reject_loss, loss)
    return loss


class BayesRiskTerminalReward(RewardFunction):
    """Terminal reward for minimizing downstream lot-decision Bayes risk."""

    requires_likelihood = False

    def __init__(
        self,
        *,
        num_zones: int,
        tau: float,
        alpha: float,
        false_accept_cost: float,
        false_reject_cost: float,
    ) -> None:
        self.num_zones = int(num_zones)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.false_accept_cost = float(false_accept_cost)
        self.false_reject_cost = float(false_reject_cost)

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        if env.t + 1 < env.horizon:
            return torch.zeros(env.batch_shape, device=env.device)
        if env.theta_0 is None:
            raise RuntimeError("Environment has not been reset.")
        accepted = downstream_accepts_lot(
            env.history,
            num_zones=self.num_zones,
            tau=self.tau,
            alpha=self.alpha,
        )
        loss = lot_loss(
            accepted,
            env.theta_0,
            tau=self.tau,
            false_accept_cost=self.false_accept_cost,
            false_reject_cost=self.false_reject_cost,
        )
        return -loss


def _clone_observation(obs: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().clone() for key, value in obs.items()}


def _flatten_rollout_observations(
    observations: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    stacked = {
        key: torch.stack([obs[key] for obs in observations]) for key in observations[0]
    }
    return {
        key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        for key, value in stacked.items()
    }


def _index_observation(
    obs: dict[str, Tensor],
    indices: Tensor,
) -> dict[str, Tensor]:
    return {key: value[indices] for key, value in obs.items()}


def _validate_args(args: Args) -> None:
    if args.horizon <= 0:
        raise ValueError("horizon must be positive.")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if args.num_training_iterations <= 0:
        raise ValueError("num_training_iterations must be positive.")
    if args.num_envs * args.horizon % args.ppo_num_minibatches != 0:
        raise ValueError("num_envs * horizon must be divisible by ppo_num_minibatches.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0.0 <= args.tau <= 1.0:
        raise ValueError("tau must be between 0 and 1.")
    if args.false_accept_cost < 0.0 or args.false_reject_cost < 0.0:
        raise ValueError("loss costs must be non-negative.")


def train(args: Args):
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    reward_fn = BayesRiskTerminalReward(
        num_zones=args.num_zones,
        tau=args.tau,
        alpha=args.alpha,
        false_accept_cost=args.false_accept_cost,
        false_reject_cost=args.false_reject_cost,
    )
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=reward_fn,
        observation_fn=PaddedHistoryObservation(args.horizon),
        batch_shape=(args.num_envs,),
    )

    dim_y = task.event_shape_y.numel()
    policy = CategoricalPolicyNet(
        task.design_space,
        dim_y=dim_y,
        outcome_bounds=(0.0, 1.0),
    ).to(device)
    value_net = RLBOEDValueNetwork(
        dim_y=dim_y,
        dim_design=task.event_shape_design.numel(),
        design_low=policy.design_low,
        design_high=policy.design_high,
        outcome_low=0.0,
        outcome_high=1.0,
    ).to(device)
    trainable_parameters = list(policy.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(trainable_parameters, lr=args.ppo_learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.horizon
    minibatch_size = batch_size // args.ppo_num_minibatches
    metrics: list[dict[str, float]] = []
    start_time = time.time()
    env_steps = 0
    completed_episodes = 0

    value_loss = torch.tensor(float("nan"))
    policy_loss = torch.tensor(float("nan"))
    entropy_loss = torch.tensor(float("nan"))
    old_approx_kl = torch.tensor(float("nan"))
    approx_kl = torch.tensor(float("nan"))

    for training_iteration in range(args.num_training_iterations):
        if args.ppo_anneal_lr:
            frac = 1.0 - training_iteration / args.num_training_iterations
            optimizer.param_groups[0]["lr"] = frac * args.ppo_learning_rate

        obs, _ = env.reset(seed=args.seed if training_iteration == 0 else None)
        next_done = torch.zeros(args.num_envs, device=device)
        rollout_obs = []
        actions = []
        logprobs = []
        rewards = torch.zeros(args.horizon, args.num_envs, device=device)
        dones = torch.zeros(args.horizon, args.num_envs, device=device)
        values = torch.zeros(args.horizon, args.num_envs, device=device)
        episodic_returns = torch.zeros(args.num_envs, device=device)

        for step in range(args.horizon):
            rollout_obs.append(_clone_observation(obs))
            dones[step] = next_done
            with torch.no_grad():
                sample = policy.sample_from_observation(obs)
                value = value_net(obs).flatten()
            actions.append(sample.action.detach())
            logprobs.append(sample.log_prob.detach())
            values[step] = value

            obs, reward, terminations, truncations, _ = env.step(sample.action)
            next_done = (terminations | truncations).float()
            rewards[step] = reward
            episodic_returns += reward
            env_steps += args.num_envs

        with torch.no_grad():
            next_value = value_net(obs).flatten()
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for step in reversed(range(args.horizon)):
                if step == args.horizon - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[step + 1]
                    nextvalues = values[step + 1]
                delta = (
                    rewards[step]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[step]
                )
                advantages[step] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.ppo_gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        b_obs = _flatten_rollout_observations(rollout_obs)
        b_actions = torch.stack(actions).reshape(
            batch_size,
            *task.event_shape_design,
        )
        b_logprobs = torch.stack(logprobs).reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.ppo_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = torch.as_tensor(
                    b_inds[start : start + minibatch_size],
                    device=device,
                )
                mb_obs = _index_observation(b_obs, mb_inds)
                newlogprob, entropy = policy.evaluate_actions_from_observation(
                    mb_obs,
                    b_actions[mb_inds],
                )
                newvalue = value_net(mb_obs).view(-1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.ppo_clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.ppo_norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - args.ppo_clip_coef,
                    1 + args.ppo_clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.ppo_clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.ppo_clip_coef,
                        args.ppo_clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    value_loss = 0.5 * torch.max(
                        v_loss_unclipped,
                        v_loss_clipped,
                    ).mean()
                else:
                    value_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    - args.ppo_ent_coef * entropy_loss
                    + args.ppo_vf_coef * value_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    args.ppo_max_grad_norm,
                )
                optimizer.step()

            if args.ppo_target_kl is not None and approx_kl > args.ppo_target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        clipfrac = float(np.mean(clipfracs)) if clipfracs else float("nan")
        completed_episodes += args.num_envs

        episode_metrics = {
            "step": float(training_iteration),
            "env_steps": float(env_steps),
            "episodes": float(completed_episodes),
            "episodic_return": episodic_returns.mean().item(),
            "bayes_risk": (-episodic_returns).mean().item(),
        }
        metrics.append(episode_metrics)

        if training_iteration % args.log_frequency == 0:
            step_metrics = {
                "step": float(training_iteration),
                "env_steps": float(env_steps),
                "sps": float(int(env_steps / (time.time() - start_time))),
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": clipfrac,
                "explained_variance": float(explained_var),
            }
            metrics.append(step_metrics)
            if args.verbose:
                print(
                    f"iteration={training_iteration}, "
                    f"env_steps={env_steps}, "
                    f"risk={episode_metrics['bayes_risk']:.4f}, "
                    f"value_loss={step_metrics['value_loss']:.4f}, "
                    f"policy_loss={step_metrics['policy_loss']:.4f}, "
                    f"entropy={step_metrics['entropy']:.4f}"
                )

    policy.eval()
    designer = PolicyDesigner(task, args.horizon, policy, deterministic=False)
    checkpoint = {
        "algorithm": "ppo_bayes_risk",
        "config": asdict(args),
        "horizon": args.horizon,
        "tau": torch.as_tensor(args.tau),
        "alpha": torch.as_tensor(args.alpha),
        "false_accept_cost": torch.as_tensor(args.false_accept_cost),
        "false_reject_cost": torch.as_tensor(args.false_reject_cost),
        "design_low": policy.design_low.detach().cpu(),
        "design_high": policy.design_high.detach().cpu(),
        "outcome_low": torch.as_tensor(0.0),
        "outcome_high": torch.as_tensor(1.0),
        "designer": policy.state_dict(),
        "policy": policy.state_dict(),
        "value_net": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_steps": env_steps,
        "training_iterations": args.num_training_iterations,
        "global_step": env_steps,
    }
    result = FitResult(designer=designer, metrics=metrics, checkpoint=checkpoint)

    if not args.save:
        return result, None

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.output_dir) / run_name / "ppo_bayes_risk_destructive_lot.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(checkpoint_path)
    return result, checkpoint_path


@torch.no_grad()
def evaluate_policy_mean_risk(args: Args, policy_net) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    reward_fn = BayesRiskTerminalReward(
        num_zones=args.num_zones,
        tau=args.tau,
        alpha=args.alpha,
        false_accept_cost=args.false_accept_cost,
        false_reject_cost=args.false_reject_cost,
    )
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=reward_fn,
        observation_fn=PaddedHistoryObservation(args.horizon),
        batch_shape=(args.eval_episodes,),
    )
    policy_net.eval()
    observation, _ = env.reset(seed=args.seed + 1)
    returns = torch.zeros(args.eval_episodes, device=device)
    for _ in range(args.horizon):
        designs = policy_net.sample_from_observation(
            observation,
            deterministic=True,
        ).action
        observation, rewards, _, _, _ = env.step(designs)
        returns += rewards

    if env.theta_0 is None:
        raise RuntimeError("Environment has not been reset.")
    accepted = downstream_accepts_lot(
        env.history,
        num_zones=args.num_zones,
        tau=args.tau,
        alpha=args.alpha,
    )
    lot_good = (env.theta_0 <= args.tau).all(dim=-1)
    false_accept = accepted & ~lot_good
    false_reject = ~accepted & lot_good
    risks = -returns
    return {
        "mean_risk": risks.mean().item(),
        "standard_error": risks.std(unbiased=True).div(args.eval_episodes**0.5).item(),
        "acceptance_rate": accepted.to(torch.float32).mean().item(),
        "false_accept_rate": false_accept.to(torch.float32).mean().item(),
        "false_reject_rate": false_reject.to(torch.float32).mean().item(),
    }


if __name__ == "__main__":
    args = parse_args()
    result, checkpoint = train(args)
    if checkpoint is not None:
        print(f"saved checkpoint to {checkpoint}")
    evaluation = evaluate_policy_mean_risk(args, result.designer.policy_net)
    print(
        "deterministic policy Bayes risk: "
        f"{evaluation['mean_risk']:.4f} +/- {evaluation['standard_error']:.4f}"
    )
    print(
        "decision rates: "
        f"accept={evaluation['acceptance_rate']:.4f}, "
        f"false_accept={evaluation['false_accept_rate']:.4f}, "
        f"false_reject={evaluation['false_reject_rate']:.4f}"
    )
