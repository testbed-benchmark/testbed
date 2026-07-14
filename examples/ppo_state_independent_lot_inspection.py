"""Run state-independent PPO with separate actor and critic optimizers."""

import argparse
from dataclasses import asdict, dataclass

import torch
from ppo_destructive_lot_inspection import (
    Args,
    decision_maker_location_probs,
    evaluate_policy_mean_risk,
    make_reward,
    make_task,
    parse_suspicious_zones,
)
from torch import Tensor, nn

from testbed.design_nets import CategoricalPolicyActor, CategoricalPolicyNet
from testbed.envs import CategoricalDesignSpace, DesignSpace
from testbed.methods import RLBOED, RLBOEDConfig
from testbed.tasks.base_task import Task


@dataclass
class StateIndependentArgs(Args):
    """Sets the separate learning rates used by the diagnostic."""

    ppo_learning_rate: float = 0.01
    ppo_critic_learning_rate: float = 2.5e-4


class StateIndependentCategoricalActor(CategoricalPolicyActor):
    """Represents one trainable logit per zone, independent of history."""

    def __init__(self, num_zones: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_zones))

    def forward(
        self,
        y_history: Tensor,
        design_history: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Expands the shared zone logits across the observation batch."""
        return self.logits.expand(y_history.shape[0], -1)


class StateIndependentRLBOED(RLBOED):
    """Runs the standard PPO update with state-independent policy logits."""

    def __init__(self, config: RLBOEDConfig, suspicious_zones: tuple[int, ...]):
        super().__init__(config)
        self.suspicious_zones = suspicious_zones

    def _make_ppo_policy(
        self,
        task: Task,
        design_space: DesignSpace,
    ) -> CategoricalPolicyNet:
        if not isinstance(design_space, CategoricalDesignSpace):
            raise TypeError("The state-independent diagnostic requires categories.")
        return CategoricalPolicyNet(
            design_space,
            actor=StateIndependentCategoricalActor(design_space.n),
            dim_y=task.event_shape_y.numel(),
            outcome_bounds=(self.config.outcome_low, self.config.outcome_high),
        )

    def _ppo_policy_metrics(
        self,
        policy: CategoricalPolicyNet,
        _observation: dict[str, Tensor],
    ) -> dict[str, float]:
        probabilities = policy.actor.logits.detach().softmax(dim=-1)
        return {
            "suspicious_zone_mass": probabilities[list(self.suspicious_zones)]
            .sum()
            .item()
        }


def parse_args() -> StateIndependentArgs:
    """Parses the controls needed for the matched PPO diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument(
        "--cuda", action=argparse.BooleanOptionalAction, default=Args.cuda
    )
    parser.add_argument("--num-training-iterations", type=int, default=2_000)
    parser.add_argument("--num-envs", type=int, default=Args.num_envs)
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=StateIndependentArgs.ppo_learning_rate,
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=StateIndependentArgs.ppo_critic_learning_rate,
    )
    parser.add_argument("--max-grad-norm", type=float, default=Args.ppo_max_grad_norm)
    parser.add_argument("--eval-episodes", type=int, default=Args.eval_episodes)
    parser.add_argument("--log-frequency", type=int, default=100)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=Args.verbose
    )
    values = vars(parser.parse_args())
    return StateIndependentArgs(
        exp_name="ppo_state_independent_matched_lot_inspection",
        seed=values["seed"],
        cuda=values["cuda"],
        save=False,
        num_training_iterations=values["num_training_iterations"],
        num_envs=values["num_envs"],
        ppo_learning_rate=values["actor_learning_rate"],
        ppo_critic_learning_rate=values["critic_learning_rate"],
        ppo_anneal_lr=False,
        ppo_max_grad_norm=values["max_grad_norm"],
        eval_episodes=values["eval_episodes"],
        log_frequency=values["log_frequency"],
        verbose=values["verbose"],
    )


def train(args: StateIndependentArgs):
    """Trains state-independent logits with the matched PPO configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    reward_fn = make_reward(args, task)
    config = RLBOEDConfig(
        outcome_low=0.0,
        outcome_high=float(args.batch_size),
        seed=args.seed,
        torch_deterministic=args.torch_deterministic,
        n_contrastive=0,
        num_training_iterations=args.num_training_iterations,
        num_envs=args.num_envs,
        gamma=args.gamma,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_actor_learning_rate=args.ppo_learning_rate,
        ppo_critic_learning_rate=args.ppo_critic_learning_rate,
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
        verbose=args.verbose,
    )
    suspicious_zones = tuple(int(zone) for zone in args.dm_suspicious_zones.split(","))
    result = StateIndependentRLBOED(config, suspicious_zones).fit(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=reward_fn,
    )
    result.checkpoint.update(
        {
            "algorithm": "ppo_state_independent_matched",
            "config": asdict(args),
            "dm_acceptance_threshold": torch.as_tensor(
                reward_fn.decision_rule.threshold
            ),
            "dm_location_probs": decision_maker_location_probs(
                args, torch.device("cpu")
            ),
        }
    )
    return result


def suspicious_zone_mass(args: Args, policy: CategoricalPolicyNet) -> float:
    """Returns total action probability assigned to suspicious zones."""
    logits = policy.actor.logits.detach()
    zones = [int(zone) for zone in args.dm_suspicious_zones.split(",")]
    return logits.softmax(dim=-1)[zones].sum().item()


def policy_from_probabilities(
    args: Args,
    probabilities: Tensor,
) -> CategoricalPolicyNet:
    """Builds a fixed state-independent policy for evaluation."""
    policy = CategoricalPolicyNet(
        CategoricalDesignSpace(args.num_zones),
        actor=StateIndependentCategoricalActor(args.num_zones),
        dim_y=1,
        outcome_bounds=(0.0, float(args.batch_size)),
    ).to(probabilities.device)
    with torch.no_grad():
        policy.actor.logits.copy_(probabilities.log())
    return policy


def print_risk(label: str, evaluation: dict[str, float]) -> None:
    """Prints a labeled Bayes-risk estimate."""
    print(
        f"{label} Bayes risk: "
        f"{evaluation['mean_risk']:.4f} +/- {evaluation['standard_error']:.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    result = train(args)
    policy = result.designer.policy_net
    probabilities = policy.actor.logits.detach().softmax(dim=-1)
    suspicious_zones = parse_suspicious_zones(
        args.dm_suspicious_zones,
        args.num_zones,
    )
    uniform_probabilities = torch.full_like(probabilities, 1.0 / args.num_zones)
    suspicious_probabilities = torch.zeros_like(probabilities)
    suspicious_probabilities[list(suspicious_zones)] = 1.0 / len(suspicious_zones)

    uniform_evaluation = evaluate_policy_mean_risk(
        args,
        policy_from_probabilities(args, uniform_probabilities),
    )
    suspicious_evaluation = evaluate_policy_mean_risk(
        args,
        policy_from_probabilities(args, suspicious_probabilities),
    )
    learned_evaluation = evaluate_policy_mean_risk(args, policy)

    print(
        "learned zone probabilities: ["
        + ", ".join(f"{probability:.4f}" for probability in probabilities.tolist())
        + "]"
    )
    print(f"suspicious-zone mass: {suspicious_zone_mass(args, policy):.4f}")
    print_risk("uniform-all-zones", uniform_evaluation)
    print_risk("uniform-suspicious-zones", suspicious_evaluation)
    print_risk("learned", learned_evaluation)
