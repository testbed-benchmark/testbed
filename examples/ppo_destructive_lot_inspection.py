"""Train PPO for a fixed conservative lot-inspection decision rule."""

import argparse
import time
from dataclasses import asdict, dataclass, field
from math import log
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from testbed.design_nets import CategoricalPolicyActor, CategoricalPolicyNet
from testbed.design_nets.modules import MLP, _initialize_ppo_linear_layers
from testbed.designers import PolicyDesigner
from testbed.envs import BEDEnv, RewardFunction
from testbed.envs.history import History
from testbed.methods.ppo import (
    Trainer as PPOTrainer,
)
from testbed.methods.ppo import (
    _set_seed,
    validate_config,
)
from testbed.tasks import DestructiveLotInspection

EVAL_EPISODES = 10_000


def _zone_one_hot(
    design_history: Tensor,
    *,
    num_zones: int,
    normalized: bool,
) -> Tensor:
    """Encodes scalar zone histories as one-hot zone identities."""
    zone = design_history.squeeze(-1)
    if normalized:
        zone = zone * (num_zones - 1)
    zone_index = zone.round().long().clamp(0, num_zones - 1)
    return F.one_hot(zone_index, num_classes=num_zones).to(design_history.dtype)


def _lot_summary_features(
    y_history: Tensor,
    design_history: Tensor,
    mask: Tensor | None,
    *,
    num_zones: int,
    horizon: int,
    normalized_designs: bool,
) -> Tensor:
    """Builds per-zone count and defect summaries for lot inspection."""
    zone_features = _zone_one_hot(
        design_history,
        num_zones=num_zones,
        normalized=normalized_designs,
    )
    if mask is not None:
        zone_features = zone_features * mask.to(dtype=zone_features.dtype).unsqueeze(-1)
        y_history = y_history * mask.to(dtype=y_history.dtype).unsqueeze(-1)
    visits = zone_features.sum(dim=1)
    defect_load = (zone_features * y_history).sum(dim=1)
    defect_rate = defect_load / visits.clamp_min(1.0)
    visit_fraction = visits / horizon
    elapsed_fraction = visits.sum(dim=-1, keepdim=True) / horizon
    return torch.cat(
        [visit_fraction, defect_rate, defect_load / horizon, elapsed_fraction],
        dim=-1,
    )


class LotInspectionCategoricalActor(CategoricalPolicyActor):
    """Categorical actor with explicit per-zone inspection summaries."""

    def __init__(
        self,
        *,
        num_zones: int,
        dim_y: int,
        horizon: int,
        initial_zone_probs: Tensor | None = None,
        dim_h: int = 128,
    ) -> None:
        """Initializes the lot-inspection actor."""
        super().__init__()
        if dim_y != 1:
            raise ValueError("LotInspectionCategoricalActor expects scalar outcomes.")
        self.num_zones = int(num_zones)
        self.horizon = int(horizon)
        self.actor_net = MLP(
            [3 * self.num_zones + 1, dim_h, dim_h],
            activation=nn.ReLU,
        )
        self.logits_head = nn.Linear(dim_h, self.num_zones)
        _initialize_ppo_linear_layers(self, output_scales={self.logits_head: 0.01})
        if initial_zone_probs is not None:
            probabilities = torch.as_tensor(
                initial_zone_probs,
                dtype=self.logits_head.bias.dtype,
            )
            probabilities = probabilities / probabilities.sum()
            with torch.no_grad():
                self.logits_head.bias.copy_(probabilities.clamp_min(1e-12).log())

    def forward(
        self,
        y_history: Tensor,
        design_history: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Computes zone logits from normalized summary histories."""
        features = _lot_summary_features(
            y_history,
            design_history,
            num_zones=self.num_zones,
            horizon=self.horizon,
            normalized_designs=True,
            mask=mask,
        )
        return self.logits_head(self.actor_net(features))


class LotInspectionValueNetwork(nn.Module):
    """State-value network with explicit per-zone inspection summaries."""

    def __init__(
        self,
        *,
        num_zones: int,
        dim_y: int,
        horizon: int,
        outcome_low: float,
        outcome_high: float,
        dim_h: int = 128,
    ) -> None:
        """Initializes the lot-inspection value network."""
        super().__init__()
        if dim_y != 1:
            raise ValueError("LotInspectionValueNetwork expects scalar outcomes.")
        self.num_zones = int(num_zones)
        self.horizon = int(horizon)
        self.emitter_net = MLP(
            [3 * self.num_zones + 1, dim_h, dim_h, 1],
            activation=nn.ReLU,
        )
        self.register_buffer("outcome_low", torch.as_tensor(outcome_low))
        self.register_buffer("outcome_high", torch.as_tensor(outcome_high))
        _initialize_ppo_linear_layers(
            self,
            output_scales={self.emitter_net.net[-1]: 1.0},
        )

    def forward(self, observation: History) -> Tensor:
        """Estimates V from one-hot zone histories and normalized outcomes."""
        outcomes = observation.outcomes
        designs = observation.designs.reshape(*observation.designs.shape, 1)
        outcomes = (outcomes - self.outcome_low) / (self.outcome_high - self.outcome_low)
        features = _lot_summary_features(
            outcomes,
            designs,
            num_zones=self.num_zones,
            horizon=self.horizon,
            normalized_designs=False,
            mask=observation.mask,
        )
        return self.emitter_net(features)


@dataclass
class FitResult:
    """Trained designer, metrics, and serializable checkpoint."""

    designer: PolicyDesigner
    metrics: list[dict[str, float]] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Saves the training checkpoint."""
        torch.save(self.checkpoint, Path(path))


@dataclass
class Args:
    exp_name: str = Path(__file__).stem
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    save: bool = True
    output_dir: str = "runs"

    num_zones: int = 16
    horizon: int = 30
    batch_size: int = 8
    length_scale: float = 1.5
    theta_base: float = 0.02
    tau: float = 0.12
    experimenter_good_probability: float = 0.6
    good_rho_loc: float = 0.08
    good_rho_scale: float = 0.02
    bad_rho_loc: float = 0.16
    bad_rho_scale: float = 0.025

    experimenter_false_accept_cost: float = 1.0
    experimenter_false_reject_cost: float = 1.0
    dm_good_probability: float = 0.4
    dm_false_accept_cost: float = 4.0
    dm_false_reject_cost: float = 1.0
    dm_suspicious_zones: str = "0,4,8,12"
    dm_suspicion_weight: float = 0.75
    dm_rho_grid_size: int = 64
    eval_episodes: int = EVAL_EPISODES

    num_training_iterations: int = 2_000
    num_envs: int = 256
    gamma: float = 1.0
    ppo_learning_rate: float = 2.5e-4
    ppo_critic_learning_rate: float = 2.5e-4
    ppo_anneal_lr: bool = False
    ppo_gae_lambda: float = 1.0
    ppo_num_minibatches: int = 4
    ppo_update_epochs: int = 4
    ppo_norm_adv: bool = True
    ppo_clip_coef: float = 0.2
    ppo_clip_vloss: bool = True
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_target_kl: float | None = None
    log_frequency: int = 10
    verbose: bool = True
    eval_deterministic: bool = False


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
    parser.add_argument("--horizon", type=int, default=Args.horizon)
    parser.add_argument("--batch-size", type=int, default=Args.batch_size)
    parser.add_argument("--length-scale", type=float, default=Args.length_scale)
    parser.add_argument("--theta-base", type=float, default=Args.theta_base)
    parser.add_argument("--tau", type=float, default=Args.tau)
    parser.add_argument(
        "--experimenter-good-probability",
        type=float,
        default=Args.experimenter_good_probability,
    )
    parser.add_argument("--good-rho-loc", type=float, default=Args.good_rho_loc)
    parser.add_argument("--good-rho-scale", type=float, default=Args.good_rho_scale)
    parser.add_argument("--bad-rho-loc", type=float, default=Args.bad_rho_loc)
    parser.add_argument("--bad-rho-scale", type=float, default=Args.bad_rho_scale)

    parser.add_argument(
        "--experimenter-false-accept-cost",
        type=float,
        default=Args.experimenter_false_accept_cost,
    )
    parser.add_argument(
        "--experimenter-false-reject-cost",
        type=float,
        default=Args.experimenter_false_reject_cost,
    )
    parser.add_argument(
        "--dm-good-probability",
        type=float,
        default=Args.dm_good_probability,
    )
    parser.add_argument(
        "--dm-false-accept-cost",
        type=float,
        default=Args.dm_false_accept_cost,
    )
    parser.add_argument(
        "--dm-false-reject-cost",
        type=float,
        default=Args.dm_false_reject_cost,
    )
    parser.add_argument(
        "--dm-suspicious-zones",
        default=Args.dm_suspicious_zones,
        help="Comma-separated zero-based zone indices.",
    )
    parser.add_argument(
        "--dm-suspicion-weight",
        type=float,
        default=Args.dm_suspicion_weight,
    )
    parser.add_argument("--dm-rho-grid-size", type=int, default=Args.dm_rho_grid_size)
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
        help="Actor learning rate.",
    )
    parser.add_argument(
        "--ppo-critic-learning-rate",
        type=float,
        default=Args.ppo_critic_learning_rate,
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
    parser.add_argument(
        "--eval-deterministic",
        action=argparse.BooleanOptionalAction,
        default=Args.eval_deterministic,
        help="Use categorical argmax actions instead of sampling at evaluation.",
    )
    return Args(**vars(parser.parse_args()))


def make_task(args: Args, device: torch.device) -> DestructiveLotInspection:
    """Builds the experimenter's data-generating task."""
    return DestructiveLotInspection(
        num_zones=args.num_zones,
        length_scale=args.length_scale,
        theta_base=args.theta_base,
        tau=args.tau,
        batch_size=args.batch_size,
        good_probability=args.experimenter_good_probability,
        good_rho_loc=args.good_rho_loc,
        good_rho_scale=args.good_rho_scale,
        bad_rho_loc=args.bad_rho_loc,
        bad_rho_scale=args.bad_rho_scale,
    ).to(device)


def make_lot_inspection_networks(args: Args, task: DestructiveLotInspection):
    """Builds zone-aware PPO actor and critic networks for this example."""
    actor = LotInspectionCategoricalActor(
        num_zones=args.num_zones,
        dim_y=task.event_shape_y.numel(),
        horizon=args.horizon,
        initial_zone_probs=decision_maker_location_probs(args, torch.device("cpu")),
    )
    policy = CategoricalPolicyNet(
        task.design_space,
        actor=actor,
        dim_y=task.event_shape_y.numel(),
        outcome_bounds=(0.0, float(args.batch_size)),
    ).to(task.device)
    value = LotInspectionValueNetwork(
        num_zones=args.num_zones,
        dim_y=task.event_shape_y.numel(),
        horizon=args.horizon,
        outcome_low=0.0,
        outcome_high=float(args.batch_size),
    ).to(task.device)
    return policy, value


def parse_suspicious_zones(value: str, num_zones: int) -> tuple[int, ...]:
    """Parses and validates zero-based suspicious zone indices."""
    zones = tuple(
        sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    )
    if any(zone < 0 or zone >= num_zones for zone in zones):
        raise ValueError(f"suspicious zones must lie between 0 and {num_zones - 1}.")
    return zones


def decision_maker_location_probs(args: Args, device: torch.device) -> Tensor:
    """Constructs the conservative decision maker's location prior."""
    suspicious = parse_suspicious_zones(args.dm_suspicious_zones, args.num_zones)
    eta = args.dm_suspicion_weight
    if not 0.0 <= eta <= 1.0:
        raise ValueError("dm_suspicion_weight must lie between 0 and 1.")
    if eta > 0.0 and not suspicious:
        raise ValueError("dm_suspicious_zones cannot be empty when eta is positive.")

    probabilities = torch.full(
        (args.num_zones,),
        (1.0 - eta) / args.num_zones,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    if suspicious:
        probabilities[list(suspicious)] += eta / len(suspicious)
    return probabilities


def lot_loss(
    accepted: Tensor,
    theta: Tensor,
    *,
    false_accept_cost: float,
    false_reject_cost: float,
) -> Tensor:
    """Evaluates the experimenter's asymmetric terminal decision loss."""
    good = theta[..., 0].bool()
    return false_accept_cost * (accepted & ~good).to(
        theta.dtype
    ) + false_reject_cost * (~accepted & good).to(theta.dtype)


class ConservativeDecisionRule:
    """Fixed posterior-threshold rule under the decision maker's prior.

    The posterior integral over ``rho`` uses deterministic midpoint quantiles
    of each class-conditional truncated normal. Location and quality class are
    summed exactly.
    """

    posterior_batch_size = 256

    def __init__(
        self,
        task: DestructiveLotInspection,
        *,
        good_probability: float,
        location_probs: Tensor,
        false_accept_cost: float,
        false_reject_cost: float,
        rho_grid_size: int,
        good_rho_loc: float,
        good_rho_scale: float,
        bad_rho_loc: float,
        bad_rho_scale: float,
    ) -> None:
        if not 0.0 < good_probability < 1.0:
            raise ValueError("dm_good_probability must lie strictly between 0 and 1.")
        if false_accept_cost < 0.0 or false_reject_cost < 0.0:
            raise ValueError("decision-maker loss costs must be non-negative.")
        if false_accept_cost + false_reject_cost <= 0.0:
            raise ValueError("at least one decision-maker loss cost must be positive.")
        if rho_grid_size <= 0:
            raise ValueError("dm_rho_grid_size must be positive.")

        self.task = task
        self.good_probability = float(good_probability)
        self.threshold = false_accept_cost / (false_accept_cost + false_reject_cost)
        self.theta, self.log_prior_weights = self._make_grid(
            location_probs=location_probs,
            rho_grid_size=rho_grid_size,
            good_rho_loc=good_rho_loc,
            good_rho_scale=good_rho_scale,
            bad_rho_loc=bad_rho_loc,
            bad_rho_scale=bad_rho_scale,
        )
        self.good_mask = self.theta[:, 0].bool()

    def _truncated_normal_grid(
        self,
        *,
        loc: float,
        scale: float,
        lower: Tensor,
        upper: Tensor,
        size: int,
    ) -> Tensor:
        reference = self.task.theta_base
        normal = torch.distributions.Normal(
            torch.zeros((), device=reference.device, dtype=reference.dtype),
            torch.ones((), device=reference.device, dtype=reference.dtype),
        )
        loc_tensor = reference.new_tensor(loc)
        scale_tensor = reference.new_tensor(scale)
        lower_cdf = normal.cdf((lower - loc_tensor) / scale_tensor)
        upper_cdf = normal.cdf((upper - loc_tensor) / scale_tensor)
        quantiles = (torch.arange(size, device=reference.device) + 0.5) / size
        probabilities = lower_cdf + quantiles * (upper_cdf - lower_cdf)
        return loc_tensor + scale_tensor * normal.icdf(probabilities)

    def _make_grid(
        self,
        *,
        location_probs: Tensor,
        rho_grid_size: int,
        good_rho_loc: float,
        good_rho_scale: float,
        bad_rho_loc: float,
        bad_rho_scale: float,
    ) -> tuple[Tensor, Tensor]:
        location_probs = torch.as_tensor(
            location_probs,
            device=self.task.device,
            dtype=torch.get_default_dtype(),
        )
        if location_probs.shape != (self.task.num_zones,):
            raise ValueError(
                f"location_probs must have shape ({self.task.num_zones},)."
            )
        if torch.any(location_probs < 0.0) or location_probs.sum() <= 0.0:
            raise ValueError("location_probs must have positive total mass.")
        location_probs = location_probs / location_probs.sum()

        class_specs = (
            (
                1.0,
                self.good_probability,
                good_rho_loc,
                good_rho_scale,
                self.task.theta_base,
                self.task.tau,
            ),
            (
                0.0,
                1.0 - self.good_probability,
                bad_rho_loc,
                bad_rho_scale,
                self.task.tau,
                self.task.tau.new_tensor(1.0),
            ),
        )
        particles = []
        log_weights = []
        locations = torch.arange(
            self.task.num_zones,
            device=self.task.device,
            dtype=torch.get_default_dtype(),
        )
        for good, class_probability, loc, scale, lower, upper in class_specs:
            rho = self._truncated_normal_grid(
                loc=loc,
                scale=scale,
                lower=lower,
                upper=upper,
                size=rho_grid_size,
            )
            location_grid, rho_grid = torch.meshgrid(locations, rho, indexing="ij")
            particles.append(
                torch.stack(
                    (
                        torch.full_like(rho_grid, good),
                        location_grid.to(rho_grid.dtype),
                        rho_grid,
                    ),
                    dim=-1,
                ).reshape(-1, 3)
            )
            log_weights.append(
                (
                    torch.log(location_probs).unsqueeze(-1)
                    + log(class_probability)
                    - log(rho_grid_size)
                )
                .expand(-1, rho_grid_size)
                .reshape(-1)
            )
        return torch.cat(particles), torch.cat(log_weights)

    def posterior_good_probability(self, history: History) -> Tensor:
        """Computes ``p_DM(G=1 | history)`` for each history in the batch."""
        if (
            len(history.batch_shape) == 1
            and history.batch_shape[0] > self.posterior_batch_size
        ):
            probabilities = []
            for start in range(0, history.batch_shape[0], self.posterior_batch_size):
                stop = start + self.posterior_batch_size
                probabilities.append(
                    self._posterior_good_probability(
                        History(
                            max_episode_length=history.max_episode_length,
                            designs=history.designs[start:stop],
                            outcomes=history.outcomes[start:stop],
                            mask=history.mask[start:stop],
                        )
                    )
                )
            return torch.cat(probabilities)
        return self._posterior_good_probability(history)

    def _posterior_good_probability(self, history: History) -> Tensor:
        batch_ndim = len(history.batch_shape)
        particle_shape = (*([1] * batch_ndim), self.theta.shape[0], 3)
        theta = self.theta.view(particle_shape)
        log_likelihood = torch.zeros(
            *history.batch_shape,
            self.theta.shape[0],
            device=self.task.device,
        )
        for t in range(history.max_episode_length):
            active = history.mask.select(batch_ndim, t).unsqueeze(-1)
            if not active.any():
                continue
            outcome = history.outcomes.select(batch_ndim, t).unsqueeze(batch_ndim)
            design = history.designs.select(batch_ndim, t).unsqueeze(batch_ndim)
            increment = self.task.outcome_logp(outcome, theta, design)
            log_likelihood += torch.where(active, increment, 0.0)

        log_joint = log_likelihood + self.log_prior_weights.view(
            *([1] * batch_ndim), -1
        )
        log_evidence = torch.logsumexp(log_joint, dim=-1)
        log_good_evidence = torch.logsumexp(log_joint[..., self.good_mask], dim=-1)
        return torch.exp(log_good_evidence - log_evidence)

    def __call__(self, history: History) -> Tensor:
        """Returns whether the fixed decision maker accepts each lot."""
        return self.posterior_good_probability(history) >= self.threshold


def make_decision_rule(
    args: Args,
    task: DestructiveLotInspection,
) -> ConservativeDecisionRule:
    """Builds the fixed conservative decision rule."""
    return ConservativeDecisionRule(
        task,
        good_probability=args.dm_good_probability,
        location_probs=decision_maker_location_probs(args, task.device),
        false_accept_cost=args.dm_false_accept_cost,
        false_reject_cost=args.dm_false_reject_cost,
        rho_grid_size=args.dm_rho_grid_size,
        good_rho_loc=args.good_rho_loc,
        good_rho_scale=args.good_rho_scale,
        bad_rho_loc=args.bad_rho_loc,
        bad_rho_scale=args.bad_rho_scale,
    )


def make_experimenter_posterior(
    args: Args,
    task: DestructiveLotInspection,
) -> ConservativeDecisionRule:
    """Builds a posterior calculator under the experimenter's prior."""
    location_probs = getattr(task.prior, "location_probs", None)
    if location_probs is None:
        location_probs = torch.full(
            (args.num_zones,),
            1.0 / args.num_zones,
            device=task.device,
            dtype=torch.get_default_dtype(),
        )
    return ConservativeDecisionRule(
        task,
        good_probability=args.experimenter_good_probability,
        location_probs=location_probs,
        false_accept_cost=1.0,
        false_reject_cost=1.0,
        rho_grid_size=args.dm_rho_grid_size,
        good_rho_loc=args.good_rho_loc,
        good_rho_scale=args.good_rho_scale,
        bad_rho_loc=args.bad_rho_loc,
        bad_rho_scale=args.bad_rho_scale,
    )


class FixedDecisionRuleTerminalReward(RewardFunction):
    """Negative experimenter loss from the fixed downstream decision rule."""

    requires_likelihood = False

    def __init__(
        self,
        decision_rule: ConservativeDecisionRule,
        risk_posterior: ConservativeDecisionRule,
        *,
        false_accept_cost: float,
        false_reject_cost: float,
    ) -> None:
        if false_accept_cost < 0.0 or false_reject_cost < 0.0:
            raise ValueError("experimenter loss costs must be non-negative.")
        self.decision_rule = decision_rule
        self.risk_posterior = risk_posterior
        self.false_accept_cost = float(false_accept_cost)
        self.false_reject_cost = float(false_reject_cost)
        self.last_accepted: Tensor | None = None
        self.last_expected_loss: Tensor | None = None

    def reset(self, env: BEDEnv, *, theta_all: Tensor | None = None) -> None:
        """Clears the cached terminal action for a new rollout."""
        del theta_all
        self.last_accepted = None
        self.last_expected_loss = None

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Returns zero until the final inspection, then negative expected loss."""
        if env.t + 1 < env.horizon:
            return torch.zeros(env.batch_shape, device=env.device)
        accepted = self.decision_rule(env.history)
        posterior_good = self.risk_posterior.posterior_good_probability(env.history)
        expected_loss = torch.where(
            accepted,
            self.false_accept_cost * (1.0 - posterior_good),
            self.false_reject_cost * posterior_good,
        )
        self.last_accepted = accepted
        self.last_expected_loss = expected_loss
        return -expected_loss


def _validate_args(args: Args) -> None:
    if args.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive.")
    if not 0.0 <= args.experimenter_good_probability <= 1.0:
        raise ValueError("experimenter_good_probability must lie between 0 and 1.")
    if (
        args.experimenter_false_accept_cost < 0.0
        or args.experimenter_false_reject_cost < 0.0
    ):
        raise ValueError("experimenter loss costs must be non-negative.")
    parse_suspicious_zones(args.dm_suspicious_zones, args.num_zones)


def make_reward(
    args: Args,
    task: DestructiveLotInspection,
) -> FixedDecisionRuleTerminalReward:
    """Builds the fixed-rule Bayes-risk reward."""
    return FixedDecisionRuleTerminalReward(
        make_decision_rule(args, task),
        make_experimenter_posterior(args, task),
        false_accept_cost=args.experimenter_false_accept_cost,
        false_reject_cost=args.experimenter_false_reject_cost,
    )


class LotInspectionPPOTrainer(PPOTrainer):
    """Adds lot-inspection-specific PPO diagnostics."""

    def __init__(self, *args, suspicious_zones: tuple[int, ...], **kwargs):
        super().__init__(*args, **kwargs)
        self.suspicious_zones = suspicious_zones

    @torch.no_grad()
    def _policy_metrics(self, observation: History) -> dict[str, float]:
        _, outcomes, designs, mask = self.policy._prepare_histories(
            observation.outcomes,
            observation.designs,
            mask=observation.mask,
        )
        probabilities = self.policy.actor(outcomes, designs, mask=mask).softmax(dim=-1)
        return {
            "suspicious_zone_mass": probabilities[:, list(self.suspicious_zones)]
            .sum(dim=-1)
            .mean()
            .item()
        }


def train(args: Args):
    """Trains the design policy against the fixed decision rule."""
    _validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    reward_fn = make_reward(args, task)
    config = OmegaConf.create(
        {
            "seed": args.seed,
            "torch_deterministic": args.torch_deterministic,
            "T": args.horizon,
            "reward": {
                "n_contrastive": 0,
                "outcome_low": 0.0,
                "outcome_high": float(args.batch_size),
            },
            "ppo": {
                "num_training_iterations": args.num_training_iterations,
                "num_envs": args.num_envs,
                "gamma": args.gamma,
                "learning_rate": args.ppo_learning_rate,
                "actor_learning_rate": args.ppo_learning_rate,
                "critic_learning_rate": args.ppo_critic_learning_rate,
                "anneal_lr": args.ppo_anneal_lr,
                "gae_lambda": args.ppo_gae_lambda,
                "num_minibatches": args.ppo_num_minibatches,
                "update_epochs": args.ppo_update_epochs,
                "norm_adv": args.ppo_norm_adv,
                "clip_coef": args.ppo_clip_coef,
                "clip_vloss": args.ppo_clip_vloss,
                "ent_coef": args.ppo_ent_coef,
                "vf_coef": args.ppo_vf_coef,
                "max_grad_norm": args.ppo_max_grad_norm,
                "target_kl": args.ppo_target_kl,
                "log_frequency": args.log_frequency,
            },
        }
    )
    validate_config(task, config, task.design_space)
    _set_seed(config.seed, config.torch_deterministic)
    policy, value_net = make_lot_inspection_networks(args, task)
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=reward_fn,
        batch_shape=(args.num_envs,),
    )
    designer = PolicyDesigner(task, args.horizon, policy, deterministic=False)
    suspicious_zones = parse_suspicious_zones(
        args.dm_suspicious_zones,
        args.num_zones,
    )
    trainer = LotInspectionPPOTrainer(
        task,
        designer,
        env,
        value_net,
        config,
        suspicious_zones=suspicious_zones,
    )
    trainer.train()
    policy.eval()
    checkpoint = trainer.build_checkpoint(step=args.num_training_iterations - 1)
    checkpoint.update(
        {
            "algorithm": "ppo_fixed_conservative_dm",
            "config": asdict(args),
            "dm_acceptance_threshold": torch.as_tensor(
                reward_fn.decision_rule.threshold
            ),
            "dm_location_probs": decision_maker_location_probs(
                args, torch.device("cpu")
            ),
        }
    )
    result = FitResult(
        designer=designer,
        metrics=trainer.metrics,
        checkpoint=checkpoint,
    )

    if not args.save:
        return result, None

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_path = Path(args.output_dir) / run_name / "ppo_fixed_rule_lot.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(checkpoint_path)
    return result, checkpoint_path


@torch.no_grad()
def evaluate_policy_mean_risk(args: Args, policy_net) -> dict[str, float]:
    """Evaluates deployed risk under the experimenter's prior."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    task = make_task(args, device)
    reward_fn = make_reward(args, task)
    env = BEDEnv(
        task,
        horizon=args.horizon,
        design_space=task.design_space,
        reward_fn=reward_fn,
        batch_shape=(args.eval_episodes,),
    )
    policy_net.eval()
    observation, _ = env.reset(seed=args.seed + 1)
    for _ in range(args.horizon):
        designs = policy_net.sample_from_observation(
            observation,
            deterministic=args.eval_deterministic,
        ).design
        observation, _, _, _, _ = env.step(designs)

    if env.theta_0 is None:
        raise RuntimeError("Environment has not been reset.")
    accepted = reward_fn.last_accepted
    if accepted is None:
        raise RuntimeError("The fixed decision rule did not produce a terminal action.")
    good = env.theta_0[..., 0].bool()
    false_accept = accepted & ~good
    false_reject = ~accepted & good
    risks = lot_loss(
        accepted,
        env.theta_0,
        false_accept_cost=args.experimenter_false_accept_cost,
        false_reject_cost=args.experimenter_false_reject_cost,
    )
    good_acceptance_rate = accepted[good].to(torch.float32).mean().item()
    bad_acceptance_rate = accepted[~good].to(torch.float32).mean().item()
    return {
        "mean_risk": risks.mean().item(),
        "standard_error": risks.std(unbiased=True).div(args.eval_episodes**0.5).item(),
        "acceptance_rate": accepted.to(torch.float32).mean().item(),
        "false_accept_rate": false_accept.to(torch.float32).mean().item(),
        "false_reject_rate": false_reject.to(torch.float32).mean().item(),
        "good_acceptance_rate": good_acceptance_rate,
        "bad_acceptance_rate": bad_acceptance_rate,
    }


if __name__ == "__main__":
    args = parse_args()
    result, checkpoint = train(args)
    if checkpoint is not None:
        print(f"saved checkpoint to {checkpoint}")
    evaluation = evaluate_policy_mean_risk(args, result.designer.policy_net)
    print(
        "deployed Bayes risk: "
        f"{evaluation['mean_risk']:.4f} +/- {evaluation['standard_error']:.4f}"
    )
    print(
        "fixed-rule decision rates: "
        f"accept={evaluation['acceptance_rate']:.4f}, "
        f"false_accept={evaluation['false_accept_rate']:.4f}, "
        f"false_reject={evaluation['false_reject_rate']:.4f}, "
        f"accept_given_good={evaluation['good_acceptance_rate']:.4f}, "
        f"accept_given_bad={evaluation['bad_acceptance_rate']:.4f}"
    )
