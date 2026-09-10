"""Train BED-focused ALINE (Huang et al., 2025).

This implementation follows https://arxiv.org/abs/2506.07259 and adapts the
authors' reference implementation at https://github.com/huangdaolang/ALINE to
TestBED's task, logging, and evaluation interfaces.
"""

import inspect
from dataclasses import dataclass
from math import prod
from numbers import Integral

import hydra
import torch
from torch import Tensor

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.rollout import Rollout
from testbed.core.setup import setup
from testbed.design_nets import AlineModel, AlineOutput
from testbed.design_objectives import SNMC, SPCE


@dataclass
class AlineRollout:
    """A batched ALINE trajectory and its training signals."""

    outcomes: Tensor
    designs: Tensor
    posterior_log_probs: Tensor
    action_log_probs: Tensor
    rewards: Tensor
    target_mask: Tensor
    final_output: AlineOutput
    inference_log_probs: Tensor | None = None


class Trainer:
    """Jointly trains ALINE's inference and acquisition components."""

    def __init__(
        self,
        task,
        model: AlineModel,
        design_space: BoxDesignSpace,
        optimizer,
        cfg,
        manager=None,
        scheduler=None,
    ) -> None:
        """Initializes the ALINE trainer."""
        validate_config(task, cfg, design_space)
        self.task = task
        self.model = model
        self.design_space = design_space
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.manager = manager
        self.best_metric = -torch.inf
        self._joint_optimizer_started = False
        self.target_masks, self.target_weights = _build_target_masks(
            cfg, model.dim_theta, model.design_low.device
        )

    @property
    def device(self) -> torch.device:
        """Returns the model device."""
        return self.model.design_low.device

    def _sample_target_mask(self) -> Tensor:
        index = torch.multinomial(self.target_weights, 1).item()
        return self.target_masks[index]

    def _as_theta_batch(self, theta: Tensor) -> Tensor:
        if torch.Size(theta.shape) == self.task.event_shape_theta:
            theta = theta.unsqueeze(0)
        if torch.Size(theta.shape[1:]) != self.task.event_shape_theta:
            raise ValueError(
                "theta must have one batch dimension and the task event shape."
            )
        if theta.device != self.device:
            raise ValueError("theta and the ALINE model must be on the same device.")
        return theta

    def _selected_log_prob(
        self, output: AlineOutput, theta: Tensor, target_mask: Tensor
    ) -> Tensor:
        theta_flat = theta.reshape(theta.shape[0], self.model.dim_theta)
        marginal_log_probs = output.posterior.log_prob(theta_flat)
        return marginal_log_probs[:, target_mask].mean(dim=1)

    @staticmethod
    def _random_action(available_mask: Tensor) -> Tensor:
        return torch.multinomial(available_mask.float(), 1).squeeze(1)

    @staticmethod
    def _policy_action(
        output: AlineOutput, *, deterministic: bool
    ) -> tuple[Tensor, Tensor]:
        policy = torch.distributions.Categorical(logits=output.acquisition_logits)
        action = (
            output.acquisition_logits.argmax(dim=1)
            if deterministic
            else policy.sample()
        )
        return action, policy.log_prob(action)

    def rollout(
        self,
        theta: Tensor,
        *,
        target_mask: Tensor | None = None,
        random_policy: bool = False,
        deterministic: bool = False,
        pool_size: int | None = None,
    ) -> AlineRollout:
        """Simulates a batch of fixed-pool ALINE trajectories."""
        cfg = self.cfg
        theta = self._as_theta_batch(theta)
        batch_size = theta.shape[0]
        if target_mask is None:
            target_mask = self._sample_target_mask()
        target_mask = target_mask.to(device=self.device, dtype=torch.bool)

        pool_size = cfg.pool_size if pool_size is None else pool_size
        candidates = self.design_space.sample(
            (batch_size, pool_size), device=self.device
        )
        available_mask = torch.ones(
            batch_size, pool_size, device=self.device, dtype=torch.bool
        )
        context_designs = candidates.new_empty(
            batch_size, 0, *self.task.event_shape_design
        )
        context_y = candidates.new_empty(batch_size, 0, *self.task.event_shape_y)

        current_output = self.model(
            context_y,
            context_designs,
            candidates,
            target_mask,
            available_mask,
        )
        current_log_prob = self._selected_log_prob(current_output, theta, target_mask)
        posterior_log_probs = []
        inference_log_probs = []
        action_log_probs = []
        rewards = []
        batch_indices = torch.arange(batch_size, device=self.device)

        for time_index in range(cfg.T):
            use_policy = not random_policy and time_index >= cfg.n_initial
            if time_index >= cfg.n_initial:
                # Reference training scores every parameter before acquisition.
                inference_log_probs.append(
                    current_output.posterior.log_prob(
                        theta.reshape(batch_size, self.model.dim_theta)
                    ).mean(dim=1)
                )
            if use_policy:
                action, action_log_prob = self._policy_action(
                    current_output, deterministic=deterministic
                )
            else:
                action = self._random_action(available_mask)
                action_log_prob = None

            design = candidates[batch_indices, action]
            outcome = self.task.sample_outcome(
                theta, design, y_history=context_y, design_history=context_designs
            ).detach()
            context_designs = torch.cat([context_designs, design.unsqueeze(1)], dim=1)
            context_y = torch.cat([context_y, outcome.unsqueeze(1)], dim=1)
            available_mask = available_mask.clone()
            available_mask[batch_indices, action] = False

            next_output = self.model(
                context_y,
                context_designs,
                candidates,
                target_mask,
                available_mask,
            )
            next_log_prob = self._selected_log_prob(next_output, theta, target_mask)
            posterior_log_probs.append(next_log_prob)
            if action_log_prob is not None and time_index < cfg.T - 1:
                action_log_probs.append(action_log_prob)
                rewards.append(next_log_prob.detach() - current_log_prob.detach())

            current_output = next_output
            current_log_prob = next_log_prob

        empty = theta.new_empty(batch_size, 0)
        return AlineRollout(
            outcomes=context_y,
            designs=context_designs,
            posterior_log_probs=torch.stack(posterior_log_probs, dim=1),
            action_log_probs=(
                torch.stack(action_log_probs, dim=1) if action_log_probs else empty
            ),
            rewards=torch.stack(rewards, dim=1) if rewards else empty,
            target_mask=target_mask,
            final_output=current_output,
            inference_log_probs=torch.stack(inference_log_probs, dim=1),
        )

    def train_step(self, step: int = 0) -> dict[str, float]:
        """Takes one warm-up or joint ALINE training step."""
        self.model.train()
        warmup = step < self.cfg.n_warmup_iter
        theta = self.task.sample_prior(self.cfg.n_outer)
        rollout = self.rollout(theta, pool_size=self.cfg.T if warmup else None)
        inference_loss = -rollout.inference_log_probs.mean()
        policy_loss = inference_loss.new_zeros(())
        mean_reward = inference_loss.new_zeros(())

        if not warmup:
            # Match train_aline.py: clip gains, discount, then standardize
            # across trajectories using the sample standard deviation.
            rewards = rollout.rewards.detach().clamp_min(0.0)
            discounts = self.cfg.gamma ** torch.arange(
                1, rewards.shape[1] + 1, device=self.device
            )
            policy_rewards = rewards * discounts
            if rewards.shape[0] > 1:
                policy_rewards = (
                    policy_rewards - policy_rewards.mean(dim=0, keepdim=True)
                ) / (policy_rewards.std(dim=0, keepdim=True) + 1e-9)
            policy_loss = -(rollout.action_log_probs * policy_rewards).mean()
            mean_reward = rewards.mean()

        loss = inference_loss + self.cfg.alpha * policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip, norm_type=float("inf")
            )
        if not warmup and not self._joint_optimizer_started:
            # The reference restarts AdamW and cosine decay at the boundary;
            # only the acquisition predictor retains the full learning rate.
            base_lr = self.optimizer.defaults["lr"]
            acquisition = list(self.model.acquisition_head.parameters())
            acquisition_ids = {id(p) for p in acquisition}
            shared = [
                p for p in self.model.parameters() if id(p) not in acquisition_ids
            ]
            self.optimizer = type(self.optimizer)(
                [
                    {"params": shared, "lr": base_lr / 5},
                    {"params": acquisition, "lr": base_lr},
                ],
                **{
                    key: value
                    for key, value in self.optimizer.defaults.items()
                    if key in inspect.signature(type(self.optimizer)).parameters
                },
            )
            if self.scheduler is not None:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, self.cfg.n_train_iter - self.cfg.n_warmup_iter),
                )
            self._joint_optimizer_started = True
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.item(),
            "inference_loss": inference_loss.item(),
            "policy_loss": policy_loss.item(),
            "reward": mean_reward.item(),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluates posterior fit and, for full targets, EIG bounds."""
        was_training = self.model.training
        self.model.eval()
        theta = self._as_theta_batch(self.task.sample_prior(self.cfg.n_outer_eval))
        target_mask = self.target_masks[0]
        rollout = self.rollout(
            theta,
            target_mask=target_mask,
            deterministic=self.cfg.eval_deterministic,
        )
        metrics = {
            "eval/posterior_nll": -rollout.posterior_log_probs[:, -1].mean().item()
        }
        if torch.all(target_mask):
            metrics["eval/spce"] = (
                SPCE(self.task, n_contrastive=self.cfg.n_inner_eval)
                .estimate_gain(Rollout(theta, rollout.outcomes, rollout.designs))
                .item()
            )
            metrics["eval/snmc"] = (
                SNMC(self.task, n_contrastive=self.cfg.n_inner_eval)
                .estimate_gain(Rollout(theta, rollout.outcomes, rollout.designs))
                .item()
            )
        self.model.train(was_training)
        return metrics

    def _maybe_save_best(self, step: int, metrics: dict[str, float]) -> None:
        if self.manager is None:
            return
        if "eval/spce" in metrics:
            metric = metrics["eval/spce"]
        else:
            metric = -metrics["eval/posterior_nll"]
        if metric <= self.best_metric:
            return
        self.best_metric = metric
        torch.save(
            {"model": self.model.state_dict(), "step": step, "metric": metric},
            self.manager.path("best.pt"),
        )
        self.manager.event_logger.info(
            f"Saving new best ALINE checkpoint: metric={metric} @ step={step}"
        )

    def train(self) -> None:
        """Runs warm-up and joint ALINE training."""
        for step in range(self.cfg.n_train_iter):
            metrics = self.train_step(step)
            metrics["step"] = step
            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step
                if self.manager and self.manager.metric_logger:
                    self.manager.metric_logger.log(eval_metrics)
                self._maybe_save_best(step, eval_metrics)

        if self.manager is not None:
            torch.save(
                {"model": self.model.state_dict(), "step": self.cfg.n_train_iter - 1},
                self.manager.path("final_step.pt"),
            )
            if self.manager.metric_logger is not None:
                self.manager.metric_logger.close()


def _build_target_masks(
    cfg, dim_theta: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    masks = []
    for subset in cfg.target_subsets:
        mask = torch.zeros(dim_theta, device=device, dtype=torch.bool)
        mask[list(subset)] = True
        masks.append(mask)
    target_masks = torch.stack(masks)
    if cfg.target_weights is None:
        weights = torch.ones(len(masks), device=device)
    else:
        weights = torch.as_tensor(
            cfg.target_weights, device=device, dtype=torch.get_default_dtype()
        )
    return target_masks, weights / weights.sum()


def build_design_space(cfg, task) -> BoxDesignSpace:
    """Builds the fixed candidate-pool design space."""
    if not isinstance(task.design_space, BoxDesignSpace):
        raise ValueError("ALINE requires a finite BoxDesignSpace on task.design_space.")
    return task.design_space


def build_model(cfg, task, design_space: BoxDesignSpace) -> AlineModel:
    """Builds the ALINE neural architecture."""
    return AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        design_low=design_space.low,
        design_high=design_space.high,
        embedding_dim=cfg.model.embedding_dim,
        feedforward_dim=cfg.model.feedforward_dim,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        n_components=cfg.model.n_components,
        min_scale=cfg.model.min_scale,
        design_center=cfg.model.get("design_center", 0.0),
        design_scale=cfg.model.get("design_scale", 1.0),
        outcome_center=cfg.model.outcome_center,
        outcome_scale=cfg.model.outcome_scale,
    )


def validate_config(task, cfg, design_space: BoxDesignSpace) -> None:
    """Rejects unsupported tasks and invalid ALINE settings."""
    if task.discrete_theta:
        raise ValueError(
            "ALINE's Gaussian-mixture posterior requires continuous theta."
        )
    if design_space.event_shape != task.event_shape_design:
        raise ValueError("The design-space event shape must match the task.")
    if not torch.all(torch.isfinite(design_space.low)) or not torch.all(
        torch.isfinite(design_space.high)
    ):
        raise ValueError("The design-space bounds must be finite.")

    positive = {
        "T": cfg.T,
        "pool_size": cfg.pool_size,
        "n_outer": cfg.n_outer,
        "n_outer_eval": cfg.n_outer_eval,
        "n_inner_eval": cfg.n_inner_eval,
        "n_train_iter": cfg.n_train_iter,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if cfg.pool_size < cfg.T:
        raise ValueError(
            "pool_size must be at least T for sampling without replacement."
        )
    if not 0 <= cfg.n_initial < cfg.T - 1:
        raise ValueError("n_initial must satisfy 0 <= n_initial < T - 1.")
    if cfg.n_warmup_iter < 0 or cfg.eval_every < 0:
        raise ValueError("n_warmup_iter and eval_every must be non-negative.")
    if cfg.gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if cfg.alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if cfg.grad_clip is not None and cfg.grad_clip <= 0:
        raise ValueError("grad_clip must be positive when provided.")

    dim_theta = prod(task.event_shape_theta)
    if not cfg.target_subsets:
        raise ValueError("target_subsets must contain at least one subset.")
    for subset in cfg.target_subsets:
        indices = list(subset)
        if any(not isinstance(index, Integral) for index in indices):
            raise ValueError("Target-subset indices must be integers.")
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(
                "Each target subset must contain unique parameter indices."
            )
        if min(indices) < 0 or max(indices) >= dim_theta:
            raise ValueError("Target-subset index is outside flattened theta.")
    if cfg.target_weights is not None:
        weights = torch.as_tensor(cfg.target_weights)
        if len(weights) != len(cfg.target_subsets):
            raise ValueError("target_weights must match target_subsets.")
        if (
            not torch.all(torch.isfinite(weights))
            or torch.any(weights < 0)
            or not torch.any(weights > 0)
        ):
            raise ValueError(
                "target_weights must be finite, non-negative, and have a positive sum."
            )


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/aline")
def main(cfg) -> None:
    """Runs ALINE training from a Hydra recipe."""
    torch.manual_seed(cfg.seed)
    manager, task = setup(cfg)
    design_space = build_design_space(cfg, task)
    validate_config(task, cfg, design_space)
    model = build_model(cfg, task, design_space).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_train_iter
    )
    trainer = Trainer(
        task,
        model,
        design_space,
        optimizer,
        cfg,
        manager=manager,
        scheduler=scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
