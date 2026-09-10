"""Train BED-focused ALINE (Huang et al., 2025).

This is based on the authors' reference implementation at https://github.com/huangdaolang/ALINE.
Unlike the authors' code however, this implementation is specialized to
targeting the EIG in all parameters.
"""

import inspect

import hydra
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.rollout import Rollout
from testbed.core.setup import setup
from testbed.design_nets import AlineModel
from testbed.design_objectives import SNMC, SPCE
from testbed.designers import AlineDesigner


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
        self.designer = AlineDesigner(
            task,
            cfg.T,
            model,
            pool_size=cfg.pool_size,
            deterministic=False,
            design_space=design_space,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.manager = manager
        self.best_metric = -torch.inf
        self._joint_optimizer_started = False

    @property
    def device(self) -> torch.device:
        """Returns the model device."""
        return self.model.design_low.device

    def train_step(self, step: int = 0) -> dict[str, float]:
        """Takes one warm-up or joint ALINE training step."""
        self.model.train()
        warmup = step < self.cfg.n_warmup_iter
        theta = self.task.sample_prior(self.cfg.n_outer)
        rollout = self.designer.rollout(
            theta,
            random_policy=warmup,
            pool_size=self.cfg.T if warmup else self.cfg.pool_size,
            return_posterior_scores=True,
        )
        # Train the posterior at every state D_0, ..., D_T. In particular, the
        # D_0 score is the baseline for the clipped first-step reward.
        posterior_log_probs = rollout.extra["posterior_log_probs"]
        inference_loss = -posterior_log_probs.mean()
        policy_loss = inference_loss.new_zeros(())
        mean_reward = inference_loss.new_zeros(())

        if not warmup:
            # Match train_aline.py: clip gains, discount, then standardize
            # across trajectories using the sample standard deviation.
            rewards = (
                posterior_log_probs[:, 1:].detach()
                - posterior_log_probs[:, :-1].detach()
            ).clamp_min(0.0)
            discounts = self.cfg.gamma ** torch.arange(
                1, rewards.shape[1] + 1, device=self.device
            )
            policy_rewards = rewards * discounts
            if rewards.shape[0] > 1:
                policy_rewards = (
                    policy_rewards - policy_rewards.mean(dim=0, keepdim=True)
                ) / (policy_rewards.std(dim=0, keepdim=True) + 1e-9)
            policy_loss = -(rollout.extra["policy_log_probs"] * policy_rewards).mean()
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
        """Evaluates posterior fit and full-parameter EIG bounds."""
        was_training = self.model.training
        self.model.eval()
        theta = self.task.sample_prior(self.cfg.n_outer_eval).reshape(
            self.cfg.n_outer_eval, *self.task.event_shape_theta
        )
        rollouts = [
            self.designer.rollout(
                theta_batch,
                deterministic=self.cfg.eval_deterministic,
                pool_size=self.cfg.eval_pool_size,
                return_posterior_scores=True,
            )
            for theta_batch in theta.split(self.cfg.eval_batch_size)
        ]
        posterior_log_probs = torch.cat(
            [rollout.extra["posterior_log_probs"] for rollout in rollouts]
        )
        outcomes = torch.cat([rollout.y_history for rollout in rollouts])
        designs = torch.cat([rollout.design_history for rollout in rollouts])
        metrics = {"eval/posterior_nll": -posterior_log_probs[:, -1].mean().item()}
        metrics["eval/spce"] = (
            SPCE(self.task, n_contrastive=self.cfg.n_inner_eval)
            .estimate_gain(Rollout(theta, outcomes, designs))
            .item()
        )
        metrics["eval/snmc"] = (
            SNMC(self.task, n_contrastive=self.cfg.n_inner_eval)
            .estimate_gain(Rollout(theta, outcomes, designs))
            .item()
        )
        self.model.train(was_training)
        return metrics

    def _maybe_save_best(self, step: int, metrics: dict[str, float]) -> None:
        if self.manager is None:
            return
        metric = metrics["eval/spce"]
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
        "eval_pool_size": cfg.eval_pool_size,
        "eval_batch_size": cfg.eval_batch_size,
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
    if cfg.eval_pool_size < cfg.T:
        raise ValueError(
            "eval_pool_size must be at least T for sampling without replacement."
        )
    if cfg.n_warmup_iter < 0 or cfg.eval_every < 0:
        raise ValueError("n_warmup_iter and eval_every must be non-negative.")
    if cfg.gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if cfg.alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if cfg.grad_clip is not None and cfg.grad_clip <= 0:
        raise ValueError("grad_clip must be positive when provided.")


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
