"""Train BED-focused ALINE (Huang et al., 2025).

This is based on the authors' reference implementation at https://github.com/huangdaolang/ALINE.
Unlike the authors' code however, this implementation is specialized to
targeting the EIG in all parameters.
"""

import time
from math import isfinite

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.setup import instantiate_design_net, setup
from testbed.core.util import restore_rng, rng_state, seed_all
from testbed.design_objectives import SPCE
from testbed.designers import AlineDesigner
from testbed.method_nets import AlineModel


class Trainer:
    """Jointly trains ALINE's inference and acquisition components."""

    def __init__(
        self,
        task,
        model: AlineModel,
        cfg,
        manager=None,
        *,
        start_time=None,
    ) -> None:
        """Initializes the ALINE trainer."""
        validate_config(task, cfg)
        self.task = task
        self.model = model
        self.designer = AlineDesigner(
            task,
            cfg.T,
            model,
            pool_size=cfg.pool_size,
            deterministic=False,
        )
        self.cfg = cfg
        self.manager = manager
        self._start_time = time.monotonic() if start_time is None else start_time
        self.completed_iterations = 0
        self.training_elapsed_seconds = None
        self.stop_reason = None
        self.best_metric = -torch.inf
        self._joint_optimizer_started = False
        self._start_warmup_optimizer()

    @property
    def device(self) -> torch.device:
        """Returns the model device."""
        return next(self.model.parameters()).device

    def _start_warmup_optimizer(self) -> None:
        # Warm-up uses one learning rate and decay over the full training horizon.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.n_train_iter
        )

    def _start_joint_optimizer(self) -> None:
        # Restart AdamW state and cosine decay at the phase boundary, as in the
        # reference. Only the acquisition head retains the full learning rate.
        acquisition = list(self.model.acquisition_head.parameters())
        acquisition_ids = {id(p) for p in acquisition}
        shared = [p for p in self.model.parameters() if id(p) not in acquisition_ids]
        base_lr = self.cfg.optimizer.lr
        self.optimizer = torch.optim.AdamW(
            [
                {"params": shared, "lr": base_lr / 5},
                {"params": acquisition, "lr": base_lr},
            ],
            lr=base_lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.cfg.n_train_iter - self.cfg.n_warmup_iter),
        )
        self._joint_optimizer_started = True

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
            self._start_joint_optimizer()
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "inference_loss": inference_loss.item(),
            "policy_loss": policy_loss.item(),
            "reward": mean_reward.item(),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluates posterior fit and the full-parameter sPCE lower bound."""
        was_training = self.model.training
        self.model.eval()
        theta = self.task.sample_prior(self.cfg.n_outer_eval).reshape(
            self.cfg.n_outer_eval, *self.task.event_shape_theta
        )
        spce = SPCE(
            self.task,
            n_contrastive=self.cfg.n_inner_eval,
            chunk_size=getattr(self.cfg, "eval_contrastive_chunk_size", 10_000),
        )
        posterior_sum = theta.new_zeros(())
        gain_sum = theta.new_zeros(())
        for theta_batch in theta.split(self.cfg.eval_batch_size):
            rollout = self.designer.rollout(
                theta_batch,
                deterministic=self.cfg.eval_deterministic,
                pool_size=self.cfg.eval_pool_size,
                return_posterior_scores=True,
            )
            posterior_sum += rollout.extra["posterior_log_probs"][:, -1].sum()
            gain_sum += spce.per_sample_gain(rollout).sum()
        # Sum individual scores so the final, possibly smaller batch has the
        # correct weight. Never combine all trajectories for contrastive scoring.
        metrics = {
            "eval/posterior_nll": (-posterior_sum / self.cfg.n_outer_eval).item(),
            "eval/spce": (gain_sum / self.cfg.n_outer_eval).item(),
        }
        self.model.train(was_training)
        return metrics

    def _maybe_save_best(
        self, step: int, metrics: dict[str, float], min_step: int = 0
    ) -> None:
        # Saves a checkpoint of the model if it has the best posterior fit seen
        # so far. NB: we maximize the mean marginal posterior log probability,
        # which is the per-step training signal; periodic eval is off by default
        # so eval/spce is not available to select on.

        # Warm-up draws random designs from a T-sized pool, so its scores are
        # not comparable with the joint phase's; min_step skips them.
        if step < min_step:
            return
        if self.manager is None:
            return
        metric = -metrics["inference_loss"]
        if metric <= self.best_metric:
            return
        self.best_metric = metric
        torch.save(
            {"design_net": self.model.state_dict(), "step": step, "metric": metric},
            self.manager.path("best.pt"),
        )
        self.manager.event_logger.info(
            f"Saving new best ALINE checkpoint: posterior_log_prob={metric} "
            f"@ step={step}"
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _deadline_reached(self) -> bool:
        if time.monotonic() - self._start_time >= _time_budget(self.cfg):
            self.stop_reason = "time_budget"
            return True
        return False

    def build_policy_checkpoint(self) -> dict:
        """Builds a deployable model snapshot without optimizer state."""
        elapsed = self.training_elapsed_seconds
        if elapsed is None:
            elapsed = time.monotonic() - self._start_time
        return {
            "design_net": self.model.state_dict(),
            "step": self.completed_iterations - 1,
            "completed_iterations": self.completed_iterations,
            "elapsed_seconds": elapsed,
        }

    def build_checkpoint(self) -> dict:
        """Builds the full state required to resume warm-up or joint training."""
        return {
            **self.build_policy_checkpoint(),
            "checkpoint_version": 1,
            "method": "aline",
            "config": OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=False
            ),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "joint_optimizer_started": self._joint_optimizer_started,
            "best_metric": self.best_metric,
            "rng_state": rng_state(),
            "stop_reason": self.stop_reason,
            "train_time_budget_seconds": _time_budget(self.cfg),
        }

    def train(self) -> None:
        """Runs warm-up and joint ALINE training."""
        self.training_elapsed_seconds = None
        self.stop_reason = "iteration_limit"
        for step in range(self.completed_iterations, self.cfg.n_train_iter):
            if self._deadline_reached():
                break
            metrics = self.train_step(step)
            self.completed_iterations = step + 1
            metrics["step"] = step
            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            self._maybe_save_best(step, metrics, min_step=self.cfg.n_warmup_iter)
            interval = getattr(self.cfg, "checkpoint_every", 0)
            if self.manager is not None and interval and (step + 1) % interval == 0:
                torch.save(
                    self.build_policy_checkpoint(),
                    self.manager.path(f"step_{step + 1:08d}.pt"),
                )

            # Complete the update and preserve its weights before stopping;
            # avoid starting a potentially expensive evaluation after the limit.
            self._synchronize()
            if self._deadline_reached():
                break
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step
                if self.manager and self.manager.metric_logger:
                    self.manager.metric_logger.log(eval_metrics)

        self._synchronize()
        self.training_elapsed_seconds = time.monotonic() - self._start_time
        if self.manager is not None:
            budget = _time_budget(self.cfg)
            self.manager.event_logger.info(
                f"Stopped: {self.stop_reason}; {self.completed_iterations} iterations, "
                f"{self.training_elapsed_seconds:.3f}s before final serialization; "
                f"budget={budget}s, "
                f"overrun={max(0.0, self.training_elapsed_seconds - budget):.3f}s."
            )
            torch.save(
                self.build_checkpoint(),
                self.manager.path("final_step.pt"),
            )
            if self.manager.metric_logger is not None:
                self.manager.metric_logger.close()


def _time_budget(cfg) -> float:
    """Returns the cumulative time limit; null means unlimited training."""
    budget = getattr(cfg, "train_time_budget_seconds", None)
    return float("inf") if budget is None else float(budget)


def load_checkpoint(
    path,
    *,
    device=None,
    train_time_budget_seconds=None,
    n_train_iter=None,
    manager=None,
) -> Trainer:
    """Reconstructs an ALINE trainer from a full ``final_step.pt`` checkpoint.

    Call ``train()`` on the result to resume. Limits are cumulative: pass a
    larger total ``train_time_budget_seconds`` or ``n_train_iter`` if exhausted.
    Use ``float("inf")`` for an unlimited time budget. Time spent between runs
    is excluded; loading and reconstruction count toward the resumed budget.
    Optimizer phase, scheduler and global RNGs are restored; exact continuation
    requires the same device/RNG setup. Supply a manager for automatic saving
    and logging, or save ``build_checkpoint()`` manually. Policy-only snapshots
    (including ``best.pt``) cannot resume training.
    """
    start_time = time.monotonic()
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "config",
        "design_net",
        "optimizer",
        "scheduler",
        "joint_optimizer_started",
        "best_metric",
        "rng_state",
        "completed_iterations",
        "elapsed_seconds",
    }
    if (
        not isinstance(state, dict)
        or state.get("checkpoint_version") != 1
        or state.get("method") != "aline"
        or not required.issubset(state)
    ):
        raise ValueError(
            "Resuming requires a full ALINE checkpoint (final_step.pt), including "
            "optimizer, scheduler and RNG state; policy-only checkpoints cannot "
            "resume training."
        )
    cfg = OmegaConf.create(state["config"])
    if device is not None:
        cfg.device = str(device)
    if train_time_budget_seconds is not None:
        cfg.train_time_budget_seconds = train_time_budget_seconds
    if n_train_iter is not None:
        cfg.n_train_iter = n_train_iter
    elapsed = state["elapsed_seconds"]
    if not isfinite(elapsed) or elapsed < 0:
        raise ValueError("Checkpoint elapsed_seconds must be finite and non-negative.")
    if _time_budget(cfg) <= elapsed:
        raise ValueError(
            "The saved training time budget is exhausted; pass "
            "train_time_budget_seconds greater than the checkpoint's "
            f"elapsed_seconds ({elapsed}). This is a total, not additional, budget."
        )
    if cfg.n_train_iter <= state["completed_iterations"]:
        raise ValueError(
            "The saved iteration limit is exhausted; pass n_train_iter greater "
            "than the checkpoint's completed_iterations."
        )
    task = instantiate(cfg.task).to(cfg.device)
    model = instantiate_design_net(cfg.design_net, task).to(cfg.device)
    model.load_state_dict(state["design_net"])
    trainer = Trainer(task, model, cfg, manager, start_time=start_time - elapsed)
    # AdamW's parameter groups differ between phases. Recreate the saved phase
    # before loading its moments and the original cosine schedule.
    if state["joint_optimizer_started"]:
        trainer._start_joint_optimizer()
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.completed_iterations = state["completed_iterations"]
    trainer.best_metric = state["best_metric"]
    restore_rng(state["rng_state"])
    return trainer


def validate_config(task, cfg) -> None:
    """Rejects unsupported tasks and invalid ALINE settings."""
    budget = getattr(cfg, "train_time_budget_seconds", None)
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float, type(None)))
        or (budget is not None and not budget > 0)
    ):
        raise ValueError(
            "train_time_budget_seconds must be a positive number of seconds, "
            "or null for no time limit."
        )
    interval = getattr(cfg, "checkpoint_every", 0)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("checkpoint_every must be a non-negative integer.")
    if task.discrete_theta:
        raise ValueError(
            "ALINE's Gaussian-mixture posterior requires continuous theta."
        )
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
    if getattr(cfg, "eval_contrastive_chunk_size", 10_000) <= 0:
        raise ValueError("eval_contrastive_chunk_size must be positive.")
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
    start_time = time.monotonic()
    seed_all(cfg.seed)
    manager, task = setup(cfg)
    model = instantiate_design_net(cfg.design_net, task).to(cfg.device)
    trainer = Trainer(task, model, cfg, manager=manager, start_time=start_time)
    trainer.train()


if __name__ == "__main__":
    main()
