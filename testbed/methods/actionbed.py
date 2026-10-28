"""Training for ActionBED (Rossa et al., 2026)."""

import inspect
import time
from math import inf, isfinite

import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from testbed.core import ActionHook
from testbed.core.setup import make_main, setup
from testbed.core.util import restore_rng, rng_state, seed_all
from testbed.design_objectives import LossReinforce
from testbed.designers import PolicyDesigner, RandomDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for ActionBED-style training."""

    def __init__(
        self,
        task,
        designer,
        action_net,
        loss_fn,
        optimizer,
        cfg,
        objective=None,
        scheduler=None,
        random_designer=None,
        manager=None,
    ):
        """Initializes the trainer.

        objective: optional RolloutObjective replacing the plain loss during
        joint training (see build_objective). Warmup and evaluation always use
        loss_fn, which is the quantity every objective's value reports anyway.
        """
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.action_net = action_net
        self.loss_fn = loss_fn
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        # used for action policy warmup
        self.random_designer = random_designer

        self.manager = manager

        # checkpointing / progress, restored by load_checkpoint
        self.best_metric = torch.inf
        self.step = -1
        self.elapsed_seconds = 0.0
        self._start_time = time.monotonic()

        # Attach an action hook directly to the designer(s), so every
        # rollout() call from them computes an action -- no need to thread
        # hooks through each individual call site.
        action_hook = ActionHook(self.action_net, use_raw_history=cfg.use_raw_history)
        self.designer.hooks.append(action_hook)
        if self.random_designer is not None:
            self.random_designer.hooks.append(action_hook)

    def train(self):
        """Full training loop."""
        self.designer.design_net.train()
        self.action_net.train()

        budget = self.cfg.train_time_budget_seconds
        self._budget = inf if budget is None else budget
        self._start_time = time.monotonic() - self.elapsed_seconds
        self.stop_reason = "iteration_limit"

        # Action warm-up
        if self.manager and self.step + 1 < self.cfg.n_action_warmup_iter:
            self.manager.event_logger.info("Starting action warmup training...")
        for step in range(self.step + 1, self.cfg.n_action_warmup_iter):
            metrics = self.action_warmup_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            if self._checkpoint_or_stop(step, metrics["action_warmup/loss"]):
                return self._finish()

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate_warmup()
                eval_metrics["step"] = step

                if self.manager:
                    self.manager.event_logger.info(
                        f"[Eval Warmup @ step {step}/{self.cfg.n_action_warmup_iter}]: loss={eval_metrics['action_warmup/eval/loss']}"
                    )
                    if self.manager.metric_logger is not None:
                        self.manager.metric_logger.log(eval_metrics)

        # Joint action-data training
        if self.manager:
            self.manager.event_logger.info("Starting joint data-action training...")
        for step in range(
            self.step + 1, self.cfg.n_action_warmup_iter + self.cfg.n_train_iter
        ):
            metrics = self.train_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            if self._checkpoint_or_stop(step, metrics["loss"]):
                break

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step

                if self.manager:
                    self.manager.event_logger.info(
                        f"[Eval @ step {step}/{self.cfg.n_action_warmup_iter + self.cfg.n_train_iter}]: loss={eval_metrics['eval/loss']}"
                    )
                    if self.manager.metric_logger is not None:
                        self.manager.metric_logger.log(eval_metrics)

                self._maybe_save_best(step, eval_metrics)

        self._finish()

    def _checkpoint_or_stop(self, step, loss):
        """Saves a periodic policy checkpoint; returns True once training should stop.

        A non-finite loss is terminal: its gradient turns every parameter NaN on
        the next optimizer step, and nothing recovers. Stopping there rather
        than running out the budget is what keeps a diverged run from burning
        its whole allocation.
        """
        self.step = step
        if not isfinite(loss):
            self.stop_reason = "nan_loss"
            return True
        interval = self.cfg.checkpoint_every
        if interval and (step + 1) % interval == 0:
            torch.save(
                self._resumable_state(step),
                self.manager.path(f"step_{step + 1:08d}.pt"),
            )
        if time.monotonic() - self._start_time >= self._budget:
            self.stop_reason = "time_budget"
            return True
        return False

    def _finish(self):
        """Saves the full training state, enabling warm-starts."""
        self.elapsed_seconds = time.monotonic() - self._start_time
        self.manager.event_logger.info(
            f"Stopped: {self.stop_reason} after {self.step + 1} steps, "
            f"{self.elapsed_seconds:.1f}s."
        )
        torch.save(
            self._resumable_state(self.step, self.stop_reason),
            self.manager.path("final_step.pt"),
        )

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

    def _resumable_state(self, step, stop_reason=None):
        """Policy weights plus the optimizer/schedule/RNG a warm start needs."""
        return {
            **self._policy_state(step),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler and self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "stop_reason": stop_reason,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "rng_state": rng_state(),
        }

    def _policy_state(self, step):
        return {
            "design_net": self.designer.design_net.state_dict(),
            "action_net": self.action_net.state_dict(),
            "step": step,
            "use_raw_history": self.designer.use_raw_history,
            "elapsed_seconds": time.monotonic() - self._start_time,
        }

    def action_warmup_step(self):
        """Takes a single training step while warming up the action network."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_outer)
        rollout = self.random_designer.rollout(theta)

        action = rollout.extra["action"]
        loss = self.loss_fn(action, theta)

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            all_params = list(self.designer.design_net.parameters()) + list(
                self.action_net.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)

        self.optimizer.step()

        return {"action_warmup/loss": loss.item()}

    def train_step(self):
        """Takes a single training step."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_outer)
        rollout = self.designer.rollout(theta)

        # The objective's value is the mean loss; its gradient additionally
        # carries the score term the plain loss misses (see build_objective).
        if self.objective is not None:
            loss = self.objective.estimate_gain(rollout)
        else:
            loss = self.loss_fn(rollout.extra["action"], theta)

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            all_params = list(self.designer.design_net.parameters()) + list(
                self.action_net.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss.item(),
        }

    def evaluate_warmup(self):
        """Evalutes warmup with a fresh batch of rollouts."""
        cfg = self.cfg

        policy_was_training = self.designer.design_net.training
        action_was_training = self.action_net.training
        self.designer.design_net.eval()
        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            rollout = self.random_designer.rollout(theta)
            action = rollout.extra["action"]
            loss = self.loss_fn(action, theta)

        self.designer.design_net.train(policy_was_training)
        self.action_net.train(action_was_training)

        return {
            "action_warmup/eval/loss": loss.item(),
        }

    def evaluate(self):
        """Evalutes training with a fresh batch of rollouts."""
        cfg = self.cfg

        policy_was_training = self.designer.design_net.training
        action_was_training = self.action_net.training
        self.designer.design_net.eval()
        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            rollout = self.designer.rollout(theta)
            action = rollout.extra["action"]
            loss = self.loss_fn(action, theta)

        self.designer.design_net.train(policy_was_training)
        self.action_net.train(action_was_training)

        return {
            "eval/loss": loss.item(),
        }

    def _maybe_save_best(self, step, metrics, min_step=0):
        # Saves a checkpoint of the model if it's the lowest training loss seen so far

        # min_step controls how long we wait before saving; useful for avoiding a ton of i/o at the start
        if step < min_step:
            return

        if self.manager is None:
            return

        step = metrics["step"]
        metric_val = metrics["eval/loss"]
        is_better = metric_val < self.best_metric

        if is_better:
            self.best_metric = metric_val

            chkpt_path = self.manager.path("best.pt")

            torch.save({**self._policy_state(step), "metric": metric_val}, chkpt_path)

            self.manager.event_logger.info("Saved new best checkpoint")


# ============================================================================
# Helper + factory functions
# ============================================================================


def build_policy_nets(cfg, task):
    """Instantiates design and action policy networks."""
    design_net = instantiate(
        cfg.design_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )

    action_net = instantiate(
        cfg.action_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )

    return design_net, action_net


def build_loss_fn(cfg, task, **overrides):
    """Instantiates the training loss.

    Most losses only depend on ``cfg.loss``, but some (e.g. LogLoss) also
    need the task itself, which the config can't express. Rather than special
    casing loss classes by name, we inspect the target's constructor and only
    supply what it actually declares. Losses needing a design prior take it
    from ``task.design_space``, so nothing extra is threaded through here.

    ``overrides`` are passed straight to the target, so a caller needing a
    particular reduction can ask for it without editing the recipe.
    """
    loss_cls = get_class(cfg.loss._target_)
    params = inspect.signature(loss_cls.__init__).parameters

    extra_kwargs = {}
    if "task" in params:
        extra_kwargs["task"] = task

    return instantiate(cfg.loss, **extra_kwargs, **overrides)


def per_sample_loss(loss_fn):
    """Turns a ``reduction="none"`` loss into the [b] per-rollout value REINFORCE needs.

    Every event dimension is averaged over; only the batch axis survives, so
    the leave-one-out baseline sees one number per rollout.
    """

    def value(action, theta):
        loss = loss_fn(action, theta)
        return loss.reshape(loss.shape[0], -1).mean(-1)

    return value


def build_objective(cfg, task):
    """Instantiates the objective for joint training, or None for the plain loss.

    "loss" backpropagates the loss as-is: correct only when the outcomes are
    reparameterizable, since that gradient sees the designs solely through the
    action net's input. "reinforce" adds the trajectory score term, which is
    the missing piece for a discrete (or otherwise non-reparameterizable)
    likelihood; it assumes a deterministic design policy.
    """
    name = cfg.objective.name
    if name == "loss":
        return None
    if name != "reinforce":
        raise ValueError(f"Unknown objective: {name}")

    return LossReinforce(
        task,
        per_sample_loss(build_loss_fn(cfg, task, reduction="none")),
        stochastic_design_policy=cfg.objective.requires_policy_log_probs,
        baseline=cfg.objective.baseline,
    )


def build_trainer(cfg, task, manager=None):
    """Seeds, then builds a fresh trainer with its networks, optimizer and loss."""
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    design_net, action_net = build_policy_nets(cfg, task)
    design_net = design_net.to(cfg.device)
    action_net = action_net.to(cfg.device)
    designer = PolicyDesigner(
        task,
        cfg.T,
        design_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
        use_raw_history=cfg.use_raw_history,
    )
    # Update config to match the designer's proposal/gradient capability.
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs

    # Setup for action-warmup; designs are drawn from the task's design space.
    random_designer = None
    if cfg.n_action_warmup_iter > 0:
        random_designer = RandomDesigner(task, cfg.T)

    # Create optimizer + LR scheduler
    all_policy_params = list(design_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_policy_params, lr=cfg.lr, betas=tuple(cfg.betas))

    scheduler = None
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    return Trainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=build_loss_fn(cfg, task),
        optimizer=optimizer,
        cfg=cfg,
        objective=build_objective(cfg, task),
        scheduler=scheduler,
        random_designer=random_designer,
        manager=manager,
    )


def load_checkpoint(
    path,
    *,
    device=None,
    train_time_budget_seconds=None,
    n_train_iter=None,
    manager=None,
):
    """Reconstructs a trainer from a ``final_step.pt`` checkpoint to resume training.

    The saved config is used, with optional overrides. Time and iteration limits
    are totals including completed training, e.g. pass
    ``train_time_budget_seconds=24 * 3600`` to continue a run stopped at 12 hours
    (``float("inf")`` removes the limit). ``trainer.train()`` requires a manager.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(state["config"])
    overrides = {
        "device": device,
        "train_time_budget_seconds": train_time_budget_seconds,
        "n_train_iter": n_train_iter,
    }
    cfg.merge_with({k: v for k, v in overrides.items() if v is not None})

    trainer = build_trainer(cfg, instantiate(cfg.task).to(cfg.device), manager)
    trainer.designer.design_net.load_state_dict(state["design_net"])
    trainer.action_net.load_state_dict(state["action_net"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(state["scheduler"])
    for name in ("step", "best_metric", "elapsed_seconds"):
        setattr(trainer, name, state[name])
    # Construction consumes randomness, so restore generators last.
    restore_rng(state["rng_state"])
    return trainer


# ============================================================================
# Entry point
# ============================================================================


def run(cfg):
    """Entry point for ActionBED training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/actionbed.py [overrides...]``.
    """
    start_time = time.monotonic()
    manager, task = setup(cfg)
    trainer = build_trainer(cfg, task, manager)
    # Setup counts toward the time budget.
    trainer.elapsed_seconds = time.monotonic() - start_time
    trainer.train()


main = make_main(run, "method/actionbed")

if __name__ == "__main__":
    main()
