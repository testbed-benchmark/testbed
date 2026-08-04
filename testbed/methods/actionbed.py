"""Training for ActionBED (Rossa et al., 2026)."""

import inspect
import os

import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from testbed.core.distributions import TorchDistribution
from testbed.core.manager import Manager
from testbed.designers import (
    ActionHook,
    PolicyDesigner,
    RandomDesigner,
    build_policy_designer,
)

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
        scheduler=None,
        random_designer=None,
        manager=None,
    ):
        """Initializes the trainer."""
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.action_net = action_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        # used for action policy warmup
        self.random_designer = random_designer

        self.manager = manager

        # checkpointing
        self.best_metric = torch.inf

        # Attach an action hook directly to the designer(s), so every
        # rollout() call from them computes an action -- no need to thread
        # hooks through each individual call site.
        action_hook = ActionHook(self.action_net)
        self.designer.hooks.append(action_hook)
        if self.random_designer is not None:
            self.random_designer.hooks.append(action_hook)

    def train(self):
        """Full training loop."""
        self.designer.design_net.train()
        self.action_net.train()

        # Action warm-up
        if self.manager and self.cfg.n_action_warmup_iter > 0:
            self.manager.event_logger.info("Starting action warmup training...")
        for step in range(self.cfg.n_action_warmup_iter):
            metrics = self.action_warmup_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

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
            self.cfg.n_action_warmup_iter,
            self.cfg.n_action_warmup_iter + self.cfg.n_train_iter,
        ):
            metrics = self.train_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

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

        save_path = self.manager.path("final_step.pt")
        torch.save(
            {
                "design_net": self.designer.design_net.state_dict(),
                "action_net": self.action_net.state_dict(),
                "step": step,
            },
            save_path,
        )

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

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

            torch.save(
                {
                    "design_net": self.designer.design_net.state_dict(),
                    "action_net": self.action_net.state_dict(),
                    "step": step,
                    "metric": metric_val,
                },
                chkpt_path,
            )

            self.manager.event_logger.info("Saved new best checkpoint")


# ============================================================================
# Helper + factory functions
# ============================================================================


def resolve_event_shape_action(task, manager):
    """Output shape for the action net.

    Until tasks model arbitrary actions, we assume the action space equals the
    theta space. A task opts out by defining `event_shape_action`; otherwise we
    fall back to event_shape_theta and warn.
    """
    if hasattr(task, "event_shape_action"):
        return task.event_shape_action

    manager.event_logger.warning(
        f"{type(task).__name__} does not define `event_shape_action`; assuming "
        f"action space == theta space (event_shape_theta="
        f"{tuple(task.event_shape_theta)})."
    )
    return task.event_shape_theta


def build_policy_nets(cfg, task, manager):
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
        event_shape_out=resolve_event_shape_action(task, manager),
    )

    return design_net, action_net


def default_design_distr(task, device, manager):
    """A Gaussian distribution over designs, for objectives that need a design prior.

    Assumes ``task.event_shape_design`` is flat.
    """
    manager.event_logger.warning(
        "Assuming a Gaussian design prior. Also assumes that task.event_shape_design is flat."
    )
    return TorchDistribution(
        torch.distributions.MultivariateNormal(
            torch.zeros(task.event_shape_design, device=device),
            torch.eye(task.event_shape_design.numel(), device=device),
        )
    )


def build_loss_fn(cfg, task, device, manager):
    """Instantiates the training loss.

    Most losses only depend on ``cfg.loss``, but some (e.g. LogLoss) also
    need runtime objects the config can't express, like the task itself or a
    design distribution to estimate an outer expectation. Rather than special
    casing loss classes by name, we inspect the target's constructor and only
    supply what it actually declares.
    """
    loss_cls = get_class(cfg.loss._target_)
    params = inspect.signature(loss_cls.__init__).parameters

    extra_kwargs = {}
    if "task" in params:
        extra_kwargs["task"] = task
    if "design_distr" in params:
        extra_kwargs["design_distr"] = default_design_distr(task, device, manager)

    return instantiate(cfg.loss, **extra_kwargs)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    """ Entry point for ActionBED training.

    Usage: python actionbed.py config='./path/to/config.yaml' save_dir='./save_path/'
    """
    cfg_cli = OmegaConf.from_cli()
    cfg_file = OmegaConf.load(cfg_cli.config)
    # Some defaults for optional args
    cfg_defaults = OmegaConf.create(
        {
            "n_action_warmup_iter": 0,
            "betas": (0.9, 0.999),
            "grad_clip": None,
            "lr_period": None,
            "lr_gamma": None,
        }
    )
    # Precedence: CLI > file > defaults
    cfg = OmegaConf.merge(cfg_defaults, cfg_file, cfg_cli)

    # Manager setup
    run_name = f"{os.environ.get('SLURM_ARRAY_JOB_ID', '0')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    manager = Manager(cfg, root=cfg.save_dir, name=run_name, timestamp=True)

    # Set seed
    torch.manual_seed(cfg.seed)

    task = instantiate(cfg.task)
    task = task.to(cfg.device)

    design_net, action_net = build_policy_nets(cfg, task, manager)
    design_net = design_net.to(cfg.device)
    action_net = action_net.to(cfg.device)
    designer = build_policy_designer(task, cfg.T, design_net)

    # Setup for action-warmup
    random_designer = None
    if cfg.n_action_warmup_iter > 0:
        design_dist = default_design_distr(task, cfg.device, manager)
        random_designer = RandomDesigner(task, cfg.T, dist=design_dist)

    # Create optimizer + LR scheduler
    all_policy_params = list(design_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_policy_params, lr=cfg.lr, betas=cfg.betas)

    scheduler = None
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    loss_fn = build_loss_fn(cfg, task, cfg.device, manager)

    trainer = Trainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        scheduler=scheduler,
        random_designer=random_designer,
        manager=manager,
    )

    trainer.train()
