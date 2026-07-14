"""Training an actions-only policy with randomized designs (Rossa et al., 2026)."""

import os

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.distributions import TorchDistribution
from testbed.core.manager import Manager
from testbed.designers import ActionHook, RandomDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for ActionBED-style training.

    This class only trains the action policy and assumes designs are drawn from a random design distribution.
    """

    def __init__(
        self,
        task,
        designer,
        action_net,
        loss_fn,
        optimizer,
        cfg,
        scheduler=None,
        manager=None,
    ):
        """Initializes the trainer."""
        assert isinstance(designer, RandomDesigner), (
            "This Trainer assumes your designer is a RandomDesigner object."
        )

        self.task = task
        self.designer = designer
        self.action_net = action_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg

        self.manager = manager

        # checkpointing
        self.best_metric = torch.inf

        self.designer.hooks.append(ActionHook(self.action_net))

    def train(self):
        """Full training loop."""
        self.action_net.train()

        if self.manager:
            self.manager.event_logger.info("Starting training...")
        for step in range(self.cfg.n_train_iter):
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
                "action_net": self.action_net.state_dict(),
                "step": step,
            },
            save_path,
        )

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

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
            torch.nn.utils.clip_grad_norm_(self.action_net.parameters(), cfg.grad_clip)

        self.optimizer.step()

        return {"loss": loss.item()}

    def evaluate(self):
        """Evalutes with a fresh batch of rollouts."""
        cfg = self.cfg

        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            rollout = self.designer.rollout(theta)
            action = rollout.extra["action"]
            loss = self.loss_fn(action, theta)

        self.action_net.train()

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


def build_policy_net(cfg, task, manager):
    """Instantiates the action policy network."""
    action_net = instantiate(
        cfg.action_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=resolve_event_shape_action(task, manager),
    )

    return action_net


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    """ Entry point for ActionBED style training of actions only.

    Usage: python action.py config='./path/to/config.yaml' save_dir='./save_path/'
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

    action_net = build_policy_net(cfg, task, manager)
    action_net = action_net.to(cfg.device)

    manager.event_logger.warning(
        "Assuming a Gaussian design prior for action warmup. Also assumes that task.event_shape_design is flat."
    )
    design_dist = TorchDistribution(
        torch.distributions.MultivariateNormal(
            torch.zeros(task.event_shape_design, device=cfg.device),
            torch.eye(task.event_shape_design.numel(), device=cfg.device),
        )
    )
    random_designer = RandomDesigner(task, cfg.T, design_dist)

    # Create optimizer + LR scheduler
    optimizer = torch.optim.Adam(action_net.parameters(), lr=cfg.lr, betas=cfg.betas)
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    loss_fn = instantiate(cfg.loss)

    trainer = Trainer(
        task=task,
        designer=random_designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        scheduler=scheduler,
        manager=manager,
    )

    trainer.train()
