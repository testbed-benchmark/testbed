"""Training an actions-only policy on top of a frozen designer."""

import hydra
import torch
from hydra.utils import instantiate

from testbed.core.load import designer_from_run, run_config
from testbed.core.setup import instantiate_action_net, setup
from testbed.core.util import seed_all
from testbed.designers import ActionHook, Designer
from testbed.methods.actionbed import build_loss_fn

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Trains an action net on rollouts from a frozen designer.

    Compatible with any :class:`Designer` since all the trainer
    needs is a ``rollout()`` method.
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
        """Initializes the trainer, freezing the designer."""
        assert isinstance(designer, Designer), (
            "This Trainer assumes your designer is a Designer object."
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

        # Freeze before attaching the hook, so the action net stays trainable.
        freeze_designer(designer)

        # resolve use of raw history
        # config takes precedence over designer; defaults to false if not found
        cfg_raw_history = getattr(cfg, "use_raw_history", None)
        if cfg_raw_history is not None:
            self.use_raw_history = bool(cfg_raw_history)
        else:
            self.use_raw_history = bool(getattr(designer, "use_raw_history", False))

        self.designer.hooks.append(
            ActionHook(self.action_net, use_raw_history=self.use_raw_history)
        )

        if manager:
            manager.event_logger.info(
                f"Frozen {type(designer).__name__}; action net conditions on "
                f"{'raw' if self.use_raw_history else 'physical'} designs."
            )

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
                        f"[Eval @ step {step}/{self.cfg.n_train_iter}]: loss={eval_metrics['eval/loss']}"
                    )
                    if self.manager.metric_logger is not None:
                        self.manager.metric_logger.log(eval_metrics)

                self._maybe_save_best(step, eval_metrics)

        if self.manager:
            self.manager.event_logger.info("Saving final checkpoint...")
        torch.save(self._policy_state(step), self.manager.path("final_step.pt"))

        if self.manager:
            self.manager.event_logger.info("Training successfully completed")

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

    def _policy_state(self, step):
        # The frozen designer rides along for the sake of completeness, under
        # the keys the method that trains it uses: "design_net" for a networked
        # designer, "raw_design" for a static one. Either may be absent.
        design_net = getattr(self.designer, "design_net", None)
        raw_design = getattr(self.designer, "raw_design", None)
        return {
            "action_net": self.action_net.state_dict(),
            "design_net": design_net and design_net.state_dict(),
            "raw_design": raw_design
            if raw_design is None
            else raw_design.detach().cpu(),
            "transform_designs": self.designer.transform_designs,
            "use_raw_history": self.use_raw_history,
            "step": step,
        }

    def action_from(self, rollout):
        """Returns the action the hook computed during ``rollout``.

        A designer whose rollout() is its own, rather than the base class's,
        may never run its hooks -- in which case there is no action to train
        on, and saying so beats a bare KeyError.
        """
        if "action" not in rollout.extra:
            raise RuntimeError(
                f"{type(self.designer).__name__}.rollout() produced no action: it "
                "does not run its hooks, so the attached ActionHook never fired."
            )
        return rollout.extra["action"]

    def train_step(self):
        """Takes a single training step."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_outer)
        rollout = self.designer.rollout(theta)

        loss = self.loss_fn(self.action_from(rollout), theta)

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.action_net.parameters(), cfg.grad_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}

    def evaluate(self):
        """Evalutes with a fresh batch of rollouts."""
        cfg = self.cfg

        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            rollout = self.designer.rollout(theta)
            loss = self.loss_fn(self.action_from(rollout), theta)

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
                {**self._policy_state(step), "metric": metric_val},
                chkpt_path,
            )

            self.manager.event_logger.info("Saved new best checkpoint")


# ============================================================================
# Helper + factory functions
# ============================================================================


def freeze_designer(designer):
    """Freezes all parameters associated with a designer (except for possibly any task params).

    Call this before attaching an action hook to only train the action net.
    The Task associated with the designer is a submodule; no need to freeze it here.
    """
    task_params = {id(p) for p in designer.task.parameters()}
    for param in designer.parameters():
        if id(param) not in task_params:
            param.requires_grad_(False)
    for module in designer.children():
        if module is not designer.task:
            module.eval()
    return designer


def adopt_run_task(cfg):
    """Takes ``task`` and ``T`` from the run whose designer we are freezing.

    A frozen designer only means anything on the task it was trained for, and
    T fixes the history length, so the source run's config wins over ours.
    """
    src = run_config(cfg.designer_run_dir)
    cfg.task = src.task
    cfg.T = src.T


def build_designer(cfg, task, manager=None):
    """Instantiates the designer to be frozen.

    Designs are drawn from the task's design space unless ``designer_run_dir``
    names a finished run, whose own designer is rebuilt and loaded instead.
    """
    if cfg.designer_run_dir is None:
        return instantiate(cfg.designer, task=task, T=cfg.T).to(cfg.device)
    return designer_from_run(cfg.designer_run_dir, task, cfg.device, manager)


def build_policy_net(cfg, task):
    """Instantiates the action policy network."""
    return instantiate_action_net(cfg.action_net, task, cfg.T)


def build_trainer(cfg, task, manager=None):
    """Seeds, then builds a fresh trainer with its designer, action net, optimizer and loss."""
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    designer = build_designer(cfg, task, manager)
    action_net = build_policy_net(cfg, task).to(cfg.device)

    optimizer = torch.optim.Adam(
        action_net.parameters(), lr=cfg.lr, betas=tuple(cfg.betas)
    )
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
        scheduler=scheduler,
        manager=manager,
    )


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/action")
def main(cfg):
    """Entry point for training an action net on top of a frozen designer.

    Designs are drawn at random unless ``designer_run_dir`` points at a
    finished run, in which case that run's designer (and its task, T, design
    net and history convention) are taken from its own saved config.

        python testbed/methods/action.py designer_run_dir=runs/<run_name>

    Everything else, the action net and the training schedule, stays ours.

    Args:
        cfg: Composed Hydra configuration for the training run.
    """
    if cfg.designer_run_dir is not None:
        adopt_run_task(cfg)
    manager, task = setup(cfg)
    build_trainer(cfg, task, manager).train()


if __name__ == "__main__":
    main()
