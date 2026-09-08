"""Training for ActionBED (Rossa et al., 2026)."""

import inspect

import hydra
import torch
from hydra.utils import get_class, instantiate

from testbed.core.setup import setup
from testbed.design_objectives import LossReinforce
from testbed.designers import (
    ActionHook,
    PolicyDesigner,
    RandomDesigner,
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

        # checkpointing
        self.best_metric = torch.inf

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
                "use_raw_history": self.designer.use_raw_history,
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

            torch.save(
                {
                    "design_net": self.designer.design_net.state_dict(),
                    "action_net": self.action_net.state_dict(),
                    "step": step,
                    "metric": metric_val,
                    "use_raw_history": self.designer.use_raw_history,
                },
                chkpt_path,
            )

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
        event_shape_out=task.action_space.event_shape,
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


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(
    version_base="1.3", config_path="../recipes", config_name="method/actionbed"
)
def main(cfg):
    """Entry point for ActionBED training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/actionbed.py [overrides...]``.
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

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
    optimizer = torch.optim.Adam(all_policy_params, lr=cfg.lr, betas=cfg.betas)

    scheduler = None
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    loss_fn = build_loss_fn(cfg, task)
    objective = build_objective(cfg, task)

    trainer = Trainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        objective=objective,
        scheduler=scheduler,
        random_designer=random_designer,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
