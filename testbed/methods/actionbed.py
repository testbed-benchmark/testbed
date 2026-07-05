"""Training for ActionBED (Rossa et al., 2026)."""

import hydra
import torch
from hydra.utils import instantiate

from testbed.core.setup import setup
from testbed.designers import PolicyDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for ActionBED-style training."""

    def __init__(
        self, task, designer, action_net, loss_fn, optimizer, cfg, manager=None
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
        self.cfg = cfg

        self.manager = manager

        # checkpointing
        self.best_metric = torch.inf

    def train(self):
        """Full training loop."""
        self.designer.policy_net.train()
        self.action_net.train()

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

        save_path = self.manager.path("final_step.pt")
        torch.save(
            {
                "design_net": self.designer.policy_net.state_dict(),
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
        y_obs, design_obs = self.designer.rollout(theta)

        action = self.action_net(y_obs, design_obs)
        loss = self.loss_fn(action, theta)

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            all_params = list(self.designer.policy_net.parameters()) + list(
                self.action_net.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)

        self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def evaluate(self):
        """Evalutes training with a fresh batch of rollouts."""
        cfg = self.cfg

        policy_was_training = self.designer.policy_net.training
        action_was_training = self.action_net.training
        self.designer.policy_net.eval()
        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            y_obs, design_obs = self.designer.rollout(theta)
            action = self.action_net(y_obs, design_obs)
            loss = self.loss_fn(action, theta)

        self.designer.policy_net.train(policy_was_training)
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
                    "design_net": self.designer.policy_net.state_dict(),
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


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="actionbed")
def main(cfg):
    """Entry point for ActionBED training.

    Usage: python testbed/methods/actionbed.py save_dir='./save_path/' [overrides...]
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    design_net, action_net = build_policy_nets(cfg, task, manager)
    design_net = design_net.to(cfg.device)
    action_net = action_net.to(cfg.device)
    designer = PolicyDesigner(task, cfg.T, design_net)

    all_policy_params = list(design_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_policy_params, lr=cfg.lr)

    loss_fn = instantiate(cfg.loss)

    trainer = Trainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        cfg=cfg,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
