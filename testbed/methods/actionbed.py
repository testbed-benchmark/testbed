"""Training for ActionBED (Rossa et al., 2026)."""

import hydra
import torch
from hydra.utils import instantiate

from testbed.core.distributions import TorchDistribution
from testbed.core.setup import setup
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.envs.design_spaces import CategoricalDesignSpace

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
        action_optimizer=None,
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
        self.design_optimizer = optimizer
        self.action_optimizer = action_optimizer or optimizer
        self.separate_optimizers = action_optimizer is not None
        self.scheduler = scheduler
        self.cfg = cfg

        # used for action policy warmup
        self.random_designer = random_designer

        self.manager = manager

        # checkpointing
        self.best_metric = torch.inf

    def train(self):
        """Full training loop."""
        self.designer.policy_net.train()
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
                "design_net": self.designer.policy_net.state_dict(),
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
        y_obs, design_obs = self.random_designer.rollout(theta)

        action = self.action_net(y_obs, design_obs)
        loss = self.loss_fn(action, theta)
        if loss.ndim > 0:
            loss = loss.mean()

        self.action_optimizer.zero_grad()
        loss.backward()

        action_grad_clip = getattr(cfg, "action_grad_clip", cfg.grad_clip)
        if action_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.action_net.parameters(), action_grad_clip
            )

        self.action_optimizer.step()

        return {"action_warmup/loss": loss.item()}

    def train_step(self):
        """Takes a single training step."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_outer)
        if self.designer.reinforce_requires_policy_log_probs:
            y_obs, design_obs, policy_log_probs = self.designer.rollout(
                theta, return_policy_log_probs=True
            )
        else:
            y_obs, design_obs = self.designer.rollout(theta)

        action = self.action_net(y_obs, design_obs)
        loss = self.loss_fn(action, theta)
        entropy = None
        if self.designer.reinforce_requires_policy_log_probs:
            if loss.ndim == 0 or loss.shape[0] != theta.shape[0]:
                raise ValueError(
                    "Discrete ActionBED policies require an unreduced loss with "
                    "one leading batch dimension."
                )
            loss_per_rollout = loss.reshape(loss.shape[0], -1).mean(dim=1)
            baseline = leave_one_out_baseline(loss_per_rollout.detach())
            score = policy_log_probs.sum(dim=-1)
            score_loss = (
                (loss_per_rollout.detach() - baseline) * (score - score.detach())
            ).mean()
            loss = loss_per_rollout.mean() + score_loss

            entropy_coef = getattr(cfg, "entropy_coef", 0.0)
            if entropy_coef != 0.0:
                entropies = []
                for t in range(design_obs.shape[1]):
                    _, step_entropy = self.designer.policy_net.evaluate_designs(
                        y_obs[:, :t], design_obs[:, :t], design_obs[:, t]
                    )
                    entropies.append(step_entropy)
                entropy = torch.stack(entropies, dim=1).mean()
                loss = loss - entropy_coef * entropy

        self.design_optimizer.zero_grad()
        if self.separate_optimizers:
            self.action_optimizer.zero_grad()
        loss.backward()

        if self.separate_optimizers:
            design_grad_clip = getattr(cfg, "design_grad_clip", cfg.grad_clip)
            action_grad_clip = getattr(cfg, "action_grad_clip", cfg.grad_clip)
            if design_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.designer.policy_net.parameters(), design_grad_clip
                )
            if action_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.action_net.parameters(), action_grad_clip
                )
        elif cfg.grad_clip is not None:
            all_params = list(self.designer.policy_net.parameters()) + list(
                self.action_net.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)

        self.design_optimizer.step()
        if self.separate_optimizers:
            self.action_optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "loss": loss_per_rollout.mean().item()
            if entropy is not None
            else loss.item()
        }
        if entropy is not None:
            metrics["entropy"] = entropy.item()
        return metrics

    def evaluate_warmup(self):
        """Evalutes warmup with a fresh batch of rollouts."""
        cfg = self.cfg

        policy_was_training = self.designer.policy_net.training
        action_was_training = self.action_net.training
        self.designer.policy_net.eval()
        self.action_net.eval()

        with torch.no_grad():
            theta = self.task.sample_prior(cfg.n_outer_eval)
            y_obs, design_obs = self.random_designer.rollout(theta)
            action = self.action_net(y_obs, design_obs)
            loss = self.loss_fn(action, theta)
            if loss.ndim > 0:
                loss = loss.mean()

        self.designer.policy_net.train(policy_was_training)
        self.action_net.train(action_was_training)

        return {
            "action_warmup/eval/loss": loss.item(),
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
            if loss.ndim > 0:
                loss = loss.mean()

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
    if task.discrete_design:
        design_net = instantiate(cfg.design_net, design_space=task.design_space)
    else:
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

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/actionbed.py [overrides...]``.
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    design_net, action_net = build_policy_nets(cfg, task, manager)
    design_net = design_net.to(cfg.device)
    action_net = action_net.to(cfg.device)
    designer = PolicyDesigner(task, cfg.T, design_net)

    # Setup for action-warmup
    random_designer = None
    if cfg.n_action_warmup_iter > 0:
        if isinstance(task.design_space, CategoricalDesignSpace):
            design_dist = TorchDistribution(
                torch.distributions.Categorical(
                    logits=torch.zeros(task.design_space.n, device=cfg.device)
                )
            )
        else:
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

    # Create separate optimizers so policy clipping cannot suppress action updates.
    design_optimizer = torch.optim.Adam(
        design_net.parameters(),
        lr=getattr(cfg, "design_lr", cfg.lr),
        betas=cfg.betas,
    )
    action_optimizer = torch.optim.Adam(
        action_net.parameters(),
        lr=getattr(cfg, "action_lr", cfg.lr),
        betas=cfg.betas,
    )

    scheduler = None
    if cfg.lr_period is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            design_optimizer, step_size=cfg.lr_period, gamma=cfg.lr_gamma
        )

    loss_fn = instantiate(cfg.loss)

    trainer = Trainer(
        task=task,
        designer=designer,
        action_net=action_net,
        loss_fn=loss_fn,
        optimizer=design_optimizer,
        cfg=cfg,
        scheduler=scheduler,
        action_optimizer=action_optimizer,
        random_designer=random_designer,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
