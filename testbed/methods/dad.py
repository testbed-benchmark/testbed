"""Training for DAD (Foster et al., 2021)."""

import time
from math import inf

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.setup import make_main, setup
from testbed.core.util import restore_rng, rng_state, seed_all
from testbed.design_objectives import SNMC, SPCE, SPCEReinforce
from testbed.designers import PolicyDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for DAD-style training."""

    def __init__(
        self, task, designer, objective, optimizer, cfg, manager=None, scheduler=None
    ):
        """Initializes the trainer."""
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.objective = objective
        self.optimizer = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.manager = manager

        # checkpointing / progress, restored by load_checkpoint
        self.best_metric = -torch.inf
        self.step = -1
        self.elapsed_seconds = 0.0
        self._start_time = time.monotonic()

    def train(self):
        """Full training loop."""
        self.designer.design_net.train()

        budget = self.cfg.train_time_budget_seconds
        self._budget = inf if budget is None else budget
        self._start_time = time.monotonic() - self.elapsed_seconds
        self.stop_reason = "iteration_limit"

        for step in range(self.step + 1, self.cfg.n_train_iter):
            metrics = self.train_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            if self._checkpoint_or_stop(step):
                break

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step

                if self.manager.metric_logger is not None:
                    self.manager.metric_logger.log(eval_metrics)

            self._maybe_save_best(step, metrics, min_step=500)

        self._finish()

    def _checkpoint_or_stop(self, step):
        """Saves a periodic policy checkpoint; returns True once the time budget is spent."""
        self.step = step
        interval = self.cfg.checkpoint_every
        if interval and (step + 1) % interval == 0:
            torch.save(
                self._policy_state(step), self.manager.path(f"step_{step + 1:08d}.pt")
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
            {
                **self._policy_state(self.step),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler and self.scheduler.state_dict(),
                "best_metric": self.best_metric,
                "stop_reason": self.stop_reason,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "rng_state": rng_state(),
            },
            self.manager.path("final_step.pt"),
        )

        if self.manager.metric_logger is not None:
            self._log_summary()
            self.manager.metric_logger.close()

    def _log_summary(self):
        """Logs how far training got, as metrics rather than only to the event log."""
        if self.step < 0:
            return
        self.manager.metric_logger.log(
            {
                "step": self.step,
                "train/steps_completed": self.step + 1,
                "train/elapsed_seconds": self.elapsed_seconds,
                "train/stop_reason": self.stop_reason,
            }
        )

    def _policy_state(self, step):
        return {
            "design_net": self.designer.design_net.state_dict(),
            "use_raw_history": self.designer.use_raw_history,
            "step": step,
            "elapsed_seconds": time.monotonic() - self._start_time,
        }

    def train_step(self):
        """Takes a single training step."""
        cfg = self.cfg

        theta0 = self.task.sample_prior(cfg.n_outer)
        rollout = self.designer.rollout(theta0)
        gain = self.objective.estimate_gain(rollout)
        loss = -gain

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.designer.design_net.parameters(), cfg.grad_clip
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "gain": gain.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self):
        """Evaluates training progress using SNMC/SPCE lower/upper bounds."""
        # TODO
        # Evaluating with a large inner sample size slows down training quite a bit.
        # It's also memory intensive and kills my local machine.

        return {"eval/spce": 0.0, "eval/snmc": 0.0}

    def _maybe_save_best(self, step, metrics, min_step=0):
        # Saves a checkpoint of the model if it's the lowest training loss seen so far
        # NB: assumes our objective is maximizing a gain

        # min_step controls how long we wait before saving; useful for avoiding a ton of i/o at the start
        if step < min_step:
            return

        if self.manager is None:
            return

        step = metrics["step"]
        metric_val = metrics["gain"]
        is_better = metric_val > self.best_metric

        if is_better:
            self.best_metric = metric_val

            chkpt_path = self.manager.path("best.pt")

            torch.save({**self._policy_state(step), "metric": metric_val}, chkpt_path)

            self.manager.event_logger.info(
                f"Saving new best checkpoint : gain={metric_val} @ step={step}"
            )


# ============================================================================
# Helper + factory functions
# ============================================================================


def build_design_net(cfg, task):
    """Instantiates the design policy network.

    Any raw-to-design reparametrization (e.g. TD's exp/sigmoid transform)
    lives on task.design_space and is applied automatically by the designer
    -- no net-level wrapper needed.
    """
    return instantiate(
        cfg.design_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def build_objective(cfg, task):
    """Instantiates the objective for training DAD.

    SNMC/SPCE need the task to construct, so it can't be a pure instantiate.
    """
    n_inner = cfg.objective.n_inner
    fused = cfg.objective.fused

    objectives = {
        "snmc": lambda: SNMC(task, n_contrastive=n_inner, fused=fused),
        "spce": lambda: SPCE(task, n_contrastive=n_inner, fused=fused),
        "reinforce": lambda: SPCEReinforce(
            task,
            n_contrastive=n_inner,
            stochastic_design_policy=cfg.objective.requires_policy_log_probs,
            baseline=cfg.objective.baseline,
            fused=fused,
        ),
    }
    name = cfg.objective.name
    return objectives[name]()


def build_trainer(cfg, task, manager=None):
    """Seeds, then builds a fresh trainer with its network, optimizer and objective."""
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    policy_net = build_design_net(cfg, task)
    policy_net = policy_net.to(cfg.device)
    designer = PolicyDesigner(
        task,
        cfg.T,
        policy_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
        use_raw_history=cfg.use_raw_history,
    )
    # Update config to match the designer's proposal/gradient capability.
    # build_objective below reads the flag, so this has to come first.
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs

    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=cfg.lr,
        betas=tuple(cfg.betas),
    )

    # Multiply the LR by lr_gamma once every lr_anneal_every steps (DAD paper D.2).
    scheduler = None
    if cfg.lr_anneal_every is not None and cfg.lr_gamma is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_anneal_every, gamma=cfg.lr_gamma
        )

    return Trainer(
        task=task,
        designer=designer,
        objective=build_objective(cfg, task),
        optimizer=optimizer,
        cfg=cfg,
        manager=manager,
        scheduler=scheduler,
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
    """Entry point for DAD training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/dad.py [overrides...]``.
    """
    start_time = time.monotonic()
    manager, task = setup(cfg)
    trainer = build_trainer(cfg, task, manager)
    # Setup counts toward the time budget.
    trainer.elapsed_seconds = time.monotonic() - start_time
    trainer.train()


main = make_main(run, "method/dad")

if __name__ == "__main__":
    main()
