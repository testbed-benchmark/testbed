"""Training for static (non-adaptive) designs, with random restarts."""

import hydra
import torch
from torch import nn

from testbed.core.rollout import Rollout
from testbed.core.setup import setup
from testbed.design_objectives import SNMC, SPCE
from testbed.designers import StaticDesigner
from testbed.eval import eval_spce_snmc

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for static designs, optimized from ``R`` random restarts.

    The static-design objective can be multi-modal, and hence sensitive
    to initialisation. Here ``R`` independently
    initialized designs are optimized simultaneously and the best one is kept.
    """

    def __init__(
        self, task, designers, objective, optimizer, cfg, manager=None, scheduler=None
    ):
        """Initializes the trainer."""
        assert all(isinstance(designer, StaticDesigner) for designer in designers), (
            "This Trainer assumes your designers are StaticDesigner objects."
        )

        self.task = task
        self.designers = nn.ModuleList(designers)
        self.objective = objective
        self.optimizer = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.manager = manager

        # checkpointing
        self.best_metric = -torch.inf

        self.n_restarts = len(self.designers)

    def train(self):
        """Full training loop, then final selection of the winning restart."""
        self.designers.train()

        for step in range(self.cfg.n_train_iter):
            metrics = self.train_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            self._maybe_save_best(step, metrics, min_step=self.cfg.save_best_after)

        final_step = self.cfg.n_train_iter - 1
        best, gains = self.select_best()
        if self.manager:
            self.manager.event_logger.info(
                f"Selected restart {best} : sPCE={gains[best]:.4f} "
                f"(restarts: {[round(g, 4) for g in gains]})"
            )
            torch.save(
                self._checkpoint(
                    best, step=final_step, gain=gains[best], gain_per_restart=gains
                ),
                self.manager.path("final_step.pt"),
            )

            if self.manager.metric_logger is not None:
                self.manager.metric_logger.log(
                    {
                        "select/gain": gains[best],
                        "select/restart": best,
                        "step": final_step,
                    }
                )
                self.manager.metric_logger.close()

    def train_step(self):
        """Takes a single training step, updating every restart."""
        cfg = self.cfg

        gains = self.restart_gains(cfg.n_outer)  # [R]
        loss = -gains.sum()

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            # Clipped per restart, not globally
            for designer in self.designers:
                torch.nn.utils.clip_grad_norm_(designer.parameters(), cfg.grad_clip)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # One scalar per restart, not a tensor: the metric loggers take scalars
        # (and W&B then gives each restart its own curve).
        per_restart = gains.detach().tolist()
        best = int(gains.argmax())
        return {
            "gain": per_restart[best],
            **{f"gain/restart_{i}": g for i, g in enumerate(per_restart)},
            "best_restart": best,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def restart_gains(self, n_outer):
        """Estimates each restart's gain from ``n_outer`` rollouts.

        The rollouts are concatenated into one batch so the objective can
        be scored in a single vectorized pass for all ``R`` restarts.

        Returns:
            Tensor: Per-restart gain estimates, shape ``[R]``.
        """
        theta = self.task.sample_prior(self.n_restarts * n_outer)
        rollouts = [
            designer.rollout(theta_i)
            for designer, theta_i in zip(self.designers, theta.chunk(self.n_restarts))
        ]
        merged = Rollout(
            theta=torch.cat([r.theta for r in rollouts]),
            y_history=torch.cat([r.y_history for r in rollouts]),
            design_history=torch.cat([r.design_history for r in rollouts]),
        )
        per_sample = self.objective.per_sample_gain(merged)
        return per_sample.reshape(self.n_restarts, n_outer).mean(dim=1)

    @torch.no_grad()
    def select_best(self):
        """Picks the winning restart by a larger, independent sPCE estimate.

        The training-time gains are often too noisy to reliably rank the restarts.

        Returns:
            tuple[int, list[float]]: The winning restart's index, and every
            restart's sPCE lower bound.
        """
        cfg = self.cfg.select
        gains = [
            eval_spce_snmc(
                self.task,
                designer,
                n_outer=cfg.n_outer,
                n_contrastive=cfg.n_contrastive,
                outer_batch_size=cfg.outer_batch_size,
                contrastive_chunk_size=cfg.contrastive_chunk_size,
                verbose=False,
                fused=cfg.fused,
            )["lower_bound"]
            for designer in self.designers
        ]
        return max(range(len(gains)), key=gains.__getitem__), gains

    def _checkpoint(self, best, step, gain, gain_per_restart=None):
        # The winner's raw design is what evaluation reloads; the others ride
        # along so a finished run can be re-ranked without retraining.
        state = {
            "raw_design": self.designers[best].raw_design.detach().cpu(),
            "raw_designs": torch.stack(
                [d.raw_design.detach().cpu() for d in self.designers]
            ),
            "design": self.designers[best].design.detach().cpu(),
            "designs": torch.stack([d.design.detach().cpu() for d in self.designers]),
            "transform_designs": self.designers[best].transform_designs,
            "best_restart": best,
            "step": step,
            "gain": gain,
        }
        if gain_per_restart is not None:
            state["gain_per_restart"] = gain_per_restart
        return state

    def _maybe_save_best(self, step, metrics, min_step=0):
        # Saves a checkpoint of the designs if this is the highest training gain
        # seen so far, across restarts.
        # NB: assumes our objective is maximizing a gain

        # min_step controls how long we wait before saving; useful for avoiding a ton of i/o at the start
        if step < min_step:
            return

        if self.manager is None:
            return

        metric_val = metrics["gain"]
        if metric_val <= self.best_metric:
            return

        self.best_metric = metric_val

        torch.save(
            self._checkpoint(metrics["best_restart"], step=step, gain=metric_val),
            self.manager.path("best.pt"),
        )

        self.manager.event_logger.info(
            f"Saving new best checkpoint : gain={metric_val} @ step={step} "
            f"(restart {metrics['best_restart']})"
        )


# ============================================================================
# Helper + factory functions
# ============================================================================


def build_objective(cfg, task):
    """Instantiates the objective for training static designs.

    Restricted to the bounds that expose a per-sample gain, which is what
    scores the restarts separately. REINFORCE has no place here: a
    StaticDesigner is deterministic, so its gain differentiates directly.
    """
    objectives = {
        "snmc": SNMC,
        "spce": SPCE,
    }
    name = cfg.objective.name
    if name not in objectives:
        raise ValueError(
            f"static designs support {sorted(objectives)} objectives; got {name!r}."
        )
    return objectives[name](
        task,
        n_contrastive=cfg.objective.n_inner,
        chunk_size=cfg.objective.chunk_size,
        fused=cfg.objective.fused,
    )


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/static")
def main(cfg):
    """Entry point for static-design training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/static.py [overrides...]``.
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    designers = [
        StaticDesigner(task, cfg.T).to(cfg.device) for _ in range(cfg.n_restarts)
    ]

    objective = build_objective(cfg, task)

    optimizer = torch.optim.Adam(
        [p for designer in designers for p in designer.parameters()],
        lr=cfg.lr,
        betas=cfg.betas,
    )

    # Multiply the LR by lr_gamma once every lr_anneal_every steps
    scheduler = None
    if cfg.lr_anneal_every is not None and cfg.lr_gamma is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_anneal_every, gamma=cfg.lr_gamma
        )

    trainer = Trainer(
        task=task,
        designers=designers,
        objective=objective,
        optimizer=optimizer,
        cfg=cfg,
        manager=manager,
        scheduler=scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
