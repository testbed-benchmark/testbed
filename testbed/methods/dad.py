"""Training for DAD (Foster et al., 2021)."""

import hydra
import torch
from hydra.utils import instantiate

from testbed.core.setup import setup
from testbed.design_objectives import SNMC, SPCE, SPCEReinforce
from testbed.designers import PolicyDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for DAD-style training."""

    def __init__(self, task, designer, objective, optimizer, cfg, manager=None):
        """Initializes the trainer."""
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.objective = objective
        self.optimizer = optimizer
        self.cfg = cfg
        self.manager = manager

        # checkpointing
        self.best_metric = -torch.inf

    def train(self):
        """Full training loop."""
        self.designer.design_net.train()

        for step in range(self.cfg.n_train_iter):
            metrics = self.train_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            # evaluation
            if self.cfg.eval_every > 0 and step % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate()
                eval_metrics["step"] = step

                if self.manager.metric_logger is not None:
                    self.manager.metric_logger.log(eval_metrics)

            self._maybe_save_best(step, metrics, min_step=500)

        save_path = self.manager.path("final_step.pt")
        torch.save(
            {
                "design_net": self.designer.design_net.state_dict(),
                "use_raw_history": self.designer.use_raw_history,
                "step": step,
            },
            save_path,
        )

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

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

        return {
            "gain": gain.item(),
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

            torch.save(
                {
                    "design_net": self.designer.design_net.state_dict(),
                    "use_raw_history": self.designer.use_raw_history,
                    "step": step,
                    "metric": metric_val,
                },
                chkpt_path,
            )

            self.manager.event_logger.info(
                f"Saving new best checkpoint : gain={metric_val} @ step={step}"
            )


# ============================================================================
# Helper + factory functions
# ============================================================================


def build_design_net(cfg, task):
    """Instantiates the design policy network.

    Any raw-to-design reparametrization (e.g. HTD's exp/sigmoid transform)
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

    objectives = {
        "snmc": lambda: SNMC(task, n_contrastive=n_inner),
        "spce": lambda: SPCE(task, n_contrastive=n_inner),
        "reinforce": lambda: SPCEReinforce(
            task,
            n_contrastive=n_inner,
            stochastic_design_policy=cfg.objective.requires_policy_log_probs,
            baseline=cfg.objective.baseline,
        ),
    }
    name = cfg.objective.name
    return objectives[name]()


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="dad")
def main(cfg):
    """Entry point for DAD training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/dad.py [overrides...]``.
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    policy_net = build_design_net(cfg, task)
    policy_net = policy_net.to(cfg.device)
    designer = PolicyDesigner(
        task,
        cfg.T,
        policy_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
    )
    # Update config to match the designer's proposal/gradient capability.
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs

    objective = build_objective(cfg, task)

    optimizer = torch.optim.Adam(
        policy_net.parameters(),
        lr=cfg.lr,
    )

    trainer = Trainer(
        task=task,
        designer=designer,
        objective=objective,
        optimizer=optimizer,
        cfg=cfg,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
