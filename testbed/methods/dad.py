"""Training for DAD (Foster et al., 2021)."""

import os

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.manager import Manager
from testbed.design_objectives import SNMC, SPCE, Reinforce
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
        self.designer.policy_net.train()

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
                "model": self.designer.policy_net.state_dict(),
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

        # Policy log probs are recorded at sample time so the value matches the
        # exact policy distribution the designs were drawn from.
        if getattr(self.objective, "requires_policy_log_probs", False):
            y_obs, design_obs, policy_log_probs = self.designer.rollout(
                theta0, return_policy_log_probs=True
            )
            gain = self.objective.estimate_gain(
                y_obs, design_obs, theta0, policy_log_probs
            )
        else:
            y_obs, design_obs = self.designer.rollout(theta0)
            gain = self.objective.estimate_gain(y_obs, design_obs, theta0)
        loss = -gain

        self.optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.designer.policy_net.parameters(), cfg.grad_clip
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
                    "model": self.designer.policy_net.state_dict(),
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
    """Instantiates the design policy network."""
    return instantiate(
        cfg.design_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def build_spce_reinforce(cfg, task, designer):
    """Instantiates a Reinforce objective with an SPCE value function."""
    spce = SPCE(task, n_contrastive=cfg.objective.n_inner)
    baseline = getattr(cfg.objective, "baseline", None)

    # Reinforce reads the estimator mode (score-function vs pathwise DAD Eq. 16)
    # from the designer; the SPCE gain is the value in both cases.
    return Reinforce(task, spce.per_sample_gain, designer, baseline=baseline)


def build_objective(cfg, task, designer=None):
    """Instantiates the objective for training DAD.

    SNMC/SPCE need the task to construct, so it can't be a pure instantiate.
    """
    objectives = {
        "snmc": lambda: SNMC(task, n_contrastive=cfg.objective.n_inner),
        "spce": lambda: SPCE(task, n_contrastive=cfg.objective.n_inner),
        "reinforce": lambda: build_spce_reinforce(cfg, task, designer),
    }
    name = cfg.objective.name
    return objectives[name]()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    """ Entry point for DAD training.

    Usage: python dad.py config='./path/to/config.yaml' save_dir='./save_path/'
    """
    conf_cli = OmegaConf.from_cli()
    cfg = OmegaConf.load(conf_cli.config)
    # sync cli args with config
    cfg = OmegaConf.merge(cfg, conf_cli)

    # Manager setup
    run_name = f"{os.environ.get('SLURM_ARRAY_JOB_ID', '0')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    manager = Manager(cfg, root=cfg.save_dir, name=run_name, timestamp=True)

    # Set seed
    torch.manual_seed(cfg.seed)

    task = instantiate(cfg.task)
    task = task.to(cfg.device)

    policy_net = build_design_net(cfg, task)
    policy_net = policy_net.to(cfg.device)
    designer = PolicyDesigner(
        task,
        cfg.T,
        policy_net,
        reparameterize_policy=getattr(cfg.objective, "reparameterize_policy", False),
    )

    objective = build_objective(cfg, task, designer)

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
