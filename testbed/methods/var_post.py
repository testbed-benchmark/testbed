"""Training for variational-posterior (Barber-Agakov) design, frozen or joint guide.

Inspired by Bayesian Experimental Design via Score Matching paper.
"""

import hydra
import torch
from hydra.utils import instantiate

from testbed.core.setup import setup
from testbed.design_objectives import GuideNLL, VarPost, VarPostReinforce
from testbed.designers import (
    PolicyDesigner,
    RandomDesigner,
    StochasticPolicyDesigner,
    build_policy_designer,
)

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for VarPost-style training.

    Two phases: a guide-pretraining phase (fits the guide net against random
    designs) followed by a policy-training phase, mirroring ActionBED's
    two-phase structure. `cfg.joint` controls whether the guide keeps
    training (rather than being frozen) during the second phase.
    """

    def __init__(
        self,
        task,
        designer,
        guide_net,
        objective,
        guide_loss,
        policy_optimizer,
        guide_optimizer,
        cfg,
        pretrain_designer,
        scheduler=None,
        manager=None,
    ):
        """Initializes the trainer."""
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.guide_net = guide_net
        self.objective = objective
        self.guide_loss = guide_loss
        self.policy_optimizer = policy_optimizer
        self.guide_optimizer = guide_optimizer
        self.cfg = cfg
        self.pretrain_designer = pretrain_designer
        self.scheduler = scheduler
        self.manager = manager

        self.joint = bool(cfg.joint)

        # checkpointing
        self.best_metric = -torch.inf

    def train(self):
        """Full training loop: guide pretraining, then policy training."""
        self.designer.design_net.train()
        self.guide_net.train()

        for step in range(self.cfg.n_guide_pretrain_iter):
            metrics = self.guide_pretrain_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

        if not self.joint:
            self._freeze_guide()

        for step in range(
            self.cfg.n_guide_pretrain_iter,
            self.cfg.n_guide_pretrain_iter + self.cfg.n_train_iter,
        ):
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
                "guide_net": self.guide_net.state_dict(),
                "step": step,
            },
            save_path,
        )

        if self.manager.metric_logger is not None:
            self.manager.metric_logger.close()

    def guide_pretrain_step(self):
        """Takes a single guide-pretraining step against selected designs - typically random designs."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_guide_outer)
        # The pretrain designer has no trainable params, so no_grad on the
        # rollout just avoids building an unnecessary graph through it. The
        # resulting theta/y_history/design_history are plain tensors either
        # way; guide_loss's forward pass below still builds a fresh graph
        # from the guide net's own parameters.
        with torch.no_grad():
            rollout = self.pretrain_designer.rollout(theta)

        loss = self.guide_loss.from_rollout(rollout)

        self.guide_optimizer.zero_grad()
        loss.backward()

        if cfg.guide_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.guide_net.parameters(), cfg.guide_grad_clip
            )

        self.guide_optimizer.step()

        return {"guide_pretrain/loss": loss.item()}

    def _freeze_guide(self):
        """Freezes the guide net after pretraining (non-joint mode)."""
        self.guide_net.eval()
        for p in self.guide_net.parameters():
            p.requires_grad_(False)

    def train_step(self):
        """Takes a single policy-training step (and, in joint mode, a guide step)."""
        cfg = self.cfg

        theta0 = self.task.sample_prior(cfg.n_outer)
        rollout = self.designer.rollout(theta0)

        # In joint mode, VarPostReinforce.estimate_gain(train_guide=True)
        # self-contains the guide's gradient. Plain VarPost already updates both
        # nets in a single pass when the policy is pathwise-differentiable
        # so it needs no such flag.
        if self.joint and isinstance(self.objective, VarPostReinforce):
            gain = self.objective.estimate_gain(rollout, train_guide=True)
        else:
            gain = self.objective.estimate_gain(rollout)
        loss = -gain

        self.policy_optimizer.zero_grad()
        if self.joint:
            self.guide_optimizer.zero_grad()

        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.designer.design_net.parameters(), cfg.grad_clip
            )
        self.policy_optimizer.step()

        if self.joint:
            if cfg.guide_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.guide_net.parameters(), cfg.guide_grad_clip
                )
            self.guide_optimizer.step()

        return {"gain": gain.item()}

    def evaluate(self):
        """Evaluates training progress using SNMC/SPCE lower/upper bounds.

        Todo -impleemntation not complete
        """
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
                    "guide_net": self.guide_net.state_dict(),
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


def build_guide_net(cfg, task):
    """Instantiates the guide (variational-posterior) network."""
    return instantiate(
        cfg.guide_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_theta=task.event_shape_theta,
    )


def build_objective(cfg, task, guide_net):
    """Instantiates the objective for training VarPost.

    VarPost/VarPostReinforce need the task and guide_net to construct, so it
    can't be a pure instantiate.
    """
    objectives = {
        "var_post": lambda: VarPost(
            task,
            guide_net,
            use_current_prior=cfg.objective.use_current_prior,
            canonicalize_theta=cfg.objective.canonicalize_theta,
        ),
        "var_post_reinforce": lambda: VarPostReinforce(
            task,
            guide_net,
            stochastic_design_policy=cfg.objective.requires_policy_log_probs,
            baseline=cfg.objective.baseline,
            use_current_prior=cfg.objective.use_current_prior,
            canonicalize_theta=cfg.objective.canonicalize_theta,
        ),
    }
    name = cfg.objective.name
    return objectives[name]()


def build_pretrain_designer(cfg, task):
    """Instantiates the RandomDesigner used to pretrain the guide net."""
    dist = (
        instantiate(
            cfg.pretrain_design_distr,
            T=cfg.T,
            event_shape_design=task.event_shape_design,
        )
        if cfg.pretrain_design_distr
        else None
    )
    return RandomDesigner(task, cfg.T, dist=dist)


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="var_post")
def main(cfg):
    """Entry point for VarPost training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/var_post.py [overrides...]``.
    """
    manager, task = setup(cfg)

    # Set seed
    torch.manual_seed(cfg.seed)

    design_net = build_design_net(cfg, task)
    design_net = design_net.to(cfg.device)
    guide_net = build_guide_net(cfg, task)
    guide_net = guide_net.to(cfg.device)

    designer = build_policy_designer(
        task,
        cfg.T,
        design_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
    )

    cfg.objective.requires_policy_log_probs = (
        isinstance(designer, StochasticPolicyDesigner)
        and not designer.reparameterize_policy
    )

    objective = build_objective(cfg, task, guide_net)
    guide_loss = GuideNLL(
        guide_net, canonicalize_theta=cfg.objective.canonicalize_theta
    )
    pretrain_designer = build_pretrain_designer(cfg, task)

    policy_optimizer = torch.optim.Adam(design_net.parameters(), lr=cfg.lr)
    guide_optimizer = torch.optim.Adam(guide_net.parameters(), lr=cfg.guide_lr)

    trainer = Trainer(
        task=task,
        designer=designer,
        guide_net=guide_net,
        objective=objective,
        guide_loss=guide_loss,
        policy_optimizer=policy_optimizer,
        guide_optimizer=guide_optimizer,
        cfg=cfg,
        pretrain_designer=pretrain_designer,
        manager=manager,
    )

    trainer.train()


if __name__ == "__main__":
    main()
