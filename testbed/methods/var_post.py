"""Training for variational-posterior (Barber-Agakov) design, frozen or joint posterior.

Inspired by Bayesian Experimental Design via Score Matching paper.
"""

import copy
import time
from functools import partial
from math import inf

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.setup import setup
from testbed.core.util import restore_rng, rng_state, seed_all
from testbed.design_objectives import BarberAgakov, BarberAgakovReinforce, posterior_nll
from testbed.designers import PolicyDesigner, RandomDesigner

# ============================================================================
# Training class
# ============================================================================


class Trainer:
    """Training class for BarberAgakov-style training.

    Two phases: a posterior-pretraining phase (fits the posterior net against random
    designs) followed by a policy-training phase, mirroring ActionBED's
    two-phase structure. `cfg.joint` controls whether the posterior keeps
    training (rather than being frozen) during the second phase.
    """

    def __init__(
        self,
        task,
        designer,
        posterior_net,
        objective,
        posterior_loss,
        policy_optimizer,
        posterior_optimizer,
        cfg,
        pretrain_designer,
        policy_scheduler=None,
        posterior_scheduler=None,
        ema_decay=None,
        manager=None,
    ):
        """Initializes the trainer.

        posterior_loss: callable mapping a rollout to the scalar loss minimized
        during posterior pretraining, e.g. `posterior_nll` bound to a posterior
        net.

        ema_decay: if set, an exponential moving average of design_net's
        parameters is tracked (`design_net_ema = ema_decay * design_net_ema +
        (1 - ema_decay) * design_net`, applied every policy step) and used
        for evaluation/checkpointing. This matches ScoreBED's implementation.
        """
        assert isinstance(designer, PolicyDesigner), (
            "This Trainer assumes your designer is a PolicyDesigner object."
        )

        self.task = task
        self.designer = designer
        self.posterior_net = posterior_net
        self.objective = objective
        self.posterior_loss = posterior_loss
        self.policy_optimizer = policy_optimizer
        self.posterior_optimizer = posterior_optimizer
        self.cfg = cfg
        self.pretrain_designer = pretrain_designer
        self.policy_scheduler = policy_scheduler
        self.posterior_scheduler = posterior_scheduler
        self.manager = manager

        self.joint = bool(cfg.joint)

        self.ema_decay = ema_decay
        self.design_net_ema = (
            copy.deepcopy(designer.design_net) if ema_decay is not None else None
        )

        # checkpointing / progress, restored by load_checkpoint
        self.best_metric = -torch.inf
        self.step = -1
        self.elapsed_seconds = 0.0
        self._start_time = time.monotonic()

    def _update_ema(self):
        """Updates design_net_ema with the current design_net parameters."""
        with torch.no_grad():
            for p_ema, p in zip(
                self.design_net_ema.parameters(), self.designer.design_net.parameters()
            ):
                p_ema.mul_(self.ema_decay).add_(p, alpha=1 - self.ema_decay)

    def train(self):
        """Full training loop: posterior pretraining, then policy training."""
        self.designer.design_net.train()
        self.posterior_net.train()

        budget = self.cfg.train_time_budget_seconds
        self._budget = inf if budget is None else budget
        self._start_time = time.monotonic() - self.elapsed_seconds
        self.stop_reason = "iteration_limit"

        # Pretraining has no gain to select a best checkpoint on, and no eval.
        for step in range(self.step + 1, self.cfg.n_posterior_pretrain_iter):
            metrics = self.posterior_pretrain_step()
            metrics["step"] = step

            if self.manager and self.manager.metric_logger:
                self.manager.metric_logger.log(metrics)

            if self._checkpoint_or_stop(step):
                return self._finish()

        # requires_grad and .training are not part of any state dict, so a resume
        # that lands in the policy phase has to re-freeze here.
        if not self.joint:
            self._freeze_posterior()

        for step in range(
            self.step + 1,
            self.cfg.n_posterior_pretrain_iter + self.cfg.n_train_iter,
        ):
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
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "posterior_optimizer": self.posterior_optimizer.state_dict(),
                "policy_scheduler": self.policy_scheduler
                and self.policy_scheduler.state_dict(),
                "posterior_scheduler": self.posterior_scheduler
                and self.posterior_scheduler.state_dict(),
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
        """Logs how far training got, as metrics rather than only to the event log.

        steps_completed spans both phases, matching the step counter, so the
        policy steps are broken out separately to stay comparable with methods
        that have no pretraining phase.
        """
        if self.step < 0:
            return
        completed = self.step + 1
        self.manager.metric_logger.log(
            {
                "step": self.step,
                "train/steps_completed": completed,
                "train/policy_steps_completed": max(
                    0, completed - self.cfg.n_posterior_pretrain_iter
                ),
                "train/elapsed_seconds": self.elapsed_seconds,
                "train/stop_reason": self.stop_reason,
            }
        )

    def _policy_state(self, step):
        state = {
            "design_net": self.designer.design_net.state_dict(),
            "posterior_net": self.posterior_net.state_dict(),
            "use_raw_history": self.designer.use_raw_history,
            "step": step,
            "elapsed_seconds": time.monotonic() - self._start_time,
        }
        # eval_policy prefers the EMA weights, so every checkpoint needs them.
        if self.design_net_ema is not None:
            state["design_net_ema"] = self.design_net_ema.state_dict()
        return state

    def posterior_pretrain_step(self):
        """Takes a single posterior-pretraining step against selected designs - typically random designs."""
        cfg = self.cfg

        theta = self.task.sample_prior(cfg.n_posterior_outer)
        # The pretrain designer has no trainable params, so no_grad on the
        # rollout just avoids building an unnecessary graph through it. The
        # resulting theta/y_history/design_history are plain tensors either
        # way; the posterior NLL below still builds a fresh graph from the
        # posterior net's own parameters.
        with torch.no_grad():
            rollout = self.pretrain_designer.rollout(theta)

        loss = self.posterior_loss(rollout)

        self.posterior_optimizer.zero_grad()
        loss.backward()

        if cfg.posterior_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.posterior_net.parameters(), cfg.posterior_grad_clip
            )

        self.posterior_optimizer.step()
        if self.posterior_scheduler is not None:
            self.posterior_scheduler.step()

        return {"posterior_pretrain/loss": loss.item()}

    def _freeze_posterior(self):
        """Freezes the posterior net after pretraining (non-joint mode)."""
        self.posterior_net.eval()
        for p in self.posterior_net.parameters():
            p.requires_grad_(False)

    def train_step(self):
        """Takes a single policy-training step (and, in joint mode, a posterior step)."""
        cfg = self.cfg

        theta0 = self.task.sample_prior(cfg.n_outer)
        rollout = self.designer.rollout(theta0)

        # In joint mode, BarberAgakovReinforce.estimate_gain(train_posterior=True)
        # self-contains the posterior's gradient. Plain BarberAgakov already updates both
        # nets in a single pass when the policy is pathwise-differentiable
        # so it needs no such flag.
        if self.joint and isinstance(self.objective, BarberAgakovReinforce):
            gain = self.objective.estimate_gain(rollout, train_posterior=True)
        else:
            gain = self.objective.estimate_gain(rollout)
        loss = -gain

        self.policy_optimizer.zero_grad()
        if self.joint:
            self.posterior_optimizer.zero_grad()

        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.designer.design_net.parameters(), cfg.grad_clip
            )
        self.policy_optimizer.step()
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()
        if self.ema_decay is not None:
            self._update_ema()

        if self.joint:
            if cfg.posterior_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.posterior_net.parameters(), cfg.posterior_grad_clip
                )
            self.posterior_optimizer.step()
            if self.posterior_scheduler is not None:
                self.posterior_scheduler.step()

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

            torch.save({**self._policy_state(step), "metric": metric_val}, chkpt_path)

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


def build_posterior_net(cfg, task):
    """Instantiates the posterior (variational-posterior) network.

    Theta canonicalization lives on the net, so both the bound and the
    pretraining NLL see one convention; set it under cfg.posterior_net.
    """
    return instantiate(
        cfg.posterior_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_theta=task.event_shape_theta,
    )


def build_objective(cfg, task, posterior_net):
    """Instantiates the objective for training BarberAgakov.

    BarberAgakov/BarberAgakovReinforce need the task and posterior_net to construct, so it
    can't be a pure instantiate.
    """
    objectives = {
        "barber_agakov": lambda: BarberAgakov(
            task,
            posterior_net,
            use_current_prior=cfg.objective.use_current_prior,
        ),
        "barber_agakov_reinforce": lambda: BarberAgakovReinforce(
            task,
            posterior_net,
            stochastic_design_policy=cfg.objective.requires_policy_log_probs,
            baseline=cfg.objective.baseline,
            use_current_prior=cfg.objective.use_current_prior,
            fused=cfg.objective.fused,
        ),
    }
    name = cfg.objective.name
    return objectives[name]()


def schedule_horizon(cfg):
    """Number of steps the LR schedules span.

    Defaults to the whole run. Set cfg.lr_schedule_iters when n_train_iter is a
    deliberately non-binding cap and train_time_budget_seconds ends the run:
    the schedules then span the steps the run is expected to reach rather than
    the cap. Keep it above that expectation, since a cosine turns back up past
    its T_max.
    """
    return cfg.get("lr_schedule_iters") or (
        cfg.n_posterior_pretrain_iter + cfg.n_train_iter
    )


def build_policy_optimizer(cfg, design_net):
    """Builds the policy Adam optimizer and its (optional) cosine-decay scheduler.

    cfg.betas: Adam (beta1, beta2); torch's (0.9, 0.999) default if null.
    cfg.policy_cosine_alpha: if set, LR cosine-decays over schedule_horizon(cfg)
    from cfg.lr down to cfg.lr * cfg.policy_cosine_alpha; no scheduler
    (constant LR) if null.
    """
    betas = tuple(cfg.betas) if cfg.get("betas") else (0.9, 0.999)
    optimizer = torch.optim.Adam(design_net.parameters(), lr=cfg.lr, betas=betas)

    scheduler = None
    if cfg.get("policy_cosine_alpha") is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=schedule_horizon(cfg),
            eta_min=cfg.lr * cfg.policy_cosine_alpha,
        )
    return optimizer, scheduler


def build_posterior_optimizer(cfg, posterior_net):
    """Builds the posterior Adam optimizer and its (optional) warmup-cosine scheduler.

    cfg.posterior_lr is the peak LR. cfg.posterior_warmup_frac (fraction of
    schedule_horizon(cfg)) is the linear-warmup length; no scheduler (constant
    LR) if null/0. After warmup, LR cosine-decays to cfg.posterior_lr_floor
    over the rest of the horizon.
    """
    betas = tuple(cfg.posterior_betas) if cfg.get("posterior_betas") else (0.9, 0.999)
    optimizer = torch.optim.Adam(
        posterior_net.parameters(), lr=cfg.posterior_lr, betas=betas
    )

    warmup_frac = cfg.get("posterior_warmup_frac") or 0.0
    if not warmup_frac:
        return optimizer, None

    total_iters = schedule_horizon(cfg)
    warmup_iters = max(1, round(warmup_frac * total_iters))
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min(1.0, t / warmup_iters)
    )
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters - warmup_iters, eta_min=cfg.posterior_lr_floor
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_iters]
    )
    return optimizer, scheduler


def build_pretrain_designer(cfg, task):
    """Instantiates the RandomDesigner used to pretrain the posterior net."""
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


def build_trainer(cfg, task, manager=None):
    """Seeds, then builds a fresh trainer with its nets, optimizers and objective."""
    seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    design_net = build_design_net(cfg, task)
    design_net = design_net.to(cfg.device)
    posterior_net = build_posterior_net(cfg, task)
    posterior_net = posterior_net.to(cfg.device)

    designer = PolicyDesigner(
        task,
        cfg.T,
        design_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
        # PolicyDesigner now defaults this to False for every net. Reproduce the
        # old per-net default before PR #37: raw-history conditioning for
        # deterministic nets, constrained history for stochastic ones.
        use_raw_history=not design_net.produces_policy_log_probs,
    )

    # build_objective below reads the flag, so this has to come first.
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs

    policy_optimizer, policy_scheduler = build_policy_optimizer(cfg, design_net)
    posterior_optimizer, posterior_scheduler = build_posterior_optimizer(
        cfg, posterior_net
    )

    return Trainer(
        task=task,
        designer=designer,
        posterior_net=posterior_net,
        objective=build_objective(cfg, task, posterior_net),
        posterior_loss=partial(posterior_nll, posterior_net),
        policy_optimizer=policy_optimizer,
        posterior_optimizer=posterior_optimizer,
        cfg=cfg,
        pretrain_designer=build_pretrain_designer(cfg, task),
        policy_scheduler=policy_scheduler,
        posterior_scheduler=posterior_scheduler,
        ema_decay=cfg.ema_decay,
        manager=manager,
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

    Both LR schedules are defined over the original total run length
    (``n_posterior_pretrain_iter + n_train_iter``) and their horizons are restored
    from the checkpoint, so an ``n_train_iter`` override does not stretch them:
    past the saved cosine ``T_max`` the learning rate turns back up. Extend a run
    with ``train_time_budget_seconds``, not ``n_train_iter``.
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
    trainer.posterior_net.load_state_dict(state["posterior_net"])
    if trainer.design_net_ema is not None:
        trainer.design_net_ema.load_state_dict(state["design_net_ema"])
    trainer.policy_optimizer.load_state_dict(state["policy_optimizer"])
    trainer.posterior_optimizer.load_state_dict(state["posterior_optimizer"])
    # Optimizers first: they carry the current lr that the schedulers step from.
    for name in ("policy_scheduler", "posterior_scheduler"):
        scheduler = getattr(trainer, name)
        if scheduler is not None:
            scheduler.load_state_dict(state[name])
    for name in ("step", "best_metric", "elapsed_seconds"):
        setattr(trainer, name, state[name])
    # Construction consumes randomness, so restore generators last.
    restore_rng(state["rng_state"])
    return trainer


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/var_post")
def main(cfg):
    """Entry point for BarberAgakov training.

    Args:
        cfg: Composed Hydra configuration for the training run.

    Run with ``python testbed/methods/var_post.py [overrides...]``.
    """
    start_time = time.monotonic()
    manager, task = setup(cfg)
    trainer = build_trainer(cfg, task, manager)
    # Setup counts toward the time budget.
    trainer.elapsed_seconds = time.monotonic() - start_time
    trainer.train()


if __name__ == "__main__":
    main()
