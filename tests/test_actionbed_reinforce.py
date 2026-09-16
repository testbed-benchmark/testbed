"""Tests for ActionBED's REINFORCE objective, checkpointing, and stop conditions.

The HTD outcomes are discrete, so the plain loss only reaches the design net
through the action net's input and misses the trajectory score term. These
cover the wiring that puts that term back, that a run using it still resumes
exactly, and that a diverged run stops instead of spending its budget.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from testbed.core.manager import Manager
from testbed.design_objectives import LossReinforce
from testbed.methods.actionbed import build_trainer, load_checkpoint

RECIPES_DIR = str(Path(__file__).resolve().parents[1] / "testbed" / "recipes")
HTD_RECIPE = "experiments/benchmarking/temporal_discounting/actionbed_logmse"


def _cfg(**overrides):
    """Composes the HTD recipe, shrunk to a few cheap steps on the CPU."""
    base = {
        "device": "cpu",
        "n_outer": 32,
        "n_outer_eval": 32,
        "n_action_warmup_iter": 2,
        "n_train_iter": 3,
        "eval_every": 0,
        "checkpoint_every": 0,
        "train_time_budget_seconds": "null",
        "action_net.dims": "[60,16,2]",
        "design_net.dim_emb": 8,
        "design_net.num_heads": 1,
        "design_net.num_blocks": 1,
        "design_net.encoder_dim_h": "[8]",
    }
    base.update(overrides)
    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        return compose(
            config_name=HTD_RECIPE,
            overrides=[f"{k}={v}" for k, v in base.items()],
        )


def _design_net_grad(trainer):
    trainer.train_step()
    return torch.cat(
        [
            p.grad.flatten()
            for p in trainer.designer.design_net.parameters()
            if p.grad is not None
        ]
    )


def test_htd_recipe_selects_the_reinforce_objective():
    trainer = build_trainer(_cfg(), instantiate(_cfg().task))

    assert isinstance(trainer.objective, LossReinforce)
    assert trainer.objective.baseline == "leave_one_out"
    # A deterministic design policy takes the pathwise branch (DAD Eq. 16).
    assert trainer.objective.stochastic_design_policy is False


def test_plain_loss_objective_is_none():
    cfg = _cfg(**{"objective.name": "loss"})
    assert build_trainer(cfg, instantiate(cfg.task)).objective is None


def test_reinforce_changes_the_design_net_gradient():
    """The score term is what the plain loss misses on a discrete likelihood."""
    reinforce = _design_net_grad(build_trainer(_cfg(), instantiate(_cfg().task)))

    cfg = _cfg(**{"objective.name": "loss"})
    plain = _design_net_grad(build_trainer(cfg, instantiate(cfg.task)))

    assert reinforce.norm() > 0
    assert not torch.allclose(reinforce, plain)


def test_unknown_objective_is_rejected():
    cfg = _cfg(**{"objective.name": "spce"})
    with pytest.raises(ValueError, match="Unknown objective: spce"):
        build_trainer(cfg, instantiate(cfg.task))


def _run(cfg, tmp_path, name):
    manager = Manager(
        cfg, root=str(tmp_path), name=name, timestamp=False, console=False
    )
    trainer = build_trainer(cfg, instantiate(cfg.task), manager)
    trainer.train()
    return trainer, Path(manager.path("final_step.pt"))


def _nan_after(trainer, n):
    """Makes the loss non-finite from call ``n`` onwards."""
    real, calls = trainer.loss_fn, []

    def loss_fn(action, theta):
        calls.append(1)
        loss = real(action, theta)
        return loss * float("nan") if len(calls) > n else loss

    trainer.loss_fn = loss_fn


def test_nan_during_joint_training_stops_the_run(tmp_path):
    cfg = _cfg(n_action_warmup_iter=0, n_train_iter=200, **{"objective.name": "loss"})
    manager = Manager(
        cfg, root=str(tmp_path), name="nan", timestamp=False, console=False
    )
    trainer = build_trainer(cfg, instantiate(cfg.task), manager)
    _nan_after(trainer, 2)

    trainer.train()

    assert trainer.stop_reason == "nan_loss"
    assert trainer.step == 2, "should stop on the first non-finite loss, not run on"
    state = torch.load(manager.path("final_step.pt"), weights_only=True)
    assert state["stop_reason"] == "nan_loss"


def test_nan_during_warmup_stops_the_run(tmp_path):
    cfg = _cfg(n_action_warmup_iter=200, n_train_iter=200, **{"objective.name": "loss"})
    manager = Manager(
        cfg, root=str(tmp_path), name="nanwarm", timestamp=False, console=False
    )
    trainer = build_trainer(cfg, instantiate(cfg.task), manager)
    _nan_after(trainer, 1)

    trainer.train()

    assert trainer.stop_reason == "nan_loss"
    assert trainer.step == 1


def test_finite_run_still_stops_on_the_iteration_limit(tmp_path):
    cfg = _cfg()
    manager = Manager(
        cfg, root=str(tmp_path), name="ok", timestamp=False, console=False
    )
    trainer = build_trainer(cfg, instantiate(cfg.task), manager)

    trainer.train()

    assert trainer.stop_reason == "iteration_limit"


def test_location_finding_truelogmse_recipe_stops_immediately(tmp_path):
    """The known-NaN recipe must not spend its wall-clock budget.

    Its Gaussian source prior puts ~50% of theta below zero, so the loss's
    log(theta + eps) is NaN from the first step.
    """
    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        cfg = compose(
            config_name="experiments/benchmarking/location_finding_2src/actionbed_truelogmse",
            overrides=[
                "device=cpu",
                "n_outer=32",
                "n_action_warmup_iter=5",
                "n_train_iter=5",
                "eval_every=0",
                "checkpoint_every=0",
                "train_time_budget_seconds=null",
            ],
        )
    manager = Manager(
        cfg, root=str(tmp_path), name="lf", timestamp=False, console=False
    )
    trainer = build_trainer(cfg, instantiate(cfg.task), manager)

    trainer.train()

    assert trainer.stop_reason == "nan_loss"
    assert trainer.step == 0


def test_reinforce_run_resumes_to_the_same_weights(tmp_path):
    """Training straight through and resuming mid-way must agree exactly.

    build_trainer rebuilds the objective from the saved config, so a resumed
    REINFORCE run has to pick it back up rather than silently falling back to
    the plain loss.
    """
    _, path = _run(_cfg(n_train_iter=3), tmp_path, "partial")
    resumed = load_checkpoint(
        path,
        n_train_iter=6,
        manager=Manager(
            _cfg(), root=str(tmp_path), name="resumed", timestamp=False, console=False
        ),
    )
    assert isinstance(resumed.objective, LossReinforce)
    resumed.train()

    straight, _ = _run(_cfg(n_train_iter=6), tmp_path, "straight")

    assert resumed.step == straight.step
    for resumed_p, straight_p in zip(
        resumed.designer.design_net.parameters(),
        straight.designer.design_net.parameters(),
    ):
        torch.testing.assert_close(resumed_p, straight_p, rtol=0, atol=0)
    for resumed_p, straight_p in zip(
        resumed.action_net.parameters(), straight.action_net.parameters()
    ):
        torch.testing.assert_close(resumed_p, straight_p, rtol=0, atol=0)
