"""Tests for the static-design training components, and its random restarts."""

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from testbed.core.manager import Manager
from testbed.design_objectives import SNMC, SPCE, SPCEReinforce
from testbed.designers import StaticDesigner
from testbed.eval.eval_static import (
    evaluate_static_checkpoint,
    load_static_designer,
)
from testbed.methods.static import Trainer, build_objective
from testbed.tasks import LocationFinding

T = 3


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_cfg(**overrides):
    select = SimpleNamespace(
        n_outer=4,
        n_contrastive=4,
        outer_batch_size=4,
        contrastive_chunk_size=4,
        fused=False,
    )
    cfg = {
        "n_outer": 4,
        "grad_clip": None,
        "n_train_iter": 2,
        "save_best_after": 0,
        "select": select,
    }
    return SimpleNamespace(**{**cfg, **overrides})


def _make_trainer(task=None, cfg=None, manager=None, n_restarts=3):
    task = task or _make_task()
    cfg = cfg or _make_cfg()
    designers = [StaticDesigner(task, T) for _ in range(n_restarts)]
    objective = SPCE(task, n_contrastive=4)
    optimizer = torch.optim.Adam(
        [p for designer in designers for p in designer.parameters()], lr=1e-2
    )
    return Trainer(task, designers, objective, optimizer, cfg, manager), designers


def test_trainer_requires_static_designers():
    task = _make_task()
    objective = SNMC(task, n_contrastive=4)
    optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)

    with pytest.raises(AssertionError):
        Trainer(task, [object()], objective, optimizer, _make_cfg())


def test_restart_gains_are_per_restart():
    trainer, designers = _make_trainer(n_restarts=3)
    gains = trainer.restart_gains(4)
    assert gains.shape == (len(designers),)


@pytest.mark.parametrize("reinforce", [False, True])
def test_restarts_do_not_share_gradients(reinforce):
    # The whole point of restarts: each design must be optimized as if it were
    # alone, so one restart's gain may not depend on another's parameters.
    # The leave-one-out baseline pools restarts but is detached, so still holds.
    trainer, designers = _make_trainer(n_restarts=3)
    if reinforce:
        trainer.task = LocationFinding(
            n_sources=1, dim_theta=2, reparameterize_outcomes=False
        )
        trainer.objective = SPCEReinforce(
            trainer.task, 4, stochastic_design_policy=False, baseline="leave_one_out"
        )
    gains = trainer.restart_gains(4)

    grads = torch.autograd.grad(
        gains[1], [d.raw_design for d in designers], allow_unused=True
    )
    assert grads[1] is not None and grads[1].abs().sum() > 0
    for other in (0, 2):
        assert grads[other] is None or torch.equal(
            grads[other], torch.zeros_like(grads[other])
        )


def test_train_step_updates_every_restart():
    trainer, designers = _make_trainer()
    before = [d.raw_design.clone() for d in designers]
    trainer.train_step()

    assert all(not torch.equal(old, d.raw_design) for old, d in zip(before, designers))


def test_train_step_returns_best_restart():
    trainer, designers = _make_trainer()
    metrics = trainer.train_step()

    per_restart = [metrics[f"gain/restart_{i}"] for i in range(len(designers))]
    assert all(isinstance(g, float) for g in per_restart)
    assert metrics["best_restart"] == per_restart.index(max(per_restart))
    assert metrics["gain"] == max(per_restart)


def test_train_step_with_grad_clip():
    trainer, _ = _make_trainer(cfg=_make_cfg(grad_clip=1.0))
    assert "gain" in trainer.train_step()


def test_select_best_returns_argmax_of_reported_gains():
    trainer, designers = _make_trainer()
    best, gains = trainer.select_best()

    assert len(gains) == len(designers)
    assert gains[best] == max(gains)


def test_build_objective_creates_spce_and_snmc():
    task = _make_task()
    for name, expected in [("spce", SPCE), ("snmc", SNMC)]:
        cfg = SimpleNamespace(
            objective=SimpleNamespace(
                name=name, n_inner=4, chunk_size=None, fused=False
            )
        )
        assert isinstance(build_objective(cfg, task), expected)


def test_build_objective_creates_deterministic_reinforce():
    task = LocationFinding(n_sources=1, dim_theta=2, reparameterize_outcomes=False)
    cfg = SimpleNamespace(
        objective=SimpleNamespace(
            name="reinforce",
            n_inner=4,
            chunk_size=None,
            fused=False,
            baseline="leave_one_out",
        )
    )
    objective = build_objective(cfg, task)
    assert isinstance(objective, SPCEReinforce)
    assert not objective.stochastic_design_policy


def test_build_objective_rejects_unknown():
    cfg = SimpleNamespace(
        objective=SimpleNamespace(name="nmc", n_inner=4, chunk_size=None, fused=False)
    )
    with pytest.raises(ValueError):
        build_objective(cfg, _make_task())


def test_checkpoint_roundtrips_through_load_static_designer():
    trainer, designers = _make_trainer()
    state = trainer._checkpoint(best=1, step=0, gain=0.0)

    assert state["raw_designs"].shape == (len(designers), T, 2)
    reloaded = load_static_designer(_make_task(), state, T)
    assert torch.equal(reloaded.raw_design, designers[1].raw_design)


def test_evaluate_static_checkpoint_scores_saved_run(tmp_path):
    trainer, _ = _make_trainer()
    torch.save(trainer._checkpoint(best=2, step=0, gain=0.0), tmp_path / "final.pt")
    task_cfg = {
        "_target_": "testbed.tasks.LocationFinding",
        "n_sources": 1,
        "dim_theta": 2,
    }
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save({"T": T, "task": task_cfg}, tmp_path / ".hydra" / "config.yaml")

    out = evaluate_static_checkpoint(
        str(tmp_path),
        checkpoint="final.pt",
        n_outer=8,
        n_contrastive=16,
        outer_batch_size=4,
        contrastive_chunk_size=8,
        fused=False,
    )

    assert out["best_restart"] == 2
    assert out["lower_bound"] <= out["upper_bound"]


def test_maybe_save_best_saves_checkpoint_on_improvement(tmp_path):
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="test",
        timestamp=False,
        console=False,
    )
    trainer, _ = _make_trainer(manager=manager)

    trainer._maybe_save_best(step=0, metrics={"gain": 5.0, "best_restart": 0})

    best_path = tmp_path / "test" / "best.pt"
    checkpoint = torch.load(best_path, weights_only=True)
    assert set(checkpoint.keys()) >= {"raw_design", "raw_designs", "best_restart"}
    assert checkpoint["gain"] == pytest.approx(5.0)

    # Worse metric (lower gain) should not overwrite the checkpoint
    trainer._maybe_save_best(step=1, metrics={"gain": 3.0, "best_restart": 1})
    assert torch.load(best_path, weights_only=True)["gain"] == pytest.approx(5.0)


def test_train_writes_final_checkpoint_and_selection_metrics(tmp_path):
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="train_test",
        timestamp=False,
        console=False,
    )
    trainer, designers = _make_trainer(manager=manager)

    trainer.train()

    run_dir = tmp_path / "train_test"
    checkpoint = torch.load(run_dir / "final_step.pt", weights_only=True)
    assert checkpoint["step"] == trainer.cfg.n_train_iter - 1
    assert len(checkpoint["gain_per_restart"]) == len(designers)
    assert checkpoint["raw_design"].shape == (T, 2)
    assert checkpoint["designs"].shape == (len(designers), T, 2)

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("gain" in record for record in records)
    assert any("select/restart" in record for record in records)
