"""Tests for the DAD training components."""

import json
from types import SimpleNamespace

import pytest
import torch

from testbed.core.manager import Manager
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SNMC, SPCE
from testbed.designers import PolicyDesigner
from testbed.methods.dad import Trainer, build_objective
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_design_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_trainer(task=None, cfg=None, manager=None):
    task = task or _make_task()
    cfg = cfg or SimpleNamespace(n_outer=4, grad_clip=None)
    policy_net = _make_design_net(task)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    objective = SNMC(task, n_contrastive=4)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    return Trainer(task, designer, objective, optimizer, cfg, manager), policy_net


def test_trainer_requires_policy_designer():
    task = _make_task()
    cfg = SimpleNamespace(n_outer=4, grad_clip=None)
    objective = SNMC(task, n_contrastive=4)
    optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1e-3)

    with pytest.raises(AssertionError):
        Trainer(task, object(), objective, optimizer, cfg)


def test_train_step_returns_gain_key():
    trainer, _ = _make_trainer()
    metrics = trainer.train_step()
    assert "gain" in metrics
    assert isinstance(metrics["gain"], float)


def test_train_step_updates_parameters():
    trainer, policy_net = _make_trainer()
    params_before = [p.clone() for p in policy_net.parameters()]
    trainer.train_step()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, policy_net.parameters())
    )


def test_train_step_with_grad_clip():
    cfg = SimpleNamespace(n_outer=4, grad_clip=1.0)
    trainer, _ = _make_trainer(cfg=cfg)
    metrics = trainer.train_step()
    assert "gain" in metrics


def test_evaluate_returns_placeholder_metrics():
    trainer, _ = _make_trainer()
    metrics = trainer.evaluate()
    assert metrics == {"eval/spce": 0.0, "eval/snmc": 0.0}


def test_build_objective_creates_snmc():
    task = _make_task()
    cfg = SimpleNamespace(objective=SimpleNamespace(name="snmc", n_inner=4))
    obj = build_objective(cfg, task)
    assert isinstance(obj, SNMC)


def test_build_objective_creates_spce():
    task = _make_task()
    cfg = SimpleNamespace(objective=SimpleNamespace(name="spce", n_inner=4))
    obj = build_objective(cfg, task)
    assert isinstance(obj, SPCE)


def test_build_objective_rejects_unknown_name():
    task = _make_task()
    cfg = SimpleNamespace(objective=SimpleNamespace(name="unknown", n_inner=4))
    with pytest.raises(KeyError):
        build_objective(cfg, task)


def test_maybe_save_best_saves_checkpoint_on_improvement(tmp_path):
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="test",
        timestamp=False,
        console=False,
    )
    trainer, _ = _make_trainer(manager=manager)

    trainer._maybe_save_best(step=0, metrics={"step": 0, "gain": 5.0})

    best_path = tmp_path / "test" / "best.pt"
    assert best_path.exists()
    checkpoint = torch.load(best_path, weights_only=True)
    assert set(checkpoint.keys()) >= {"model", "step", "metric"}
    assert checkpoint["metric"] == pytest.approx(5.0)
    assert checkpoint["step"] == 0

    # Worse metric (lower gain) should not overwrite the checkpoint
    trainer._maybe_save_best(step=1, metrics={"step": 1, "gain": 3.0})
    checkpoint_after = torch.load(best_path, weights_only=True)
    assert checkpoint_after["metric"] == pytest.approx(5.0)


def test_train_runs_with_periodic_evaluation(tmp_path):
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="train_test",
        timestamp=False,
        console=False,
    )
    cfg = SimpleNamespace(n_outer=4, grad_clip=None, n_train_iter=3, eval_every=1)
    trainer, _ = _make_trainer(cfg=cfg, manager=manager)

    trainer.train()

    run_dir = tmp_path / "train_test"
    checkpoint = torch.load(run_dir / "final_step.pt", weights_only=True)
    assert checkpoint["step"] == cfg.n_train_iter - 1
    assert "model" in checkpoint

    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("gain" in record for record in records)
    assert any("eval/spce" in record for record in records)
