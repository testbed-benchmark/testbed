"""Tests for rebuilding a trained designer from a run directory."""

import pytest
import torch
from omegaconf import OmegaConf

from testbed.core.load import designer_from_run, load_designer, run_config
from testbed.design_nets import DeepSetPolicyNet
from testbed.designers import PolicyDesigner, RandomDesigner, StaticDesigner
from testbed.tasks import LocationFinding

T = 3


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _write_run(tmp_path, cfg, state, hydra_only=True):
    """Writes a finished run directory: its saved config and final checkpoint."""
    if hydra_only:
        (tmp_path / ".hydra").mkdir()
        cfg_path = tmp_path / ".hydra" / "config.yaml"
    else:
        cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)
    torch.save(state, tmp_path / "final_step.pt")
    return tmp_path


def _policy_run(tmp_path, task, *, use_raw_history=True, **kwargs):
    design_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )
    cfg = {
        "T": T,
        "task": {"_target_": "testbed.tasks.LocationFinding", "n_sources": 1},
        "design_net": {"_target_": "testbed.design_nets.DeepSetPolicyNet"},
        "final_eval": {"designer": {"_target_": "testbed.designers.PolicyDesigner"}},
    }
    state = {"design_net": design_net.state_dict(), "use_raw_history": use_raw_history}
    return _write_run(tmp_path, cfg, state, **kwargs), design_net


def test_designer_from_run_rebuilds_a_policy_with_its_own_architecture(tmp_path):
    task = _make_task()
    run_dir, design_net = _policy_run(tmp_path, task)

    designer = designer_from_run(run_dir, task)

    assert isinstance(designer, PolicyDesigner)
    # The run's config, not ours, decides the architecture and the coordinates.
    assert designer.T == T
    assert designer.use_raw_history is True
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            design_net.state_dict().values(), designer.design_net.state_dict().values()
        )
    )


def test_designer_from_run_rebuilds_a_static_design(tmp_path):
    task = _make_task()
    design = torch.randn(T, *task.event_shape_design)
    cfg = {
        "T": T,
        "task": {"_target_": "testbed.tasks.LocationFinding", "n_sources": 1},
        "final_eval": {"designer": {"_target_": "testbed.designers.StaticDesigner"}},
    }
    run_dir = _write_run(
        tmp_path, cfg, {"raw_design": design, "transform_designs": True}
    )

    designer = designer_from_run(run_dir, task)

    assert isinstance(designer, StaticDesigner)
    assert torch.equal(designer.raw_design.detach(), design)


def test_run_config_prefers_what_the_run_actually_built(tmp_path):
    # The Manager's copy records post-startup edits; .hydra's is the fallback.
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(OmegaConf.create({"T": 1}), tmp_path / ".hydra" / "config.yaml")
    assert run_config(tmp_path).T == 1

    OmegaConf.save(OmegaConf.create({"T": 2}), tmp_path / "config.yaml")
    assert run_config(tmp_path).T == 2


def test_run_config_raises_without_a_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_config(tmp_path)


def test_static_design_loads_from_a_tensor_a_path_or_a_static_checkpoint(tmp_path):
    task = _make_task()
    design = torch.randn(T, *task.event_shape_design)

    path = tmp_path / "design.pt"
    torch.save(design, path)
    ckpt = tmp_path / "best.pt"
    torch.save({"raw_design": design, "transform_designs": True, "step": 3}, ckpt)

    for source in (design, str(path), str(ckpt)):
        designer = StaticDesigner(task, T)
        load_designer(designer, source)
        assert torch.equal(designer.raw_design.detach(), design)


def test_static_design_rejects_a_mismatched_shape():
    task = _make_task()
    designer = StaticDesigner(task, T)
    with pytest.raises(ValueError, match="expects"):
        load_designer(designer, torch.randn(T + 1, *task.event_shape_design))


def test_load_designer_rejects_a_designer_with_nothing_to_load():
    task = _make_task()
    with pytest.raises(ValueError, match="nothing to load"):
        load_designer(RandomDesigner(task, T), torch.zeros(1))


def test_designer_from_run_keeps_the_designers_own_settings(tmp_path):
    # A run's final_eval.designer may carry settings of its own (ALINE's
    # candidate pool, say); they must survive the rebuild.
    task = _make_task()
    run_dir, _ = _policy_run(tmp_path, task)
    cfg = run_config(run_dir)
    cfg.final_eval.designer.reparameterize_policy = True
    OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")

    designer = designer_from_run(run_dir, task)

    assert designer.reparameterize_policy is True


def test_designer_from_run_prefers_ema_weights(tmp_path):
    # methods.var_post saves an EMA copy alongside the raw weights, and
    # evaluation scores the EMA; a frozen designer should match it.
    task = _make_task()
    run_dir, design_net = _policy_run(tmp_path, task)
    ema = {k: torch.zeros_like(v) for k, v in design_net.state_dict().items()}
    state = torch.load(run_dir / "final_step.pt", weights_only=True)
    torch.save({**state, "design_net_ema": ema}, run_dir / "final_step.pt")

    designer = designer_from_run(run_dir, task)

    assert all(
        torch.equal(v, torch.zeros_like(v))
        for v in designer.design_net.state_dict().values()
    )
