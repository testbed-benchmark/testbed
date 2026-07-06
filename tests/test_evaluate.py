"""Tests for the standalone checkpoint evaluation script."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from testbed.methods.evaluate import (
    _batched_gain,
    evaluate_bounds,
    infer_method,
    load_run,
    main,
)

TASK_CFG = {"_target_": "testbed.tasks.LocationFinding", "n_sources": 1, "dim_theta": 2}
DESIGN_NET_CFG = {
    "_target_": "testbed.design_nets.DeepSetPolicyNet",
    "dim_emb": 8,
    "encoder_dim_h": 8,
}


def _eval_cfg(run_dir, tmp_path, **overrides):
    cfg = {
        "run_dir": str(run_dir),
        "n_inner": 10,
        "n_rollouts": 8,
        "n_runs": 2,
        "seed": 0,
        "batch_size": 3,
        "device": "cpu",
        "results_dir": str(tmp_path / "results"),
        "n_speed_rollouts": 2,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _make_dad_run(tmp_path):
    from testbed.methods.dad import build_design_net

    run_dir = tmp_path / "dad_run"
    run_dir.mkdir()
    cfg = OmegaConf.create({"task": TASK_CFG, "T": 2, "design_net": DESIGN_NET_CFG})
    OmegaConf.save(cfg, run_dir / "config.yaml")

    from hydra.utils import instantiate

    task = instantiate(cfg.task)
    policy_net = build_design_net(cfg, task)
    torch.save({"model": policy_net.state_dict(), "step": 5}, run_dir / "best.pt")
    return run_dir, policy_net


def _make_actionbed_run(tmp_path):
    from hydra.utils import instantiate

    from testbed.methods.actionbed import build_policy_nets

    run_dir = tmp_path / "actionbed_run"
    run_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "task": TASK_CFG,
            "T": 2,
            "design_net": DESIGN_NET_CFG,
            "action_net": DESIGN_NET_CFG,
            "loss": {"_target_": "torch.nn.MSELoss"},
        }
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")

    task = instantiate(cfg.task)
    shim = SimpleNamespace(event_logger=SimpleNamespace(warning=lambda *a, **k: None))
    design_net, action_net = build_policy_nets(cfg, task, shim)
    torch.save(
        {
            "design_net": design_net.state_dict(),
            "action_net": action_net.state_dict(),
            "step": 7,
        },
        run_dir / "best.pt",
    )
    return run_dir


def _make_rlboed_run(tmp_path):
    from hydra.utils import instantiate

    from testbed.methods.rlboed import build_design_space, make_networks

    run_dir = tmp_path / "rlboed_run"
    run_dir.mkdir()
    cfg = OmegaConf.create(
        {
            "task": TASK_CFG,
            "T": 2,
            "design_space": {"low": -2.0, "high": 3.0},
            "reward": {"n_contrastive": 2, "outcome_low": -3.0, "outcome_high": 10.0},
            "sac": {
                "policy_lr": 1e-4,
                "q_lr": 3e-4,
                "policy_init_std": 3**-0.5,
                "alpha": None,
                "initial_alpha": 1.0,
                "target_entropy": None,
            },
        }
    )
    task = instantiate(cfg.task)
    design_space = build_design_space(cfg, task)
    actor = make_networks(task, design_space, cfg, torch.device("cpu")).actor
    torch.save(
        {
            "config": OmegaConf.to_container(cfg),
            "designer": actor.state_dict(),
            "actor": actor.state_dict(),
            "step": 3,
        },
        run_dir / "best.pt",
    )
    return run_dir


def test_infer_method():
    assert infer_method({"model": {}}) == "dad"
    assert infer_method({"design_net": {}, "action_net": {}}) == "actionbed"
    assert infer_method({"config": {}, "designer": {}}) == "rlboed"
    # rlboed checkpoints also carry model-like keys; "config" wins
    assert infer_method({"config": {}, "model": {}}) == "rlboed"
    with pytest.raises(ValueError):
        infer_method({"weights": {}})


def test_load_run_dad_round_trip(tmp_path):
    run_dir, source_net = _make_dad_run(tmp_path)
    loaded = load_run(run_dir, device="cpu")

    assert loaded.method == "dad"
    assert loaded.designer.deterministic
    for source, restored in zip(
        source_net.parameters(), loaded.designer.policy_net.parameters()
    ):
        assert torch.equal(source, restored)


@pytest.mark.parametrize(
    "make_run, method",
    [
        (_make_dad_run, "dad"),
        (_make_actionbed_run, "actionbed"),
        (_make_rlboed_run, "rlboed"),
    ],
)
def test_main_end_to_end(tmp_path, make_run, method):
    made = make_run(tmp_path)
    run_dir = made[0] if isinstance(made, tuple) else made
    payload = main(_eval_cfg(run_dir, tmp_path))

    assert payload["method"] == method
    assert payload["task"] == "location_finding"
    assert payload["eval"]["seeds"] == [0, 1]
    for stats in (payload["metrics"]["spce"], payload["metrics"]["snmc"]):
        assert math.isfinite(stats["mean"]) and math.isfinite(stats["se"])
    assert math.isfinite(payload["metrics"]["proposal_time_ms"]["median"])
    assert ("decision_loss" in payload["metrics"]) == (method == "actionbed")

    assert (run_dir / "eval.json").exists()
    results_path = tmp_path / "results" / "location_finding" / f"{method}.json"
    written = json.loads(results_path.read_text())
    assert written["metrics"] == payload["metrics"]


def test_batched_gain_matches_unbatched():
    # A deterministic objective isolates the weighted-mean batching math from
    # the RNG consumed by contrastive draws.
    class MeanTheta:
        def estimate_gain(self, y_obs, design_obs, theta0):
            return theta0.mean()

    theta0 = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    y_obs = torch.zeros(7, 2, 1)
    design_obs = torch.zeros(7, 2, 1)

    full = _batched_gain(MeanTheta(), y_obs, design_obs, theta0, batch_size=100)
    batched = _batched_gain(MeanTheta(), y_obs, design_obs, theta0, batch_size=3)
    assert full == pytest.approx(theta0.mean().item())
    assert batched == pytest.approx(full)


def test_evaluate_bounds_records(tmp_path):
    run_dir, _ = _make_dad_run(tmp_path)
    loaded = load_run(run_dir, device="cpu")
    records = evaluate_bounds(loaded, _eval_cfg(run_dir, tmp_path, n_inner=32))

    assert len(records) == 2
    for record in records:
        # per-sample sPCE is deterministically capped at log(L + 1)
        assert record["spce"] <= math.log(32 + 1) + 1e-5
        assert math.isfinite(record["snmc"])
