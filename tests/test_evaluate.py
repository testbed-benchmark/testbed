"""Tests for the standalone checkpoint evaluation script."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from testbed.eval.evaluate import (
    CONFIG_DIR,
    evaluate,
    evaluate_bounds,
    infer_method,
    load_run,
)

TASK_CFG = {"_target_": "testbed.tasks.LocationFinding", "n_sources": 1, "dim_theta": 2}
DESIGN_NET_CFG = {
    "_target_": "testbed.design_nets.DeepSetPolicyNet",
    "dim_emb": 8,
    "encoder_dim_h": 8,
}


def _make_task_instance():
    from hydra.utils import instantiate

    return instantiate(OmegaConf.create(TASK_CFG))


def _eval_cfg(run_dir, tmp_path, **overrides):
    # Compose from the same config/default.yaml the CLI uses, then set a small
    # test protocol on top. This exercises the real config, not a hand-built dict.
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="default")
    cfg.run_dir = str(run_dir)
    cfg.n_inner = 10
    cfg.n_rollouts = 8
    cfg.batch_size = 3
    cfg.device = "cpu"
    cfg.results_dir = str(tmp_path / "results")
    cfg.n_speed_rollouts = 2
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


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
def test_evaluate_end_to_end(tmp_path, make_run, method):
    made = make_run(tmp_path)
    run_dir = made[0] if isinstance(made, tuple) else made
    payload = evaluate(_eval_cfg(run_dir, tmp_path))

    assert payload["method"] == method
    assert payload["task"] == "location_finding"
    assert payload["eval"]["seed"] == 0
    assert payload["eval"]["n_rollouts"] == 8
    for stats in (payload["metrics"]["spce"], payload["metrics"]["snmc"]):
        assert math.isfinite(stats["mean"]) and math.isfinite(stats["se"])
    assert math.isfinite(payload["metrics"]["proposal_time_ms"]["median"])
    assert ("decision_loss" in payload["metrics"]) == (method == "actionbed")

    assert (run_dir / "eval.json").exists()
    results_path = tmp_path / "results" / "location_finding" / f"{method}.json"
    written = json.loads(results_path.read_text())
    assert written["metrics"] == payload["metrics"]


def test_per_sample_gain_matches_estimate_gain():
    # The refactor must leave the scalar estimate unchanged: mean of the
    # per-sample gains equals estimate_gain under the same contrastive draws.
    from testbed.design_objectives import SNMC, SPCE

    task = _make_task_instance()
    theta0 = task.sample_prior(6)
    y_obs = torch.zeros(6, 2, 1)
    design_obs = torch.zeros(6, 2, 2)

    for objective in (SPCE(task, n_contrastive=8), SNMC(task, n_contrastive=8)):
        torch.manual_seed(0)
        per_sample = objective.per_sample_gain(y_obs, design_obs, theta0)
        torch.manual_seed(0)
        scalar = objective.estimate_gain(y_obs, design_obs, theta0)
        assert per_sample.shape == (6,)
        assert torch.allclose(per_sample.mean(), scalar)


def test_evaluate_bounds_pools_all_rollouts(tmp_path):
    run_dir, _ = _make_dad_run(tmp_path)
    loaded = load_run(run_dir, device="cpu")
    samples = evaluate_bounds(loaded, _eval_cfg(run_dir, tmp_path, n_inner=32))

    # one pooled value per rollout, so mean/SE follow the paper's definition
    assert len(samples["spce"]) == 8
    assert len(samples["snmc"]) == 8
    for spce_value in samples["spce"]:
        # per-sample sPCE is deterministically capped at log(L + 1)
        assert spce_value <= math.log(32 + 1) + 1e-5
