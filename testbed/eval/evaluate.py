"""Standalone checkpoint evaluation.

Loads a saved training run (DAD, RL-BOED, or ActionBED), rolls out the policy,
and reports sPCE/sNMC bounds, deployment speed, and method-specific metrics.

Usage:
    python testbed/eval/evaluate.py run_dir=<dir> [checkpoint=best.pt]
        [n_inner=5000] [n_rollouts=10000] [n_runs=10] [seed=0] [batch_size=100]
"""

import copy
import json
import logging
import re
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.design_objectives import SNMC, SPCE
from testbed.designers import PolicyDesigner
from testbed.methods.actionbed import build_policy_nets
from testbed.methods.dad import build_design_net
from testbed.methods.rlboed import build_design_space, make_networks

DEFAULTS = OmegaConf.create(
    {
        "run_dir": "???",
        "checkpoint": "best.pt",
        "method": None,  # dad | actionbed | rlboed; None infers from the checkpoint
        "n_inner": 5000,  # contrastive samples L (DAD paper eval protocol)
        "n_rollouts": 10000,  # total rollouts, split over n_runs seeded runs
        "n_runs": 10,
        "seed": 0,
        "batch_size": 100,  # outer mini-batch; bounds memory at batch_size * (L+1)
        "device": None,  # None -> cuda if available else cpu
        "results_dir": "results",  # None -> skip the results/ write
        "n_speed_rollouts": 10,
        "wandb": {
            "enabled": False,
            "project": "testbed",
            "entity": None,
            "mode": "online",
            "tags": None,
            "group": None,
        },
    }
)


# ============================================================================
# Loading a saved run
# ============================================================================


@dataclass
class LoadedRun:
    """A task/designer/config rebuilt from a saved training run."""

    method: str
    task: object
    designer: PolicyDesigner
    cfg: object
    checkpoint: dict
    extras: dict = field(default_factory=dict)


def infer_method(checkpoint: dict) -> str:
    """Infers the training method from checkpoint keys."""
    if "config" in checkpoint:
        return "rlboed"
    if "design_net" in checkpoint and "action_net" in checkpoint:
        return "actionbed"
    if "model" in checkpoint:
        return "dad"
    raise ValueError(f"Unrecognized checkpoint keys: {sorted(checkpoint)}")


def _load_dad(cfg, checkpoint, device):
    task = instantiate(cfg.task).to(device)
    policy_net = build_design_net(cfg, task).to(device)
    policy_net.load_state_dict(checkpoint["model"])
    policy_net.eval()
    designer = PolicyDesigner(task, cfg.T, policy_net, deterministic=True)
    return task, designer, {}


def _load_actionbed(cfg, checkpoint, device):
    task = instantiate(cfg.task).to(device)
    shim = SimpleNamespace(event_logger=logging.getLogger("evaluate"))
    design_net, action_net = build_policy_nets(cfg, task, shim)
    design_net = design_net.to(device)
    action_net = action_net.to(device)
    design_net.load_state_dict(checkpoint["design_net"])
    action_net.load_state_dict(checkpoint["action_net"])
    design_net.eval()
    action_net.eval()
    designer = PolicyDesigner(task, cfg.T, design_net, deterministic=True)
    return task, designer, {"action_net": action_net, "loss_fn": instantiate(cfg.loss)}


def _load_rlboed(cfg, checkpoint, device):
    task = instantiate(cfg.task).to(device)
    design_space = build_design_space(cfg, task)
    actor = make_networks(task, design_space, cfg, torch.device(device)).actor
    actor.load_state_dict(checkpoint["designer"])
    actor.eval()
    designer = PolicyDesigner(task, cfg.T, actor, deterministic=True)
    return task, designer, {}


LOADERS = {"dad": _load_dad, "actionbed": _load_actionbed, "rlboed": _load_rlboed}


def load_run(run_dir, checkpoint_name="best.pt", method=None, device="cpu"):
    """Rebuilds task and deterministic designer from a saved run directory."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / checkpoint_name, weights_only=True)
    method = method or infer_method(checkpoint)

    if method == "rlboed":
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        cfg = OmegaConf.load(run_dir / "config.yaml")

    task, designer, extras = LOADERS[method](cfg, checkpoint, device)
    return LoadedRun(method, task, designer, cfg, checkpoint, extras)


# ============================================================================
# Metrics
# ============================================================================


def _batched_gain(objective, y_obs, design_obs, theta0, batch_size):
    """Mean estimate_gain over outer mini-batches, weighted by batch size."""
    n_outer = theta0.shape[0]
    total = 0.0
    for start in range(0, n_outer, batch_size):
        stop = min(start + batch_size, n_outer)
        gain = objective.estimate_gain(
            y_obs[start:stop], design_obs[start:stop], theta0[start:stop]
        )
        total += gain.item() * (stop - start)
    return total / n_outer


def evaluate_bounds(loaded: LoadedRun, cfg) -> list[dict]:
    """Runs seeded eval rollouts and computes per-run sPCE/sNMC (and loss)."""
    task, designer = loaded.task, loaded.designer
    n_outer = cfg.n_rollouts // cfg.n_runs
    if n_outer < 1:
        raise ValueError("n_rollouts must be at least n_runs")

    spce = SPCE(task, n_contrastive=cfg.n_inner)
    snmc = SNMC(task, n_contrastive=cfg.n_inner)

    records = []
    for run_idx in range(cfg.n_runs):
        torch.manual_seed(cfg.seed + run_idx)
        with torch.no_grad():
            theta0 = task.sample_prior(n_outer)
            y_obs, design_obs = designer.rollout(theta0)
            record = {
                "run": run_idx,
                "seed": cfg.seed + run_idx,
                "spce": _batched_gain(spce, y_obs, design_obs, theta0, cfg.batch_size),
                "snmc": _batched_gain(snmc, y_obs, design_obs, theta0, cfg.batch_size),
            }
            if "action_net" in loaded.extras:
                action = loaded.extras["action_net"](y_obs, design_obs)
                record["decision_loss"] = loaded.extras["loss_fn"](
                    action, theta0
                ).item()
        records.append(record)
    return records


def measure_proposal_time(loaded: LoadedRun, cfg) -> float:
    """Median per-design proposal time in ms on CPU (DAD paper speed test)."""
    task = instantiate(loaded.cfg.task)  # cpu copy
    policy_net = copy.deepcopy(loaded.designer.policy_net).to("cpu").eval()
    designer = PolicyDesigner(task, loaded.cfg.T, policy_net, deterministic=True)

    torch.manual_seed(cfg.seed)
    n_warmup = 2
    step_times = []
    with torch.no_grad():
        for rollout_idx in range(n_warmup + cfg.n_speed_rollouts):
            theta = task.sample_prior(1)
            y_list, design_list = [], []
            for _ in range(loaded.cfg.T):
                y_hist = (
                    torch.stack(y_list, dim=1)
                    if y_list
                    else torch.empty(1, 0, *task.event_shape_y)
                )
                design_hist = (
                    torch.stack(design_list, dim=1)
                    if design_list
                    else torch.empty(1, 0, *task.event_shape_design)
                )
                start = time.perf_counter()
                design = designer.propose_design(y_hist, design_hist)
                elapsed = time.perf_counter() - start
                if rollout_idx >= n_warmup:
                    step_times.append(elapsed)
                y_list.append(task.sample_outcome(theta, design))
                design_list.append(design)
    return statistics.median(step_times) * 1000.0


# ============================================================================
# Aggregation and output
# ============================================================================


def _mean_se(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "se": se}


def _task_slug(cfg) -> str:
    class_name = cfg.task._target_.rsplit(".", 1)[-1]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=5,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def build_payload(loaded: LoadedRun, records: list[dict], speed_ms: float, cfg):
    """Assembles the results JSON payload."""
    metrics = {
        "spce": _mean_se([record["spce"] for record in records]),
        "snmc": _mean_se([record["snmc"] for record in records]),
        "proposal_time_ms": {"median": speed_ms},
    }
    if "decision_loss" in records[0]:
        metrics["decision_loss"] = _mean_se(
            [record["decision_loss"] for record in records]
        )

    return {
        "task": _task_slug(loaded.cfg),
        "task_config": OmegaConf.to_container(loaded.cfg.task),
        "method": loaded.method,
        "T": int(loaded.cfg.T),
        "metrics": metrics,
        "eval": {
            "n_inner": int(cfg.n_inner),
            "n_rollouts": int(cfg.n_rollouts),
            "n_runs": int(cfg.n_runs),
            "seeds": [record["seed"] for record in records],
            "batch_size": int(cfg.batch_size),
            "deterministic": True,
            "device": str(cfg.device),
            "num_threads_speed": torch.get_num_threads(),
            "checkpoint": str(cfg.checkpoint),
            "checkpoint_step": loaded.checkpoint.get("step"),
        },
        "run_dir": str(Path(cfg.run_dir).resolve()),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _print_report(records: list[dict], payload: dict):
    """Prints per-run values and the summary, with rich if available."""
    metrics = payload["metrics"]
    rows = [
        (
            str(record["run"]),
            f"{record['spce']:.4f}",
            f"{record['snmc']:.4f}",
            f"{record['snmc'] - record['spce']:.4f}",
            f"{record['decision_loss']:.4f}" if "decision_loss" in record else "-",
        )
        for record in records
    ]
    summary = (
        f"sPCE {metrics['spce']['mean']:.4f} ± {metrics['spce']['se']:.4f} | "
        f"sNMC {metrics['snmc']['mean']:.4f} ± {metrics['snmc']['se']:.4f} | "
        f"proposal {metrics['proposal_time_ms']['median']:.2f} ms"
    )

    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"{payload['method']} on {payload['task']}")
        for column in ("Run", "sPCE", "sNMC", "Width", "Loss"):
            table.add_column(column, justify="right")
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)
        console.print(summary)
    except ImportError:
        print(f"{payload['method']} on {payload['task']}")
        for row in rows:
            print("  run " + " | ".join(row))
        print(summary)


def _write_results(payload: dict, cfg):
    """Writes eval.json into the run dir and results/<task>/<method>.json."""
    run_path = Path(cfg.run_dir) / "eval.json"
    run_path.write_text(json.dumps(payload, indent=2) + "\n")

    if cfg.results_dir is None:
        return
    results_path = Path(cfg.results_dir) / payload["task"] / f"{payload['method']}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {results_path}")


def _make_wandb_logger(cfg, payload):
    """Builds a WandbLogger, degrading to None on any failure."""
    if not cfg.wandb.enabled:
        return None
    try:
        from testbed.core.logging import WandbLogger

        return WandbLogger(
            cfg,
            run_name=f"eval_{payload['method']}_{payload['task']}",
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            mode=cfg.wandb.mode,
            group=cfg.wandb.group,
        )
    except Exception as err:  # noqa: BLE001 -- eval must not die on wandb issues
        print(f"wandb logging disabled ({err})")
        return None


# ============================================================================
# Entry point
# ============================================================================


def main(cfg):
    """Evaluates one saved run and writes/prints/logs the results."""
    cfg = OmegaConf.merge(DEFAULTS, cfg)
    if cfg.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = load_run(
        cfg.run_dir,
        checkpoint_name=cfg.checkpoint,
        method=cfg.method,
        device=cfg.device,
    )
    records = evaluate_bounds(loaded, cfg)
    speed_ms = measure_proposal_time(loaded, cfg)
    payload = build_payload(loaded, records, speed_ms, cfg)

    _print_report(records, payload)
    _write_results(payload, cfg)

    logger = _make_wandb_logger(cfg, payload)
    if logger is not None:
        for record in records:
            logger.log({**record, "step": record["run"]})
        summary = {
            f"{name}_{key}": value
            for name, stats in payload["metrics"].items()
            for key, value in stats.items()
        }
        logger.log(summary)
        logger.close()

    return payload


if __name__ == "__main__":
    main(OmegaConf.from_cli())
