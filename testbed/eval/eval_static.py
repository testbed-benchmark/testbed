"""Standalone sPCE/sNMC evaluation for a static-design checkpoint.

Reloads the winning restart's `"raw_design"` written by
`methods.static.Trainer` into a `StaticDesigner`, then scores it exactly as
`eval_policy` scores a policy.

Usage:
    python -m testbed.eval.eval_static <run_dir>

<run_dir> is a static training run's output directory (containing
.hydra/config.yaml and the checkpoint file). Checkpoint, sample sizes and
device all come from the run's saved `final_eval` block
(`final_eval.n_outer=32`).
"""

import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.designers import StaticDesigner
from testbed.eval.base import eval_spce_snmc


def load_static_designer(task, state, T):
    """Rebuilds the StaticDesigner holding a checkpoint's winning design."""
    designer = StaticDesigner(
        task, T, transform_designs=state.get("transform_designs", True)
    )
    with torch.no_grad():
        designer.raw_design.copy_(state["raw_design"])
    return designer


def evaluate_static_checkpoint(
    run_dir: str,
    checkpoint: str = "final_step.pt",
    n_outer: int = 2048,
    n_contrastive: int = 500_000,
    outer_batch_size: int = 256,
    contrastive_chunk_size: int = 2_000,
    device: str = "cpu",
    fused: bool = True,
):
    """Computes mean +/- SEM sPCE/sNMC bounds for a saved static design."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    task = instantiate(cfg.task).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    designer = load_static_designer(task, state, cfg.T)

    out = eval_spce_snmc(
        task,
        designer,
        n_outer=n_outer,
        n_contrastive=n_contrastive,
        outer_batch_size=outer_batch_size,
        contrastive_chunk_size=contrastive_chunk_size,
        fused=fused,
    )
    out["best_restart"] = state["best_restart"]
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m testbed.eval.eval_static <run_dir>")

    run_dir = Path(sys.argv[1])
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    settings = OmegaConf.select(cfg, "final_eval")
    if settings is None:
        raise SystemExit(
            f"{run_dir} has no final_eval block; add `- /final_eval: default` to "
            f"the method recipe's defaults."
        )

    print(
        evaluate_static_checkpoint(
            str(run_dir),
            checkpoint=settings.checkpoint,
            n_outer=settings.n_outer,
            n_contrastive=settings.n_contrastive,
            outer_batch_size=settings.outer_batch_size,
            contrastive_chunk_size=settings.contrastive_chunk_size,
            device=cfg.device,
            fused=settings.fused,
        )
    )
