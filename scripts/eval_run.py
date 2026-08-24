"""
scripts/eval_run.py — post-training evaluation for a run directory
==================================================================
Computes sPCE/sNMC bounds for a finished run's checkpoint and writes them to
``<run_dir>/eval.json``. When the run was logged to W&B, the bounds are added
to that same run rather than a new one, so each seed stays a single row.

``scripts/sweep_run.py`` calls this automatically after a successful training
run. It is also runnable by hand on any run directory::

    python scripts/eval_run.py --run-dir runs/location_finding_2src_actionbed_seed3_...

The W&B run id is read from ``<run_dir>/wandb_run_id``, which sweep_run.py
writes; pass ``--wandb-run-id`` to override it, or ``--no-wandb`` to only write
the JSON.

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from testbed.eval import evaluate_checkpoint


def log_to_wandb(run_dir: Path, run_id: str, results: dict) -> None:
    """Adds eval results to the training run, as summary columns and history."""
    import wandb

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    run = wandb.init(
        id=run_id,
        resume="must",
        project=OmegaConf.select(cfg, "wandb.project"),
        entity=OmegaConf.select(cfg, "wandb.entity"),
    )
    scalars = {f"eval/{k}": v for k, v in results.items() if isinstance(v, (int, float))}
    run.log(scalars)
    # Summary is what surfaces as a column in the runs table.
    run.summary.update(scalars)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="final_step.pt")
    parser.add_argument("--n-outer", type=int, default=2048)
    parser.add_argument("--n-contrastive", type=int, default=1_000_000)
    parser.add_argument("--outer-batch-size", type=int, default=256)
    parser.add_argument("--contrastive-chunk-size", type=int, default=200_000)
    parser.add_argument("--use-raw-weights", action="store_true")
    parser.add_argument("--device", default=None,
                        help="Default: the device the run was trained on.")
    parser.add_argument("--wandb-run-id", default=None,
                        help="Default: read from <run_dir>/wandb_run_id.")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    device = args.device or cfg.device

    results = evaluate_checkpoint(
        str(run_dir),
        checkpoint=args.checkpoint,
        n_outer=args.n_outer,
        n_contrastive=args.n_contrastive,
        outer_batch_size=args.outer_batch_size,
        contrastive_chunk_size=args.contrastive_chunk_size,
        use_raw_weights=args.use_raw_weights,
        device=device,
    )
    (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(results)

    run_id = args.wandb_run_id
    if run_id is None:
        id_path = run_dir / "wandb_run_id"
        run_id = id_path.read_text().strip() if id_path.exists() else None
    if not args.no_wandb and run_id:
        try:
            log_to_wandb(run_dir, run_id, results)
        except Exception as error:
            print(f"could not add eval metrics to wandb run {run_id}: {error}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()