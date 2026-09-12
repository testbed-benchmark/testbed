"""Post-training evaluation for a run directory.

Computes sPCE/sNMC bounds for a finished run's checkpoint and writes them to
``<run_dir>/eval.json``. When the run was logged to W&B, the bounds are added
to that same run rather than a new one, so each seed stays a single row.

``scripts/sweep_run.py`` calls this automatically after a successful training
run. It is also runnable by hand on any run directory::

    python scripts/eval_run.py --run-dir runs/location_finding_2src_actionbed_seed3_...

Sizes come from the run's saved ``final_eval`` config block, so they are set in
the recipe and overridden like any other Hydra value
(``final_eval.n_outer=32``), and ``final_eval.enabled=false`` skips evaluation.

The W&B run id is read from ``<run_dir>/wandb_run_id``, which sweep_run.py
writes; pass ``--wandb-run-id`` to override it, or ``--no-wandb`` to only write
the JSON.

``--n-contrastive`` / ``--n-outer`` override the saved sizes for a one-off
re-evaluation, e.g. re-scoring a finished run at a larger L to check whether its
sPCE bound was saturated at log(L+1). ``--prefix`` namespaces both the metric
keys and the JSON file, so a re-eval lands beside the original ``eval/*``
columns instead of overwriting them::

    python scripts/eval_run.py --run-dir runs/<run_name> \
        --n-contrastive 10000000 --n-outer 512 --prefix eval_L1e7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from testbed.eval import evaluate_checkpoint


def log_to_wandb(run_dir: Path, run_id: str, results: dict, prefix: str) -> None:
    """Adds eval results to the training run, as summary columns and history."""
    import wandb

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    run = wandb.init(
        id=run_id,
        resume="must",
        project=OmegaConf.select(cfg, "wandb.project"),
        entity=OmegaConf.select(cfg, "wandb.entity"),
    )
    scalars = {
        f"{prefix}/{k}": v for k, v in results.items() if isinstance(v, (int, float))
    }
    run.log(scalars)
    # Summary is what surfaces as a column in the runs table.
    run.summary.update(scalars)
    run.finish()


def main() -> None:
    """Evaluates one run directory and records the bounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--device", default=None, help="Default: the device the run was trained on."
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Default: read from <run_dir>/wandb_run_id.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate even when the run's final_eval.enabled is false.",
    )
    parser.add_argument(
        "--n-contrastive",
        type=int,
        default=None,
        help="Override final_eval.n_contrastive (L).",
    )
    parser.add_argument(
        "--n-outer", type=int, default=None, help="Override final_eval.n_outer."
    )
    parser.add_argument(
        "--prefix",
        default="eval",
        help="Namespace for the metric keys and the JSON file.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    # Settings come from the run's own config, so they are set and overridden
    settings = OmegaConf.select(cfg, "final_eval")
    if settings is None:
        raise SystemExit(
            f"{run_dir} has no final_eval block; add `- /final_eval: default` to "
            f"the method recipe's defaults."
        )
    if not settings.enabled and not args.force:
        print("final_eval.enabled is false; skipping")
        return

    results = evaluate_checkpoint(
        str(run_dir),
        checkpoint=settings.checkpoint,
        n_outer=args.n_outer or settings.n_outer,
        n_contrastive=args.n_contrastive or settings.n_contrastive,
        outer_batch_size=settings.outer_batch_size,
        contrastive_chunk_size=settings.contrastive_chunk_size,
        use_raw_weights=settings.use_raw_weights,
        device=args.device or cfg.device,
    )
    (run_dir / f"{args.prefix}.json").write_text(json.dumps(results, indent=2))
    print(results)

    run_id = args.wandb_run_id
    if run_id is None:
        id_path = run_dir / "wandb_run_id"
        run_id = id_path.read_text().strip() if id_path.exists() else None
    if not args.no_wandb and run_id:
        try:
            log_to_wandb(run_dir, run_id, results, args.prefix)
        except Exception as error:
            print(
                f"could not add eval metrics to wandb run {run_id}: {error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
