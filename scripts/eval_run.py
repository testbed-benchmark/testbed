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

``--checkpoint`` scores a file other than the recipe's
``final_eval.checkpoint``, such as one of the periodic ``step_*.pt`` policies
RL-BOED writes when ``checkpoint_every`` is set. Scoring a non-default
checkpoint namespaces its own output (``eval_step_00010000.json`` and
``eval_step_00010000/*``), so a mid-training score cannot overwrite the run's
headline ``eval/*`` columns; pass ``--prefix`` to choose the namespace
yourself::

    python scripts/eval_run.py --run-dir runs/<run_name> \
        --checkpoint step_00010000.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf

from testbed.designers import StaticDesigner
from testbed.eval import evaluate_checkpoint, evaluate_static_checkpoint


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
        "--checkpoint",
        default=None,
        help="Checkpoint filename within the run directory. "
        "Default: the run's final_eval.checkpoint.",
    )
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
        default=None,
        help="Namespace for the metric keys and the JSON file. Default: "
        "'eval', or 'eval_<checkpoint stem>' when --checkpoint is not the "
        "one the run would be evaluated with.",
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

    # Scoring a checkpoint other than the run's own would otherwise overwrite
    # its headline eval/* columns and eval.json with a mid-training number.
    checkpoint = args.checkpoint or settings.checkpoint
    prefix = args.prefix
    if prefix is None:
        prefix = (
            "eval"
            if checkpoint == settings.checkpoint
            else f"eval_{Path(checkpoint).stem}"
        )

    kwargs = {
        "checkpoint": checkpoint,
        "n_outer": args.n_outer or settings.n_outer,
        "n_contrastive": args.n_contrastive or settings.n_contrastive,
        "outer_batch_size": settings.outer_batch_size,
        "contrastive_chunk_size": settings.contrastive_chunk_size,
        "device": args.device or cfg.device,
        # run configs saved before final_eval.fused existed default to fused
        "fused": settings.get("fused", True),
    }
    designer = OmegaConf.select(settings, "designer._target_")
    if designer and issubclass(get_class(designer), StaticDesigner):
        results = evaluate_static_checkpoint(str(run_dir), **kwargs)
    else:
        results = evaluate_checkpoint(
            str(run_dir), use_raw_weights=settings.use_raw_weights, **kwargs
        )
    results_path = run_dir / f"{prefix}.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"{checkpoint} -> {results_path}")
    print(results)

    run_id = args.wandb_run_id
    if run_id is None:
        id_path = run_dir / "wandb_run_id"
        run_id = id_path.read_text().strip() if id_path.exists() else None
    if not args.no_wandb and run_id:
        try:
            log_to_wandb(run_dir, run_id, results, prefix)
        except Exception as error:
            print(
                f"could not add eval metrics to wandb run {run_id}: {error}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
