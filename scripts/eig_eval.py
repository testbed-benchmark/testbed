"""Eval finished runs SNMC/SPCE.

Computes sPCE/sNMC bounds for a finished run's checkpoint and writes them to
``<run_dir>/eval.json`` or --out if specified.

Runnable on any run directory::

    python scripts/eig_eval.py \\
        --run-dir runs/location_finding_2src_actionbed_seed3_... \\
        --n_outer <N> \\
        --n_contrastive <L> \\
        --outer_batch_size <M> \\
        --out <path.json>

Sizes default to the run's saved ``final_eval`` config block; ``--n-outer``,
``--n-contrastive`` and ``--outer-batch-size`` override them for this
evaluation only. ``scripts/eig_eval.sbatch`` runs this over a list of run
directories as a SLURM array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from testbed.eval import evaluate_checkpoint


def main() -> None:
    """Evaluates one run directory and records the bounds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--device", default=None, help="Default: the device the run was trained on."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate even when the run's final_eval.enabled is false.",
    )
    parser.add_argument(
        "--n-outer",
        "--n_outer",
        type=int,
        default=None,
        help="Default: the run's final_eval.n_outer.",
    )
    parser.add_argument(
        "--n-contrastive",
        "--n_contrastive",
        type=int,
        default=None,
        help="Default: the run's final_eval.n_contrastive.",
    )
    parser.add_argument(
        "--outer-batch-size",
        "--outer_batch_size",
        type=int,
        default=None,
        help="Default: the run's final_eval.outer_batch_size.",
    )
    parser.add_argument(
        "--out",
        default="eval.json",
        help="Results file, relative to the run dir (default: eval.json).",
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
        outer_batch_size=args.outer_batch_size or settings.outer_batch_size,
        contrastive_chunk_size=settings.contrastive_chunk_size,
        use_raw_weights=settings.use_raw_weights,
        device=args.device or cfg.device,
    )
    (run_dir / args.out).write_text(json.dumps(results, indent=2))
    print(results)


if __name__ == "__main__":
    main()
