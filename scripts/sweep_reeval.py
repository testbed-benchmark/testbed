r"""Re-evaluates every run in a finished W&B sweep.

Resolves a sweep id to its runs, maps each one back to its run directory, and
calls ``scripts/eval_run.py`` on it. The bounds land on the original W&B run, so
the re-eval joins the row it belongs to rather than opening a new one.

The run directory is the W&B run's display name under the run's own
``save_dir``: ``sweep_run.py`` names the directory and Manager copies that
basename into the W&B run name, so the two always agree. The W&B run id comes
from the API rather than ``<run_dir>/wandb_run_id``.

Pass ``--n-contrastive`` / ``--n-outer`` / ``--prefix`` straight through to
``eval_run.py``: the prefix namespaces the new metrics, so re-scoring at a
larger L sits beside the original ``eval/*`` columns instead of overwriting
them.

Usage (whole sweep in one process, or one run per SLURM array task via
``--index``; see ``scripts/reeval.sbatch``)::

    python scripts/sweep_reeval.py 1dpl3rg2 --project location_finding_2src \
        --n-contrastive 10000000 --n-outer 512 --prefix eval_L1e7

    python scripts/sweep_reeval.py 1dpl3rg2 --project location_finding_2src --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(REPO_ROOT, "scripts", "eval_run.py")


def resolve_sweep_path(sweep: str, entity: str, project: str | None) -> str:
    """Expands a bare sweep id into ``<entity>/<project>/<sweep_id>``."""
    if sweep.count("/") == 2:
        return sweep
    if project is None:
        raise SystemExit(
            f"{sweep} is a bare sweep id; pass --project, or give the full "
            f"<entity>/<project>/<sweep_id> path"
        )
    return f"{entity}/{project}/{sweep}"


def sweep_runs(sweep_path: str):
    """Returns the sweep's runs, ordered by name so --index is stable."""
    import wandb

    return sorted(wandb.Api().sweep(sweep_path).runs, key=lambda run: run.name)


def run_directory(run, save_dir: str | None) -> str:
    """Returns the run directory for a W&B run."""
    root = save_dir if save_dir is not None else run.config["save_dir"]
    return os.path.join(root, run.name)


def eval_command(run, run_dir: str, args) -> list[str]:
    """Builds the eval_run.py invocation for one run."""
    command = [
        sys.executable,
        EVAL_SCRIPT,
        "--run-dir",
        run_dir,
        "--wandb-run-id",
        run.id,
        "--prefix",
        args.prefix,
    ]
    if args.n_contrastive is not None:
        command += ["--n-contrastive", str(args.n_contrastive)]
    if args.n_outer is not None:
        command += ["--n-outer", str(args.n_outer)]
    if args.device is not None:
        command += ["--device", args.device]
    if args.force:
        command.append("--force")
    return command


def main() -> None:
    """Re-evaluates one sweep's runs, or the single run named by --index."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sweep", help="Sweep id, or the full <entity>/<project>/<sweep_id> path."
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "rainml-testbed"),
        help="Default: $WANDB_ENTITY, else rainml-testbed.",
    )
    parser.add_argument(
        "--project", default=None, help="Required when the sweep id is bare."
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Root holding the run directories. Default: each run's own save_dir.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Evaluate only this run, by position in the name-sorted list. "
        "Surplus array tasks exit 0.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print the run mapping and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running."
    )
    parser.add_argument("--n-contrastive", type=int, default=None)
    parser.add_argument("--n-outer", type=int, default=None)
    parser.add_argument("--prefix", default="eval")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate even when a run's final_eval.enabled is false.",
    )
    args = parser.parse_args()

    sweep_path = resolve_sweep_path(args.sweep, args.entity, args.project)
    runs = sweep_runs(sweep_path)
    if not runs:
        raise SystemExit(f"{sweep_path} has no runs")

    if args.list:
        print(f"{sweep_path}: {len(runs)} runs")
        for index, run in enumerate(runs):
            run_dir = run_directory(run, args.save_dir)
            present = "ok" if os.path.isdir(run_dir) else "MISSING"
            print(f"  [{index}] {run.id} {run.state:9s} {present:7s} {run_dir}")
        return

    if args.index is not None:
        # Mirrors agent.sbatch: an array range wider than the work is not an error.
        if args.index >= len(runs):
            print(f"index {args.index} beyond {len(runs)} runs; nothing to do")
            return
        runs = [runs[args.index]]

    failures = []
    for run in runs:
        run_dir = run_directory(run, args.save_dir)
        if run.state != "finished":
            print(f"skipping {run.name}: state is {run.state}", file=sys.stderr)
            continue
        if not os.path.isdir(run_dir):
            print(f"skipping {run.name}: no run dir at {run_dir}", file=sys.stderr)
            continue

        command = eval_command(run, run_dir, args)
        print(f"→ {' '.join(command)}")
        if args.dry_run:
            continue
        if subprocess.call(command) != 0:
            failures.append(run.name)

    if failures:
        print(f"eval failed for: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
