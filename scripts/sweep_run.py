r"""Sweep-agent wrapper for TestBED seed sweeps.

A W&B sweep agent invokes this script once per queued seed. It translates the
queue item into Hydra overrides, shells out to the recipe's entry point, and
then evaluates the resulting checkpoint in the same job.

The sweep YAML pins ``--recipe``; the agent appends the swept parameters via
the ``${args_no_hyphens}`` macro, which writes them as Hydra overrides
(``seed=3 lr=1e-3``). Any parameter can be swept this way. ``seed`` must be
among them, since it names the run.

Run directories are named ``<task>_<method>_seed<n>_<timestamp>_<jobid>_<taskid>``,
following the hydra_run group's convention; Manager copies that basename into
the W&B run name. The path is resolved here rather than left to Hydra's
``${now:}`` so that the eval step can be pointed at it. The wrapper also fixes
the W&B run id up front (``WANDB_RUN_ID``) so eval metrics join the training
run instead of opening a new one.

Evaluation is best-effort: a failure is reported but does not fail the run,
since the checkpoint is already saved. Pass ``--no-eval`` to skip it.

Set ``WANDB_ENTITY`` to log to a team rather than your default entity.

Usage (normally via ``wandb agent``, but runnable by hand from the repo root)::

    python scripts/sweep_run.py \
        --recipe experiments/location_finding_2src/actionbed seed=3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES_DIR = os.path.join(REPO_ROOT, "testbed", "recipes")


def resolve_save_dir(recipe: str, overrides: list[str]) -> str:
    """Returns the save_dir the training run will use."""
    for override in reversed(overrides):
        if override.startswith("save_dir="):
            return override.split("=", 1)[1]

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        return compose(config_name=recipe).save_dir


def main() -> None:
    """Trains one seed of a recipe, then evaluates its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe path under testbed/recipes, e.g. "
        "experiments/location_finding_2src/actionbed.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Entry point under testbed/methods (default: the recipe's basename).",
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="Skip the post-training evaluation."
    )
    parser.add_argument("--eval-n-outer", type=int, default=None)
    parser.add_argument("--eval-n-contrastive", type=int, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides forwarded to the entry point, including the "
        "swept parameters. Must contain seed=<n>.",
    )
    args = parser.parse_args()

    task = os.path.basename(os.path.dirname(args.recipe))
    method = args.method or os.path.basename(args.recipe)

    # The seed is swept like any other parameter, but it also names the run, so
    # read it back out of the overrides.
    seed = next(
        (
            override.split("=", 1)[1]
            for override in reversed(args.overrides)
            if override.startswith("seed=")
        ),
        None,
    )
    if seed is None:
        print("no seed=<n> among the overrides", file=sys.stderr)
        sys.exit(2)

    entry_point = os.path.join(REPO_ROOT, "testbed", "methods", f"{method}.py")
    if not os.path.exists(entry_point):
        print(f"no entry point at {entry_point}", file=sys.stderr)
        sys.exit(2)

    # The SLURM ids point back at the job's slurm-<jobid>_<taskid>.out, and are
    # 0_0 off a cluster.
    run_name = "_".join(
        [
            task,
            method,
            f"seed{seed}",
            time.strftime("%Y-%m-%d_%H-%M-%S"),
            os.environ.get("SLURM_ARRAY_JOB_ID", "0"),
            os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
        ]
    )
    run_dir = os.path.join(resolve_save_dir(args.recipe, args.overrides), run_name)

    command = [
        sys.executable,
        entry_point,
        "--config-name",
        args.recipe,
        "wandb.enabled=true",
        f"wandb.tags=[seed{seed}]",
        f"hydra.run.dir={run_dir}",
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        command.append(f"wandb.entity={entity}")
    command.extend(args.overrides)

    # Fixing the id here is what lets eval_run.py reopen this run afterwards;
    # wandb.init picks it up from the environment.
    env = dict(os.environ)
    wandb_run_id = env.setdefault("WANDB_RUN_ID", uuid.uuid4().hex[:16])

    print(f"→ {' '.join(command)}")
    train_status = subprocess.call(command, env=env)
    if train_status != 0 or args.no_eval:
        sys.exit(train_status)

    # Recorded so a later manual eval_run.py lands on the same W&B run.
    with open(os.path.join(run_dir, "wandb_run_id"), "w") as handle:
        handle.write(wandb_run_id)

    eval_command = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "eval_run.py"),
        "--run-dir",
        run_dir,
        "--wandb-run-id",
        wandb_run_id,
    ]
    if args.eval_n_outer is not None:
        eval_command += ["--n-outer", str(args.eval_n_outer)]
    if args.eval_n_contrastive is not None:
        eval_command += ["--n-contrastive", str(args.eval_n_contrastive)]

    print(f"→ {' '.join(eval_command)}")
    eval_status = subprocess.call(eval_command, env=env)
    if eval_status != 0:
        # Training succeeded and the checkpoint is on disk, so don't fail the
        # run; eval_run.py can be re-run against the directory.
        print(
            f"eval failed ({eval_status}); checkpoint kept at {run_dir}",
            file=sys.stderr,
        )

    sys.exit(train_status)


if __name__ == "__main__":
    main()
