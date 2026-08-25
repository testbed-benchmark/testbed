"""sPCE/sNMC EIG bounds for a random designer.

Usage:
    python -m testbed.eval.eval_random <task|all> [--n_outer N] ...

<task> names a recipe in testbed/recipes/task/, or `all` to sweep ALL_TASKS
below. Results are written as JSON to --out_dir and printed as LaTeX table
rows. The sample sizes default to testbed/recipes/final_eval/default.yaml
(see `--help`).

Note that you can specify 'all' as the task, in which case the eval will be run
for all of the tasks specified in ALL_TASKS.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.designers import RandomDesigner
from testbed.eval import eval_spce_snmc

ALL_TASKS = [
    "location_finding_2src",
    "temporal_discounting",
    "constant_elasticity_of_substitution",
]

SIZE_ARGS = {
    "n_outer": "outer theta samples; avoid changing from the default",
    "n_contrastive": "contrastive samples per outer sample; avoid changing "
    "from the default",
    "outer_batch_size": "outer samples per batch; set by memory/compute",
    "contrastive_chunk_size": "contrastive samples scored at once; set by "
    "memory/compute",
}


def set_seed(seed):
    """Seeds every rng the eval path can reach.

    Note this won't give you full determinism: results still vary with the
    batch/chunk sizes (they change the draw sequence) and across hardware.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def eval_random(task_cfg, eval_cfg):
    """Computes mean +/- SEM sPCE/sNMC bounds for a random designer."""
    task = instantiate(task_cfg.task).to(eval_cfg.device)
    designer = RandomDesigner(task, task_cfg.T)
    return eval_spce_snmc(
        task,
        designer,
        n_outer=eval_cfg.n_outer,
        n_contrastive=eval_cfg.n_contrastive,
        outer_batch_size=eval_cfg.outer_batch_size,
        contrastive_chunk_size=eval_cfg.contrastive_chunk_size,
        verbose=eval_cfg.verbose,
    )


def eval_tasks(task_names, eval_cfg):
    """Evaluates each task, keyed by task name.

    The seed is reset per task, so a task's numbers don't depend on which
    tasks ran before it.
    """
    results = {}
    for name in task_names:
        set_seed(eval_cfg.seed)
        task_cfg = OmegaConf.load(f"testbed/recipes/task/{name}.yaml")
        results[name] = eval_random(task_cfg, eval_cfg)
    return results


def save_results(results, out_dir, seed):
    """Writes the results dict to <out_dir>/random_eval_seed<seed>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"random_eval_seed{seed}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def latex_rows(results, decimals):
    r"""Formats results as `task & \MeanSD{mean}{sem} & ...` rows."""

    def cell(mean, se):
        return f"\\MeanSD{{{mean:.{decimals}f}}}{{{se:.{decimals}f}}}"

    rows = ["Task & NMC LB & EIG UB \\"]
    for name, r in results.items():
        label = name.replace("_", r"\_")
        rows.append(
            f"{label} & {cell(r['lower_bound'], r['lower_bound_se'])} "
            f"& {cell(r['upper_bound'], r['upper_bound_se'])} \\\\"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    defaults = OmegaConf.load("testbed/recipes/final_eval/default.yaml")

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "task", help="recipe name under testbed/recipes/task/, or `all`"
    )
    for name, help_text in SIZE_ARGS.items():
        parser.add_argument(
            f"--{name}",
            type=int,
            default=defaults[name],
            help=help_text,
        )
    parser.add_argument("--device", default="cpu", help="torch device; gpu recommended")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tqdm progress bar (auto-silenced when stderr isn't a terminal)",
    )
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    parser.add_argument("--out_dir", default="results/eval/", help="results directory")
    parser.add_argument(
        "--decimals", type=int, default=3, help="decimals in the LaTeX output"
    )

    args = parser.parse_args()

    task_names = ALL_TASKS if args.task == "all" else [args.task]
    results = eval_tasks(task_names, args)

    print("")
    print(f"saved {save_results(results, args.out_dir, args.seed)}")
    print("")
    print(latex_rows(results, args.decimals))
