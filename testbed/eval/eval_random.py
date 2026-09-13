"""sPCE/sNMC EIG bounds for a random designer.

Usage:
    python -m testbed.eval.eval_random <task> [--n_outer N] ...

<task> names a recipe in testbed/recipes/task/. One (task, seed) is evaluated
per invocation and written to <out_dir>/<task>/seed<n>.jsonl. The sample sizes
default to testbed/recipes/final_eval/default.yaml (see `--help`).

scripts/random_eval.sbatch holds the task and seed lists, running one
(task, seed) per array element.
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


def set_seed(seed):
    """Seeds every rng the eval path can reach.

    Note this won't give you full determinism: results still vary with the
    batch/chunk sizes (they change the draw sequence) and across hardware.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def eval_random(task_cfg, eval_cfg):
    """Computes sPCE/sNMC bounds for a random designer, for one seed."""
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
        fused=eval_cfg.fused,
    )


def save_result(task_name, seed, record, out_dir):
    """Writes the result to <out_dir>/<task>/seed<n>.jsonl."""
    path = Path(out_dir) / task_name / f"seed{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task": task_name, "seed": seed, **record}) + "\n")
    return path


if __name__ == "__main__":
    defaults = OmegaConf.load("testbed/recipes/final_eval/default.yaml")

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("task", help="recipe name under testbed/recipes/task/")
    parser.add_argument(
        "--n_outer",
        type=int,
        default=defaults["n_outer"],
        help="outer theta samples; avoid changing from the default",
    )
    parser.add_argument(
        "--n_contrastive",
        type=int,
        default=defaults["n_contrastive"],
        help="contrastive samples per outer sample; avoid changing from the default",
    )
    parser.add_argument(
        "--outer_batch_size",
        type=int,
        default=defaults["outer_batch_size"],
        help="outer samples per batch; set by memory/compute",
    )
    parser.add_argument(
        "--contrastive_chunk_size",
        type=int,
        default=defaults["contrastive_chunk_size"],
        help="contrastive samples scored at once; set by memory/compute",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device; cpu is ~90x slower here",
    )
    parser.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        default=defaults["fused"],
        help="score each chunk in one compiled call instead of a loop over T",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tqdm progress bar (auto-silenced when stderr isn't a terminal)",
    )
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    parser.add_argument("--out_dir", default="runs/random/", help="results directory")

    args = parser.parse_args()

    set_seed(args.seed)
    task_cfg = OmegaConf.load(f"testbed/recipes/task/{args.task}.yaml")
    record = eval_random(task_cfg, args)

    print()
    print(f"saved {save_result(args.task, args.seed, record, args.out_dir)}")
