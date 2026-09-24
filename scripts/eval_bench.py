"""Wall-clock and peak GPU memory of eval_spce_snmc at one (L, chunk, fused) point.

Run each point in a fresh process with an empty TORCHINDUCTOR_CACHE_DIR, so a
fused run pays a cold compile; scripts/eval_bench.sbatch does both. Writes one
JSON record to --out, recording an OOM rather than failing.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.designers import RandomDesigner
from testbed.eval import eval_spce_snmc

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--n-contrastive", type=int, required=True)
parser.add_argument(
    "--chunk-size",
    type=lambda s: None if s == "none" else int(s),
    default=200_000,
    help="'none' scores all L draws at once",
)
parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--n-outer", type=int, default=2_048)
parser.add_argument("--outer-batch-size", type=int, default=256)
parser.add_argument("--task", default="location_finding_2src")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
task_cfg = OmegaConf.load(f"testbed/recipes/task/{args.task}.yaml")
task = instantiate(task_cfg.task).to("cuda")
designer = RandomDesigner(task, task_cfg.T)

# initialise the CUDA context up front so it isn't billed as compilation
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
try:
    result = eval_spce_snmc(
        task,
        designer,
        n_outer=args.n_outer,
        n_contrastive=args.n_contrastive,
        outer_batch_size=args.outer_batch_size,
        contrastive_chunk_size=args.chunk_size,
        verbose=False,
        fused=args.fused,
    )
    torch.cuda.synchronize()
except torch.cuda.OutOfMemoryError:
    result = {"oom": True}

record = {
    **{k: v for k, v in vars(args).items() if k != "out"},
    "seconds": time.perf_counter() - start,
    "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
    "gpu": torch.cuda.get_device_name(),
    "gpu_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
    "torch": torch.__version__,
    **result,
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record) + "\n")
print(json.dumps(record))
