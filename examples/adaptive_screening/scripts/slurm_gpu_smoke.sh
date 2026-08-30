#!/bin/bash

#SBATCH --job-name=adaptive-screening-smoke
#SBATCH --partition=standard-gpu
#SBATCH --clusters=srf_gpu_01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-adaptive-screening-smoke-%j.out

set -eu

# Submit from the project root.
runs_dir="$(pwd)/runs/adaptive_screening_smoke"
python=".venv/bin/python"

if [[ ! -x "${python}" ]]; then
  echo "Cannot execute ${python} from $(pwd)" >&2
  exit 127
fi

mkdir -p "${runs_dir}"
"${python}" examples/adaptive_screening/scripts/gpu_smoke.py \
  --config examples/adaptive_screening/configs/main.yaml \
  --gate examples/adaptive_screening/outputs/pilot/pilot_gate.json
