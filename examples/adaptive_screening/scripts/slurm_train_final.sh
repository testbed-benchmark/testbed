#!/bin/bash

#SBATCH --job-name=adaptive-screening-final
#SBATCH --partition=standard-gpu
#SBATCH --clusters=srf_gpu_01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-adaptive-screening-final-%A_%a.out

set -eu

# Submit from the project root.
runs_dir="$(pwd)/runs/adaptive_screening_final"
python=".venv/bin/python"

if [[ ! -x "${python}" ]]; then
  echo "Cannot execute ${python} from $(pwd)" >&2
  exit 127
fi

"${python}" -c 'import jax; devices = jax.devices(); assert any(device.platform == "gpu" for device in devices), devices; print(devices)'
mkdir -p "${runs_dir}"

kappas=(0 2.5 3.0)
seeds=(404 505)
kappa_index=$((SLURM_ARRAY_TASK_ID / 2))
seed_index=$((SLURM_ARRAY_TASK_ID % 2))
kappa="${kappas[${kappa_index}]}"
seed="${seeds[${seed_index}]}"

"${python}" examples/adaptive_screening/scripts/train_policy.py \
  --config examples/adaptive_screening/configs/main.yaml \
  --gate examples/adaptive_screening/outputs/pilot/pilot_gate.json \
  --optimization-gate examples/adaptive_screening/outputs/optimization_check/optimization_gate.json \
  --train-kappa "${kappa}" \
  --seeds "${seed}" \
  --output "${runs_dir}"
