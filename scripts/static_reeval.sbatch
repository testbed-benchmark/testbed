#!/bin/bash
# scripts/static_reeval.sbatch — re-evaluate finished static runs at L=1e9, one GPU per seed.
#
# Loads runs/static/<task>/seed<id>/final_step.pt and writes eval_1e9.json beside
# it. Every other eval setting comes from the run's saved final_eval block.
#
#   sbatch scripts/static_reeval.sbatch constant_elasticity_of_substitution
#
#SBATCH --partition=low.p
#SBATCH --nodelist=markov-2
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --array=0-4
#SBATCH --job-name=static-reeval
#SBATCH --output=runs/static/logs/reeval-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

TASK=${1:?usage: sbatch scripts/static_reeval.sbatch <task>}
RUN="${ROOT:-runs/static}/$TASK/seed$SLURM_ARRAY_TASK_ID"
OUT=eval_1e9
PYTHON=/home/kerrigag/miniconda3/envs/wbed/bin/python  # conda activate hits a pyenv shim

if [[ -f "$RUN/$OUT.json" ]]; then
  echo "$RUN/$OUT.json exists; nothing to do"
  exit 0
fi

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) run=$RUN commit=$(git rev-parse --short HEAD)"
nvidia-smi

"$PYTHON" scripts/eval_run.py \
  --run-dir "$RUN" \
  --no-wandb \
  --n-contrastive "${N_CONTRASTIVE:-1000000000}" \
  --prefix "$OUT"
