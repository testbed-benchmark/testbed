#!/bin/bash
# scripts/reeval.sbatch — re-evaluate a finished sweep as a SLURM array job.
#
# Each array task re-evaluates one run of the sweep, indexed by position in the
# name-sorted run list, so the runs are re-scored in parallel. Surplus tasks
# find nothing at their index and exit immediately, so the array range never has
# to match the run count exactly.
#
#   sbatch --array=0-4 scripts/reeval.sbatch \
#       rainml-testbed/location_finding_2src/1dpl3rg2 \
#       --n-contrastive 10000000 --n-outer 512 --prefix eval_L1e7
#
# Everything after the sweep path is forwarded to scripts/sweep_reeval.py.
# Check the mapping first from the login node:
#
#   python scripts/sweep_reeval.py <sweep_path> --list
#
#SBATCH --job-name=testbed-reeval
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Repo-root venv by default; override with VENV_PATH=... for another location.
VENV_PATH="${VENV_PATH:-.venv}"
[[ -f "$VENV_PATH/bin/activate" ]] || {
  echo "no venv at $VENV_PATH (set VENV_PATH)" >&2
  exit 1
}
source "$VENV_PATH/bin/activate"

export WANDB_ENTITY="${WANDB_ENTITY:-rainml-testbed}"

SWEEP=${1:?usage: sbatch --array=0-N scripts/reeval.sbatch <entity>/<project>/<sweep_id> [sweep_reeval.py args]}
shift

export WANDB_DIR="$SLURM_SUBMIT_DIR/logs/wandb-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WANDB_DIR"

python scripts/sweep_reeval.py "$SWEEP" --index "$SLURM_ARRAY_TASK_ID" "$@"
