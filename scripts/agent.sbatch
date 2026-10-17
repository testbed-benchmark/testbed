#!/bin/bash
# scripts/agent.sbatch — run W&B sweep agents as a SLURM array job.
#
# Each array task claims one seed from the sweep queue, so the array runs the
# seeds in parallel. Surplus tasks find an empty queue and exit immediately, so
# the array range never has to match the amount of outstanding work.
#
#   sbatch --array=0-4 scripts/agent.sbatch <entity>/location_finding_2src/<sweep_id>
#
# Raise --count (below) when there are more seeds than array tasks, and extend
# --time to cover count x single-run duration.
#
#SBATCH --job-name=testbed-sweep
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

SWEEP=${1:?usage: sbatch --array=0-N scripts/agent.sbatch <entity>/<project>/<sweep_id>}

export WANDB_DIR="$SLURM_SUBMIT_DIR/logs/wandb-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WANDB_DIR"

# --count 1 keeps one array task to one seed. Without it a single agent drains
# the whole queue sequentially while the other tasks exit with nothing to do.
wandb agent --count 1 "$SWEEP"
