#!/bin/bash
# scripts/eig_eval.sbatch — sPCE/sNMC eval of finished runs, one array task per run.
#
# Write the run directories to eval one per line (blank lines and #comments are
# ignored), then launch an array with one task per line:
#
#   cat > logs/eig_runs.txt <<'EOF'
#   runs/location_finding_2src_actionbed_seed0_...
#   runs/location_finding_2src_actionbed_seed1_...
#   EOF
#   sbatch --array=0-1 scripts/eig_eval.sbatch logs/eig_runs.txt
#
# Task i evaluates line i, so the array range must match the list; surplus tasks
# fail loudly rather than silently evaluating nothing.
#
#SBATCH --job-name=eig-eval
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail

# Eval sizes, shared by every task so the runs stay comparable. Override per
# submission with e.g. N_CONTRASTIVE=10000000 sbatch ...
N_OUTER="${N_OUTER:-2048}"
N_CONTRASTIVE="${N_CONTRASTIVE:-10000000}"
OUT="${OUT:-1e7_eig_eval.json}"

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

CONDA_ENV="${CONDA_ENV:-testbed}"
CONDA_BASE="${CONDA_BASE:-/home/u6sb/gavin.u6sb/miniforge3}"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || {
  echo "no conda at $CONDA_BASE (set CONDA_BASE)" >&2
  exit 1
}
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

RUN_LIST=${1:?usage: sbatch --array=0-N scripts/eig_eval.sbatch <run_list.txt>}

mapfile -t RUNS < <(grep -v '^\s*\(#\|$\)' "$RUN_LIST")
RUN_DIR="${RUNS[${SLURM_ARRAY_TASK_ID:?run this as an array job}]:?task index past the end of $RUN_LIST}"

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) run=$RUN_DIR"

python scripts/eig_eval.py \
  --run-dir "$RUN_DIR" \
  --n-outer "$N_OUTER" \
  --n-contrastive "$N_CONTRASTIVE" \
  --out "$OUT" \
  --force
