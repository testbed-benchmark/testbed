#!/bin/bash
# scripts/random_eval.sbatch — sPCE/sNMC eval of the random-designer baseline.
#
# One array element per (task, seed), writing runs/random/<task>/seed<n>.jsonl:
#
#   sbatch scripts/random_eval.sbatch
#
# The array size must cover len(TASKS) * len(SEEDS); --array below is sized for
# the 9 tasks x 5 seeds set below, and the job fails fast if the grid outgrows
# it. To evaluate a subset, override the grid at submission:
#
#   TASKS=location_finding_2src sbatch --array=0-4 scripts/random_eval.sbatch
#
#SBATCH --job-name=random-eval
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-44

set -euo pipefail

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

OUT_DIR="${OUT_DIR:-runs/random}"
SEEDS=(${SEEDS:-0 1 2 3 4})
TASKS=(${TASKS:-
  location_finding_2src
  temporal_discounting
  constant_elasticity_of_substitution
  michaelis_menten
  pharmacokinetic
  stochastic_pendulum
  double_link
  cart_pole
  gravimetry
})

i="${SLURM_ARRAY_TASK_ID:?Submit as an array job}"
n=$((${#TASKS[@]} * ${#SEEDS[@]}))
((i < n)) || {
  echo "array index $i outside the ${#TASKS[@]} x ${#SEEDS[@]} grid; " \
       "resubmit with --array=0-$((n - 1))" >&2
  exit 1
}
TASK="${TASKS[$((i / ${#SEEDS[@]}))]}"
SEED="${SEEDS[$((i % ${#SEEDS[@]}))]}"

# Missing recipes are the common failure here, and the check is far cheaper
# than discovering it after the allocation starts.
[[ -f "testbed/recipes/task/$TASK.yaml" ]] || {
  echo "no recipe testbed/recipes/task/$TASK.yaml" >&2
  exit 1
}

# Fail loudly rather than falling back to CPU: the fused CUDA path is ~90x
# faster, so a silent CPU run turns 25 minutes into a day and a half.
DEVICE="${DEVICE:-cuda}"
python -c "import torch; assert torch.cuda.is_available()" || {
  echo "no visible GPU; set DEVICE=cpu to run anyway" >&2
  exit 1
}

echo "job=${SLURM_ARRAY_JOB_ID}_${i} host=$(hostname) task=$TASK seed=$SEED"

# Sample sizes and `fused` come from testbed/recipes/final_eval/default.yaml,
# which eval_random reads for its own defaults; don't duplicate them here.
python -m testbed.eval.eval_random "$TASK" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR"
