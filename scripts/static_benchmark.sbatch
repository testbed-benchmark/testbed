#!/bin/bash
# scripts/static_benchmark.sbatch — static designs on the benchmarking tasks, seeds 0-4.
#
# One array task per (task, seed), index = 5 * task_index + seed. Each writes
# runs/static/<task>/seed<id>/: the Hydra config, training logs and metrics,
# final_step.pt (the selected design and every restart's), and eval.json.
#
#   mkdir -p runs/static/logs && sbatch scripts/static_benchmark.sbatch
#
# Extra arguments are forwarded to training as Hydra overrides, e.g. a smoke test:
#
#   OUT_ROOT=runs/static_smoke sbatch --array=15 scripts/static_benchmark.sbatch \
#       n_train_iter=20 select.n_contrastive=1000 final_eval.n_outer=64
#
# Jobs run whatever is checked out when they start, so leave the checkout alone
# until the array finishes.
#
#SBATCH --partition=low.p
#SBATCH --nodelist=markov-2
#SBATCH --gpus=1
#SBATCH --mem=48gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-24%8
#SBATCH --job-name=static-benchmark
#SBATCH --output=runs/static/logs/slurm-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

TASKS=(
  location_finding_2src
  pharmacokinetic
  michaelis_menten
  constant_elasticity_of_substitution
  temporal_discounting
)
N_SEEDS=5
TASK=${TASKS[SLURM_ARRAY_TASK_ID / N_SEEDS]}
SEED=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
RUN="${OUT_ROOT:-runs/static}/$TASK/seed$SEED"
PYTHON=/home/kerrigag/miniconda3/envs/wbed/bin/python  # conda activate hits a pyenv shim

if [[ -f "$RUN/eval.json" ]]; then
  echo "$RUN already evaluated; nothing to do"
  exit 0
fi

# Training cannot resume, so a preempted attempt restarts from scratch. Keep its
# logs aside rather than appending a second run to them.
if [[ -d "$RUN" && ! -f "$RUN/final_step.pt" ]]; then
  mv "$RUN" "$RUN.preempted-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1

cp "$0" "$RUN/job.sbatch"
cat > "$RUN/metadata.json" <<EOF
{
  "experiment_name": "static_benchmark",
  "purpose": "static-design sPCE/sNMC baselines on the benchmarking tasks",
  "task": "$TASK",
  "seed": $SEED,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(git rev-parse --short HEAD)",
  "git_branch": "$(git branch --show-current)",
  "git_dirty": $([[ -z "$(git status --porcelain)" ]] && echo false || echo true),
  "host": "$(hostname)",
  "python": "$PYTHON",
  "job_id": "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}",
  "extra_overrides": "$*"
}
EOF

echo "job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) task=$TASK seed=$SEED"
nvidia-smi

if [[ ! -f "$RUN/final_step.pt" ]]; then
  "$PYTHON" -m testbed.cli.static \
    --config-name "experiments/benchmarking/$TASK/static" \
    seed="$SEED" \
    hydra.run.dir="$RUN" \
    wandb.enabled=false \
    final_eval.n_contrastive=1000000 \
    "$@"
fi

"$PYTHON" scripts/eval_run.py --run-dir "$RUN" --no-wandb
