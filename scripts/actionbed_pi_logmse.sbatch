#!/bin/bash
# scripts/actionbed_pi_logmse.sbatch — ActionBED LF 2-source with the swap-min
# Log-MSE loss (PermutationInvariantLogMSELoss), one job per variant:
#
#   mkdir -p logs
#   sbatch scripts/actionbed_pi_logmse.sbatch loss_only   # loss swap only
#   sbatch scripts/actionbed_pi_logmse.sbatch loss_adam   # + lr 1e-3, betas (0.9, 0.999)
#
# Runs land in runs/lf2src_actionbed_pi_logmse/<variant>/. The wandb groups carry
# an _isambard suffix, keeping these apart from the cancelled markov-2 attempts.
#
#SBATCH --job-name=actionbed-pi-logmse
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

VARIANT=${1:?usage: sbatch scripts/actionbed_pi_logmse.sbatch <loss_only|loss_adam>}
case "$VARIANT" in
  loss_only) EXTRA=() ;;
  loss_adam) EXTRA=(lr=1e-3 'betas=[0.9,0.999]') ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

# Repo-root venv by default; override with VENV_PATH=... for another location.
VENV_PATH="${VENV_PATH:-.venv}"
[[ -f "$VENV_PATH/bin/activate" ]] || {
  echo "no venv at $VENV_PATH (set VENV_PATH)" >&2
  exit 1
}
source "$VENV_PATH/bin/activate"

export WANDB_ENTITY="${WANDB_ENTITY:-rainml-testbed}"
export WANDB_DIR="$SLURM_SUBMIT_DIR/logs/wandb-$SLURM_JOB_ID"
mkdir -p "$WANDB_DIR"

echo "job=$SLURM_JOB_ID variant=$VARIANT host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi

python scripts/sweep_run.py \
  --recipe experiments/location_finding_2src/actionbed \
  seed=42 \
  save_dir=runs/lf2src_actionbed_pi_logmse/$VARIANT \
  wandb.group=actionbed_pi_logmse_${VARIANT}_isambard \
  loss._target_=testbed.design_objectives.PermutationInvariantLogMSELoss \
  ${EXTRA[@]+"${EXTRA[@]}"}
