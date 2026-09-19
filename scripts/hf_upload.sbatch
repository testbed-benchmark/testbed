#!/bin/bash
# scripts/hf_upload.sbatch — push or pull a sweep's run directories as a SLURM job.
#
# One task handles the whole sweep: scripts/sweep_hf_upload.py loops over the
# runs itself, so this is not an array job and needs no GPU.
#
#   sbatch scripts/hf_upload.sbatch upload \
#       rainml-testbed/architecture_location_finding_2src/gubvjvzm
#   sbatch scripts/hf_upload.sbatch download \
#       rainml-testbed/architecture_location_finding_2src/gubvjvzm
#
# Everything after the sweep path is forwarded to scripts/sweep_hf_upload.py,
# so --ignore '*.pt', --overwrite and the rest work here too. Check what would
# move first, from the login node:
#
#   python scripts/sweep_hf_upload.py upload <sweep_path> --dry-run
#
# huggingface_hub is not a testbed dependency, so this uses its own venv:
#   python -m venv .hf_env && .hf_env/bin/pip install wandb huggingface_hub
# Authenticate before submitting (the job cannot prompt): huggingface-cli login
# or HF_TOKEN, plus wandb login or WANDB_API_KEY for uploads.
#
#SBATCH --job-name=testbed-hf
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

# Repo-root venv holding huggingface_hub; override with VENV_PATH=... .
VENV_PATH="${VENV_PATH:-.hf_env}"
[[ -f "$VENV_PATH/bin/activate" ]] || {
  echo "no venv at $VENV_PATH (set VENV_PATH)" >&2
  exit 1
}
source "$VENV_PATH/bin/activate"

export WANDB_ENTITY="${WANDB_ENTITY:-rainml-testbed}"

MODE=${1:?usage: sbatch scripts/hf_upload.sbatch <upload|download> <entity>/<project>/<sweep_id> [sweep_hf_upload.py args]}
SWEEP=${2:?usage: sbatch scripts/hf_upload.sbatch <upload|download> <entity>/<project>/<sweep_id> [sweep_hf_upload.py args]}
shift 2

python scripts/sweep_hf_upload.py "$MODE" "$SWEEP" "$@"
