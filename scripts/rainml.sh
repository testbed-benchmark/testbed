#!/bin/bash
#SBATCH --job-name=testbed            # this is the job name
#SBATCH --cluster=swan                  # always specify the server
#SBATCH --partition=standard-rainml-gpu # and the partition
#SBATCH --output=testbed_%A_%a.out      # output file name
#SBATCH --time=0-05:00:00                 # time limit
#SBATCH  --gres=gpu:Ampere_H100_80GB:1 --mem=40G

set -euo pipefail

source /etc/profile.d/modules.sh
module purge
module use /apps/modulefiles/Core

BITBUCKET_ROOT="${BITBUCKET_ROOT:-/bitbucket/$USER}"

# Activate the Python environment. 
#   ENV_TYPE=venv sbatch --array=0-N scripts/rainml.sh <sweep>
# Stats IT Docs: https://ithelp.stats.ox.ac.uk/userguides/hpc/modules.html
ENV_TYPE="${ENV_TYPE:-conda}"

case "$ENV_TYPE" in
  conda)
    TESTBED_ENV="${TESTBED_ENV:-$BITBUCKET_ROOT/conda-envs/testbed}"
    module load conda/miniforge3
    source /apps/conda/miniforge3/etc/profile.d/conda.sh
    conda activate "$TESTBED_ENV"
    ;;
  venv)
    TESTBED_ENV="${TESTBED_ENV:-$BITBUCKET_ROOT/venvs/testbed}"
    module load "python/${PYTHON_VERSION:-3.11.15}"
    source "$TESTBED_ENV/bin/activate"
    ;;
  *)
    echo "ENV_TYPE must be 'conda' or 'venv', got: $ENV_TYPE" >&2
    exit 1
    ;;
esac

# Confirm activation
echo "Active environment ($ENV_TYPE): $(which python)"

SWEEP=${1:?usage: sbatch --array=0-N scripts/rainml.sh <entity>/<project>/<sweep_id>}

cd "$BITBUCKET_ROOT/testbed"

git pull

# you should have already logged into wandb on the compute node
# for rainml the login creds will be stored in .netrc file on head node also
export WANDB_ENTITY="${WANDB_ENTITY:?WANDB_ENTITY must be set}"


# --count 1 keeps one array task to one seed. Without it a single agent drains
# the whole queue sequentially while the other tasks exit with nothing to do.
wandb agent --count 1 "$SWEEP"