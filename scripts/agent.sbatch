#!/bin/bash
# scripts/agent.sbatch — run W&B sweep agents as a SLURM array job.
#
# Each array task claims one seed from the sweep queue, so the array runs the
# seeds in parallel. Surplus tasks find an empty queue and exit immediately, so
# the array range never has to match the amount of outstanding work.
#
#   sbatch --array=0-4 scripts/agent.sbatch rainml-testbed/location_finding_2src/<sweep_id>
#
# Raise --count (below) when there are more seeds than array tasks, and extend
# --time to cover count x single-run duration.
#
#SBATCH --job-name=testbed-sweep
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
## Cluster-specific; uncomment and set:
##SBATCH --partition=
##SBATCH --account=

set -euo pipefail

SWEEP=${1:?usage: sbatch --array=0-N scripts/agent.sbatch <entity>/<project>/<sweep_id>}

mkdir -p logs
cd "$(dirname "$0")/.."  # sweep YAMLs name scripts/sweep_run.py relative to the repo root

# The agent needs to reach api.wandb.ai from the compute node to be handed a
# seed; WANDB_MODE=offline cannot substitute, as there is no server to ask.
export WANDB_ENTITY=${WANDB_ENTITY:-rainml-testbed}

# --count 1 keeps one array task to one seed. Without it a single agent drains
# the whole queue sequentially while the other tasks exit with nothing to do.
wandb agent --count 1 "$SWEEP"