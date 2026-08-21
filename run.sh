#!/usr/bin/bash
set -euo pipefail

# This is for Isambard AI

# ============================================================
# Experiment configuration
# Edit this section for each run
# ============================================================

PROJECT_DIR="$(pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"

RUN_NAME="check-gpus"

RUN_DIR="${PROJECT_DIR}/runs/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "${RUN_DIR}"

COMMAND=(
	.venv/bin/python test_script.py
)

# ============================================================
# Slurm configuration
# ============================================================

TIME="00:05:00"
MEM="24G"

# ============================================================
# Save reproducibility information
# ============================================================

# Exact command
{
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${RUN_DIR}/command.sh"

chmod +x "${RUN_DIR}/command.sh"

# Git information
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
    git status --short > "${RUN_DIR}/git_status.txt"
    git diff > "${RUN_DIR}/git_diff.patch"
fi

# ============================================================
# Generate the actual Slurm script
# ============================================================

JOB_SCRIPT="${RUN_DIR}/job.slurm"

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/bash

#SBATCH --job-name=${RUN_NAME}
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${RUN_DIR}/stdout.log
#SBATCH --error=${RUN_DIR}/stderr.log

set -euo pipefail

echo "Job ID: \$SLURM_JOB_ID"
echo "Host: \$(hostname)"
echo "Started: \$(date)"
echo "Project directory: ${PROJECT_DIR}"
echo "Run directory: ${RUN_DIR}"

cd "${PROJECT_DIR}"

EOF

# Append the command with correct shell quoting.
{
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} >> "${JOB_SCRIPT}"

# ============================================================
# Submit
# ============================================================

JOB_ID="$(sbatch --parsable "${JOB_SCRIPT}")"

echo "${JOB_ID}" > "${RUN_DIR}/slurm_job_id.txt"

echo "Submitted Slurm job ${JOB_ID}"
echo "Run directory: ${RUN_DIR}"
