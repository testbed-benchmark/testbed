#!/bin/bash
# scripts/eval_bench.sbatch — wall-clock and peak memory of eval_spce_snmc vs L,
# chunked/unchunked x fused/eager, on location_finding_2src with a random designer.
#
# Each point runs in its own process with an empty inductor cache, so fused runs
# pay a cold compile. Writes runs/eval_bench/<name>/points/*.json and skips points
# already written, so a requeue resumes rather than restarts.
#
#   sbatch scripts/eval_bench.sbatch pilot
#   LS="1000000000" CHUNK=200000 sbatch scripts/eval_bench.sbatch full
#
#SBATCH --partition=low.p
#SBATCH --nodelist=markov-2
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --job-name=eval-bench
#SBATCH --output=runs/eval_bench/logs/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

OUT=runs/eval_bench/${1:?usage: sbatch scripts/eval_bench.sbatch <name>}/points
LS=${LS:-"1000 10000 100000 200000 400000 1000000 2000000 4000000 10000000"}
CHUNK=${CHUNK:-100000}  # largest round chunk the eager path fits in 24 GB at b=256
PYTHON=/home/kerrigag/miniconda3/envs/wbed/bin/python  # conda activate hits a pyenv shim

echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi

for L in $LS; do
  for chunk in $CHUNK none; do
    for fused in fused no-fused; do
      out=$OUT/L${L}_chunk${chunk}_${fused}.json
      [[ -f $out ]] && continue
      cache=$(mktemp -d -p /home/kerrigag/claude/scratch inductor.XXXXXX)
      TORCHINDUCTOR_CACHE_DIR=$cache TRITON_CACHE_DIR=$cache/triton "$PYTHON" scripts/eval_bench.py \
        --n-contrastive "$L" --chunk-size "$chunk" --"$fused" --out "$out" \
        || echo "failed: $out"
      rm -rf "$cache"
    done
  done
done
