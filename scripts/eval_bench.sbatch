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

DIR=runs/eval_bench/${1:?usage: sbatch scripts/eval_bench.sbatch <name>}
LS=${LS:-"1000 10000 100000 200000 400000 1000000 2000000 4000000 10000000"}
CHUNK=${CHUNK:-100000}  # largest round chunk the eager path fits in 24 GB at b=256
PYTHON=/home/kerrigag/miniconda3/envs/wbed/bin/python  # conda activate hits a pyenv shim

echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi

# Runs eval_bench.py with an empty inductor cache, so its compile is cold.
bench() {
  local cache status=0
  cache=$(mktemp -d -p /home/kerrigag/claude/scratch inductor.XXXXXX)
  TORCHINDUCTOR_CACHE_DIR=$cache TRITON_CACHE_DIR=$cache/triton "$PYTHON" scripts/eval_bench.py "$@" || status=$?
  rm -rf "$cache"
  return "$status"
}

# Pages in the lazily imported compile stack, which a cold filesystem otherwise
# bills to whichever fused point runs first.
bench --n-contrastive 1000 --out /dev/null >/dev/null || echo "warmup failed"

for L in $LS; do
  for chunk in $CHUNK none; do
    for fused in fused no-fused; do
      out=$DIR/points/L${L}_chunk${chunk}_${fused}.json
      [[ -f $out ]] && continue
      bench --n-contrastive "$L" --chunk-size "$chunk" --"$fused" --out "$out" || echo "failed: $out"
    done
  done
done

"$PYTHON" scripts/plot_eval_bench.py "$DIR" || echo "plotting failed; run scripts/plot_eval_bench.py $DIR by hand"
