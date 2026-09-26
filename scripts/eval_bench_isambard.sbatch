#!/bin/bash
# scripts/eval_bench_isambard.sbatch — Isambard-AI version of
# scripts/eval_bench.sbatch (which targets markov-2).
#
# Wall-clock and peak memory of eval_spce_snmc vs L, chunked/unchunked x
# fused/eager, on location_finding_2src with a random designer. Each point runs
# in its own process with an empty inductor cache, so fused runs pay a cold
# compile. Writes runs/eval_bench/<name>/points/*.json and skips points already
# written, so a resubmission resumes rather than restarts. Use a fresh <name>
# per CHUNK, or the plot mixes chunk sizes.
#
#   mkdir -p runs/eval_bench/logs
#   sbatch scripts/eval_bench_isambard.sbatch gh200
#
# The eager L=1e9 point dominates: ~8.6h extrapolated on markov-2's A5000, so
# --time covers a GH200 no faster than that. Max walltime on Isambard is 24h.
#
#SBATCH --job-name=eval-bench
#SBATCH --output=runs/eval_bench/logs/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

DIR=runs/eval_bench/${1:?usage: sbatch scripts/eval_bench_isambard.sbatch <name>}
LS=${LS:-"1000 10000 100000 200000 400000 1000000 2000000 4000000 10000000 100000000 1000000000"}
CHUNK=${CHUNK:-200000}  # the paper's eval setting

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

echo "job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi

# Runs eval_bench.py with an empty inductor cache, so its compile is cold.
bench() {
  local cache status=0
  cache=$(mktemp -d)
  TORCHINDUCTOR_CACHE_DIR=$cache TRITON_CACHE_DIR=$cache/triton python scripts/eval_bench.py "$@" || status=$?
  rm -rf "$cache"
  return "$status"
}

# Pages in the lazily imported compile stack, which a cold filesystem otherwise
# bills to whichever fused point runs first (43s rather than 16s here).
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

python scripts/plot_eval_bench.py "$DIR" || echo "plotting failed; run scripts/plot_eval_bench.py $DIR by hand"
