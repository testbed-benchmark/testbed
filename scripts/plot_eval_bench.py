"""Plots scripts/eval_bench.sbatch results: peak memory and wall-clock vs L.

    python scripts/plot_eval_bench.py runs/eval_bench/pilot

Writes eval_bench.{pdf,png} into the run directory and prints the numbers the
text quotes: compile cost, per-draw cost, break-even L and speedup.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

run = Path(sys.argv[1])
points = [json.loads(p.read_text()) for p in run.glob("points/*.json")]

COLOR = {True: "#2a78d6", False: "#eb6834"}  # fused, eager
GPU_GB = points[0]["gpu_gb"]


def curve(fused, chunked):
    """Rows for one configuration, sorted by L."""
    rows = [
        r
        for r in points
        if r["fused"] == fused and (r["chunk_size"] is not None) == chunked
    ]
    return sorted(rows, key=lambda r: r["n_contrastive"])


def ran(fused, chunked=True):
    """``curve`` without the runs that ran out of memory."""
    return [r for r in curve(fused, chunked) if not r.get("oom")]


def fit(rows):
    """Least-squares ``seconds = intercept + slope * L`` in relative error."""
    L, t = (np.array([r[k] for r in rows]) for k in ("n_contrastive", "seconds"))
    slope, intercept = np.polyfit(L, t, 1, w=1 / t)
    return intercept, slope


plt.rcParams.update(
    {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (ax_mem, ax_time) = plt.subplots(
    1, 2, figsize=(6.5, 2.4), sharex=True, layout="constrained"
)
for fused in (True, False):
    for chunked in (True, False):
        rows = curve(fused, chunked)
        ok = ran(fused, chunked)
        L = [r["n_contrastive"] for r in ok]
        style = dict(
            color=COLOR[fused],
            ls="-" if chunked else "--",
            marker="o",
            ms=4,
            mfc=COLOR[fused] if chunked else "white",
            lw=1.5,
            label=f"{'fused' if fused else 'eager'}, {'chunked' if chunked else 'unchunked'}",
        )
        ax_mem.plot(L, [r["peak_gb"] for r in ok], **style)
        ax_time.plot(L, [r["seconds"] for r in ok], **style)
        oom = [r["n_contrastive"] for r in rows if r.get("oom")]
        if oom:
            ax_mem.plot(min(oom), GPU_GB, "x", color=COLOR[fused], ms=7, mew=1.5)

(c_f, a_f), (c_e, a_e) = fit(ran(True)), fit(ran(False))
breakeven = (c_f - c_e) / (a_e - a_f)
ax_mem.axhline(GPU_GB, color="0.6", lw=0.8, ls=":")
ax_mem.text(
    1.2 * min(r["n_contrastive"] for r in points),
    GPU_GB * 1.15,
    "GPU memory (× = OOM)",
    color="0.4",
)
ax_time.axvline(breakeven, color="0.6", lw=0.8, ls=":")
ax_time.text(
    breakeven * 1.2,
    0.97,
    "break-even",
    color="0.4",
    va="top",
    transform=ax_time.get_xaxis_transform(),
)

for ax, ylabel in ((ax_mem, "peak GPU memory (GB)"), (ax_time, "wall-clock (s)")):
    ax.set(xscale="log", yscale="log", xlabel="contrastive samples $L$", ylabel=ylabel)
    ax.grid(alpha=0.3, lw=0.5)
ax_time.legend(frameon=False, fontsize=7)
for ext in ("pdf", "png"):
    fig.savefig(run / f"eval_bench.{ext}", dpi=200)

n_outer = points[0]["n_outer"]
fused, eager = (
    {r["n_contrastive"]: r["seconds"] for r in ran(f)} for f in (True, False)
)
L_max = max(fused.keys() & eager.keys())
print(f"compile ≈ {c_f - c_e:.1f} s")
print(
    f"per (outer, draw): fused {a_f / n_outer:.2e} s, eager {a_e / n_outer:.2e} s ({a_e / a_f:.0f}x)"
)
print(f"break-even L ≈ {breakeven:.2g}")
print(
    f"at L={L_max:.0e}: fused {fused[L_max]:.0f} s, eager {eager[L_max]:.0f} s ({eager[L_max] / fused[L_max]:.1f}x)"
)
