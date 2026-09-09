"""Posterior plots for location finding.

One panel per inference step, showing the posterior over source locations
against the true ones. Any method whose rollout carries ``posterior_history``
can use this. Both hooks that infer a posterior keep it only under
``store_posterior_history``: ``RefinementHook`` once per refinement,
``PosteriorUpdateHook`` once per step.

Plots two components of ``dim_theta``, so a search space of higher dimension is
shown as a 2D slice through it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..inference.is_sampler import effective_sample_size

# Enough to read the shape of a posterior without drawing ten million points.
N_PLOT_DEFAULT = 2_000
SOURCE_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def _project(tensor, dims):
    """Selects components ``dims`` of the last axis, as a CPU array."""
    return tensor[..., list(dims)].detach().cpu().numpy()


def _axis_limits(arrays, pad=0.5):
    """Bounding box of arrays of 2D points, padded.

    Shared across panels so that contraction between steps is visible rather
    than rescaled away.
    """
    points = np.concatenate([array.reshape(-1, 2) for array in arrays])
    lower = points.min(axis=0) - pad
    upper = points.max(axis=0) + pad
    return (lower[0], upper[0]), (lower[1], upper[1])


def plot_posterior_history(
    rollout,
    path,
    dims=(0, 1),
    n_plot=N_PLOT_DEFAULT,
    show_designs=True,
    title=None,
):
    """Writes a figure with one posterior panel per inference step.

    Args:
        rollout: A batch-1 rollout carrying ``extra["posterior_history"]``.
        path: Where to write the figure.
        dims: The two components of ``dim_theta`` to plot.
        n_plot: Posterior draws per panel. Drawn with replacement against the
            importance weights, so a degenerate posterior visibly collapses onto
            a handful of points.
        show_designs: Mark the designs measured up to each step.
        title: Figure title.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = rollout.extra.get("posterior_history")
    if not history:
        raise ValueError(
            "rollout.extra has no posterior_history to plot. The hook that "
            "inferred the posteriors needs store_posterior_history=True "
            "(SequentialDesigner takes it under the same name)."
        )
    if len(dims) != 2:
        raise ValueError(f"dims must name exactly two components; got {dims}.")
    if rollout.theta.shape[0] != 1:
        raise ValueError(
            f"Expected a single trajectory; got a batch of {rollout.theta.shape[0]}. "
            "Plot each rollout separately."
        )
    dim_theta = rollout.theta.shape[-1]
    if max(dims) >= dim_theta:
        raise ValueError(
            f"dims={dims} but theta has only {dim_theta} components per source."
        )

    # {step count: posterior}, ordered by when inference ran.
    steps = sorted(history)
    torch.manual_seed(0)  # a fixed subsample, so panels are comparable
    posteriors = [
        _project(history[step].sample(n_samples=n_plot), dims) for step in steps
    ]

    true_sources = _project(rollout.theta[0], dims)  # [n_sources, 2]
    designs = _project(rollout.design_history[0], dims) if show_designs else None
    xlim, ylim = _axis_limits([true_sources, *posteriors])

    n_cols = min(len(steps), 4)
    n_rows = math.ceil(len(steps) / n_cols)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        # The extra height is room for the figure title above the panel titles.
        figsize=(3.2 * n_cols, 3.2 * n_rows + 0.4),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = [axis for row in axes for axis in row]

    for axis, step, posterior_points in zip(flat_axes, steps, posteriors):
        for source in range(true_sources.shape[0]):
            colour = SOURCE_COLOURS[source % len(SOURCE_COLOURS)]
            axis.scatter(
                posterior_points[:, source, 0],
                posterior_points[:, source, 1],
                s=4,
                alpha=0.15,
                color=colour,
                linewidths=0,
            )
            axis.scatter(
                true_sources[source, 0],
                true_sources[source, 1],
                marker="*",
                s=220,
                color=colour,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
        if designs is not None:
            # Only what had been measured when this posterior was inferred.
            axis.scatter(
                designs[:step, 0],
                designs[:step, 1],
                marker="x",
                s=18,
                color="0.35",
                linewidths=0.8,
                zorder=2,
            )
        ess = effective_sample_size(history[step])
        subtitle = f"after {step} steps"
        if ess is not None:
            subtitle += f"   ESS {ess:,.0f}"
        axis.set_title(subtitle, fontsize=9)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal")
        axis.tick_params(labelsize=7)

    for axis in flat_axes[len(steps) :]:
        axis.axis("off")

    handles = [
        plt.Line2D([], [], marker="*", markersize=12, linestyle="none",
                   color="0.4", markeredgecolor="black", label="true source"),
        plt.Line2D([], [], marker="o", markersize=5, linestyle="none",
                   color="0.4", label="posterior draw"),
    ]
    if designs is not None:
        handles.append(
            plt.Line2D([], [], marker="x", markersize=6, linestyle="none",
                       color="0.35", label="design measured")
        )
    flat_axes[0].legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.8)

    if title:
        figure.suptitle(title, fontsize=11)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
