"""Plotting helpers. Task-specific, and matplotlib is an optional dependency."""

from .location_finding import (
    plot_posterior_history,
    report_posterior_figures,
    write_trajectory_figures,
)

__all__ = [
    "plot_posterior_history",
    "report_posterior_figures",
    "write_trajectory_figures",
]
