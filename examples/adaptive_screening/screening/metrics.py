"""Per-episode metrics and genuinely paired uncertainty estimates."""

import jax.numpy as jnp
import numpy as np

from .action_rules import bayes_action, fixed_action, selected_value


def decision_metrics(mean, sd, theta, kappa):
    action = fixed_action(mean, sd, kappa)
    reward = selected_value(mean, action)
    value = jnp.maximum(0, jnp.max(mean, axis=-1))
    utility = selected_value(theta, action)
    return {
        "return": reward,
        "bayes_value": value,
        "incompatibility": value - reward,
        "action": action,
        "disagreement": (action != bayes_action(mean)).astype(jnp.float64),
        "default": (action == 0).astype(jnp.float64),
        "beneficial_deployment": ((action > 0) & (utility > 0)).astype(jnp.float64),
        "selected_utility": utility,
        "selected_sd": selected_value(sd, action),
    }


def trajectory_metrics(trajectory, theta, kappa, config):
    metrics = decision_metrics(trajectory.mean, trajectory.sd, theta, kappa)
    action = metrics["action"]
    deployed = action > 0
    index = jnp.maximum(action - 1, 0)
    first = jnp.take_along_axis(trajectory.first_leader, index[:, None], axis=-1)[:, 0]
    ever = deployed & (first <= config.T)
    allocated = (trajectory.actions == (action - 1)[:, None]) & deployed[:, None]
    metrics.update(
        {
            "distinct": jnp.sum(trajectory.counts > 0, axis=-1).astype(jnp.float64),
            "budget_fraction": jnp.mean(allocated, axis=-1, dtype=jnp.float64),
            "last_six_fraction": jnp.mean(
                allocated[:, -min(6, config.T) :], axis=-1, dtype=jnp.float64
            ),
            "first_leader_round": jnp.where(ever, first, 0).astype(jnp.float64),
            "selected_ever_leader": ever.astype(jnp.float64),
            "selected_never_leader": (deployed & ~ever).astype(jnp.float64),
        }
    )
    return metrics


def mean_interval(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Intervals require at least two finite paired observations")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "episodes": int(values.size),
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def summarize(metrics, prior_max):
    metrics = {k: np.asarray(v) for k, v in metrics.items()}
    for key, values in metrics.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite episode metric: {key}")
    result = {k: float(v.mean()) for k, v in metrics.items() if k != "action"}
    result["J"] = result.pop("return")
    result["risk"] = prior_max - result["J"]
    result["bayes_component"] = prior_max - result["bayes_value"]
    result["decomposition_error"] = abs(
        result["risk"] - result["bayes_component"] - result["incompatibility"]
    )
    if result["decomposition_error"] > 1e-10:
        raise ValueError("Risk decomposition does not close")
    result["episodes"] = len(metrics["return"])
    result["J_se"] = mean_interval(metrics["return"])["se"]
    deployed = int(np.sum(metrics["default"] == 0))
    result["selected_sd_conditional"] = (
        float(metrics["selected_sd"].sum() / deployed) if deployed else None
    )
    if "first_leader_round" in metrics:
        ever = int(metrics["selected_ever_leader"].sum())
        result["first_leader_round_conditional"] = (
            float(metrics["first_leader_round"].sum() / ever) if ever else None
        )
        result["selected_never_leader_conditional"] = (
            float(metrics["selected_never_leader"].sum() / deployed)
            if deployed
            else None
        )
    return result
