"""Tune on validation only, freeze, then evaluate both independent test seeds."""

import itertools
import platform
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from .action_rules import terminal_return
from .common_randomness import observation_prefix, permuted, pregenerate
from .config import provenance
from .environment import explore_terminal, rollout, static_terminal
from .io import file_hash, make_manifest, write_json, write_table
from .metrics import mean_interval, summarize, trajectory_metrics
from .policies import Policy
from .quadrature import expected_prior_max

POLICY_NAMES = (
    "ucb_bayes",
    "ucb_rule_aware",
    "explore_certify",
    "nonadaptive",
    "eig",
    "random",
)


def log(message):
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


@partial(jax.jit, static_argnames=("config",))
def allocation_returns(prefix, parameters, config):
    """Batch the exact two-stage terminal calculation over all requested kappas."""

    def one(params):
        k, S, margin, kappa = params
        mean, sd = explore_terminal(
            prefix, config, k.astype(jnp.int32), S.astype(jnp.int32), margin > 0, kappa
        )
        return terminal_return(mean, sd, kappa)

    rewards = jax.vmap(one)(parameters)
    return jnp.sum(rewards, axis=-1), jnp.sum(rewards**2, axis=-1)


@partial(jax.jit, static_argnames=("config",))
def subset_returns(prefix, sizes, config):
    def one(k):
        mean, sd = static_terminal(prefix, config, k)
        return jax.vmap(lambda kappa: terminal_return(mean, sd, kappa))(
            jnp.array(config.kappa_grid)
        )

    rewards = jax.vmap(one)(sizes)
    return jnp.sum(rewards, axis=-1), jnp.sum(rewards**2, axis=-1)


def tuning_summary(total, squares, n):
    mean = total / n
    variance = np.maximum(0, (squares - total**2 / n) / (n - 1))
    return {"J": float(mean), "J_se": float(np.sqrt(variance / n)), "episodes": n}


def select_best(rows, fields, tolerance):
    """All configurations within < tolerance of the maximum share a tie."""
    best = max(row["J"] for row in rows)
    tied = [row for row in rows if row["J"] == best or best - row["J"] < tolerance]

    def key(row):
        return tuple(
            (0 if row[f] == "mean" else 1) if f == "h" else row[f] for f in fields
        )

    return min(tied, key=key)


def tune(config, data, output, metadata, batch_size):
    grids = config.tuning_grids()
    kappas = config.kappa_grid
    ucb_rows = []
    n = len(data.theta)
    for beta in config.beta_grid:
        totals, squares = np.zeros(len(kappas)), np.zeros(len(kappas))
        for start in range(0, n, batch_size):
            result = rollout(
                data.subset(start, start + batch_size), config, Policy("ucb", beta=beta)
            )
            rewards = np.asarray(
                jax.vmap(lambda k: terminal_return(result.mean, result.sd, k))(
                    jnp.array(kappas)
                )
            )
            totals += rewards.sum(-1)
            squares += (rewards**2).sum(-1)
        ucb_rows.extend(
            {"beta": beta, "kappa": kappa, **tuning_summary(totals[j], squares[j], n)}
            for j, kappa in enumerate(kappas)
        )
    write_table(output / "ucb_validation.csv", ucb_rows, metadata)
    log(f"Validation: completed all {len(config.beta_grid)} UCB coefficients")

    entries = [
        {**p, "kappa": kappa} for p in grids["explore_certify"] for kappa in kappas
    ]
    parameters = np.array(
        [[p["k"], p["S"], int(p["h"] == "margin"), p["kappa"]] for p in entries],
        dtype=np.float64,
    )
    totals, squares = np.zeros(len(entries)), np.zeros(len(entries))
    static_total, static_sq = (
        np.zeros((config.d, len(kappas))),
        np.zeros((config.d, len(kappas))),
    )
    # Bounded working set: 24 configurations by one episode batch, not grid x dataset.
    grid_batch = 24
    for start in range(0, n, batch_size):
        prefix = observation_prefix(data.subset(start, start + batch_size))
        for p in range(0, len(entries), grid_batch):
            partial_sum, partial_sq = allocation_returns(
                prefix, parameters[p : p + grid_batch], config
            )
            totals[p : p + grid_batch] += np.asarray(partial_sum)
            squares[p : p + grid_batch] += np.asarray(partial_sq)
        a, b = subset_returns(prefix, jnp.arange(1, config.d + 1), config)
        static_total += np.asarray(a)
        static_sq += np.asarray(b)
        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= n:
            log(
                f"Validation: exhaustive two-stage/static grids, {min(start + batch_size, n)}/{n} episodes"
            )
    explore_rows = [
        {**p, **tuning_summary(totals[j], squares[j], n)} for j, p in enumerate(entries)
    ]
    static_rows = [
        {
            "k": k,
            "kappa": kappa,
            **tuning_summary(static_total[k - 1, j], static_sq[k - 1, j], n),
        }
        for k in range(1, config.d + 1)
        for j, kappa in enumerate(kappas)
    ]
    write_table(output / "explore_certify_validation.csv", explore_rows, metadata)
    write_table(output / "nonadaptive_validation.csv", static_rows, metadata)
    selected = {}
    for kappa in kappas:

        def subset(rows):
            return [row for row in rows if row["kappa"] == kappa]

        selected[str(kappa)] = {
            "ucb": select_best(
                subset(ucb_rows), ("beta",), config.validation_tie_tolerance
            ),
            "explore_certify": select_best(
                subset(explore_rows), ("S", "k", "h"), config.validation_tie_tolerance
            ),
            "nonadaptive": select_best(
                subset(static_rows), ("k",), config.validation_tie_tolerance
            ),
        }
    return {"beta_bayes": selected["0.0"]["ucb"]["beta"], "selected": selected}


def frozen_policies(frozen, kappa):
    chosen = frozen["selected"][str(kappa)]
    etc = chosen["explore_certify"]
    return {
        "ucb_bayes": Policy("ucb", beta=frozen["beta_bayes"]),
        "ucb_rule_aware": Policy("ucb", beta=chosen["ucb"]["beta"]),
        "explore_certify": Policy(
            "explore_certify", k=etc["k"], S=etc["S"], h=etc["h"], kappa=kappa
        ),
        "nonadaptive": Policy("nonadaptive", k=chosen["nonadaptive"]["k"]),
        "eig": Policy("eig"),
        "random": Policy("random"),
    }


def evaluate_dataset(config, data, frozen, metadata, dataset, batch_size):
    assignments = [
        (k, name, policy)
        for k in config.kappa_grid
        for name, policy in frozen_policies(frozen, k).items()
    ]
    distinct_policies = list(dict.fromkeys(p for _, _, p in assignments))
    metrics = {(k, name): [] for k, name, _ in assignments}
    for policy in distinct_policies:
        users = [(k, name) for k, name, p in assignments if p == policy]
        for start in range(0, len(data.theta), batch_size):
            part = data.subset(start, start + batch_size)
            trajectory = rollout(part, config, policy)
            theta = permuted(part).theta
            for kappa, name in users:
                values = trajectory_metrics(trajectory, theta, kappa, config)
                metrics[(kappa, name)].append(
                    {k: np.asarray(v) for k, v in values.items()}
                )
    rows, samples = [], {}
    for kappa, name, policy in assignments:
        chunks = metrics.pop((kappa, name))
        values = {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
        rows.append(
            {
                "seed": dataset.seed,
                "kappa": kappa,
                "policy": name,
                "policy_parameters": str(policy.to_dict()),
                **summarize(values, metadata["expected_prior_max"]["value"]),
            }
        )
        for metric in ("return", "distinct"):
            samples[f"{kappa}|{name}|{metric}"] = values[metric]
    return rows, samples


def comparisons(samples_by_seed, config):
    rows = []
    groups = {str(seed): samples for seed, samples in samples_by_seed.items()}
    groups["pooled"] = {
        key: np.concatenate([sample[key] for sample in samples_by_seed.values()])
        for key in next(iter(samples_by_seed.values()))
    }
    pairs = list(itertools.combinations(POLICY_NAMES, 2))
    # Put rule-aware UCB on the left for the primary gate comparisons.
    pairs = [(b, a) if b == "ucb_rule_aware" else (a, b) for a, b in pairs]
    for seed, samples in groups.items():
        for kappa in config.kappa_grid:
            for left, right in pairs:
                for metric in ("return", "distinct"):
                    difference = (
                        samples[f"{kappa}|{left}|{metric}"]
                        - samples[f"{kappa}|{right}|{metric}"]
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "kappa": kappa,
                            "left": left,
                            "right": right,
                            "metric": metric,
                            **mean_interval(difference),
                        }
                    )
    return rows


def run_pilot(config, output, batch_size=1000):
    output = Path(output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("The non-neural pilot must run locally on CPU")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            "Use an empty output directory; frozen pilot artifacts will not be overwritten"
        )
    metadata = provenance(config)
    metadata.update(
        {
            "expected_prior_max": expected_prior_max(config.d, config.mu, config.tau),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "backend": "cpu",
            "dtype": "float64",
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "batch_size": batch_size,
        }
    )
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True)
    )
    (output / "environment_hash.txt").write_text(metadata["environment_hash"] + "\n")
    write_json(output / "metadata.json", metadata)
    log("Pre-generating fixed validation episodes on CPU")
    validation = pregenerate(config, config.validation, batch_size)
    np.savez_compressed(
        output / f"episodes_{config.validation.seed}.npz", **validation._asdict()
    )
    frozen = tune(config, validation, output, metadata, batch_size)
    del validation
    frozen.update(
        {
            "environment_hash": metadata["environment_hash"],
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "validation_tables": {
                name: file_hash(output / name)
                for name in (
                    "ucb_validation.csv",
                    "explore_certify_validation.csv",
                    "nonadaptive_validation.csv",
                )
            },
        }
    )
    write_json(output / "frozen_hyperparameters.json", frozen)
    frozen_hash = file_hash(output / "frozen_hyperparameters.json")
    log("All hyperparameters frozen; test datasets have not yet been generated")
    rows, samples_by_seed = [], {}
    for dataset in config.tests:
        if file_hash(output / "frozen_hyperparameters.json") != frozen_hash:
            raise ValueError("Frozen hyperparameters changed during evaluation")
        log(
            f"Pre-generating and evaluating test seed {dataset.seed} ({dataset.episodes} episodes)"
        )
        data = pregenerate(config, dataset, batch_size)
        np.savez_compressed(output / f"episodes_{dataset.seed}.npz", **data._asdict())
        seed_rows, samples = evaluate_dataset(
            config, data, frozen, metadata, dataset, batch_size
        )
        rows.extend(seed_rows)
        samples_by_seed[dataset.seed] = samples
        np.savez_compressed(output / f"episode_metrics_{dataset.seed}.npz", **samples)
        del data
        log(f"Completed frozen evaluation for seed {dataset.seed}")
    write_table(output / "test_metrics_by_seed.csv", rows, metadata)
    write_table(
        output / "paired_comparisons.csv",
        comparisons(samples_by_seed, config),
        metadata,
    )
    # The source hash is audit evidence, but is not part of the later gate match.
    # Check it here to reject code edits made while a pilot is running.
    current = provenance(config)
    if (
        current["environment_hash"] != metadata["environment_hash"]
        or current["source_hash"] != metadata["source_hash"]
    ):
        raise ValueError(
            "Experiment source changed during the pilot; results cannot open the gate"
        )
    filenames = [
        "config_resolved.yaml",
        "environment_hash.txt",
        "metadata.json",
        "frozen_hyperparameters.json",
        "ucb_validation.csv",
        "explore_certify_validation.csv",
        "nonadaptive_validation.csv",
        "test_metrics_by_seed.csv",
        "paired_comparisons.csv",
    ]
    write_json(
        output / "result_manifest.json", make_manifest(output, filenames, metadata)
    )
    log("Pilot complete. Run check_gate.py to make the go/no-go decision.")
    return pd.DataFrame(rows)
