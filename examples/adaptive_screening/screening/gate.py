"""Fail-closed implementation of every Section 11 gate criterion."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config, provenance
from .io import read_json, verify_manifest, write_json
from .pilot import POLICY_NAMES, select_best

PAIRS = ((2.0, 2.5), (2.5, 3.0))


def paired_row(frame, kappa, left, right, metric="return", seed="pooled"):
    selection = frame[
        (frame.kappa == kappa)
        & (frame.left == left)
        & (frame.right == right)
        & (frame.metric == metric)
        & (frame.seed.astype(str) == str(seed))
    ]
    if len(selection) != 1:
        raise ValueError(
            f"Missing/duplicate paired comparison: {seed}, {kappa}, {left}, {right}, {metric}"
        )
    return selection.iloc[0]


def pooled_metric(frame, kappa, policy, metric):
    selection = frame[(frame.kappa == kappa) & (frame.policy == policy)]
    return float(np.average(selection[metric], weights=selection.episodes))


def check_criteria(config, metrics, paired):
    decisions = []
    for low, high in PAIRS:
        checks = []

        def add(name, passed, actual, requirement, kappa=None, seed=None):
            checks.append(
                {
                    "name": name,
                    "pass": bool(passed),
                    "actual": actual,
                    "requirement": requirement,
                    "kappa": kappa,
                    "seed": seed,
                }
            )

        add(
            "primary_protocol",
            config == Config(),
            config == Config(),
            "Exact preregistered primary configuration, datasets and tuning grids",
        )
        if low not in config.kappa_grid or high not in config.kappa_grid:
            add(
                "pair_available",
                False,
                list(config.kappa_grid),
                f"Both {low} and {high} must be evaluated",
            )
            decisions.append({"pair": [low, high], "pass": False, "checks": checks})
            continue
        etc_checks = []
        for kappa in (low, high):
            gain = paired_row(paired, kappa, "ucb_rule_aware", "ucb_bayes")
            add(
                "pooled_rule_awareness_gain",
                gain["mean"] >= 0.02,
                float(gain["mean"]),
                ">= 0.02",
                kappa,
            )
            add(
                "pooled_rule_awareness_ci",
                gain.ci_low > 0,
                float(gain.ci_low),
                "95% paired CI lower > 0",
                kappa,
            )
            for dataset in config.tests:
                seed_gain = paired_row(
                    paired, kappa, "ucb_rule_aware", "ucb_bayes", seed=dataset.seed
                )
                add(
                    "seed_rule_awareness_gain",
                    seed_gain["mean"] >= 0.015,
                    float(seed_gain["mean"]),
                    ">= 0.015",
                    kappa,
                    dataset.seed,
                )
                add(
                    "no_seed_sign_reversal",
                    seed_gain["mean"] > 0,
                    float(seed_gain["mean"]),
                    "> 0",
                    kappa,
                    dataset.seed,
                )
            etc = paired_row(paired, kappa, "ucb_rule_aware", "explore_certify")
            etc_checks.append(
                {
                    "kappa": kappa,
                    "mean": float(etc["mean"]),
                    "ci_low": float(etc.ci_low),
                    "pass": bool(etc["mean"] >= 0.02 and etc.ci_low > 0),
                }
            )
            static = paired_row(paired, kappa, "ucb_rule_aware", "nonadaptive")
            add(
                "adaptive_over_static",
                static["mean"] > 0 and static.ci_low > 0,
                {"mean": float(static["mean"]), "ci_low": float(static.ci_low)},
                "Mean and 95% paired CI lower > 0",
                kappa,
            )
            for policy in ("ucb_bayes", "ucb_rule_aware"):
                rate = pooled_metric(metrics, kappa, policy, "default")
                add(
                    f"non_saturated_{policy}",
                    rate <= 0.85,
                    rate,
                    "Default rate <= 0.85",
                    kappa,
                )
        add(
            "adaptive_over_explore_certify",
            any(c["pass"] for c in etc_checks),
            etc_checks,
            "At either pair value, gain >= 0.02 and 95% paired CI lower > 0",
        )
        breadth = paired_row(
            paired, high, "ucb_rule_aware", "ucb_bayes", metric="distinct"
        )
        add(
            "reduced_search_breadth",
            breadth["mean"] <= -1 and breadth.ci_high < 0,
            {"mean": float(breadth["mean"]), "ci_high": float(breadth.ci_high)},
            "Aware minus Bayes <= -1 and 95% paired CI upper < 0",
            high,
        )
        disagreement = pooled_metric(metrics, high, "eig", "disagreement")
        add(
            "eig_action_disagreement",
            disagreement >= 0.1,
            disagreement,
            ">= 0.10",
            high,
        )
        decisions.append(
            {
                "pair": [low, high],
                "pass": all(c["pass"] for c in checks),
                "checks": checks,
            }
        )
    passing = next(
        (decision["pair"] for decision in decisions if decision["pass"]), None
    )
    return {
        "pass": passing is not None,
        "passing_pair": passing,
        "candidate_pairs": decisions,
    }


def validate_tables(directory, config, metadata):
    directory = Path(directory)
    metrics = pd.read_csv(
        directory / "test_metrics_by_seed.csv", float_precision="round_trip"
    )
    paired = pd.read_csv(
        directory / "paired_comparisons.csv",
        dtype={"seed": str},
        float_precision="round_trip",
    )
    tables = {
        name: pd.read_csv(directory / name, float_precision="round_trip")
        for name in (
            "ucb_validation.csv",
            "explore_certify_validation.csv",
            "nonadaptive_validation.csv",
        )
    }
    for frame in [metrics, paired, *tables.values()]:
        for key in ("environment_hash", "code_commit", "source_hash"):
            if set(frame[key]) != {metadata[key]}:
                raise ValueError(f"Table provenance mismatch: {key}")
    expected = {
        (ds.seed, k, p): ds.episodes
        for ds in config.tests
        for k in config.kappa_grid
        for p in POLICY_NAMES
    }
    if (
        len(metrics) != len(expected)
        or metrics.duplicated(["seed", "kappa", "policy"]).any()
    ):
        raise ValueError("Incomplete or duplicate frozen test metrics")
    for row in metrics.itertuples():
        if expected.get((row.seed, row.kappa, row.policy)) != row.episodes:
            raise ValueError("Unexpected test seed, policy, kappa, or episode count")
    required = [
        "J",
        "risk",
        "bayes_component",
        "incompatibility",
        "default",
        "disagreement",
        "distinct",
        "episodes",
    ]
    if not np.isfinite(metrics[required].to_numpy()).all():
        raise ValueError("Nonfinite test metrics")
    if (
        np.max(np.abs(metrics.risk - metrics.bayes_component - metrics.incompatibility))
        > 1e-10
    ):
        raise ValueError("Test risk decomposition failed")
    if (
        len(paired) != 3 * len(config.kappa_grid) * 15 * 2
        or paired.duplicated(["seed", "kappa", "left", "right", "metric"]).any()
    ):
        raise ValueError("Incomplete or duplicate paired comparisons")
    if not np.isfinite(
        paired[["episodes", "mean", "se", "ci_low", "ci_high"]].to_numpy()
    ).all():
        raise ValueError("Nonfinite paired comparisons")
    expected_n = {str(ds.seed): ds.episodes for ds in config.tests}
    expected_n["pooled"] = sum(ds.episodes for ds in config.tests)
    for row in paired.itertuples():
        if expected_n.get(row.seed) != row.episodes or row.se < 0:
            raise ValueError("Wrong paired sample count or standard error")
        if (
            abs(row.ci_low - (row.mean - 1.96 * row.se)) > 1e-12
            or abs(row.ci_high - (row.mean + 1.96 * row.se)) > 1e-12
        ):
            raise ValueError("Invalid paired confidence interval")
    grids = config.tuning_grids()
    definitions = {
        "ucb_validation.csv": (
            "ucb",
            ("beta",),
            [{"beta": beta} for beta in config.beta_grid],
        ),
        "explore_certify_validation.csv": (
            "explore_certify",
            ("S", "k", "h"),
            grids["explore_certify"],
        ),
        "nonadaptive_validation.csv": (
            "nonadaptive",
            ("k",),
            [{"k": k} for k in grids["nonadaptive_k"]],
        ),
    }
    frozen = read_json(directory / "frozen_hyperparameters.json")
    if frozen["environment_hash"] != metadata["environment_hash"]:
        raise ValueError("Frozen hyperparameter provenance mismatch")
    from .io import file_hash

    for filename, (name, fields, grid) in definitions.items():
        frame = tables[filename]
        grid_expected = {
            (kappa, *(point[field] for field in fields))
            for point in grid
            for kappa in config.kappa_grid
        }
        actual = list(frame[["kappa", *fields]].itertuples(index=False, name=None))
        if len(actual) != len(grid_expected) or set(actual) != grid_expected:
            raise ValueError(f"Incomplete validation grid: {filename}")
        if (frame.episodes != config.validation.episodes).any() or not np.isfinite(
            frame[["J", "J_se"]]
        ).all().all():
            raise ValueError("Invalid validation metrics")
        if frozen["validation_tables"][filename] != file_hash(directory / filename):
            raise ValueError("Validation table changed after freezing")
        for kappa in config.kappa_grid:
            best = select_best(
                frame[frame.kappa == kappa].to_dict("records"),
                fields,
                config.validation_tie_tolerance,
            )
            if any(
                best[field] != frozen["selected"][str(kappa)][name][field]
                for field in fields
            ):
                raise ValueError(
                    "Frozen hyperparameters were not selected by validation tie rules"
                )
    if frozen["beta_bayes"] != frozen["selected"]["0.0"]["ucb"]["beta"]:
        raise ValueError("Bayes coefficient was not frozen at zero mismatch")
    return metrics, paired


def load_evidence(directory, config=None):
    directory = Path(directory)
    manifest = verify_manifest(directory)
    recorded_config = load_config(directory / "config_resolved.yaml")
    required_files = {
        "config_resolved.yaml",
        "environment_hash.txt",
        "metadata.json",
        "frozen_hyperparameters.json",
        "ucb_validation.csv",
        "explore_certify_validation.csv",
        "nonadaptive_validation.csv",
        "test_metrics_by_seed.csv",
        "paired_comparisons.csv",
    }
    if set(manifest["files"]) != required_files:
        raise ValueError("Pilot manifest does not cover all required evidence files")
    config = recorded_config if config is None else config
    metadata = read_json(directory / "metadata.json")
    # Reconstruct the certified environment with the revision and source
    # snapshot recorded when the pilot ran. Later commits may add training code
    # or commit these artifacts without invalidating the evidence.
    recorded_identity = {
        "code_commit": metadata["code_commit"],
        "source_hash": metadata["source_hash"],
    }
    expected_hash = provenance(config, recorded_identity)["environment_hash"]
    hashes = [
        metadata["environment_hash"],
        manifest["environment_hash"],
        (directory / "environment_hash.txt").read_text().strip(),
    ]
    if config != recorded_config or any(h != expected_hash for h in hashes):
        raise ValueError(
            "Environment hash mismatch; a gate from another configuration or code version is invalid"
        )
    metrics, paired = validate_tables(directory, config, metadata)
    return config, metadata, manifest, metrics, paired


def check_gate(directory, output=None):
    directory = Path(directory)
    output = directory / "pilot_gate.json" if output is None else Path(output)
    try:
        config, metadata, manifest, metrics, paired = load_evidence(directory)
        result = {
            **check_criteria(config, metrics, paired),
            "environment_hash": metadata["environment_hash"],
            "pilot_result_hash": manifest["pilot_result_hash"],
            "code_commit": metadata["code_commit"],
            "source_hash": metadata["source_hash"],
            "pilot_directory": os.path.relpath(
                directory.resolve(), output.parent.resolve()
            ),
        }
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"pass": False, "passing_pair": None, "error": str(error)}
        write_json(output, result)
        (directory / "pilot_failure_report.md").write_text(
            f"# Pilot integrity failure\n\n{error}\n\nNeural training disabled. Correct the implementation/data issue before scientific review.\n"
        )
        return result
    write_json(output, result)
    from .reports import write_reports

    write_reports(directory, config, metadata, metrics, paired, result)
    return result


def require_passing_gate(config, gate_path):
    """Future training must call this BEFORE importing/initializing neural code."""
    gate_path = Path(gate_path)
    try:
        gate = read_json(gate_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            "NEURAL TRAINING DISABLED: missing or unreadable gate"
        ) from error
    if not isinstance(gate, dict) or gate.get("pass") is not True:
        raise ValueError("NEURAL TRAINING DISABLED: pilot gate has not passed")
    directory = gate_path.parent / gate.get("pilot_directory", ".")
    try:
        _, _, manifest, metrics, paired = load_evidence(directory, config)
    except (ValueError, KeyError, OSError, TypeError) as error:
        raise ValueError(
            "NEURAL TRAINING DISABLED: pilot evidence is missing, invalid, or incompatible"
        ) from error
    if gate.get("environment_hash") != manifest["environment_hash"]:
        raise ValueError("NEURAL TRAINING DISABLED: environment hash mismatch")
    actual = check_criteria(config, metrics, paired)
    if (
        not actual["pass"]
        or actual["passing_pair"] != gate.get("passing_pair")
        or manifest["pilot_result_hash"] != gate.get("pilot_result_hash")
    ):
        raise ValueError(
            "NEURAL TRAINING DISABLED: gate does not match verified pilot evidence"
        )
    return [0.0, *actual["passing_pair"]]
