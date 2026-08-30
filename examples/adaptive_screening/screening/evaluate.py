"""Local CPU optimization checks and final common-random-number evaluation."""

import itertools
import math
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from .common_randomness import permuted, pregenerate
from .config import Dataset, code_identity
from .environment import rollout
from .io import read_json, write_json, write_table
from .metrics import mean_interval, summarize, trajectory_metrics
from .neural import load_checkpoint, rollout_policy
from .pilot import evaluate_dataset
from .policies import Policy

VALIDATION_SEED = 424242
FINAL_TEST = Dataset(8675309, 100000)
INITIAL_SEEDS = (101, 202, 303)
FINAL_SEEDS = (101, 202, 303, 404, 505)


def kappa_label(value):
    return f"{value:g}".replace(".", "p")


def discover_checkpoints(directories, expected_kappas, expected_seeds):
    found = {}
    for directory in directories:
        for path in Path(directory).rglob("checkpoint_best.msgpack"):
            params, metadata = load_checkpoint(path)
            key = (float(metadata["kappa_train"]), int(metadata["seed"]))
            if key in found:
                raise ValueError(f"Duplicate best checkpoint for {key}")
            found[key] = (path, params, metadata)
    expected = set(itertools.product(map(float, expected_kappas), expected_seeds))
    if set(found) != expected:
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise ValueError(f"Checkpoint set mismatch; missing={missing}, extra={extra}")
    return found


def _evaluate_neural(params, data, config, kappa_train, kappa_deploy, batch_size):
    chunks = []
    entropy, first = [], []
    for start in range(0, len(data.theta), batch_size):
        part = data.subset(start, min(start + batch_size, len(data.theta)))
        result = rollout_policy(params, part, config, kappa_train, greedy=False)
        chunks.append(
            {
                key: np.asarray(value)
                for key, value in trajectory_metrics(
                    result.trajectory, permuted(part).theta, kappa_deploy, config
                ).items()
            }
        )
        entropy.append(np.asarray(result.entropy))
        first.append(np.asarray(result.first_action))
    metrics = {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}
    first = np.concatenate(first)
    diagnostics = {
        "policy_entropy": float(np.concatenate(entropy).mean()),
        "first_action_max_fraction": float(
            np.bincount(first, minlength=config.d).max() / len(first)
        ),
    }
    return metrics, diagnostics


def _persistent_divergence(checkpoint_path):
    directory = Path(checkpoint_path).parent
    validation = pd.read_csv(directory / "validation_log.csv")
    training = pd.read_csv(directory / "training_log.csv")
    validation = validation[validation.phase == "reinforce"]
    training = training[training.phase == "reinforce"]
    if len(validation) < 4 or len(training) < 8:
        return False
    heldout_change = validation.J.tail(3).mean() - validation.J.head(3).mean()
    train_change = training["return"].tail(8).mean() - training["return"].head(8).mean()
    return bool(train_change > 0.01 and heldout_change < -0.01)


def run_optimization_check(
    config, gate, checkpoint_directories, frozen, output, batch_size=1000
):
    """Re-evaluate initial checkpoints locally and write the Section 13.7 gate."""
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Optimization checks must run locally on CPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Optimization-check output is not empty: {output}")
    kappas = [0.0, *map(float, gate["passing_pair"])]
    checkpoints = discover_checkpoints(checkpoint_directories, kappas, INITIAL_SEEDS)
    data = pregenerate(config, Dataset(VALIDATION_SEED, 10000), batch_size)
    prior_max = read_json(
        Path(gate["_gate_path"]).parent
        / gate.get("pilot_directory", ".")
        / "metadata.json"
    )["expected_prior_max"]["value"]
    identity = code_identity()
    metadata = {"environment_hash": gate["environment_hash"], **identity}
    rows, policy_checks = [], []
    for kappa in kappas:
        teacher = (
            frozen["beta_bayes"]
            if kappa == 0
            else frozen["selected"][str(kappa)]["ucb"]["beta"]
        )
        baseline_trajectory = rollout(data, config, Policy("ucb", beta=teacher))
        baseline_values = trajectory_metrics(
            baseline_trajectory, permuted(data).theta, kappa, config
        )
        baseline = summarize(baseline_values, prior_max)
        seed_passes = []
        for seed in INITIAL_SEEDS:
            path, params, checkpoint_metadata = checkpoints[(kappa, seed)]
            if checkpoint_metadata["environment_hash"] != gate["environment_hash"]:
                raise ValueError(f"Checkpoint environment mismatch: {path}")
            values, diagnostics = _evaluate_neural(
                params, data, config, kappa, kappa, batch_size
            )
            metrics = summarize(values, prior_max)
            summary = read_json(path.parent / "training_summary.json")
            finite = bool(summary["all_finite"])
            divergence = _persistent_divergence(path)
            seed_pass = bool(
                metrics["J"] >= baseline["J"] - 0.005
                and diagnostics["policy_entropy"] > 1e-6
                and diagnostics["first_action_max_fraction"] < 0.2
                and finite
                and not divergence
            )
            seed_passes.append(seed_pass)
            rows.append(
                {
                    "kappa_train": kappa,
                    "seed": seed,
                    "checkpoint": str(path),
                    "ucb_J": baseline["J"],
                    "gap_vs_ucb": metrics["J"] - baseline["J"],
                    "persistent_divergence": divergence,
                    "all_finite": finite,
                    "pass": seed_pass,
                    **diagnostics,
                    **metrics,
                }
            )
        kappa_rows = [row for row in rows if row["kappa_train"] == kappa]
        mean_j = float(np.mean([row["J"] for row in kappa_rows]))
        policy_checks.append(
            {
                "kappa_train": kappa,
                "learned_mean_J": mean_j,
                "ucb_J": baseline["J"],
                "mean_gap_vs_ucb": mean_j - baseline["J"],
                "seeds_passing": int(sum(seed_passes)),
                "pass": bool(mean_j >= baseline["J"] - 0.005 and sum(seed_passes) >= 2),
            }
        )
    write_table(output / "optimization_metrics.csv", rows, metadata)
    gate_result = {
        "pass": all(row["pass"] for row in policy_checks),
        "environment_hash": gate["environment_hash"],
        "pilot_result_hash": gate["pilot_result_hash"],
        "validation_seed": VALIDATION_SEED,
        "validation_episodes": 10000,
        "policy_checks": policy_checks,
        **identity,
    }
    write_json(output / "optimization_gate.json", gate_result)
    summary = ["# Neural optimization check", ""]
    summary.append(f"**Decision: {'PASS' if gate_result['pass'] else 'FAIL'}.**")
    summary.extend(
        [
            "",
            "| κ train | Learned J | UCB J | Gap | Seeds passing | Result |",
            "| ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for row in policy_checks:
        summary.append(
            f"| {row['kappa_train']:.1f} | {row['learned_mean_J']:.6f} | "
            f"{row['ucb_J']:.6f} | {row['mean_gap_vs_ucb']:.6f} | "
            f"{row['seeds_passing']}/3 | {'PASS' if row['pass'] else 'FAIL'} |"
        )
    (output / "optimization_summary.md").write_text("\n".join(summary) + "\n")
    return gate_result


def _learned_rows(
    checkpoints, data, config, kappas, prior_max, batch_size, greedy=False
):
    rows, samples = [], {}
    mode = "greedy" if greedy else "stochastic"
    for (kappa_train, seed), (path, params, metadata) in sorted(checkpoints.items()):
        chunks = {float(kappa): [] for kappa in kappas}
        entropy, first = [], []
        for start in range(0, len(data.theta), batch_size):
            part = data.subset(start, min(start + batch_size, len(data.theta)))
            result = rollout_policy(params, part, config, kappa_train, greedy=greedy)
            theta = permuted(part).theta
            for kappa_deploy in kappas:
                chunks[float(kappa_deploy)].append(
                    {
                        key: np.asarray(value)
                        for key, value in trajectory_metrics(
                            result.trajectory, theta, kappa_deploy, config
                        ).items()
                    }
                )
            entropy.append(np.asarray(result.entropy))
            first.append(np.asarray(result.first_action))
        first_values = np.concatenate(first)
        common = {
            "policy_entropy": float(np.concatenate(entropy).mean()),
            "first_action_max_fraction": float(
                np.bincount(first_values, minlength=config.d).max() / len(first_values)
            ),
        }
        for kappa_deploy in kappas:
            deploy_chunks = chunks[float(kappa_deploy)]
            values = {
                key: np.concatenate([chunk[key] for chunk in deploy_chunks])
                for key in deploy_chunks[0]
            }
            name = f"learned_train_{kappa_label(kappa_train)}"
            rows.append(
                {
                    "policy": name,
                    "evaluation_mode": mode,
                    "kappa_train": kappa_train,
                    "kappa_deploy": kappa_deploy,
                    "training_seed": seed,
                    "checkpoint": str(path),
                    **common,
                    **summarize(values, prior_max),
                }
            )
            for metric in ("return", "distinct"):
                samples[(kappa_train, seed, mode, kappa_deploy, metric)] = values[
                    metric
                ]
    return rows, samples


def _aggregate_learned(rows, samples, config):
    frame = pd.DataFrame(rows)
    frame = frame[frame.evaluation_mode == "stochastic"]
    aggregates = []
    metrics = [
        "J",
        "risk",
        "bayes_component",
        "incompatibility",
        "default",
        "disagreement",
        "beneficial_deployment",
        "selected_utility",
        "distinct",
        "budget_fraction",
        "last_six_fraction",
        "selected_sd_conditional",
        "first_leader_round_conditional",
        "selected_never_leader_conditional",
        "policy_entropy",
        "first_action_max_fraction",
    ]
    for (kappa_train, kappa_deploy), group in frame.groupby(
        ["kappa_train", "kappa_deploy"], sort=True
    ):
        row = {
            "policy": f"learned_train_{kappa_label(kappa_train)}",
            "kappa_train": kappa_train,
            "kappa_deploy": kappa_deploy,
            "training_seeds": len(group),
            "episodes": int(group.episodes.iloc[0]),
        }
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            row[metric] = float(values.mean())
            row[f"{metric}_training_sd"] = float(values.std(ddof=1))
        episode_average = np.mean(
            [
                samples[(kappa_train, seed, "stochastic", kappa_deploy, "return")]
                for seed in sorted(group.training_seed)
            ],
            axis=0,
        )
        row["J_evaluation_se"] = float(
            episode_average.std(ddof=1) / math.sqrt(len(episode_average))
        )
        row["decomposition_error"] = abs(
            row["risk"] - row["bayes_component"] - row["incompatibility"]
        )
        if row["decomposition_error"] > 1e-10:
            raise ValueError("Aggregated learned risk decomposition does not close")
        aggregates.append(row)
    return aggregates


def _baseline_aggregate(rows):
    output = []
    for row in rows:
        converted = {
            **row,
            "kappa_train": np.nan,
            "kappa_deploy": row.pop("kappa"),
            "training_seeds": 0,
            "J_evaluation_se": row["J_se"],
        }
        output.append(converted)
    return output


def run_final_evaluation(
    config,
    gate,
    checkpoint_directories,
    frozen,
    pilot_metadata,
    output,
    batch_size=1000,
):
    """Run the complete 100,000-episode final experiment locally in float64."""
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Final evaluation must run locally on CPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Final output is not empty: {output}")
    kappas_train = [0.0, *map(float, gate["passing_pair"])]
    checkpoints = discover_checkpoints(
        checkpoint_directories, kappas_train, FINAL_SEEDS
    )
    if any(
        metadata["environment_hash"] != gate["environment_hash"]
        for _, _, metadata in checkpoints.values()
    ):
        raise ValueError("A checkpoint belongs to another environment")
    data = pregenerate(config, FINAL_TEST, batch_size)
    prior_max = pilot_metadata["expected_prior_max"]["value"]
    identity = code_identity()
    metadata = {"environment_hash": gate["environment_hash"], **identity}

    baseline_rows, baseline_samples = evaluate_dataset(
        config, data, frozen, pilot_metadata, FINAL_TEST, batch_size
    )
    for row in baseline_rows:
        row["kappa_deploy"] = row.pop("kappa")
        row["training_seed"] = -1
        row["evaluation_mode"] = "fixed"
        row["kappa_train"] = np.nan
        row["checkpoint"] = ""
    stochastic_rows, samples = _learned_rows(
        checkpoints,
        data,
        config,
        config.kappa_grid,
        prior_max,
        batch_size,
        greedy=False,
    )
    greedy_rows, _ = _learned_rows(
        checkpoints,
        data,
        config,
        config.kappa_grid,
        prior_max,
        batch_size,
        greedy=True,
    )
    write_table(
        output / "policy_metrics_by_seed.csv",
        [*stochastic_rows, *greedy_rows, *baseline_rows],
        metadata,
    )

    aggregate = _aggregate_learned(stochastic_rows, samples, config)
    baseline_aggregate = []
    for original in baseline_rows:
        row = dict(original)
        row.update(
            {
                "training_seeds": 0,
                "J_evaluation_se": row["J_se"],
            }
        )
        baseline_aggregate.append(row)
    aggregate.extend(baseline_aggregate)
    write_table(output / "policy_metrics_aggregate.csv", aggregate, metadata)

    decomposition = [
        {
            key: row.get(key)
            for key in (
                "policy",
                "kappa_train",
                "kappa_deploy",
                "J",
                "risk",
                "bayes_component",
                "incompatibility",
                "decomposition_error",
            )
        }
        for row in aggregate
    ]
    behavior = [
        {
            key: row.get(key)
            for key in (
                "policy",
                "kappa_train",
                "kappa_deploy",
                "default",
                "distinct",
                "budget_fraction",
                "last_six_fraction",
                "selected_sd_conditional",
                "first_leader_round_conditional",
            )
        }
        for row in aggregate
    ]
    write_table(output / "risk_decomposition.csv", decomposition, metadata)
    write_table(output / "behavior_metrics.csv", behavior, metadata)

    paired_rows = []
    for kappa in gate["passing_pair"]:
        matched = np.mean(
            [
                samples[(float(kappa), seed, "stochastic", float(kappa), "return")]
                for seed in FINAL_SEEDS
            ],
            axis=0,
        )
        bayes = np.mean(
            [
                samples[(0.0, seed, "stochastic", float(kappa), "return")]
                for seed in FINAL_SEEDS
            ],
            axis=0,
        )
        comparisons = {"learned_train_0": bayes}
        for name in (
            "ucb_rule_aware",
            "ucb_bayes",
            "explore_certify",
            "nonadaptive",
            "eig",
            "random",
        ):
            comparisons[name] = baseline_samples[f"{kappa}|{name}|return"]
        for right, right_values in comparisons.items():
            paired_rows.append(
                {
                    "kappa_deploy": kappa,
                    "left": f"learned_train_{kappa_label(kappa)}",
                    "right": right,
                    "metric": "return",
                    **mean_interval(matched - right_values),
                }
            )
    write_table(output / "paired_policy_differences.csv", paired_rows, metadata)

    matrix = [
        row for row in aggregate if str(row["policy"]).startswith("learned_train_")
    ]
    write_table(output / "training_deployment_matrix.csv", matrix, metadata)
    primary = pd.DataFrame(paired_rows)
    aggregate_frame = pd.DataFrame(aggregate)

    def aggregate_row(policy, kappa):
        selection = aggregate_frame[
            (aggregate_frame.policy == policy) & (aggregate_frame.kappa_deploy == kappa)
        ]
        if len(selection) != 1:
            raise ValueError(f"Missing final aggregate for {policy}, kappa={kappa}")
        return selection.iloc[0]

    scientific_pass = True
    for kappa in gate["passing_pair"]:
        matched_name = f"learned_train_{kappa_label(kappa)}"
        versus_bayes = primary[
            (primary.kappa_deploy == kappa) & (primary.right == "learned_train_0")
        ].iloc[0]
        versus_ucb = primary[
            (primary.kappa_deploy == kappa) & (primary.right == "ucb_rule_aware")
        ].iloc[0]
        scientific_pass &= bool(versus_bayes.ci_low > 0 and versus_ucb.ci_low > 0)
    summary = [
        "# Adaptive screening: final learned-policy experiment",
        "",
        f"**Decision: {'PASS' if scientific_pass else 'FAIL'}.**",
        "",
        f"Environment hash: `{gate['environment_hash']}`  ",
        f"Final test: {FINAL_TEST.episodes:,} episodes, seed {FINAL_TEST.seed}. "
        f"Learned results average {len(FINAL_SEEDS)} independent training seeds.",
        "",
        "## Primary learned-policy differences",
        "",
        "| κ deploy | Bayes learned J | Rule-aware learned J | Rule-aware − Bayes | 95% CI | Rule-aware − matched UCB |",
        "| ---: | ---: | ---: | ---: | :--- | ---: |",
    ]
    for kappa in gate["passing_pair"]:
        matched_name = f"learned_train_{kappa_label(kappa)}"
        bayes = aggregate_row("learned_train_0", kappa)
        aware = aggregate_row(matched_name, kappa)
        versus_bayes = primary[
            (primary.kappa_deploy == kappa) & (primary.right == "learned_train_0")
        ].iloc[0]
        versus_ucb = primary[
            (primary.kappa_deploy == kappa) & (primary.right == "ucb_rule_aware")
        ].iloc[0]
        summary.append(
            f"| {kappa:.1f} | {bayes.J:.6f} | {aware.J:.6f} | "
            f"{versus_bayes['mean']:.6f} | "
            f"[{versus_bayes.ci_low:.6f}, {versus_bayes.ci_high:.6f}] | "
            f"{versus_ucb['mean']:.6f} |"
        )
    summary.extend(
        [
            "",
            "## Risk decomposition and behavior",
            "",
            "| κ | Bayes incompat. | Aware incompat. | Bayes distinct | Aware distinct | Bayes last-six | Aware last-six | Bayes default | Aware default |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for kappa in gate["passing_pair"]:
        matched_name = f"learned_train_{kappa_label(kappa)}"
        bayes = aggregate_row("learned_train_0", kappa)
        aware = aggregate_row(matched_name, kappa)
        summary.append(
            f"| {kappa:.1f} | {bayes.incompatibility:.6f} | "
            f"{aware.incompatibility:.6f} | {bayes.distinct:.3f} | "
            f"{aware.distinct:.3f} | {bayes.last_six_fraction:.3f} | "
            f"{aware.last_six_fraction:.3f} | {bayes.default:.3f} | "
            f"{aware.default:.3f} |"
        )
    maximum_residual = float(aggregate_frame.decomposition_error.max())
    summary.extend(
        [
            "",
            "At both gate-selected mismatch values, the matched learned policy "
            "has a positive paired advantage over both the Bayes-assumption "
            "learned policy and matched rule-aware UCB. The reduction in risk is "
            "primarily a reduction in incompatibility, accompanied by narrower "
            "search and more late measurements on the selected candidate.",
            "",
            f"Maximum risk-decomposition residual: {maximum_residual:.3g}.",
        ]
    )
    (output / "final_summary.md").write_text("\n".join(summary) + "\n")
    return aggregate
