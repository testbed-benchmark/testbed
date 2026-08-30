import json

import numpy as np
import pandas as pd
import pytest

from screening.common_randomness import generate, permuted
from screening.environment import rollout
from screening.gate import check_gate, require_passing_gate, validate_tables
from screening.io import read_json, verify_manifest
from screening.metrics import summarize, trajectory_metrics
from screening.pilot import POLICY_NAMES, run_pilot
from screening.policies import Policy
from screening.quadrature import expected_prior_max


def test_all_policies_256_episodes_all_kappas(small_config):
    config = small_config
    data = generate(config, 9812, 256)
    policies = [
        Policy("eig"),
        Policy("random"),
        Policy("ucb", beta=2.0),
        Policy("nonadaptive", k=3),
    ]
    prior = expected_prior_max(config.d, config.mu, config.tau)["value"]
    for kappa in config.kappa_grid:
        for policy in [
            *policies,
            Policy("explore_certify", k=3, S=4, h="mean", kappa=kappa),
            Policy("explore_certify", k=3, S=4, h="margin", kappa=kappa),
        ]:
            trajectory = rollout(data, config, policy)
            for array in trajectory:
                assert np.isfinite(array).all()
            assert (trajectory.sd > 0).all()
            assert (trajectory.actions >= 0).all() and (
                trajectory.actions < config.d
            ).all()
            metrics = trajectory_metrics(
                trajectory, permuted(data).theta, kappa, config
            )
            assert (metrics["action"] >= 0).all() and (
                metrics["action"] <= config.d
            ).all()
            assert summarize(metrics, prior)["decomposition_error"] < 1e-10


def test_complete_smoke_outputs_frozen_tuning_and_fail_closed(small_config, tmp_path):
    output = tmp_path / "pilot"
    run_pilot(small_config, output, batch_size=256)
    manifest = verify_manifest(output)
    result = check_gate(output)
    assert result["pass"] is False  # A small smoke run can never authorize training.
    assert "candidate_pairs" in result, result
    assert result["pilot_result_hash"] == manifest["pilot_result_hash"]
    required = [
        "config_resolved.yaml",
        "environment_hash.txt",
        "ucb_validation.csv",
        "explore_certify_validation.csv",
        "nonadaptive_validation.csv",
        "test_metrics_by_seed.csv",
        "paired_comparisons.csv",
        "pilot_summary.md",
        "pilot_gate.json",
        "pilot_failure_report.md",
        "pilot_overview.png",
        "validation_tuning_curves.png",
    ]
    assert all((output / name).is_file() for name in required)
    metadata = read_json(output / "metadata.json")
    metrics, paired = validate_tables(output, small_config, metadata)
    assert len(metrics) == 2 * 6 * 6
    assert set(metrics.policy) == set(POLICY_NAMES)
    assert (metrics.episodes == 256).all()
    assert {
        "J",
        "risk",
        "bayes_component",
        "incompatibility",
        "default",
        "distinct",
        "budget_fraction",
        "last_six_fraction",
        "selected_sd_conditional",
        "first_leader_round_conditional",
        "environment_hash",
        "code_commit",
        "source_hash",
    } <= set(metrics.columns)
    assert {"mean", "se", "ci_low", "ci_high", "episodes"} <= set(paired.columns)
    zeros = paired[
        (paired.kappa == 0)
        & (paired.left == "ucb_rule_aware")
        & (paired.right == "ucb_bayes")
    ]
    np.testing.assert_array_equal(zeros["mean"], 0)
    np.testing.assert_array_equal(zeros["se"], 0)
    with pytest.raises(FileExistsError):
        run_pilot(small_config, output)
    # A clean checkout need not carry the large reproducible episode caches.
    for path in output.glob("*.npz"):
        path.unlink()
    without_caches = check_gate(output)
    assert without_caches["pass"] is False
    assert "candidate_pairs" in without_caches and "error" not in without_caches
    # A forged boolean cannot turn smoke data into a passing gate.
    result["pass"] = True
    result["passing_pair"] = [2.5, 3.0]
    (output / "pilot_gate.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="verified pilot evidence"):
        require_passing_gate(small_config, output / "pilot_gate.json")
    # Mutating a frozen table invalidates its content hash and closes the gate.
    with (output / "paired_comparisons.csv").open("a") as stream:
        stream.write("\n")
    result = check_gate(output)
    assert result["pass"] is False and "hash mismatch" in result["error"]


def test_paired_pooled_se_is_from_episode_differences(small_config):
    from screening.pilot import comparisons

    samples = {}
    for ds in small_config.tests:
        rng = np.random.default_rng(ds.seed)
        samples[ds.seed] = {
            f"{k}|{p}|{m}": rng.normal(size=ds.episodes)
            for k in small_config.kappa_grid
            for p in POLICY_NAMES
            for m in ("return", "distinct")
        }
    frame = pd.DataFrame(comparisons(samples, small_config))
    row = frame[
        (frame.seed == "pooled")
        & (frame.kappa == 2.5)
        & (frame.left == "ucb_rule_aware")
        & (frame.right == "ucb_bayes")
        & (frame.metric == "return")
    ].iloc[0]
    differences = np.concatenate(
        [
            data["2.5|ucb_rule_aware|return"] - data["2.5|ucb_bayes|return"]
            for data in samples.values()
        ]
    )
    assert row["mean"] == differences.mean()
    assert row.se == differences.std(ddof=1) / np.sqrt(len(differences))
