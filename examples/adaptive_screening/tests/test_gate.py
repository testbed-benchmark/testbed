import json
from dataclasses import replace

import pandas as pd
import pytest

from screening.config import Config, Dataset, provenance
from screening.gate import check_criteria, require_passing_gate
from screening.pilot import select_best


def passing_evidence():
    config = Config()
    metrics, paired = [], []
    for kappa in config.kappa_grid:
        for ds in config.tests:
            for policy in ("ucb_bayes", "ucb_rule_aware", "eig"):
                metrics.append(
                    {
                        "seed": ds.seed,
                        "episodes": ds.episodes,
                        "kappa": kappa,
                        "policy": policy,
                        "default": 0.4,
                        "disagreement": 0.2,
                    }
                )
        for seed in ("pooled", *(str(ds.seed) for ds in config.tests)):
            for other, mean in (
                ("ucb_bayes", 0.03),
                ("explore_certify", 0.03),
                ("nonadaptive", 0.1),
            ):
                paired.append(
                    {
                        "seed": seed,
                        "kappa": kappa,
                        "left": "ucb_rule_aware",
                        "right": other,
                        "metric": "return",
                        "mean": mean,
                        "se": 0.001,
                        "ci_low": mean - 0.00196,
                        "ci_high": mean + 0.00196,
                    }
                )
            paired.append(
                {
                    "seed": seed,
                    "kappa": kappa,
                    "left": "ucb_rule_aware",
                    "right": "ucb_bayes",
                    "metric": "distinct",
                    "mean": -2.0,
                    "se": 0.01,
                    "ci_low": -2.0196,
                    "ci_high": -1.9804,
                }
            )
    return config, pd.DataFrame(metrics), pd.DataFrame(paired)


def test_pair_order_and_second_pair_fallback():
    config, metrics, paired = passing_evidence()
    assert check_criteria(config, metrics, paired)["passing_pair"] == [2.0, 2.5]
    mask = (
        (paired.kappa == 2.0)
        & (paired.right == "ucb_bayes")
        & (paired.metric == "return")
    )
    paired.loc[mask, "mean"] = 0.01
    result = check_criteria(config, metrics, paired)
    assert result["passing_pair"] == [2.5, 3.0]
    assert result["candidate_pairs"][0]["pass"] is False


@pytest.mark.parametrize(
    "condition",
    [
        "pooled_gain",
        "seed_gain",
        "pooled_ci",
        "seed_sign",
        "etc",
        "static",
        "default_bayes",
        "default_aware",
        "breadth_size",
        "breadth_ci",
        "disagreement",
        "primary_protocol",
    ],
)
def test_each_gate_failure_is_enforced(condition):
    config, metrics, paired = passing_evidence()
    bayes_return = (paired.right == "ucb_bayes") & (paired.metric == "return")
    if condition == "pooled_gain":
        paired.loc[bayes_return & (paired.seed == "pooled"), "mean"] = 0.01999
    elif condition == "seed_gain":
        paired.loc[bayes_return & (paired.seed == "314159"), "mean"] = 0.01499
    elif condition == "pooled_ci":
        paired.loc[bayes_return & (paired.seed == "pooled"), "ci_low"] = 0.0
    elif condition == "seed_sign":
        paired.loc[bayes_return & (paired.seed == "161803"), "mean"] = -0.01
    elif condition == "etc":
        paired.loc[paired.right == "explore_certify", "mean"] = 0.01999
    elif condition == "static":
        paired.loc[paired.right == "nonadaptive", "ci_low"] = 0.0
    elif condition.startswith("default"):
        policy = "ucb_bayes" if condition.endswith("bayes") else "ucb_rule_aware"
        metrics.loc[metrics.policy == policy, "default"] = 0.85001
    elif condition == "breadth_size":
        paired.loc[paired.metric == "distinct", "mean"] = -0.99999
    elif condition == "breadth_ci":
        paired.loc[paired.metric == "distinct", "ci_high"] = 0.0
    elif condition == "disagreement":
        metrics.loc[metrics.policy == "eig", "disagreement"] = 0.09999
    else:
        config = replace(config, validation=Dataset(271828, 19999))
    result = check_criteria(config, metrics, paired)
    assert result["pass"] is False
    assert all(
        any(not check["pass"] for check in pair["checks"])
        for pair in result["candidate_pairs"]
    )


def test_inclusive_thresholds_and_either_etc_value():
    config, metrics, paired = passing_evidence()
    paired.loc[(paired.metric == "return") & (paired.right == "ucb_bayes"), "mean"] = (
        0.02
    )
    paired.loc[
        (paired.metric == "return")
        & (paired.seed == "314159")
        & (paired.right == "ucb_bayes"),
        "mean",
    ] = 0.015
    paired.loc[paired.metric == "distinct", "mean"] = -1.0
    paired.loc[paired.right == "explore_certify", "mean"] = 0.02
    paired.loc[(paired.right == "explore_certify") & (paired.kappa == 2), "mean"] = 0.0
    metrics["default"] = 0.85
    metrics["disagreement"] = 0.1
    assert check_criteria(config, metrics, paired)["passing_pair"] == [2.0, 2.5]


def test_validation_ties_choose_simpler_with_strict_tolerance():
    assert (
        select_best(
            [{"beta": 0.2, "J": 1.0}, {"beta": 0.4, "J": 1.00005}], ("beta",), 1e-4
        )["beta"]
        == 0.2
    )
    assert (
        select_best([{"k": 1, "J": 0.0}, {"k": 2, "J": 1e-4}], ("k",), 1e-4)["k"] == 2
    )
    rows = [
        {"k": 2, "S": 5, "h": "mean", "J": 1.0},
        {"k": 3, "S": 4, "h": "margin", "J": 1.0},
        {"k": 3, "S": 4, "h": "mean", "J": 1.0},
    ]
    winner = select_best(rows, ("S", "k", "h"), 1e-4)
    assert (winner["S"], winner["k"], winner["h"]) == (4, 3, "mean")


def test_provenance_includes_config_all_grids_datasets_commit_and_source():
    config = Config()
    identity = {"code_commit": "a", "source_hash": "b"}
    base = provenance(config, identity)
    assert base == provenance(config, identity)
    assert len(base["tuning_grids"]["explore_certify"]) == 496
    variants = [
        replace(config, sigma=0.8),
        replace(config, validation=Dataset(1, 20000)),
        replace(config, tests=(Dataset(314159, 49999), config.tests[1])),
        replace(config, beta_grid=(0.0, 1.0, 2.0)),
        replace(config, kappa_grid=(0.0, 2.0, 2.5)),
        replace(config, validation_tie_tolerance=1e-5),
    ]
    assert all(
        provenance(v, identity)["environment_hash"] != base["environment_hash"]
        for v in variants
    )
    changed_commit = provenance(config, {**identity, "code_commit": "x"})
    assert changed_commit["environment_hash"] != base["environment_hash"]
    assert base["source_hash"] == "b"


def test_post_gate_source_changes_do_not_change_environment_hash():
    config = Config()
    before = provenance(config, {"code_commit": "a", "source_hash": "pilot"})
    after = provenance(config, {"code_commit": "a", "source_hash": "pilot-plus-neural"})
    assert before["environment_hash"] == after["environment_hash"]
    assert before["source_hash"] != after["source_hash"]


def test_training_guard_rejects_missing_false_and_mismatched(tmp_path):
    path = tmp_path / "gate.json"
    with pytest.raises(ValueError, match="missing"):
        require_passing_gate(Config(), path)
    path.write_text(json.dumps({"pass": False}))
    with pytest.raises(ValueError, match="not passed"):
        require_passing_gate(Config(), path)
    path.write_text("[]")
    with pytest.raises(ValueError, match="not passed"):
        require_passing_gate(Config(), path)
    path.write_text(
        json.dumps({"pass": True, "environment_hash": "another-environment"})
    )
    with pytest.raises(ValueError, match="evidence is missing"):
        require_passing_gate(Config(), path)
