"""Non-neural diagnostic reports and tuning curves; no paper claims about RL."""

import json

import numpy as np
import pandas as pd

from .io import read_json


def markdown_table(headers, rows):
    def cell(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(directory, config, metadata, metrics, paired, result):
    from .gate import paired_row, pooled_metric

    frozen = read_json(directory / "frozen_hyperparameters.json")
    lines = [
        "# Adaptive screening: non-neural go/no-go pilot",
        "",
        f"**Decision: {'PASS' if result['pass'] else 'FAIL — neural training disabled'}.**",
        f"Passing pair: {result['passing_pair']}.",
        "",
        f"Environment hash: `{metadata['environment_hash']}`  ",
        f"Code commit: `{metadata['code_commit']}`  ",
        f"Source snapshot hash (audit only; includes uncommitted experiment code): `{metadata['source_hash']}`",
        "",
        f"All computations ran locally on CPU in float64. Prior maximum from quadrature: **{metadata['expected_prior_max']['value']:.12f}**.",
        f"Validation: {config.validation.episodes:,} episodes, seed {config.validation.seed}. Tests: "
        + ", ".join(f"{ds.episodes:,} episodes, seed {ds.seed}" for ds in config.tests)
        + ".",
        "",
        "All configurations were tuned on validation, then frozen before either test dataset was generated. Confidence intervals use paired episode differences on the common latent utilities and candidate-specific observation noise. Pooled intervals concatenate both independent test sets; they do not average seed-level standard errors.",
        "",
        "## Frozen hyperparameters",
        "",
    ]
    hyper_rows = []
    for k in config.kappa_grid:
        p = frozen["selected"][str(k)]
        etc = p["explore_certify"]
        hyper_rows.append(
            [
                k,
                frozen["beta_bayes"],
                p["ucb"]["beta"],
                etc["k"],
                etc["S"],
                etc["h"],
                p["nonadaptive"]["k"],
            ]
        )
    lines += [
        markdown_table(
            ["κ", "Bayes β", "Aware β", "ETC k", "ETC S", "ETC leader", "Static k"],
            hyper_rows,
        ),
        "",
        "## Rule-awareness gains on frozen tests",
        "",
    ]
    gain_rows = []
    for k in config.kappa_grid:
        gain = paired_row(paired, k, "ucb_rule_aware", "ucb_bayes")
        seed_gains = [
            float(
                paired_row(paired, k, "ucb_rule_aware", "ucb_bayes", seed=ds.seed)[
                    "mean"
                ]
            )
            for ds in config.tests
        ]
        gain_rows.append(
            [
                k,
                pooled_metric(metrics, k, "ucb_bayes", "J"),
                pooled_metric(metrics, k, "ucb_rule_aware", "J"),
                gain["mean"],
                gain.se,
                gain.ci_low,
                gain.ci_high,
                *seed_gains,
            ]
        )
    lines += [
        markdown_table(
            [
                "κ",
                "Bayes J",
                "Aware J",
                "Gain",
                "Paired SE",
                "95% lower",
                "95% upper",
                *[str(ds.seed) for ds in config.tests],
            ],
            gain_rows,
        ),
        "",
        "## Adaptive advantage and behavior",
        "",
    ]
    behavior = []
    for k in config.kappa_grid:
        etc = paired_row(paired, k, "ucb_rule_aware", "explore_certify")
        static = paired_row(paired, k, "ucb_rule_aware", "nonadaptive")
        best_adaptive = max(
            pooled_metric(metrics, k, p, "J")
            for p in ("ucb_bayes", "ucb_rule_aware", "explore_certify")
        )
        behavior.append(
            [
                k,
                etc["mean"],
                etc.ci_low,
                static["mean"],
                static.ci_low,
                best_adaptive - pooled_metric(metrics, k, "nonadaptive", "J"),
                pooled_metric(metrics, k, "ucb_bayes", "default"),
                pooled_metric(metrics, k, "ucb_rule_aware", "default"),
                pooled_metric(metrics, k, "eig", "disagreement"),
                pooled_metric(metrics, k, "ucb_bayes", "distinct"),
                pooled_metric(metrics, k, "ucb_rule_aware", "distinct"),
            ]
        )
    lines += [
        markdown_table(
            [
                "κ",
                "Aware−ETC",
                "CI lower",
                "Aware−static",
                "CI lower",
                "Best frozen adaptive−static",
                "Bayes default",
                "Aware default",
                "EIG disagreement",
                "Bayes distinct",
                "Aware distinct",
            ],
            behavior,
        ),
        "",
        "The best frozen adaptive gap is descriptive only; no policy is selected or retuned on these test results.",
        "",
        "## Pooled policy metrics",
        "",
    ]
    table_rows = []
    columns = [
        "J",
        "risk",
        "bayes_component",
        "incompatibility",
        "default",
        "disagreement",
        "distinct",
        "budget_fraction",
        "last_six_fraction",
    ]
    for k in config.kappa_grid:
        for policy in (
            "ucb_bayes",
            "ucb_rule_aware",
            "explore_certify",
            "nonadaptive",
            "eig",
            "random",
        ):
            table_rows.append(
                [
                    k,
                    policy,
                    *[pooled_metric(metrics, k, policy, metric) for metric in columns],
                ]
            )
    lines += [
        markdown_table(["κ", "Policy", *columns], table_rows),
        "",
        "## Exact gate conditions",
        "",
    ]
    for pair in result["candidate_pairs"]:
        lines += [f"### Pair {pair['pair']}: {'PASS' if pair['pass'] else 'FAIL'}", ""]
        rows = [
            [
                c["name"],
                c["kappa"],
                c["seed"],
                "PASS" if c["pass"] else "FAIL",
                json.dumps(c["actual"], sort_keys=True),
                c["requirement"],
            ]
            for c in pair["checks"]
        ]
        lines += [
            markdown_table(
                ["Condition", "κ", "Seed", "Result", "Observed", "Required"], rows
            ),
            "",
        ]
    lines += [
        "## Metric conventions and scope",
        "",
        "Budget and last-six fractions are unconditional, assigning zero when the default is selected. Selected posterior SD is conditional on deployment. First-leader round counts post-observation rounds 1..T, taking the largest candidate posterior mean (excluding the default), and is conditional on deployment and ever being that leader. A separate never-leader rate records deployed candidates that were never the posterior-mean leader. Empty conditional groups are left blank in CSVs; raw episode metrics remain finite.",
        "",
        f"Maximum risk-decomposition residual: {metrics.decomposition_error.max():.3g}. At κ=0 the UCB objectives and their frozen settings coincide exactly.",
        "",
        "This deliverable ends at the non-neural gate. No neural policy was initialized or trained, and no HPC jobs were submitted. A passing pilot supports proceeding to the learned-policy experiment; it is not evidence about learned-policy performance.",
        "",
    ]
    if not result["pass"]:
        lines += [
            "## Required scientific review",
            "",
            "At least one mandatory condition failed for each permitted pair. Stop here: do not change κ, the budget, the prior, or the environment without scientific review. Do not start neural training.",
            "",
        ]
    report = "\n".join(lines)
    (directory / "pilot_summary.md").write_text(report)
    if not result["pass"]:
        (directory / "pilot_failure_report.md").write_text(report)
    make_plots(directory, config, metrics, paired, metadata)


def make_plots(directory, config, metrics, paired, metadata):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .gate import paired_row, pooled_metric

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for policy, label in (
        ("ucb_bayes", "Bayes UCB"),
        ("ucb_rule_aware", "Rule-aware UCB"),
        ("explore_certify", "Explore then certify"),
        ("nonadaptive", "Tuned static"),
        ("eig", "EIG round robin"),
        ("random", "Random"),
    ):
        axes[0].plot(
            config.kappa_grid,
            [pooled_metric(metrics, k, policy, "risk") for k in config.kappa_grid],
            marker="o",
            label=label,
        )
    gains = [
        paired_row(paired, k, "ucb_rule_aware", "ucb_bayes") for k in config.kappa_grid
    ]
    axes[1].errorbar(
        config.kappa_grid,
        [row["mean"] for row in gains],
        yerr=[1.96 * row.se for row in gains],
        marker="o",
        capsize=4,
    )
    axes[1].axhline(0.02, color="gray", linestyle="--", label="Pooled gain threshold")
    for policy, label in (
        ("ucb_bayes", "Bayes UCB"),
        ("ucb_rule_aware", "Rule-aware UCB"),
    ):
        axes[2].plot(
            config.kappa_grid,
            [pooled_metric(metrics, k, policy, "distinct") for k in config.kappa_grid],
            marker="o",
            label=label,
        )
    for ax, title, ylabel in zip(
        axes,
        ("Deployment risk", "Rule-awareness gain (95% paired CI)", "Search breadth"),
        ("Risk (lower is better)", "J aware − J Bayes", "Mean distinct candidates"),
        strict=True,
    ):
        ax.set(title=title, xlabel="Eligibility multiplier κ", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(directory / "pilot_overview.png", dpi=180)
    fig.savefig(
        directory / "pilot_overview.pdf",
        metadata={
            "Subject": f"environment_hash={metadata['environment_hash']}; code_commit={metadata['code_commit']}"
        },
    )
    plt.close(fig)
    fig, axes = plt.subplots(
        3,
        len(config.kappa_grid),
        figsize=(3.2 * len(config.kappa_grid), 8),
        layout="constrained",
        squeeze=False,
    )
    ucb = pd.read_csv(directory / "ucb_validation.csv")
    etc = pd.read_csv(directory / "explore_certify_validation.csv")
    static = pd.read_csv(directory / "nonadaptive_validation.csv")
    for column, kappa in enumerate(config.kappa_grid):
        u = ucb[ucb.kappa == kappa]
        e = etc[etc.kappa == kappa]
        s = static[static.kappa == kappa]
        axes[0, column].plot(u.beta, u.J, marker=".")
        axes[0, column].set(title=f"κ = {kappa:g}: UCB", xlabel="β")
        for h, marker in (("mean", "o"), ("margin", "x")):
            points = e[e.h == h]
            axes[1, column].scatter(
                points.S,
                points.J,
                c=points.k,
                cmap="viridis",
                vmin=1,
                vmax=config.d,
                marker=marker,
                s=9,
                alpha=0.5,
                label=h,
            )
        axes[1, column].set(
            title="Explore then certify (all k)", xlabel="Search budget S"
        )
        axes[1, column].legend(fontsize=7)
        axes[2, column].plot(s.k, s.J, marker=".")
        axes[2, column].set(title="Nonadaptive subset", xlabel="Subset size k")
        for ax in axes[:, column]:
            ax.set_ylabel("Validation J")
            ax.grid(alpha=0.2)
    fig.savefig(directory / "validation_tuning_curves.png", dpi=180)
    plt.close(fig)
