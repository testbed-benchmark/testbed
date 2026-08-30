"""Required final figures and LaTeX table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "learned_train_0": "#1f77b4",
    "matched_learned": "#d62728",
    "ucb_bayes": "#4c78a8",
    "ucb_rule_aware": "#e45756",
    "eig": "#72b7b2",
    "nonadaptive": "#b279a2",
}


def _row(frame, policy, kappa_deploy):
    selection = frame[(frame.policy == policy) & (frame.kappa_deploy == kappa_deploy)]
    if len(selection) != 1:
        raise ValueError(f"Missing aggregate result for {policy}, kappa={kappa_deploy}")
    return selection.iloc[0]


def make_figures(results, output):
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results / "policy_metrics_aggregate.csv")
    learned = frame[frame.policy.str.startswith("learned_train_")]
    trained = sorted(learned.kappa_train.unique())
    high = [value for value in trained if value > 0]
    if len(high) != 2:
        raise ValueError("Figures require zero and the two gate-selected policies")

    fig, axis = plt.subplots(figsize=(7.1, 4.4))
    series = [
        ("learned_train_0", "Bayes-assumption learned", "o-"),
        ("ucb_bayes", "Bayes-tuned UCB", "s--"),
        ("ucb_rule_aware", "Matched rule-aware UCB", "D--"),
        ("eig", "EIG round-robin", "^:"),
        ("nonadaptive", "Best nonadaptive", "v:"),
    ]
    for policy, label, style in series:
        values = frame[frame.policy == policy].sort_values("kappa_deploy")
        axis.plot(
            values.kappa_deploy,
            values.risk,
            style,
            color=COLORS[policy],
            label=label,
            linewidth=1.7,
            markersize=5,
        )
    matched = pd.DataFrame(
        [
            _row(
                frame,
                f"learned_train_{str(kappa).replace('.', 'p').rstrip('0').rstrip('p')}",
                kappa,
            )
            for kappa in high
        ]
    )
    axis.plot(
        matched.kappa_deploy,
        matched.risk,
        "o",
        color=COLORS["matched_learned"],
        label="Matched rule-aware learned",
        markersize=7,
    )
    axis.set(xlabel=r"Eligibility multiplier $\kappa$", ylabel="Deployment risk")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "main_risk_vs_mismatch.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=True)
    for axis, kappa in zip(axes, high, strict=True):
        matched_name = (
            f"learned_train_{str(kappa).replace('.', 'p').rstrip('0').rstrip('p')}"
        )
        rows = [_row(frame, "learned_train_0", kappa), _row(frame, matched_name, kappa)]
        bayes = [row.bayes_component for row in rows]
        incompatible = [row.incompatibility for row in rows]
        axis.bar([0, 1], bayes, color="#4c78a8", label="Bayes component")
        axis.bar(
            [0, 1],
            incompatible,
            bottom=bayes,
            color="#f58518",
            label="Incompatibility",
        )
        axis.set_xticks([0, 1], ["Bayes\nlearned", "Rule-aware\nlearned"])
        axis.set_title(rf"$\kappa={kappa:g}$")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Deployment risk")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "main_risk_decomposition.pdf")
    plt.close(fig)

    large = max(high)
    matched_name = (
        f"learned_train_{str(large).replace('.', 'p').rstrip('0').rstrip('p')}"
    )
    comparisons = [
        ("κ=0 Bayes learned", _row(frame, "learned_train_0", 0.0)),
        (f"κ={large:g} Bayes learned", _row(frame, "learned_train_0", large)),
        (f"κ={large:g} rule-aware", _row(frame, matched_name, large)),
    ]
    metrics = [
        ("distinct", "Distinct candidates"),
        ("budget_fraction", "Budget fraction on selected"),
        ("last_six_fraction", "Last-six fraction on selected"),
        ("default", "Default-selection rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    colors = ["#4c78a8", "#9ecae9", "#e45756"]
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        values = [getattr(row, metric) for _, row in comparisons]
        axis.bar(np.arange(3), values, color=colors)
        axis.set_xticks(
            np.arange(3), [name for name, _ in comparisons], rotation=18, ha="right"
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "main_policy_behavior.pdf")
    plt.close(fig)

    policies = [
        ("Bayes-assumption learned", "learned_train_0"),
        ("Rule-aware learned", None),
        ("Bayes-tuned UCB", "ucb_bayes"),
        ("Rule-aware UCB", "ucb_rule_aware"),
        ("Explore-then-certify", "explore_certify"),
        ("Best nonadaptive", "nonadaptive"),
        ("EIG round-robin", "eig"),
        ("Random", "random"),
    ]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"$\kappa$ & Design policy & $J_\kappa$ & Risk & Bayes & Incompat. & Default & Distinct \\",
        r"\midrule",
    ]
    for kappa in high:
        matched_name = (
            f"learned_train_{str(kappa).replace('.', 'p').rstrip('0').rstrip('p')}"
        )
        for label, policy in policies:
            row = _row(frame, matched_name if policy is None else policy, kappa)
            lines.append(
                f"{kappa:g} & {label} & {row.J:.4f} & {row.risk:.4f} & "
                f"{row.bayes_component:.4f} & {row.incompatibility:.4f} & "
                f"{row.default:.3f} & {row.distinct:.2f} \\\\"
            )
        if kappa != high[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "main_results_table.tex").write_text("\n".join(lines) + "\n")
