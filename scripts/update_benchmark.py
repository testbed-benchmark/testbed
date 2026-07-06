"""Regenerates the README benchmark table from results/*.json.

Usage:
    python scripts/update_benchmark.py [--check] [--results-dir results] [--readme README.md]

--check exits 1 without writing when the README is out of sync.
"""

import argparse
import json
import sys
from pathlib import Path

START_MARKER = "<!-- benchmark:start -->"
END_MARKER = "<!-- benchmark:end -->"


def load_results(results_dir) -> list[dict]:
    """Loads all result JSONs, sorted by (task, method) for stable output."""
    payloads = [
        json.loads(path.read_text())
        for path in sorted(Path(results_dir).glob("*/*.json"))
    ]
    return sorted(payloads, key=lambda payload: (payload["task"], payload["method"]))


def _fmt_mean_se(stats: dict | None) -> str:
    if stats is None:
        return "-"
    return f"{stats['mean']:.3f} ± {stats['se']:.3f}"


def _fmt_row(payload: dict) -> str:
    metrics = payload["metrics"]
    speed = metrics.get("proposal_time_ms")
    checkpoint = payload["eval"]["checkpoint"]
    step = payload["eval"].get("checkpoint_step")
    commit = (payload.get("git_commit") or "")[:8] or "-"
    cells = [
        payload["method"],
        _fmt_mean_se(metrics.get("spce")),
        _fmt_mean_se(metrics.get("snmc")),
        f"{speed['median']:.2f}" if speed else "-",
        _fmt_mean_se(metrics.get("decision_loss")),
        f"{checkpoint} @ {step}" if step is not None else checkpoint,
        commit,
    ]
    return "| " + " | ".join(cells) + " |"


def build_table_markdown(payloads: list[dict]) -> str:
    """Renders one markdown table per task."""
    if not payloads:
        return "_No results yet. Run `python testbed/eval/evaluate.py run_dir=...` to add some._"

    header = (
        "| Method | sPCE (lower) | sNMC (upper) | Proposal time (ms, CPU) "
        "| Decision loss | Checkpoint | Commit |\n"
        "|---|---|---|---|---|---|---|"
    )
    blocks = []
    for task in sorted({payload["task"] for payload in payloads}):
        task_payloads = [p for p in payloads if p["task"] == task]
        horizon = task_payloads[0]["T"]
        rows = "\n".join(_fmt_row(payload) for payload in task_payloads)
        blocks.append(f"### {task} (T={horizon})\n\n{header}\n{rows}")
    return "\n\n".join(blocks)


def sync_readme(readme_path, results_dir, check=False) -> bool:
    """Rewrites the marker block; returns True when the README was in sync."""
    readme = Path(readme_path)
    content = readme.read_text()
    if START_MARKER not in content or END_MARKER not in content:
        raise ValueError(f"{readme_path} is missing the benchmark markers")

    prefix, rest = content.split(START_MARKER, 1)
    _, suffix = rest.split(END_MARKER, 1)
    table = build_table_markdown(load_results(results_dir))
    updated = f"{prefix}{START_MARKER}\n{table}\n{END_MARKER}{suffix}"

    in_sync = updated == content
    if not in_sync and not check:
        readme.write_text(updated)
    return in_sync


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    in_sync = sync_readme(args.readme, args.results_dir, check=args.check)
    if args.check and not in_sync:
        print("README benchmark table is out of sync; run scripts/update_benchmark.py")
        sys.exit(1)
    print("README benchmark table is up to date" if in_sync else "README updated")


if __name__ == "__main__":
    main()
