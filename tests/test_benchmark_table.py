"""Tests for the README benchmark table generator."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.update_benchmark import (  # noqa: E402
    END_MARKER,
    START_MARKER,
    build_table_markdown,
    load_results,
    sync_readme,
)

REPO_ROOT = Path(__file__).parents[1]


def _fake_payload(method, spce=1.0):
    return {
        "task": "location_finding",
        "method": method,
        "T": 10,
        "metrics": {
            "spce": {"mean": spce, "se": 0.01},
            "snmc": {"mean": spce + 0.5, "se": 0.02},
            "proposal_time_ms": {"median": 1.23},
        },
        "eval": {"checkpoint": "best.pt", "checkpoint_step": 100},
        "git_commit": "abcdef1234567890",
    }


def _make_results_dir(tmp_path):
    results_dir = tmp_path / "results"
    task_dir = results_dir / "location_finding"
    task_dir.mkdir(parents=True)
    (task_dir / "dad.json").write_text(json.dumps(_fake_payload("dad", spce=2.0)))
    (task_dir / "rlboed.json").write_text(json.dumps(_fake_payload("rlboed", spce=1.5)))
    return results_dir


def _make_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START_MARKER}\nstale\n{END_MARKER}\n\ntail\n")
    return readme


def test_build_table_markdown(tmp_path):
    payloads = load_results(_make_results_dir(tmp_path))
    table = build_table_markdown(payloads)

    assert "### location_finding (T=10)" in table
    assert (
        "| dad | 2.000 ± 0.010 | 2.500 ± 0.020 | 1.23 | - | best.pt @ 100 | abcdef12 |"
        in table
    )
    assert table.index("| dad |") < table.index("| rlboed |")


def test_build_table_markdown_empty():
    assert "No results yet" in build_table_markdown([])


def test_sync_readme_updates_and_is_idempotent(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    readme = _make_readme(tmp_path)

    assert not sync_readme(readme, results_dir)  # was stale, now rewritten
    content = readme.read_text()
    assert "stale" not in content
    assert "| dad |" in content
    assert content.startswith("# Title") and content.endswith("tail\n")

    assert sync_readme(readme, results_dir)  # second pass: already in sync


def test_sync_readme_check_mode_does_not_write(tmp_path):
    results_dir = _make_results_dir(tmp_path)
    readme = _make_readme(tmp_path)

    assert not sync_readme(readme, results_dir, check=True)
    assert "stale" in readme.read_text()


def test_sync_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError):
        sync_readme(readme, tmp_path)


def test_repo_readme_in_sync():
    # CI drift guard: the committed README must match the committed results/.
    assert sync_readme(REPO_ROOT / "README.md", REPO_ROOT / "results", check=True)
