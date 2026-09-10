"""Regression test for top-level `import testbed` exposing subpackages."""

import importlib

import pytest

SUBPACKAGES = [
    "core",
    "design_nets",
    "design_objectives",
    "designers",
    "envs",
    "inference",
    "metrics",
    "methods",
    "tasks",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_accessible_after_bare_import(name):
    """`import testbed` alone should expose `testbed.<name>` as an attribute."""
    import testbed

    assert hasattr(testbed, name)
    assert getattr(testbed, name) is importlib.import_module(f"testbed.{name}")


TASKS = [
    "CES",
    "HyperbolicTemporalDiscounting",
    "LinearRegression",
    "LocationFinding",
    "MichaelisMenten",
    "PreyPopulation",
    "StochasticPendulum",
    "Task",
]


@pytest.mark.parametrize("name", TASKS)
def test_task_accessible_from_tasks_subpackage(name):
    """Each task class should be importable directly from `testbed.tasks`."""
    from testbed import tasks

    assert hasattr(tasks, name)
