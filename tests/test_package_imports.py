"""Regression test for top-level `import testbed` exposing subpackages."""

import importlib

import pytest

SUBPACKAGES = [
    "core",
    "design_nets",
    "design_objectives",
    "designers",
    "envs",
    "eval",
    "inference",
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
    "CartPole",
    "DoubleLink",
    "Gravimetry",
    "LinearRegression",
    "LocationFinding",
    "MichaelisMenten",
    "Pharmacokinetic",
    "PreyPopulation",
    "StochasticPendulum",
    "Task",
    "TemporalDiscounting",
]


@pytest.mark.parametrize("name", TASKS)
def test_task_accessible_from_tasks_subpackage(name):
    """Each task class should be importable directly from `testbed.tasks`."""
    from testbed import tasks

    assert hasattr(tasks, name)


def test_old_task_names_still_instantiate():
    """Saved run configs target the pre-rename class names."""
    from hydra.utils import instantiate

    from testbed import tasks

    for old, new in [
        ("HyperbolicTemporalDiscounting", tasks.TemporalDiscounting),
        ("PharmacoKinetic", tasks.Pharmacokinetic),
    ]:
        assert isinstance(instantiate({"_target_": f"testbed.tasks.{old}"}), new)
