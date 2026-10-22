"""Hydra command-line entry points for the training methods.

Each module wraps the matching ``testbed.methods.<name>.run`` in
``@hydra.main``, composing its default recipe from ``testbed/recipes``::

    python -m testbed.cli.<name> [overrides...]

These need the optional ``[train]`` dependencies: ``pip install testbed[train]``.
"""

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_RECIPES = Path(__file__).resolve().parent.parent / "recipes"


def _hydra_version():
    """Returns hydra-core's ``(major, minor)``, or None if it is not installed."""
    try:
        match = re.match(r"(\d+)\.(\d+)", version("hydra-core"))
    except PackageNotFoundError:
        return None
    return (int(match[1]), int(match[2])) if match else None


_HYDRA_VERSION = _hydra_version()
if _HYDRA_VERSION is None:
    raise ImportError(
        "testbed.cli needs hydra-core, an optional dependency: "
        "pip install testbed[train]"
    )
if sys.version_info >= (3, 14) and _HYDRA_VERSION < (1, 4):
    raise RuntimeError(
        "hydra-core<1.4 cannot run its command line on Python 3.14 (argparse rejects "
        "its help strings). Launch training with Python 3.13 or earlier."
    )


def make_main(run, config_name):
    """Wraps a method's ``run(cfg)`` in ``@hydra.main`` for the given recipe.

    The recipes path is absolute: a relative one would resolve against
    ``run``'s module as a ``pkg://`` path, and ``testbed/recipes`` is not a
    package.
    """
    import hydra

    return hydra.main(
        version_base="1.3", config_path=str(_RECIPES), config_name=config_name
    )(run)
