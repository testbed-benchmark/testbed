"""Hydra command-line entry points for the training methods.

Each module wraps the matching ``testbed.methods.<name>.run`` in
``@hydra.main``, composing its default recipe from ``testbed/recipes``::

    python -m testbed.cli.<name> [overrides...]

These need the optional ``[train]`` dependencies: ``pip install testbed[train]``.
"""

import sys
from importlib.metadata import version

if sys.version_info >= (3, 14) and tuple(
    int(part) for part in version("hydra-core").split(".")[:2]
) < (1, 4):
    raise RuntimeError(
        "hydra-core<1.4 cannot run its command line on Python 3.14 (argparse rejects "
        "its help strings). Launch training with Python 3.13 or earlier."
    )
