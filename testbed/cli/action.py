"""Hydra entry point for action-net training.

Run with ``python -m testbed.cli.action [overrides...]``.
"""

import hydra

from testbed.methods.action import run


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/action")
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.action.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
