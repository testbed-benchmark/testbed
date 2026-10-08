"""Hydra entry point for DAD training.

Run with ``python -m testbed.cli.dad [overrides...]``.
"""

import hydra

from testbed.methods.dad import run


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/dad")
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.dad.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
