"""Hydra entry point for ALINE training.

Run with ``python -m testbed.cli.aline [overrides...]``.
"""

import hydra

from testbed.methods.aline import run


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/aline")
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.aline.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
