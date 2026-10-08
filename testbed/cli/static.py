"""Hydra entry point for static-design training.

Run with ``python -m testbed.cli.static [overrides...]``.
"""

import hydra

from testbed.methods.static import run


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/static")
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.static.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
