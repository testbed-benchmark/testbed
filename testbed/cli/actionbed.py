"""Hydra entry point for ActionBED training.

Run with ``python -m testbed.cli.actionbed [overrides...]``.
"""

import hydra

from testbed.methods.actionbed import run


@hydra.main(
    version_base="1.3", config_path="../recipes", config_name="method/actionbed"
)
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.actionbed.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
