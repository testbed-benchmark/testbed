"""Hydra entry point for RL-BOED (SAC) training.

Run with ``python -m testbed.cli.rlboed [overrides...]``.
"""

import hydra

from testbed.methods.rlboed import run


@hydra.main(
    version_base="1.3",
    config_path="../recipes",
    config_name="experiments/location_finding/rlboed",
)
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.rlboed.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
