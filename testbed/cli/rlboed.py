"""Hydra entry point for RL-BOED (SAC) training.

Run with ``python -m testbed.cli.rlboed [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.rlboed import run

main = make_main(run, "experiments/location_finding/rlboed")

if __name__ == "__main__":
    main()
