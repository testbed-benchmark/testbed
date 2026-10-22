"""Hydra entry point for DAD training.

Run with ``python -m testbed.cli.dad [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.dad import run

main = make_main(run, "method/dad")

if __name__ == "__main__":
    main()
