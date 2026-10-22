"""Hydra entry point for ALINE training.

Run with ``python -m testbed.cli.aline [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.aline import run

main = make_main(run, "method/aline")

if __name__ == "__main__":
    main()
