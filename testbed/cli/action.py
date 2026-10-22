"""Hydra entry point for action-net training.

Run with ``python -m testbed.cli.action [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.action import run

main = make_main(run, "method/action")

if __name__ == "__main__":
    main()
