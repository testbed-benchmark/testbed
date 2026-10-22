"""Hydra entry point for ActionBED training.

Run with ``python -m testbed.cli.actionbed [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.actionbed import run

main = make_main(run, "method/actionbed")

if __name__ == "__main__":
    main()
