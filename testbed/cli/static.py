"""Hydra entry point for static-design training.

Run with ``python -m testbed.cli.static [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.static import run

main = make_main(run, "method/static")

if __name__ == "__main__":
    main()
