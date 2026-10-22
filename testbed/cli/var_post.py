"""Hydra entry point for variational-posterior (Barber-Agakov) training.

Run with ``python -m testbed.cli.var_post [overrides...]``.
"""

from testbed.cli import make_main
from testbed.methods.var_post import run

main = make_main(run, "method/var_post")

if __name__ == "__main__":
    main()
