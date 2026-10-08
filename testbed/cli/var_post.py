"""Hydra entry point for variational-posterior (Barber-Agakov) training.

Run with ``python -m testbed.cli.var_post [overrides...]``.
"""

import hydra

from testbed.methods.var_post import run


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/var_post")
def main(cfg):
    """Composes the recipe and hands it to ``testbed.methods.var_post.run``."""
    run(cfg)


if __name__ == "__main__":
    main()
