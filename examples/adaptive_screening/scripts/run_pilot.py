"""Run the complete non-neural pilot on the local CPU."""

import argparse
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.config import load_config  # noqa: E402
from screening.pilot import run_pilot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="CPU memory control only; does not alter episodes or tuning grids",
    )
    args = parser.parse_args()
    run_pilot(load_config(args.config), args.output, args.batch_size)


if __name__ == "__main__":
    main()
