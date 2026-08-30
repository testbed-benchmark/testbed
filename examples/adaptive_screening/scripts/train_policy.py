"""Fail-closed training boundary; this go/no-go deliverable has no neural trainer."""

import argparse
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.config import load_config  # noqa: E402
from screening.gate import require_passing_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--train-kappa", default="gate_pair_and_zero", choices=["gate_pair_and_zero"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate pilot provenance without initializing or training a neural policy",
    )
    args = parser.parse_args()
    try:
        values = require_passing_gate(load_config(args.config), args.gate)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.verify_only:
        print(
            f"Verified pilot permits training kappas {values}; no neural policy initialized or trained."
        )
        return
    parser.error(
        "This deliverable implements only the go/no-go pilot. Neural training is not implemented; use --verify-only to check the gate."
    )


if __name__ == "__main__":
    main()
