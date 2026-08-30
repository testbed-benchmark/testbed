"""Gate-guarded neural-policy training entry point."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "false")
os.environ["ADAPTIVE_SCREENING_TRAINING"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.config import load_config  # noqa: E402
from screening.gate import require_passing_gate  # noqa: E402
from screening.io import read_json  # noqa: E402


def kappa_label(value):
    return f"{value:g}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--train-kappa",
        default="gate_pair_and_zero",
        help="gate_pair_and_zero or one gate-authorized numeric value",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--optimization-gate",
        type=Path,
        help="Required before training final seeds 404 and 505",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate pilot provenance without importing or initializing neural code",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        values = require_passing_gate(config, args.gate)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.verify_only:
        print(f"Verified pilot permits training kappas {values}.")
        return
    if args.output is None:
        parser.error("--output is required for training")
    if args.train_kappa == "gate_pair_and_zero":
        kappas = values
    else:
        try:
            requested = float(args.train_kappa)
        except ValueError:
            parser.error("--train-kappa must be gate_pair_and_zero or a number")
        if requested not in values:
            parser.error(f"Training kappa {requested} was not authorized by the gate")
        kappas = [requested]
    allowed_seeds = {101, 202, 303, 404, 505}
    if not set(args.seeds) <= allowed_seeds:
        parser.error(f"Seeds must be selected from {sorted(allowed_seeds)}")
    if set(args.seeds) & {404, 505}:
        if args.optimization_gate is None:
            parser.error("Final seeds require --optimization-gate")
        optimization = read_json(args.optimization_gate)
        if optimization.get("pass") is not True or optimization.get(
            "environment_hash"
        ) != read_json(args.gate).get("environment_hash"):
            parser.error("Final seeds require a passing optimization gate")

    gate = read_json(args.gate)
    pilot_directory = args.gate.parent / gate.get("pilot_directory", ".")
    frozen = read_json(pilot_directory / "frozen_hyperparameters.json")
    from screening.train import train_one  # noqa: E402

    for kappa in kappas:
        teacher = (
            frozen["beta_bayes"]
            if kappa == 0
            else frozen["selected"][str(kappa)]["ucb"]["beta"]
        )
        for seed in args.seeds:
            destination = args.output / f"kappa_{kappa_label(kappa)}" / f"seed_{seed}"
            print(
                f"Training kappa={kappa:g}, seed={seed}, UCB beta={teacher:g} -> {destination}",
                flush=True,
            )
            train_one(config, gate, kappa, seed, teacher, destination)


if __name__ == "__main__":
    main()
