"""Run local CPU optimization checks or the fixed final evaluation."""

import argparse
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.config import load_config  # noqa: E402
from screening.evaluate import (  # noqa: E402
    run_final_evaluation,
    run_optimization_check,
)
from screening.gate import require_passing_gate  # noqa: E402
from screening.io import read_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--mode", choices=("validation_check", "final_test"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--optimization-gate",
        type=Path,
        default=Path("outputs/optimization_check/optimization_gate.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require_passing_gate(config, args.gate)
    gate = read_json(args.gate)
    gate["_gate_path"] = str(args.gate.resolve())
    pilot = args.gate.parent / gate.get("pilot_directory", ".")
    frozen = read_json(pilot / "frozen_hyperparameters.json")
    metadata = read_json(pilot / "metadata.json")
    if args.mode == "validation_check":
        result = run_optimization_check(
            config,
            gate,
            args.checkpoints,
            frozen,
            args.output,
            args.batch_size,
        )
        print(f"NEURAL OPTIMIZATION CHECK: {'PASS' if result['pass'] else 'FAIL'}")
        raise SystemExit(0 if result["pass"] else 1)
    optimization_path = args.optimization_gate
    if not optimization_path.is_absolute():
        optimization_path = Path(__file__).resolve().parents[1] / optimization_path
    optimization = read_json(optimization_path)
    if (
        optimization.get("pass") is not True
        or optimization.get("environment_hash") != gate["environment_hash"]
    ):
        parser.error("Final evaluation requires a matching passing optimization gate")
    run_final_evaluation(
        config,
        gate,
        args.checkpoints,
        frozen,
        metadata,
        args.output,
        args.batch_size,
    )
    print(f"FINAL EVALUATION COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
