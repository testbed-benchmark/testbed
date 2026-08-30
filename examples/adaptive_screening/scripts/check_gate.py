"""Verify frozen pilot evidence and write its go/no-go decision."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adaptive-screening-matplotlib")
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.gate import check_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_gate(args.pilot_dir, args.output)
    print(
        "PILOT GATE: PASS"
        if result["pass"]
        else "PILOT GATE: FAIL -- NEURAL TRAINING DISABLED"
    )
    if "error" in result:
        print(result["error"], file=sys.stderr)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
