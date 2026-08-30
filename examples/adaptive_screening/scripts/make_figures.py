"""Create the three required final panels and main LaTeX table."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screening.plots import make_figures  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figures(args.results, args.output)
    print(f"FIGURES COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
