"""Generate WM-811K EDA figures and takeaway notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="data/raw/LSWMD.pkl")
    parser.add_argument("--output-dir", default="results/figures/eda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place LSWMD.pkl there first."
        )

    from src.evaluation.eda import run_all_eda

    run_all_eda(raw_path=raw_path, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
