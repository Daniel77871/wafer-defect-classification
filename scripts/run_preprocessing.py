"""Build WM-811K splits and processed NumPy arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import get_labeled_subset, load_wm811k
from src.data.preprocess import ENCODINGS, build_arrays
from src.data.split import stratified_split
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="data/raw/LSWMD.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["gray", "onehot", "mask"],
        choices=sorted(ENCODINGS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    raw_path = Path(args.raw_path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Place LSWMD.pkl there first."
        )

    processed_dir = ensure_dir(PROJECT_ROOT / "data/processed")
    df = get_labeled_subset(load_wm811k(str(raw_path))).reset_index(drop=True)
    splits = stratified_split(df, seed=args.seed)

    summary_path = PROJECT_ROOT / "data/splits/split_summary.csv"
    summary = pd.read_csv(summary_path)
    print(summary.to_string(index=False))

    for encoding in args.encodings:
        for split_name, indices in tqdm(
            splits.items(),
            desc=f"Saving {encoding}",
            total=len(splits),
        ):
            x, y = build_arrays(df, indices, encoding=encoding)
            output_path = processed_dir / f"{encoding}_{split_name}.npz"
            np.savez_compressed(
                output_path,
                X=x,
                y=y,
                indices=np.asarray(indices, dtype=np.int64),
                encoding=encoding,
            )


if __name__ == "__main__":
    main()
