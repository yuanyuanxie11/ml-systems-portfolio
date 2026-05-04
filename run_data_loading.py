#!/usr/bin/env python3
"""Build `data/ratings.parquet` and `data/probe.parquet` from Netflix Prize raw files."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_recommender.data_io import run_data_loading  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Netflix Prize → Parquet loading pipeline")
    p.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="1.0 = full corpus; 0.1 = 10%% of movie files for quick dev",
    )
    args = p.parse_args()
    run_data_loading(ROOT, sample_fraction=args.sample_fraction)


if __name__ == "__main__":
    main()
