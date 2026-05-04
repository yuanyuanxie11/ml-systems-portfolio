#!/usr/bin/env python3
"""Print EDA summary tables for ratings.parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_recommender.eda import run_eda  # noqa: E402

if __name__ == "__main__":
    run_eda(ROOT)
