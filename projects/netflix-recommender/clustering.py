#!/usr/bin/env python3
"""Run user/movie clustering; writes PNGs + algorithm_comparison.csv under outputs/04_clustering/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_recommender.clustering_job import run_clustering  # noqa: E402

if __name__ == "__main__":
    run_clustering(ROOT)
