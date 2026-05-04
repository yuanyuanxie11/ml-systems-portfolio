#!/usr/bin/env python3
"""Train/tune hybrid recommender (SVD + item–item residuals on probe). Long-running on full data."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_recommender.recommendation import run_recommendation_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-sample-n", type=int, default=50_000)
    ap.add_argument("--skip-nmf", action="store_true")
    ap.add_argument("--skip-hybrid", action="store_true")
    ap.add_argument("--knn-residual-weight", type=float, default=0.3)
    ap.add_argument("--residual-sample", type=int, default=None, help="Cap rows for residual fit (dev only)")
    args = ap.parse_args()
    run_recommendation_pipeline(
        ROOT,
        tune_sample_n=args.tune_sample_n,
        run_nmf=not args.skip_nmf,
        run_hybrid=not args.skip_hybrid,
        knn_residual_weight=args.knn_residual_weight,
        residual_sample=args.residual_sample,
    )


if __name__ == "__main__":
    main()
