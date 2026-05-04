#!/usr/bin/env python3
"""RFM segmentation + figures (expects data/ratings.parquet; optional cross-tab with 04 outputs)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from netflix_recommender.rfm import run_rfm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plotly-html", action="store_true", help="Skip Plotly HTML (smaller / headless)")
    args = ap.parse_args()
    run_rfm(ROOT, write_plotly_html=not args.no_plotly_html)


if __name__ == "__main__":
    main()
