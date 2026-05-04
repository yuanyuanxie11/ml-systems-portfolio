"""Lightweight EDA summary from `data/ratings.parquet` (notebook 02 narrative, no heavy plots)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_eda(project_root: Path) -> None:
    path = project_root / "data" / "ratings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_data_loading.py first.")
    ratings = pd.read_parquet(path)
    print("Shape:", ratings.shape)
    print("Columns:", list(ratings.columns))
    print("\nHead:")
    print(ratings.head())
    print("\nDescribe (numeric):")
    print(ratings.describe())
    if "Date" in ratings.columns:
        d = pd.to_datetime(ratings["Date"])
        print("\nDate range:", d.min(), "→", d.max())
    print("\nRatings value counts (top 10):")
    print(ratings["Rating"].value_counts().head(10))
