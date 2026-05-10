"""Load ratings Parquet and write schema-normalized preprocess artifact."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"CustomerID", "MovieID", "Rating", "Date"}


def run_preprocess(
    ratings_path: Path,
    output_path: Path,
    _: dict[str, Any],
) -> Path:
    if not ratings_path.is_file():
        raise FileNotFoundError(ratings_path)

    df = pd.read_parquet(ratings_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Ratings frame missing columns: {sorted(missing)}")

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.dropna(subset=["CustomerID", "MovieID", "Rating", "Date"])
    df["Rating"] = df["Rating"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Preprocessed ratings: %s rows -> %s", f"{len(df):,}", output_path)
    return output_path
