"""Data ingestion — delegate to netflix_recommender ETL."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_ingestion(cfg: dict[str, Any], paths: dict[str, Path]) -> Path:
    """
    Produce data/ratings.parquet under project root.
    Returns path to ratings parquet.
    """
    root = Path(cfg["_resolved_project_root"])
    data_dir = root / cfg["paths"]["data_dir"]
    ratings_path = data_dir / cfg["paths"]["ratings_filename"]
    ing = cfg["ingestion"]

    if ing.get("skip_if_ratings_exist") and ratings_path.is_file():
        logger.info("Skipping ingestion; found existing %s", ratings_path)
        paths["ingestion_marker"].write_text(
            f"skipped_existing\n{ratings_path}\n",
            encoding="utf-8",
        )
        return ratings_path

    if not ing.get("enabled", True):
        if not ratings_path.is_file():
            raise FileNotFoundError(f"Ingestion disabled but ratings missing: {ratings_path}")
        logger.info("Ingestion disabled in config; using %s", ratings_path)
        paths["ingestion_marker"].write_text(
            f"disabled_using_existing\n{ratings_path}\n",
            encoding="utf-8",
        )
        return ratings_path

    from netflix_recommender.data_io import run_data_loading

    logger.info("Running ETL sample_fraction=%s", ing.get("sample_fraction", 1.0))
    run_data_loading(
        root,
        sample_fraction=float(ing.get("sample_fraction", 1.0)),
        dataset_dir_name=str(cfg["paths"].get("dataset_dir", "dataset")),
        output_dir_name=str(cfg["paths"].get("data_dir", "data")),
    )

    if not ratings_path.is_file():
        raise FileNotFoundError(f"ETL finished but ratings not found: {ratings_path}")

    paths["ingestion_marker"].write_text(str(ratings_path) + "\n", encoding="utf-8")
    return ratings_path
