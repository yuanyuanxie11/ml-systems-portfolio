"""Run-scoped artifact directory layout."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_run_id() -> str:
    """UTC timestamp + short uuid for unique artifact folders."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def ensure_run_dir(base: Path, run_id: str) -> Path:
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def standard_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "preprocessed_parquet": run_dir / "preprocessed_ratings.parquet",
        "processed_dataset": run_dir / "processed_dataset.parquet",
        "model": run_dir / "model.joblib",
        "test_holdout": run_dir / "test_holdout.npz",
        "metrics": run_dir / "metrics.json",
        "roc_curve": run_dir / "roc_curve.png",
        "config_snapshot": run_dir / "config_resolved.yaml",
        "ingestion_marker": run_dir / "ingestion_complete.txt",
    }
