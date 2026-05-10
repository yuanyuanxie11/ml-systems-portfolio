"""Upload run artifacts to S3 (bucket from config or env; no hardcoded credentials)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPLOAD_NAMES = (
    "preprocessed_ratings.parquet",
    "processed_dataset.parquet",
    "model.joblib",
    "metrics.json",
    "roc_curve.png",
    "config_resolved.yaml",
    "test_holdout.npz",
    "ingestion_complete.txt",
)


def upload_run(run_dir: Path, cfg: dict[str, Any]) -> list[str]:
    s3_cfg = cfg.get("s3", {})
    if not s3_cfg.get("upload_enabled", False):
        logger.info("S3 upload disabled (s3.upload_enabled false).")
        return []

    bucket = (s3_cfg.get("bucket") or "").strip()
    if not bucket:
        logger.info("No S3 bucket configured; skipping upload.")
        return []

    prefix = (s3_cfg.get("prefix") or "churn-pipeline").strip().strip("/")

    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 required for S3 upload. pip install boto3") from e

    client = boto3.client("s3")
    keys: list[str] = []

    for name in UPLOAD_NAMES:
        local = run_dir / name
        if not local.is_file():
            continue
        key = f"{prefix}/{run_dir.name}/{name}"
        logger.info("Uploading s3://%s/%s", bucket, key)
        client.upload_file(str(local), bucket, key)
        keys.append(f"s3://{bucket}/{key}")

    logger.info("Uploaded %d objects.", len(keys))
    return keys
