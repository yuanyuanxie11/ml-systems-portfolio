"""Load and validate churn pipeline YAML; merge environment overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_KEYS = {
    "project",
    "paths",
    "ingestion",
    "churn",
    "split",
    "feature_columns",
    "model",
    "artifacts",
    "s3",
}


class ConfigError(ValueError):
    """Invalid or incomplete configuration."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    return raw


def validate_config(cfg: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_KEYS - set(cfg.keys())
    if missing:
        raise ConfigError(f"Missing config sections: {sorted(missing)}")
    cols = cfg["feature_columns"]
    if not isinstance(cols, list) or not cols:
        raise ConfigError("feature_columns must be a non-empty list")
    for sub in REQUIRED_TOP_KEYS:
        if sub == "feature_columns":
            continue
        if not isinstance(cfg[sub], dict):
            raise ConfigError(f"Section '{sub}' must be a mapping")


def merge_env(cfg: dict[str, Any]) -> dict[str, Any]:
    """Non-destructive merge for runtime env overrides."""
    out = deepcopy(cfg)
    bucket = os.environ.get("S3_BUCKET", "").strip()
    if bucket:
        out.setdefault("s3", {})
        out["s3"]["bucket"] = bucket
    upload_enabled = os.environ.get("S3_UPLOAD_ENABLED")
    if upload_enabled is not None:
        normalized = upload_enabled.strip().lower()
        out.setdefault("s3", {})
        out["s3"]["upload_enabled"] = normalized in {"1", "true", "yes", "y", "on"}
    s3_prefix = os.environ.get("S3_PREFIX", "").strip()
    if s3_prefix:
        out.setdefault("s3", {})
        out["s3"]["prefix"] = s3_prefix
    return out


def resolve_project_root(cfg: dict[str, Any], pipeline_file: Path) -> Path:
    """Resolve project root from config and pipeline.py location."""
    raw = cfg.get("project", {}).get("root", ".")
    pr = Path(raw)
    if pr.is_absolute():
        return pr.resolve()
    # Default: repo root (parent of pipeline.py)
    return (pipeline_file.parent / pr).resolve()


def merged_config(config_path: Path, pipeline_script: Path) -> dict[str, Any]:
    cfg = merge_env(load_yaml(config_path))
    validate_config(cfg)
    cfg["_resolved_project_root"] = str(resolve_project_root(cfg, pipeline_script))
    return cfg


def save_config_snapshot(cfg: dict[str, Any], destination: Path) -> None:
    """Write YAML without internal keys."""
    snap = deepcopy(cfg)
    snap.pop("_resolved_project_root", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(snap, f, default_flow_style=False, sort_keys=False)
