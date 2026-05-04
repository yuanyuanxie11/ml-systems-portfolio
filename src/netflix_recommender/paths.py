from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root: contains `src/`, `data/`, `dataset/`, `outputs/`, and optional `offline_pipeline.ipynb`."""
    return Path(__file__).resolve().parent.parent.parent
