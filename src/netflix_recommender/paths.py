from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root: contains `src/`, `data/`, `dataset/`, `notebooks/`, `outputs/`."""
    return Path(__file__).resolve().parent.parent.parent
