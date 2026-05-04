from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the `projects/netflix-recommender` directory (contains notebooks/, data/, dataset/)."""
    return Path(__file__).resolve().parent.parent.parent
