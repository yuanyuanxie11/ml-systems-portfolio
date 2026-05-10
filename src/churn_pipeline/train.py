"""Train sklearn classifier from processed_dataset.parquet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TrainError(RuntimeError):
    """Training cannot proceed (e.g. single class)."""


def _make_model(model_cfg: dict[str, Any]):
    kind = str(model_cfg.get("type", "logistic_regression")).lower().replace("-", "_")
    params = dict(model_cfg.get("params") or {})
    if kind in ("logistic_regression", "logreg"):
        return LogisticRegression(**params)
    if kind in ("random_forest", "randomforestclassifier"):
        return RandomForestClassifier(**params)
    raise TrainError(f"Unknown model.type: {model_cfg.get('type')}")


def run_train(
    processed_parquet: Path,
    cfg: dict[str, Any],
    model_out: Path,
    test_npz_out: Path,
) -> tuple[Any, int]:
    df = pd.read_parquet(processed_parquet)
    if df.empty:
        raise TrainError("Processed dataset is empty.")

    feats = cfg["feature_columns"]
    labels = df["churned"]
    if labels.nunique() < 2:
        raise TrainError("Target has a single class; cannot train classifier.")

    X = df[feats].values
    y = labels.values

    split_cfg = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(split_cfg["test_size"]),
        random_state=int(split_cfg["random_state"]),
        stratify=y,
    )

    model = _make_model(cfg["model"])
    model.fit(X_train, y_train)
    logger.info(
        "Fitted %s train=%s test=%s",
        type(model).__name__,
        len(y_train),
        len(y_test),
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feats}, model_out)

    test_npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(test_npz_out, X_test=X_test, y_test=y_test)
    return model, len(y_test)
