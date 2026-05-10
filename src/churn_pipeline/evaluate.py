"""Classification metrics + ROC curve on held-out split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def run_evaluate(
    model_path: Path,
    test_npz_path: Path,
    metrics_path: Path,
    roc_path: Path,
) -> dict[str, float]:
    bundle = joblib.load(model_path)
    model = bundle["model"]

    if not test_npz_path.is_file():
        raise FileNotFoundError(f"Missing test set: {test_npz_path}")

    z = np.load(test_npz_path)
    X_test, y_test = z["X_test"], z["y_test"]

    y_hat = model.predict(X_test)
    proba = _predict_proba_positive(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_hat)),
        "precision": float(precision_score(y_test, y_hat, zero_division=0)),
        "recall": float(recall_score(y_test, y_hat, zero_division=0)),
        "f1": float(f1_score(y_test, y_hat, zero_division=0)),
    }
    try:
        fpr, tpr, _ = roc_curve(y_test, proba)
        metrics["roc_auc"] = float(auc(fpr, tpr))
        _save_roc(fpr, tpr, roc_path)
    except ValueError:
        metrics["roc_auc"] = float("nan")
        logger.warning("ROC AUC unavailable (possibly single-class in slice).")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Metrics: %s", metrics)
    return metrics


def _predict_proba_positive(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(features)
        return p[:, 1]
    logits = getattr(model, "decision_function")(features)
    z = np.clip(logits, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _save_roc(fpr: np.ndarray, tpr: np.ndarray, roc_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve — churn proxy")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
