"""Training step: happy and unhappy paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline.train import TrainError, run_train


@pytest.fixture()
def trivial_cfg(tmp_path: Path) -> dict:
    return {
        "feature_columns": [
            "rating_count",
            "rating_mean",
        ],
        "split": {"test_size": 0.5, "random_state": 0},
        "model": {"type": "logistic_regression", "params": {"max_iter": 200}},
    }


def test_run_train_writes_model_and_holdout(tmp_path: Path, trivial_cfg: dict) -> None:
    df = pd.DataFrame(
        {
            "CustomerID": range(40),
            "churned": [0] * 20 + [1] * 20,
            "rating_count": list(range(40)),
            "rating_mean": [3.5] * 40,
            "rating_std": [1.0] * 40,
            "five_star_pct": [0.1] * 40,
            "date_range_days": [100] * 40,
            "most_common_rating": [4] * 40,
        }
    )
    pq = tmp_path / "proc.parquet"
    df.to_parquet(pq, index=False)

    model_p = tmp_path / "model.joblib"
    npz_p = tmp_path / "holdout.npz"
    _, n_test = run_train(pq, trivial_cfg, model_p, npz_p)
    assert model_p.is_file()
    assert npz_p.is_file()
    assert n_test > 0


def test_run_train_single_class_raises(tmp_path: Path, trivial_cfg: dict) -> None:
    df = pd.DataFrame(
        {
            "CustomerID": range(30),
            "churned": [0] * 30,
            "rating_count": list(range(30)),
            "rating_mean": [3.5] * 30,
            "rating_std": [1.0] * 30,
            "five_star_pct": [0.1] * 30,
            "date_range_days": [100] * 30,
            "most_common_rating": [4] * 30,
        }
    )
    pq = tmp_path / "proc.parquet"
    df.to_parquet(pq, index=False)

    trivial_cfg["feature_columns"] = [
        "rating_count",
        "rating_mean",
        "rating_std",
        "five_star_pct",
        "date_range_days",
        "most_common_rating",
    ]
    with pytest.raises(TrainError):
        run_train(pq, trivial_cfg, tmp_path / "m.joblib", tmp_path / "t.npz")
