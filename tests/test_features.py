"""Feature engineering and labels: happy and unhappy paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline.features import FeatureBuildError, build_processed_dataset


def test_build_processed_dataset_known_labels(tmp_path: Path) -> None:
    # Reference 2006-06-01, horizon 30d -> feature cutoff 2006-05-02
    pre = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2],
            "MovieID": [1, 2, 3, 4],
            "Rating": [4, 5, 3, 2],
            "Date": pd.to_datetime(
                ["2006-03-01", "2006-04-01", "2006-03-01", "2006-05-15"]
            ),
        }
    )
    pq = tmp_path / "pre.parquet"
    out = tmp_path / "processed.parquet"
    pre.to_parquet(pq, index=False)

    churn_cfg = {"reference_date": "2006-06-01", "horizon_days": 30, "min_prior_ratings": 1}
    feat_cols = [
        "rating_count",
        "rating_mean",
        "rating_std",
        "five_star_pct",
        "date_range_days",
        "most_common_rating",
    ]

    df = build_processed_dataset(pq, churn_cfg, feat_cols, out)
    assert df["CustomerID"].tolist() == [1, 2]
    row1 = df.loc[df["CustomerID"] == 1].iloc[0]
    assert int(row1["churned"]) == 1
    row2 = df.loc[df["CustomerID"] == 2].iloc[0]
    assert int(row2["churned"]) == 0


def test_build_processed_dataset_empty_raises(tmp_path: Path) -> None:
    pq = tmp_path / "empty.parquet"
    pd.DataFrame(
        columns=["CustomerID", "MovieID", "Rating", "Date"],
    ).to_parquet(pq, index=False)
    out = tmp_path / "processed.parquet"
    churn_cfg = {"reference_date": "2006-06-01", "horizon_days": 10, "min_prior_ratings": 5}
    with pytest.raises(FeatureBuildError):
        build_processed_dataset(
            pq,
            churn_cfg,
            [
                "rating_count",
                "rating_mean",
                "rating_std",
                "five_star_pct",
                "date_range_days",
                "most_common_rating",
            ],
            out,
        )
