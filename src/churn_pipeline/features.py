"""Leakage-aware user features + churn labels (inactive in horizon window)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuildError(RuntimeError):
    """Cannot build features (usually no eligible users)."""


def resolve_reference_date(
    ratings_date_max: pd.Timestamp,
    churn_cfg: dict[str, Any],
) -> pd.Timestamp:
    raw = churn_cfg.get("reference_date")
    if raw is None or (isinstance(raw, str) and str(raw).lower() in {"", "null", "none"}):
        return pd.Timestamp(ratings_date_max)
    return pd.Timestamp(raw)


def build_processed_dataset(
    preprocessed_parquet: Path,
    churn_cfg: dict[str, Any],
    feature_columns: list[str],
    output_parquet: Path,
) -> pd.DataFrame:
    """
    T_feat_end = reference_date - horizon_days.
    Features from ratings with Date < T_feat_end.
    churned = no ratings with T_feat_end <= Date <= reference_date,
              and count(features window) >= min_prior_ratings.
    """
    ratings = pd.read_parquet(preprocessed_parquet)
    if ratings.empty:
        raise FeatureBuildError("Preprocessed ratings are empty.")

    ratings["Date"] = pd.to_datetime(ratings["Date"])
    horizon = int(churn_cfg["horizon_days"])
    min_prior = int(churn_cfg["min_prior_ratings"])

    reference_date = resolve_reference_date(ratings["Date"].max(), churn_cfg)
    feat_end = reference_date - pd.Timedelta(days=horizon)

    feats_df = ratings[ratings["Date"] < feat_end]
    lbl_df = ratings[(ratings["Date"] >= feat_end) & (ratings["Date"] <= reference_date)]

    ratings_in_horizon = set(lbl_df["CustomerID"].unique())

    user_grp = feats_df.groupby("CustomerID")

    uf = (
        user_grp["Rating"]
        .agg(rating_count="count", rating_mean="mean", rating_std="std")
        .fillna(0)
    )

    grp_pairs = feats_df.groupby(["CustomerID", "Rating"], observed=True)
    freq = grp_pairs.size().reset_index(name="count")
    most_common = (
        freq.sort_values("count", ascending=False)
        .drop_duplicates("CustomerID")
        .set_index("CustomerID")["Rating"]
        .rename("most_common_rating")
    )
    uf = uf.join(most_common, how="left")
    uf["most_common_rating"] = uf["most_common_rating"].fillna(-1)

    five_star_cnt = feats_df[feats_df["Rating"] == 5].groupby("CustomerID").size().rename("_5s")
    uf = uf.join(five_star_cnt, how="left").fillna({"_5s": 0})
    uf["five_star_pct"] = uf["_5s"] / uf["rating_count"].replace(0, np.nan).fillna(1)
    uf.drop(columns=["_5s"], inplace=True)

    dr = feats_df.groupby("CustomerID")["Date"].agg(["min", "max"])
    uf["date_range_days"] = (dr["max"] - dr["min"]).dt.days.fillna(0).astype(int)

    uf = uf.reset_index()
    uf = uf[uf["rating_count"] >= min_prior]
    if uf.empty:
        raise FeatureBuildError(
            f"No users with ≥{min_prior} ratings strictly before feature cutoff {feat_end.date()}."
        )

    uf["churned"] = (~uf["CustomerID"].isin(ratings_in_horizon)).astype(int)

    unknown = set(feature_columns) - set(uf.columns)
    if unknown:
        raise ValueError(f"Unknown feature_columns: {sorted(unknown)}")

    uf = uf[["CustomerID", "churned"] + feature_columns].copy()

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    uf.to_parquet(output_parquet, index=False)
    pos_rate = uf["churned"].mean()
    logger.info(
        "Processed dataset: %s users churn_rate=%.3f cutoff_feature=%s ref=%s",
        f"{len(uf):,}",
        pos_rate,
        feat_end.date(),
        reference_date.date(),
    )
    return uf
