"""RFM-style user segmentation (notebook 05) — matplotlib figures + optional Plotly HTML."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def quintile_score(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)


def assign_rfm_segment(row) -> str:
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Best Customers"
    if r >= 3 and f >= 4 and m >= 4:
        return "Loyal Customers"
    if m >= 4:
        return "Big Spenders"
    if r <= 2 and f >= 3 and m >= 3:
        return "At Risk"
    if r <= 2 and f <= 2 and m <= 2:
        return "Lost Cheap Customers"
    if r <= 2:
        return "Lost Customers"
    if r >= 4 and m >= 4 and f <= 3:
        return "Potential Loyal"
    if r >= 4 and m <= 2:
        return "Need Attention"
    if r <= 2 and f >= 3:
        return "Hibernating"
    return "Others"


def run_rfm(project_root: Path, *, write_plotly_html: bool = True) -> None:
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "05_rfm_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir = project_root / "outputs" / "04_clustering"

    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    if not pd.api.types.is_datetime64_any_dtype(ratings["Date"]):
        ratings["Date"] = pd.to_datetime(ratings["Date"])
    print("Ratings:", f"{len(ratings):,}", "rows.")
    print("Date range:", ratings["Date"].min(), "to", ratings["Date"].max())

    reference_date = pd.Timestamp("2005-12-31")
    user_agg = ratings.groupby("CustomerID").agg(
        last_date=("Date", "max"),
        F=("Rating", "count"),
        M=("Rating", "mean"),
    ).reset_index()
    user_agg["R_days"] = (reference_date - user_agg["last_date"]).dt.days
    user_agg["R"] = user_agg["R_days"].astype(int)
    rfm = user_agg[["CustomerID", "R", "F", "M"]].copy()
    print(rfm.head(10))
    print(rfm.describe().round(2))

    rfm["R_score"] = quintile_score(-rfm["R"])
    rfm["F_score"] = quintile_score(rfm["F"])
    rfm["M_score"] = quintile_score(rfm["M"])
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    rfm["segment"] = rfm.apply(assign_rfm_segment, axis=1)
    segment_counts = rfm["segment"].value_counts().sort_values(ascending=False)
    n_users = len(rfm)
    for seg, cnt in segment_counts.items():
        print(f"  {seg}: {cnt:,} ({100 * cnt / n_users:.1f}%)")

    if write_plotly_html:
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            plot_df = rfm.copy()
            if len(plot_df) > 25000:
                plot_df = plot_df.sample(n=25000, random_state=42)
            fig = px.scatter_3d(
                plot_df,
                x="R",
                y="F",
                z="M",
                color="segment",
                title="RFM 3D — Recency, Frequency, Monetary by segment",
                labels={"R": "Recency (days)", "F": "Frequency", "M": "Monetary (avg rating)"},
                opacity=0.6,
                height=700,
            )
            fig.write_html(output_dir / "rfm_3d_scatter.html")

            labels = ["All"] + segment_counts.index.tolist()
            parents = [""] + ["All"] * len(segment_counts)
            values = [0] + segment_counts.values.tolist()
            fig_treemap = go.Figure(go.Treemap(labels=labels, parents=parents, values=values))
            fig_treemap.update_layout(title="RFM segment sizes (treemap)")
            fig_treemap.write_html(output_dir / "rfm_segment_treemap.html")
        except ImportError:
            print("Plotly not installed; skipping HTML exports.")

    heatmap_data = rfm.groupby(["R_score", "F_score"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Segment count heatmap: R_score × F_score")
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_heatmap_RF.png", dpi=150)
    plt.close()

    order_seg = segment_counts.index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, ylabel in zip(axes, ["R", "F", "M"], ["Recency (days)", "Frequency", "Monetary (avg rating)"]):
        sns.boxplot(data=rfm, x="segment", y=col, order=order_seg, ax=ax, palette="Set3")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_boxplot_by_segment.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(segment_counts)), segment_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(segment_counts))))
    ax.set_xticks(range(len(segment_counts)))
    ax.set_xticklabels(segment_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Number of users")
    ax.set_title("RFM segment sizes")
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_segment_sizes.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    rfm["RFM_score"].hist(bins=13, range=(2.5, 15.5), ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("RFM composite score (3–15)")
    ax.set_ylabel("Number of users")
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_score_distribution.png", dpi=150)
    plt.close()

    user_clusters_path = clustering_dir / "user_clusters.parquet"
    if user_clusters_path.exists():
        user_clusters = pd.read_parquet(user_clusters_path)
        if user_clusters.index.name == "CustomerID" or "CustomerID" not in user_clusters.columns:
            user_clusters = user_clusters.reset_index()
        if "CustomerID" not in user_clusters.columns and "index" in user_clusters.columns:
            user_clusters = user_clusters.rename(columns={"index": "CustomerID"})
        rfm_with_cluster = rfm.merge(
            user_clusters[["CustomerID", "cluster_name"]].drop_duplicates(),
            on="CustomerID",
            how="left",
        )
        rfm_with_cluster["cluster_name"] = rfm_with_cluster["cluster_name"].fillna("(not in 04)")
        cross = pd.crosstab(rfm_with_cluster["segment"], rfm_with_cluster["cluster_name"], margins=True)
        print("Cross-tabulation RFM × clustering:")
        print(cross)
        cross_pct = cross.div(cross.iloc[:, :-1].sum(axis=1), axis=0).iloc[:-1, :-1] * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cross_pct, annot=True, fmt=".0f", cmap="Blues", ax=ax)
        ax.set_title("RFM segment × clustering (% within RFM segment)")
        plt.tight_layout()
        plt.savefig(output_dir / "rfm_vs_clustering_heatmap.png", dpi=150)
        plt.close()

    rfm.to_parquet(output_dir / "rfm_segments.parquet", index=False)
    print("Saved:", output_dir / "rfm_segments.parquet")
