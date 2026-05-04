"""CLI entry: `python -m netflix_recommender <command>` (after `pip install -e .`)."""

from __future__ import annotations

import argparse


def _cmd_data_loading(ns: argparse.Namespace) -> None:
    from .data_io import run_data_loading
    from .paths import project_root

    run_data_loading(project_root(), sample_fraction=ns.sample_fraction)


def _cmd_eda(_: argparse.Namespace) -> None:
    from .eda import run_eda
    from .paths import project_root

    run_eda(project_root())


def _cmd_recommendation(ns: argparse.Namespace) -> None:
    from .paths import project_root
    from .recommendation import run_recommendation_pipeline

    run_recommendation_pipeline(
        project_root(),
        tune_sample_n=ns.tune_sample_n,
        run_nmf=not ns.skip_nmf,
        run_hybrid=not ns.skip_hybrid,
        knn_residual_weight=ns.knn_residual_weight,
        residual_sample=ns.residual_sample,
    )


def _cmd_clustering(_: argparse.Namespace) -> None:
    from .clustering_job import run_clustering
    from .paths import project_root

    run_clustering(project_root())


def _cmd_rfm(ns: argparse.Namespace) -> None:
    from .paths import project_root
    from .rfm import run_rfm

    run_rfm(project_root(), write_plotly_html=not ns.no_plotly_html)


def main() -> None:
    root = argparse.ArgumentParser(
        prog="python -m netflix_recommender",
        description="Netflix Prize pipeline — install with: pip install -e . && pip install -r requirements.txt",
    )
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data-loading", help="Raw Kaggle files → ratings.parquet + probe.parquet")
    p.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="1.0 = full corpus; 0.1 = 10%% of movie files for quick dev",
    )
    p.set_defaults(func=_cmd_data_loading)

    sub.add_parser("eda", help="Print EDA summary tables for ratings.parquet").set_defaults(func=_cmd_eda)

    r = sub.add_parser("recommendation", help="Tune/train hybrid recommender (long on full data)")
    r.add_argument("--tune-sample-n", type=int, default=50_000)
    r.add_argument("--skip-nmf", action="store_true")
    r.add_argument("--skip-hybrid", action="store_true")
    r.add_argument("--knn-residual-weight", type=float, default=0.3)
    r.add_argument("--residual-sample", type=int, default=None, help="Cap rows for residual fit (dev only)")
    r.set_defaults(func=_cmd_recommendation)

    sub.add_parser("clustering", help="User/movie clustering → outputs/04_clustering/").set_defaults(
        func=_cmd_clustering
    )

    f = sub.add_parser("rfm", help="RFM segmentation + figures")
    f.add_argument("--no-plotly-html", action="store_true", help="Skip Plotly HTML (smaller / headless)")
    f.set_defaults(func=_cmd_rfm)

    args = root.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
