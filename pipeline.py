#!/usr/bin/env python3
"""End-to-end churn proxy pipeline (single command).

Usage:
  python pipeline.py --config config/churn_pipeline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("pipeline")

    parser = argparse.ArgumentParser(description="Churn proxy ML pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/churn_pipeline.yaml"),
        help="YAML configuration path",
    )
    args = parser.parse_args()

    pipeline_py = Path(__file__).resolve()
    sys.path.insert(0, str(pipeline_py.parent / "src"))

    from churn_pipeline.artifacts import ensure_run_dir, new_run_id, standard_paths
    from churn_pipeline.config import ConfigError, merged_config, save_config_snapshot
    from churn_pipeline.evaluate import run_evaluate
    from churn_pipeline.features import FeatureBuildError, build_processed_dataset
    from churn_pipeline.ingestion import run_ingestion
    from churn_pipeline.preprocess import run_preprocess
    from churn_pipeline.s3_upload import upload_run
    from churn_pipeline.train import TrainError, run_train

    try:
        cfg = merged_config(args.config.resolve(), pipeline_py)
        root = Path(cfg["_resolved_project_root"])
        art_base = root / cfg["artifacts"]["base_dir"]
        run_id = new_run_id()
        run_dir = ensure_run_dir(art_base, run_id)
        paths = standard_paths(run_dir)

        save_config_snapshot(cfg, paths["config_snapshot"])
        log.info("Run dir: %s", run_dir)

        ratings_path = root / cfg["paths"]["data_dir"] / cfg["paths"]["ratings_filename"]
        if cfg["ingestion"].get("enabled") or cfg["ingestion"].get("skip_if_ratings_exist"):
            ratings_path = run_ingestion(cfg, paths)

        run_preprocess(ratings_path, paths["preprocessed_parquet"], cfg)

        build_processed_dataset(
            paths["preprocessed_parquet"],
            cfg["churn"],
            cfg["feature_columns"],
            paths["processed_dataset"],
        )

        run_train(paths["processed_dataset"], cfg, paths["model"], paths["test_holdout"])

        run_evaluate(
            paths["model"],
            paths["test_holdout"],
            paths["metrics"],
            paths["roc_curve"],
        )

        urls = upload_run(run_dir, cfg)
        if urls:
            for u in urls:
                log.info("Uploaded: %s", u)

        log.info("Pipeline completed successfully.")
        return 0

    except ConfigError as e:
        log.error("Configuration error: %s", e)
        return 2
    except (FileNotFoundError, TrainError, FeatureBuildError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Pipeline failed.")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
