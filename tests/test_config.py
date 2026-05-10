"""Config load: happy and unhappy paths."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from churn_pipeline.config import ConfigError, load_yaml, merge_env, validate_config


def test_load_valid_yaml_and_validate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            project: { root: "." }
            paths: {}
            ingestion: {}
            churn: {}
            split: {}
            feature_columns: [a]
            model: {}
            artifacts: {}
            s3: {}
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path)
    validate_config(cfg)


def test_merge_env_s3_bucket_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            project: {}
            paths: {}
            ingestion: {}
            churn: {}
            split: {}
            feature_columns: [x]
            model: {}
            artifacts: {}
            s3: { bucket: "" }
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_yaml(cfg_path)
    validate_config(cfg)
    monkeypatch.setenv("S3_BUCKET", "from-env")
    assert merge_env(cfg)["s3"]["bucket"] == "from-env"


def test_validate_raises_on_missing_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("project: {}\npaths: {}\n", encoding="utf-8")
    cfg = load_yaml(cfg_path)
    with pytest.raises(ConfigError):
        validate_config(cfg)
