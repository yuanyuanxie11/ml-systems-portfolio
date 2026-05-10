"""Preprocessing: happy and unhappy paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn_pipeline.preprocess import run_preprocess


def test_preprocess_writes_parquet(tmp_path: Path) -> None:
    inp = tmp_path / "raw.parquet"
    out = tmp_path / "out.parquet"
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "MovieID": [10, 20],
            "Rating": [3, 4],
            "Date": pd.to_datetime(["2005-01-01", "2005-06-01"]),
        }
    )
    df.to_parquet(inp, index=False)
    run_preprocess(inp, out, {})
    got = pd.read_parquet(out)
    assert len(got) == 2
    assert pd.api.types.is_datetime64_any_dtype(got["Date"])


def test_preprocess_missing_column_raises(tmp_path: Path) -> None:
    inp = tmp_path / "bad.parquet"
    pd.DataFrame({"CustomerID": [1]}).to_parquet(inp, index=False)
    out = tmp_path / "out.parquet"
    with pytest.raises(ValueError):
        run_preprocess(inp, out, {})
