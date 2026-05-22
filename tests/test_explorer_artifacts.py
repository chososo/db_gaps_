"""Verify Explorer artifacts (series_index + per-code JSON) are well-formed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.data.universe import load_universe
from db_gaps.report.html_builder import _write_series_json


def _synthetic_prices_for_universe(n=400, seed=0):
    uni = load_universe()
    codes = uni.codes
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        100 * (1 + rng.normal(0.0003, 0.011, size=(n, len(codes)))).cumprod(axis=0),
        index=idx,
        columns=codes,
    )


def test_series_json_export(tmp_path):
    prices = _synthetic_prices_for_universe()
    summary = _write_series_json(prices, tmp_path)

    # Index
    idx_path = tmp_path / "data" / "series_index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx["n_codes"] == 188
    assert idx["n_with_data"] == 188
    sample = idx["series"][0]
    for k in ["code", "name", "risk_type", "asset_class", "sub_category", "aum_eok", "start", "end", "n_obs"]:
        assert k in sample

    # Per-series files exist for all codes
    files = list((tmp_path / "data" / "series").glob("*.json"))
    assert len(files) == 188

    # A specific file is well-formed
    one = json.loads(files[0].read_text(encoding="utf-8"))
    assert set(one.keys()) == {"code", "name", "d", "c"}
    assert len(one["d"]) == len(one["c"])
    assert len(one["d"]) > 0


def test_handles_missing_data(tmp_path):
    """If a code has no price column, index still lists it with n_obs=0 and no file."""
    uni = load_universe()
    # Only first 5 codes get data
    keep = uni.codes[:5]
    n = 200
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2024-01-02", periods=n)
    prices = pd.DataFrame(
        100 * (1 + rng.normal(0.0003, 0.011, size=(n, len(keep)))).cumprod(axis=0),
        index=idx,
        columns=keep,
    )
    _write_series_json(prices, tmp_path)
    meta = json.loads((tmp_path / "data" / "series_index.json").read_text(encoding="utf-8"))
    assert meta["n_codes"] == 188
    assert meta["n_with_data"] == 5
    n_files = len(list((tmp_path / "data" / "series").glob("*.json")))
    assert n_files == 5
