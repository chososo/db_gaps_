"""Daily update pipeline: fetch → build processed price/return matrices → write meta."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..utils.config import load_settings, resolve_path
from ..utils.logging import get_logger
from .fetcher import FetchResult, fetch_all
from .universe import load_universe

LOG = get_logger("db_gaps.pipeline")


def _raw_dir() -> Path:
    return resolve_path(load_settings()["data"]["raw_dir"])


def _processed_dir() -> Path:
    p = resolve_path(load_settings()["data"]["processed_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_dir() -> Path:
    p = resolve_path(load_settings()["data"]["meta_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_processed_matrix(field: str = "close") -> pd.DataFrame:
    """Stitch per-code parquet files into one wide DataFrame: index=date, columns=code."""
    universe = load_universe()
    series_map: dict[str, pd.Series] = {}
    for code in universe.codes:
        p = _raw_dir() / f"{code}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "date" not in df.columns or field not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")[field].astype(float)
        series_map[code] = s
    if not series_map:
        return pd.DataFrame()
    matrix = pd.concat(series_map, axis=1).sort_index()
    matrix.index.name = "date"
    return matrix


def write_processed_artifacts() -> dict[str, int]:
    """Write the standard processed artifacts: close prices, simple/log returns."""
    out = _processed_dir()
    prices = build_processed_matrix("close")
    prices.to_parquet(out / "prices_close.parquet")

    # Simple daily returns
    rets = prices.pct_change()
    rets.to_parquet(out / "returns_simple.parquet")

    # Log returns
    import numpy as np

    log_rets = np.log(prices / prices.shift(1))
    log_rets.to_parquet(out / "returns_log.parquet")

    LOG.info(
        "Processed: prices=%dx%d returns saved to %s",
        prices.shape[0],
        prices.shape[1],
        out,
    )
    return {
        "rows": int(prices.shape[0]),
        "cols": int(prices.shape[1]),
    }


def _write_meta(results: list[FetchResult], shape_info: dict[str, int]) -> None:
    meta = {
        "run_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_codes": len(results),
        "n_ok": sum(r.ok for r in results),
        "n_fail": sum(not r.ok for r in results),
        "rows_processed": shape_info.get("rows", 0),
        "cols_processed": shape_info.get("cols", 0),
        "failures": [
            {"code": r.code, "error": r.error} for r in results if not r.ok
        ],
        "per_code": [
            {
                "code": r.code,
                "source": r.source,
                "rows_new": r.rows_new,
                "rows_total": r.rows_total,
                "ok": r.ok,
            }
            for r in results
        ],
    }
    out = _meta_dir() / "last_run.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Meta written to %s (ok=%d fail=%d)", out, meta["n_ok"], meta["n_fail"])


def run_daily_update(force_full: bool = False) -> dict:
    """Fetch all universe codes, build processed matrices, write meta."""
    results = fetch_all(force_full=force_full)
    shape = write_processed_artifacts()
    _write_meta(results, shape)
    return {
        "ok": sum(r.ok for r in results),
        "fail": sum(not r.ok for r in results),
        "rows": shape.get("rows", 0),
        "cols": shape.get("cols", 0),
    }
