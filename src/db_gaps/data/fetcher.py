"""ETF price fetcher (pykrx primary, yfinance fallback).

For each ETF code we fetch OHLCV from the earliest available date up to today,
store it as a Parquet file under ``data/raw/{code}.parquet``, and resume from
last stored date on subsequent runs.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.config import load_settings, resolve_path
from ..utils.logging import get_logger
from .universe import Universe, load_universe

LOG = get_logger("db_gaps.fetcher")


@dataclass
class FetchResult:
    code: str
    rows_new: int
    rows_total: int
    source: str
    ok: bool
    error: str | None = None


def _raw_path(code: str) -> Path:
    settings = load_settings()
    raw_dir = resolve_path(settings["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{code}.parquet"


def _load_existing(code: str) -> pd.DataFrame | None:
    p = _raw_path(code)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        return df
    except Exception as e:
        LOG.warning("Failed to read %s: %s", p, e)
        return None


def _save(code: str, df: pd.DataFrame) -> None:
    p = _raw_path(code)
    out = df.copy()
    out.index.name = "date"
    out = out.reset_index()
    out.to_parquet(p, index=False)


def _fetch_pykrx(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV via pykrx. Returns df indexed by datetime with cols open/high/low/close/volume."""
    from pykrx import stock

    df = stock.get_etf_ohlcv_by_date(start, end, code)
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
        "거래대금": "value",
        "기초지수": "nav_index",
    }
    df = df.rename(columns=rename)
    df.index = pd.to_datetime(df.index)
    keep = [c for c in ["open", "high", "low", "close", "volume", "value", "nav_index"] if c in df.columns]
    return df[keep].astype(float)


def _fetch_yfinance(code: str, start: str, end: str) -> pd.DataFrame:
    """Fallback fetch via yfinance. Tries .KS then .KQ suffix."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    for suffix in (".KS", ".KQ"):
        try:
            t = yf.Ticker(f"{code}{suffix}")
            df = t.history(start=start, end=end, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df.index = pd.to_datetime(df.index).tz_localize(None)
                keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                return df[keep].astype(float)
        except Exception as e:
            LOG.debug("yfinance %s%s failed: %s", code, suffix, e)
    return pd.DataFrame()


def _to_str(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d.replace("-", "")
    return d.strftime("%Y%m%d")


def fetch_one(
    code: str,
    start: str | None = None,
    end: str | None = None,
    force_full: bool = False,
) -> FetchResult:
    """Fetch (incrementally) one ETF code and persist."""
    settings = load_settings()
    if start is None:
        start = settings["data"]["start_date"]
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    existing = None if force_full else _load_existing(code)
    fetch_start = start
    if existing is not None and not existing.empty:
        last = existing.index.max().date()
        # 약간의 안전 마진 - 마지막 날 하루 겹치도록
        fetch_start = (last - timedelta(days=1)).strftime("%Y-%m-%d")

    start_s, end_s = _to_str(fetch_start), _to_str(end)
    src = "pykrx"
    new_df = pd.DataFrame()

    last_err = None
    for attempt in range(settings["data"]["max_retries"]):
        try:
            new_df = _fetch_pykrx(code, start_s, end_s)
            break
        except Exception as e:
            last_err = e
            LOG.warning("pykrx %s attempt %d failed: %s", code, attempt + 1, e)
            time.sleep(settings["data"]["retry_backoff_sec"] * (attempt + 1))

    if new_df.empty:
        src = "yfinance"
        try:
            new_df = _fetch_yfinance(code, fetch_start, end)
        except Exception as e:
            last_err = e

    if new_df.empty and (existing is None or existing.empty):
        return FetchResult(code=code, rows_new=0, rows_total=0, source=src, ok=False, error=str(last_err))

    if existing is not None and not existing.empty:
        merged = pd.concat([existing, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        rows_new = len(merged) - len(existing)
    else:
        merged = new_df.sort_index()
        rows_new = len(merged)

    _save(code, merged)
    return FetchResult(code=code, rows_new=int(rows_new), rows_total=int(len(merged)), source=src, ok=True)


def fetch_all(
    codes: Iterable[str] | None = None,
    workers: int | None = None,
    force_full: bool = False,
) -> list[FetchResult]:
    """Fetch all codes (defaults to full universe) in parallel."""
    settings = load_settings()
    if codes is None:
        codes = load_universe().codes
    if workers is None:
        workers = settings["data"]["workers"]

    results: list[FetchResult] = []
    codes = list(codes)
    LOG.info("Fetching %d ETFs with %d workers", len(codes), workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, c, force_full=force_full): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status = "OK " if r.ok else "FAIL"
                LOG.info("[%s] %s src=%-8s new=%-4d total=%d", status, code, r.source, r.rows_new, r.rows_total)
            except Exception as e:
                results.append(FetchResult(code=code, rows_new=0, rows_total=0, source="?", ok=False, error=str(e)))
                LOG.error("[FAIL] %s exception: %s", code, e)
    return results
