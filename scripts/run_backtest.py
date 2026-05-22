#!/usr/bin/env python
"""Run a backtest for a registered strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from db_gaps.backtest import BacktestConfig, run_backtest, rolling_3m_backtest
from db_gaps.data.pipeline import _processed_dir
from db_gaps.strategy import available, get
from db_gaps.utils.logging import get_logger

LOG = get_logger("db_gaps.backtest_cli")


def _load_prices() -> pd.DataFrame:
    p = _processed_dir() / "prices_close.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run scripts/fetch_daily.py first.")
    return pd.read_parquet(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="equal_weight", help=f"Strategy name. Available: {available()}")
    ap.add_argument("--rebalance", default=None, help="daily/weekly/monthly/quarterly")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--rolling-3m", action="store_true", help="Run rolling 3-month walk-forward.")
    args = ap.parse_args()

    px = _load_prices().dropna(how="all", axis=1)
    cfg = BacktestConfig.from_settings()
    if args.rebalance:
        cfg.rebalance = args.rebalance
    if args.lookback:
        cfg.lookback_days = args.lookback
    cfg.name = args.strategy

    strat = get(args.strategy)

    if args.rolling_3m:
        df = rolling_3m_backtest(strat, px, cfg)
        print(df.to_string(index=False))
        return 0

    res = run_backtest(strat, px, cfg)
    print(pd.Series(res.metrics, name=args.strategy).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
