import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.backtest import BacktestConfig, run_backtest, rolling_3m_backtest
from db_gaps.strategy import get


def _synthetic_prices(n_days=600, n_assets=10, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=(n_days, n_assets))
    idx = pd.bdate_range("2022-01-01", periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)


def test_equal_weight_runs():
    px = _synthetic_prices()
    cfg = BacktestConfig(rebalance="monthly", lookback_days=120, name="ew")
    res = run_backtest(get("equal_weight"), px, cfg)
    assert len(res.equity) > 0
    assert "sharpe" in res.metrics


def test_rolling_3m():
    px = _synthetic_prices(n_days=900)
    cfg = BacktestConfig(rebalance="monthly", lookback_days=120, name="ew")
    df = rolling_3m_backtest(get("equal_weight"), px, cfg, test_window_days=63)
    assert not df.empty
    assert "test_start" in df.columns


def test_risk_parity_runs():
    px = _synthetic_prices()
    cfg = BacktestConfig(rebalance="monthly", lookback_days=120, name="rp")
    res = run_backtest(get("risk_parity_naive"), px, cfg)
    assert res.metrics["vol"] >= 0
