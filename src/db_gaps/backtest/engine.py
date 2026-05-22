"""Backtest engine.

Workflow per rebalance date:
  1) Strategy produces weights from a look-back window of returns.
  2) Weights are applied for the following days until the next rebalance.
  3) Transaction costs (round-trip bps) are charged on weight changes.

Designed so:
  * Rebalance frequency is pluggable (weekly/monthly/quarterly).
  * Lookback / windows are config driven.
  * Rolling 3-month forward backtest is a thin wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from ..utils.config import load_settings
from .metrics import performance_metrics
from .rebalancer import RebalanceFreq, rebalance_dates

# A strategy is a callable: (returns_history: DataFrame, prices_history: DataFrame, asof: Timestamp)
#                          -> weights: Series indexed by asset code.
StrategyFn = Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.Series]


@dataclass
class BacktestConfig:
    start: str | None = None
    end: str | None = None
    rebalance: RebalanceFreq = "monthly"
    lookback_days: int = 252
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    initial_capital: float = 1.0
    risk_free_rate: float = 0.0
    name: str = "strategy"

    @classmethod
    def from_settings(cls, **overrides) -> "BacktestConfig":
        s = load_settings()["backtest"]
        cfg = cls(
            rebalance=s["rebalance_frequency"],
            lookback_days=s["train_lookback_days"],
            cost_bps=s["cost_bps"],
            slippage_bps=s["slippage_bps"],
            initial_capital=s["initial_capital"],
            risk_free_rate=s["risk_free_rate"],
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class BacktestResult:
    config: BacktestConfig
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame          # index=date, columns=asset
    turnover: pd.Series            # per rebalance L1 weight change
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics, name=self.config.name)


def _apply_costs(prev_w: pd.Series, new_w: pd.Series, cost_bps: float, slippage_bps: float) -> float:
    """Return fractional cost charged to equity on rebalance."""
    common = prev_w.index.union(new_w.index)
    pw = prev_w.reindex(common, fill_value=0.0)
    nw = new_w.reindex(common, fill_value=0.0)
    turnover = float((nw - pw).abs().sum())
    total_bps = (cost_bps + slippage_bps)
    return turnover * total_bps / 1e4, turnover


def run_backtest(
    strategy: StrategyFn,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a single backtest.

    Parameters
    ----------
    strategy
        Function returning weights (pd.Series) for an asset universe.
    prices
        Wide DataFrame of close prices (date x code).
    config
        Backtest parameters.
    """
    if config is None:
        config = BacktestConfig.from_settings()

    px = prices.sort_index().dropna(how="all")
    if config.start:
        px = px.loc[pd.to_datetime(config.start):]
    if config.end:
        px = px.loc[:pd.to_datetime(config.end)]
    if len(px) < config.lookback_days + 2:
        raise ValueError(
            f"Not enough data: have {len(px)} rows, need > lookback {config.lookback_days}"
        )

    rets = px.pct_change().fillna(0.0)

    rb_dates = rebalance_dates(px.index, freq=config.rebalance)
    # Start applying weights only after we have enough lookback
    rb_dates = [d for d in rb_dates if px.index.get_loc(d) >= config.lookback_days]
    if not rb_dates:
        raise ValueError("No rebalance dates after lookback period.")

    equity = pd.Series(index=px.index, dtype=float)
    weight_records: list[pd.Series] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []

    current_w = pd.Series(0.0, index=px.columns)
    eq = config.initial_capital
    next_rb_idx = 0

    for i, dt in enumerate(px.index):
        # Rebalance at the START of the day (using up-to-yesterday data)
        if next_rb_idx < len(rb_dates) and dt == rb_dates[next_rb_idx]:
            hist_end = i  # use returns/prices STRICTLY BEFORE dt
            lb_start = max(0, hist_end - config.lookback_days)
            ret_hist = rets.iloc[lb_start:hist_end]
            px_hist = px.iloc[lb_start:hist_end]
            try:
                new_w = strategy(ret_hist, px_hist, dt)
            except Exception:
                new_w = current_w.copy()
            new_w = new_w.reindex(px.columns, fill_value=0.0).fillna(0.0)
            # Apply long-only & re-normalize defensively
            new_w = new_w.clip(lower=0.0)
            s = new_w.sum()
            if s > 0:
                new_w = new_w / s
            cost_frac, turn = _apply_costs(current_w, new_w, config.cost_bps, config.slippage_bps)
            eq *= (1.0 - cost_frac)
            current_w = new_w
            turnover_records.append((dt, turn))
            next_rb_idx += 1

        # Apply current weights to today's returns
        day_r = float((current_w * rets.iloc[i]).sum())
        eq *= (1.0 + day_r)
        equity.iloc[i] = eq
        weight_records.append(current_w.rename(dt))

    weights_df = pd.DataFrame(weight_records)
    weights_df.index.name = "date"
    equity = equity.dropna()
    daily_ret = equity.pct_change().fillna(0.0)
    turnover_s = pd.Series({d: t for d, t in turnover_records}, name="turnover")

    metrics = performance_metrics(equity, daily_ret, rf=config.risk_free_rate)
    return BacktestResult(
        config=config,
        equity=equity,
        returns=daily_ret,
        weights=weights_df,
        turnover=turnover_s,
        metrics=metrics,
    )


def rolling_3m_backtest(
    strategy: StrategyFn,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    test_window_days: int | None = None,
) -> pd.DataFrame:
    """Walk-forward 3-month-window evaluation.

    Each iteration trains/decides using `lookback_days` and then runs the
    strategy out-of-sample for ``test_window_days`` (default: 63 ≈ 3 months).
    Returns a DataFrame of per-window metrics + average row.
    """
    if config is None:
        config = BacktestConfig.from_settings()
    if test_window_days is None:
        test_window_days = load_settings()["backtest"]["test_window_days"]

    px = prices.sort_index().dropna(how="all")
    rets = px.pct_change().fillna(0.0)
    rows: list[dict] = []
    n = len(px)
    i = config.lookback_days
    while i + test_window_days <= n:
        train_end = i
        test_end = i + test_window_days
        sub_px = px.iloc[max(0, train_end - config.lookback_days): test_end]
        sub_cfg = BacktestConfig(
            start=sub_px.index[0].strftime("%Y-%m-%d"),
            end=sub_px.index[-1].strftime("%Y-%m-%d"),
            rebalance=config.rebalance,
            lookback_days=config.lookback_days,
            cost_bps=config.cost_bps,
            slippage_bps=config.slippage_bps,
            initial_capital=1.0,
            risk_free_rate=config.risk_free_rate,
            name=f"{config.name}_w{i}",
        )
        try:
            res = run_backtest(strategy, sub_px, sub_cfg)
            row = {
                "window_start": sub_px.index[train_end - max(0, train_end - config.lookback_days)],
                "test_start": px.index[train_end],
                "test_end": px.index[test_end - 1],
                **res.metrics,
            }
            rows.append(row)
        except Exception as e:
            rows.append({
                "test_start": px.index[train_end],
                "test_end": px.index[test_end - 1],
                "error": str(e),
            })
        i += test_window_days  # non-overlapping 3-month windows

    df = pd.DataFrame(rows)
    if not df.empty and "cagr" in df.columns:
        avg = df[["cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_ratio"]].mean(numeric_only=True)
        avg["test_start"] = "AVG"
        avg["test_end"] = ""
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df
