"""Performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualization_factor(periods_per_year: int = 252) -> int:
    return periods_per_year


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    rf_per = rf / periods_per_year
    excess = returns - rf_per
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    rf_per = rf / periods_per_year
    excess = returns - rf_per
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def calmar(equity: pd.Series, returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def hit_ratio(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def performance_metrics(
    equity: pd.Series,
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    return {
        "cagr": cagr(equity, periods_per_year),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, rf, periods_per_year),
        "sortino": sortino(returns, rf, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, returns, periods_per_year),
        "hit_ratio": hit_ratio(returns),
        "n_days": int(len(returns)),
    }
