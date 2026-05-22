"""Value at Risk (Historical, Parametric, Monte Carlo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute weighted portfolio daily returns from asset returns and weights."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns.fillna(0.0) @ w


def var_historical(
    portfolio_rets: pd.Series, alpha: float = 0.95, horizon_days: int = 1
) -> float:
    """Historical VaR (positive number = loss magnitude)."""
    if len(portfolio_rets) == 0:
        return 0.0
    scaled = portfolio_rets * np.sqrt(horizon_days)
    q = np.quantile(scaled.dropna(), 1 - alpha)
    return float(-q)


def var_parametric(
    portfolio_rets: pd.Series, alpha: float = 0.95, horizon_days: int = 1
) -> float:
    """Gaussian parametric VaR."""
    mu = portfolio_rets.mean()
    sigma = portfolio_rets.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    var_1d = -(mu + z * sigma)
    return float(var_1d * np.sqrt(horizon_days))


def var_monte_carlo(
    portfolio_rets: pd.Series,
    alpha: float = 0.95,
    horizon_days: int = 1,
    n_paths: int = 10000,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR assuming Normal returns calibrated to history.

    For longer horizons, sums ``horizon_days`` simulated daily returns per path.
    """
    if len(portfolio_rets) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    mu = portfolio_rets.mean()
    sigma = portfolio_rets.std(ddof=1)
    sims = rng.normal(mu, sigma, size=(n_paths, horizon_days)).sum(axis=1)
    q = np.quantile(sims, 1 - alpha)
    return float(-q)


def var_table(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_levels=(0.95, 0.99),
    horizon_days=(1, 5, 21),
    methods=("historical", "parametric", "monte_carlo"),
    mc_paths: int = 10000,
    mc_seed: int | None = None,
) -> pd.DataFrame:
    """Full VaR grid as a DataFrame (method x alpha x horizon)."""
    pr = portfolio_returns(returns, weights).dropna()
    rows = []
    for m in methods:
        for a in confidence_levels:
            for h in horizon_days:
                if m == "historical":
                    v = var_historical(pr, a, h)
                elif m == "parametric":
                    v = var_parametric(pr, a, h)
                elif m == "monte_carlo":
                    v = var_monte_carlo(pr, a, h, mc_paths, mc_seed)
                else:
                    continue
                rows.append({"method": m, "alpha": a, "horizon_days": h, "VaR": v})
    return pd.DataFrame(rows)
