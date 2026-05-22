"""Expected Shortfall (CVaR)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .var import portfolio_returns


def expected_shortfall(
    portfolio_rets: pd.Series,
    alpha: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
    n_paths: int = 10000,
    seed: int | None = None,
) -> float:
    """ES (positive number = expected loss magnitude beyond VaR)."""
    pr = portfolio_rets.dropna()
    if len(pr) == 0:
        return 0.0

    if method == "historical":
        scaled = pr * np.sqrt(horizon_days)
        q = np.quantile(scaled, 1 - alpha)
        tail = scaled[scaled <= q]
        return float(-tail.mean()) if len(tail) else float(-q)

    if method == "parametric":
        mu = pr.mean()
        sigma = pr.std(ddof=1)
        z = stats.norm.ppf(1 - alpha)
        es_1d = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))
        return float(es_1d * np.sqrt(horizon_days))

    if method == "monte_carlo":
        rng = np.random.default_rng(seed)
        mu = pr.mean()
        sigma = pr.std(ddof=1)
        sims = rng.normal(mu, sigma, size=(n_paths, horizon_days)).sum(axis=1)
        q = np.quantile(sims, 1 - alpha)
        tail = sims[sims <= q]
        return float(-tail.mean()) if len(tail) else float(-q)

    raise ValueError(f"Unknown ES method: {method}")


def es_table(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence_levels=(0.95, 0.99),
    horizon_days=(1, 5, 21),
    methods=("historical", "parametric", "monte_carlo"),
    mc_paths: int = 10000,
    mc_seed: int | None = None,
) -> pd.DataFrame:
    pr = portfolio_returns(returns, weights).dropna()
    rows = []
    for m in methods:
        for a in confidence_levels:
            for h in horizon_days:
                e = expected_shortfall(pr, a, h, m, mc_paths, mc_seed)
                rows.append({"method": m, "alpha": a, "horizon_days": h, "ES": e})
    return pd.DataFrame(rows)
