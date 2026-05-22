"""Monte Carlo portfolio path simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_portfolio_paths(
    returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int = 63,
    n_paths: int = 10000,
    method: str = "multivariate_normal",
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate ``n_paths`` cumulative-return paths over ``horizon_days``.

    method:
      * ``multivariate_normal``: sample daily returns ~ N(mu, Sigma) from history
      * ``bootstrap``: i.i.d. resample whole daily-return rows from history
    Returns DataFrame shape (horizon_days, n_paths) of portfolio levels starting at 1.0.
    """
    rng = np.random.default_rng(seed)
    R = returns.dropna(how="all").fillna(0.0)
    w = weights.reindex(R.columns).fillna(0.0).values
    n_assets = R.shape[1]

    if method == "multivariate_normal":
        mu = R.mean().values
        cov = R.cov().values
        # Sample shape: (n_paths, horizon_days, n_assets)
        # Memory-conscious loop over horizon
        port_rets = np.empty((horizon_days, n_paths))
        for t in range(horizon_days):
            draws = rng.multivariate_normal(mu, cov, size=n_paths)
            port_rets[t, :] = draws @ w
    elif method == "bootstrap":
        arr = R.values
        idx = rng.integers(0, len(arr), size=(horizon_days, n_paths))
        port_rets = np.empty((horizon_days, n_paths))
        for t in range(horizon_days):
            port_rets[t, :] = arr[idx[t]] @ w
    else:
        raise ValueError(f"Unknown method: {method}")

    levels = np.cumprod(1.0 + port_rets, axis=0)
    return pd.DataFrame(levels, columns=[f"path_{i}" for i in range(n_paths)])


def mc_summary(paths: pd.DataFrame, alpha_levels=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Return a summary of MC path terminal values."""
    terminal = paths.iloc[-1] - 1.0  # return = level - 1
    s = {
        "mean": float(terminal.mean()),
        "median": float(terminal.median()),
        "std": float(terminal.std(ddof=1)),
        "min": float(terminal.min()),
        "max": float(terminal.max()),
    }
    for a in alpha_levels:
        s[f"q{int(a * 100)}"] = float(terminal.quantile(a))
    return pd.DataFrame([s])
