"""Risk-exposure and correlation views."""

from __future__ import annotations

import pandas as pd

from ..data.universe import load_universe


def asset_class_exposure(weights: pd.Series, by: str = "asset_class") -> pd.Series:
    """Aggregate portfolio weights by a universe column (asset_class / sub_category / risk_type)."""
    uni = load_universe().df.set_index("code")
    if by not in uni.columns:
        raise ValueError(f"'{by}' not in universe columns: {list(uni.columns)}")
    w = weights.copy()
    w.index = w.index.astype(str)
    aligned = pd.DataFrame({"weight": w}).join(uni[[by]], how="left")
    return aligned.groupby(by, dropna=False)["weight"].sum().sort_values(ascending=False)


def correlation_matrix(returns: pd.DataFrame, lookback: int | None = None) -> pd.DataFrame:
    df = returns.dropna(how="all")
    if lookback:
        df = df.iloc[-lookback:]
    return df.corr()


def rolling_volatility(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    import numpy as np

    return returns.rolling(window).std() * np.sqrt(252)
