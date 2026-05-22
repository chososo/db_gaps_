"""Rebalance schedule helpers."""

from __future__ import annotations

from typing import Literal

import pandas as pd

RebalanceFreq = Literal["daily", "weekly", "monthly", "quarterly"]


def rebalance_dates(index: pd.DatetimeIndex, freq: RebalanceFreq = "monthly") -> pd.DatetimeIndex:
    """Return the subset of trading-day timestamps on which we rebalance.

    Strategy: pick the first trading day in each calendar period that the
    requested frequency implies. ``daily`` returns the whole index.
    """
    if freq == "daily":
        return index
    if len(index) == 0:
        return index

    if freq == "weekly":
        key = index.to_series().dt.to_period("W")
    elif freq == "monthly":
        key = index.to_series().dt.to_period("M")
    elif freq == "quarterly":
        key = index.to_series().dt.to_period("Q")
    else:
        raise ValueError(f"Unsupported rebalance frequency: {freq}")

    # First trading day per period
    return pd.DatetimeIndex(index.to_series().groupby(key).first().values)
