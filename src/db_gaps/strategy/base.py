"""Strategy base class.

A strategy is anything that, given a window of historical asset prices/returns
and an as-of timestamp, returns a target-weight Series indexed by asset code.
Concrete strategies subclass ``Strategy`` and implement ``decide``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def decide(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
    ) -> pd.Series:
        """Return target weights (will be normalized to sum to 1, clipped >= 0)."""

    def __call__(self, returns: pd.DataFrame, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        w = self.decide(returns, prices, asof)
        if not isinstance(w, pd.Series):
            raise TypeError("Strategy.decide must return a pandas Series.")
        return w


def constrain_weights(
    w: pd.Series,
    max_weight: float = 0.20,
    min_weight: float = 0.0,
    long_only: bool = True,
) -> pd.Series:
    """Apply per-asset weight constraints and renormalize.

    Caps each weight at ``max_weight``, redistributing the overflow
    proportionally among uncapped assets up to convergence.
    """
    w = w.copy().astype(float)
    if long_only:
        w = w.clip(lower=0.0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    for _ in range(50):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * (w[free] / w[free].sum())
    w = w.clip(lower=min_weight)
    if w.sum() > 0:
        w = w / w.sum()
    return w
