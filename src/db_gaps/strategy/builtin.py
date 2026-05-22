"""Built-in baseline strategies (placeholders until the user's strategy is finalized).

These exist primarily so the backtest/risk/report pipeline can be exercised
end-to-end. The user's actual TAA strategy will be added as a separate
module/page and registered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Strategy, constrain_weights
from .registry import register


@register("equal_weight")
@dataclass
class EqualWeight(Strategy):
    name: str = "equal_weight"
    universe: list[str] | None = None
    max_weight: float = 0.20

    def decide(self, returns, prices, asof):
        cols = self.universe or list(returns.columns)
        w = pd.Series(1.0, index=cols)
        return constrain_weights(w, max_weight=self.max_weight)


@register("risk_parity_naive")
@dataclass
class RiskParityNaive(Strategy):
    """Naive inverse-volatility weighting (good enough for diagnostic baseline)."""
    name: str = "risk_parity_naive"
    lookback: int = 60
    max_weight: float = 0.20

    def decide(self, returns, prices, asof):
        r = returns.iloc[-self.lookback:]
        vol = r.std(ddof=1).replace(0.0, np.nan)
        inv = 1.0 / vol
        inv = inv.fillna(0.0)
        if inv.sum() == 0:
            return pd.Series(0.0, index=returns.columns)
        w = inv / inv.sum()
        return constrain_weights(w, max_weight=self.max_weight)


@register("momentum_topn")
@dataclass
class MomentumTopN(Strategy):
    """Top-N by 6-month total return, equal-weighted among selected."""
    name: str = "momentum_topn"
    lookback: int = 126
    top_n: int = 10
    max_weight: float = 0.20

    def decide(self, returns, prices, asof):
        if len(prices) < self.lookback:
            return pd.Series(0.0, index=returns.columns)
        mom = prices.iloc[-1] / prices.iloc[-self.lookback] - 1.0
        mom = mom.dropna()
        winners = mom.nlargest(self.top_n).index
        w = pd.Series(0.0, index=returns.columns)
        w.loc[winners] = 1.0 / max(len(winners), 1)
        return constrain_weights(w, max_weight=self.max_weight)


@register("min_variance")
@dataclass
class MinVariance(Strategy):
    """Long-only minimum-variance via simple iterative shrinkage.

    Closed-form long-only MV needs a QP solver; here we use a fast heuristic:
    inv-cov diagonal weighted, then renormalize. Good enough for a baseline.
    """
    name: str = "min_variance"
    lookback: int = 120
    shrinkage: float = 0.10
    max_weight: float = 0.20

    def decide(self, returns, prices, asof):
        r = returns.iloc[-self.lookback:].dropna(axis=1, how="any")
        if r.shape[1] < 2:
            return pd.Series(0.0, index=returns.columns)
        cov = r.cov().values
        cov = (1 - self.shrinkage) * cov + self.shrinkage * np.diag(np.diag(cov))
        try:
            inv = np.linalg.pinv(cov)
            ones = np.ones(cov.shape[0])
            raw = inv @ ones
        except np.linalg.LinAlgError:
            raw = np.ones(cov.shape[0])
        raw = np.clip(raw, 0, None)
        if raw.sum() == 0:
            raw = np.ones_like(raw)
        w = pd.Series(raw / raw.sum(), index=r.columns)
        return constrain_weights(w.reindex(returns.columns, fill_value=0.0), max_weight=self.max_weight)
