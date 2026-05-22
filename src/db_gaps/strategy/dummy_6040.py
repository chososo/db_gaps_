"""Dummy 60/40 strategy.

A simple, deterministic baseline TAA strategy that demonstrates the full
pipeline (data → strategy → backtest → risk → report) before the real
strategy is finalised.

Allocation rule
---------------
* 60% to risk assets / 40% to safe assets.
* Within each side: weight is split equally across ``asset_class`` buckets.
* Within each ``asset_class`` bucket: pick the top ``top_n_per_class`` ETFs
  by AUM (as listed in ``universe.csv``) and weight them equally.
* Final weights are passed through ``constrain_weights`` (per-asset cap,
  long-only, sum-to-one).

Why this is a good "dummy"
--------------------------
* No look-ahead — uses only the universe metadata, not the history at ``asof``.
* Diversifies across all 10 sub-classes (5 risk + 5 safe).
* Easy to read and to override (subclass and change ``risk_weight``,
  ``top_n_per_class``, or ``aum_min_eok``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.universe import load_universe
from .base import Strategy, constrain_weights
from .registry import register


@register("dummy_6040")
@dataclass
class Dummy6040(Strategy):
    name: str = "dummy_6040"
    risk_weight: float = 0.60
    top_n_per_class: int = 3   # top-N largest-AUM ETFs per asset_class bucket
    aum_min_eok: float = 100.0 # exclude ETFs with AUM below 100억
    max_weight: float = 0.20

    def _bucket_weights(self, uni_side: pd.DataFrame, side_weight: float) -> pd.Series:
        """Equal weight across asset_class buckets, then top-N by AUM within each."""
        buckets = uni_side["asset_class"].dropna().unique()
        if len(buckets) == 0 or side_weight <= 0:
            return pd.Series(dtype=float)
        per_bucket = side_weight / len(buckets)
        weights: dict[str, float] = {}
        for b in buckets:
            sub = uni_side[uni_side["asset_class"] == b]
            sub = sub[sub["aum_eok"].fillna(0) >= self.aum_min_eok]
            if sub.empty:
                # fall back to any ETF in the bucket if AUM filter wipes it out
                sub = uni_side[uni_side["asset_class"] == b]
            picks = sub.sort_values("aum_eok", ascending=False).head(self.top_n_per_class)
            if picks.empty:
                continue
            w_each = per_bucket / len(picks)
            for code in picks["code"].astype(str):
                weights[code] = weights.get(code, 0.0) + w_each
        return pd.Series(weights, dtype=float)

    def decide(self, returns, prices, asof):
        uni = load_universe().df
        # Only allocate to assets actually present in the backtest universe
        available = set(map(str, returns.columns))
        uni = uni[uni["code"].astype(str).isin(available)].copy()

        risk_w = self._bucket_weights(uni[uni["risk_type"] == "risk"], self.risk_weight)
        safe_w = self._bucket_weights(uni[uni["risk_type"] == "safe"], 1.0 - self.risk_weight)

        w = pd.Series(0.0, index=returns.columns)
        for code, ww in pd.concat([risk_w, safe_w]).groupby(level=0).sum().items():
            if code in w.index:
                w.loc[code] = ww

        return constrain_weights(w, max_weight=self.max_weight)
