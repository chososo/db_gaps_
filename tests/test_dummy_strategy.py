import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.data.universe import load_universe
from db_gaps.strategy import available, get


def test_dummy_registered():
    assert "dummy_6040" in available()


def test_dummy_weights_against_full_universe():
    uni = load_universe()
    codes = uni.codes
    # Synthetic returns/prices over the full 188-code universe
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.bdate_range("2024-01-01", periods=n)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, size=(n, len(codes))), index=idx, columns=codes)
    prices = (1 + rets).cumprod() * 100

    strat = get("dummy_6040")
    w = strat(rets, prices, idx[-1])

    # Basic shape
    assert isinstance(w, pd.Series)
    assert len(w) == len(codes)
    # Long-only
    assert (w >= -1e-12).all()
    # Sums to ~1
    assert abs(w.sum() - 1.0) < 1e-6
    # Per-asset cap
    assert w.max() <= 0.20 + 1e-9
    # Allocates to both sides
    side = uni.df.set_index("code")["risk_type"]
    risk_total = w[side[w.index] == "risk"].sum()
    safe_total = w[side[w.index] == "safe"].sum()
    assert risk_total > 0.30
    assert safe_total > 0.20


def test_dummy_handles_partial_universe():
    """If only a subset of codes have data, weights only go to those."""
    uni = load_universe()
    codes = uni.codes[:30]
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=250)
    rets = pd.DataFrame(rng.normal(0.0003, 0.01, size=(250, len(codes))), index=idx, columns=codes)
    prices = (1 + rets).cumprod() * 100
    strat = get("dummy_6040")
    w = strat(rets, prices, idx[-1])
    assert set(w[w > 0].index).issubset(set(codes))
