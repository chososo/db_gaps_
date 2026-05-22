import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.risk import var_table, es_table, simulate_portfolio_paths


def _ret_panel(n=500, k=5, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, size=(n, k)),
        index=pd.bdate_range("2022-01-01", periods=n),
        columns=[f"A{i}" for i in range(k)],
    )


def test_var_grid():
    R = _ret_panel()
    w = pd.Series(1 / R.shape[1], index=R.columns)
    df = var_table(R, w)
    assert (df["VaR"] >= 0).all()
    assert {"method", "alpha", "horizon_days"}.issubset(df.columns)


def test_es_grid():
    R = _ret_panel()
    w = pd.Series(1 / R.shape[1], index=R.columns)
    df = es_table(R, w)
    assert (df["ES"] >= 0).all()


def test_mc_paths():
    R = _ret_panel()
    w = pd.Series(1 / R.shape[1], index=R.columns)
    paths = simulate_portfolio_paths(R, w, horizon_days=21, n_paths=500, seed=42)
    assert paths.shape == (21, 500)
    assert (paths.iloc[0] > 0).all()
