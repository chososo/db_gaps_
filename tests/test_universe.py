import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db_gaps.data.universe import load_universe


def test_universe_188():
    uni = load_universe()
    assert len(uni) == 188, f"expected 188 ETFs, got {len(uni)}"
    assert (uni.df["risk_type"] == "risk").sum() == 138
    assert (uni.df["risk_type"] == "safe").sum() == 50


def test_universe_codes_unique():
    uni = load_universe()
    assert uni.df["code"].is_unique


def test_subsets():
    uni = load_universe()
    assert len(uni.risk_assets) == 138
    assert len(uni.safe_assets) == 50
