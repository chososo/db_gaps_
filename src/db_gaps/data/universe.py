"""ETF universe loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..utils.config import load_settings, resolve_path


@dataclass(frozen=True)
class Universe:
    """ETF universe wrapper around a DataFrame.

    Columns: code, name, risk_type, asset_class, sub_category, aum_eok
    """

    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    @property
    def codes(self) -> list[str]:
        return self.df["code"].astype(str).tolist()

    @property
    def risk_assets(self) -> "Universe":
        return Universe(self.df[self.df["risk_type"] == "risk"].reset_index(drop=True))

    @property
    def safe_assets(self) -> "Universe":
        return Universe(self.df[self.df["risk_type"] == "safe"].reset_index(drop=True))

    def by_asset_class(self, asset_class: str) -> "Universe":
        return Universe(self.df[self.df["asset_class"] == asset_class].reset_index(drop=True))

    def by_codes(self, codes: Iterable[str]) -> "Universe":
        wanted = set(map(str, codes))
        return Universe(self.df[self.df["code"].astype(str).isin(wanted)].reset_index(drop=True))

    def name_map(self) -> dict[str, str]:
        return dict(zip(self.df["code"].astype(str), self.df["name"]))


def load_universe(path: str | Path | None = None) -> Universe:
    """Load the ETF universe from CSV."""
    if path is None:
        settings = load_settings()
        path = resolve_path(settings["data"]["universe_csv"])
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6).where(df["code"].str.len() <= 6, df["code"])
    return Universe(df)
