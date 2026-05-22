"""Build a multi-page static HTML site for GitHub Pages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..backtest import BacktestConfig, run_backtest, rolling_3m_backtest
from ..data.pipeline import _meta_dir, _processed_dir
from ..data.universe import load_universe
from ..risk import (
    asset_class_exposure,
    correlation_matrix,
    es_table,
    simulate_portfolio_paths,
    var_table,
)
from ..risk.monte_carlo import mc_summary
from ..strategy import available as available_strategies
from ..strategy import get as get_strategy
from ..utils.config import load_settings, resolve_path
from ..utils.logging import get_logger
from . import plots as P

LOG = get_logger("db_gaps.report")

PAGES = [
    {"key": "dashboard", "label": "Dashboard", "href": "index.html"},
    {"key": "universe", "label": "Universe (188)", "href": "universe.html"},
    {"key": "backtest", "label": "Backtest", "href": "backtest.html"},
    {"key": "risk", "label": "Risk · VaR/ES/MC", "href": "risk.html"},
    {"key": "portfolio", "label": "Portfolio Follow-up", "href": "portfolio.html"},
]


def _env() -> Environment:
    tpl_dir = Path(__file__).parent / "templates"
    return Environment(loader=FileSystemLoader(tpl_dir), autoescape=select_autoescape(["html"]))


def _render(env: Environment, current: str, title: str, body: str) -> str:
    tpl = env.get_template("base.html")
    return tpl.render(
        title=title,
        body=body,
        pages=PAGES,
        current=current,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _df_to_html(df: pd.DataFrame, fmt: dict | None = None, max_rows: int | None = None) -> str:
    d = df.copy() if max_rows is None else df.head(max_rows).copy()
    if fmt:
        for col, f in fmt.items():
            if col in d.columns:
                d[col] = d[col].map(f)
    return d.to_html(index=False, classes="data", border=0)


# --------------------------- page builders ---------------------------


def _load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    pdir = _processed_dir()
    prices = pd.read_parquet(pdir / "prices_close.parquet") if (pdir / "prices_close.parquet").exists() else pd.DataFrame()
    returns = pd.read_parquet(pdir / "returns_simple.parquet") if (pdir / "returns_simple.parquet").exists() else pd.DataFrame()
    return prices, returns


def _read_meta() -> dict:
    p = _meta_dir() / "last_run.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def page_dashboard(prices: pd.DataFrame, returns: pd.DataFrame) -> str:
    meta = _read_meta()
    uni = load_universe()
    n_codes = len(uni)
    n_ok = meta.get("n_ok", 0)
    n_fail = meta.get("n_fail", 0)
    last = meta.get("run_at_utc", "—")

    by_class = uni.df.groupby("asset_class").size().sort_values(ascending=False)
    by_risk = uni.df.groupby("risk_type").size()

    kpis = "".join([
        _kpi("Universe", f"{n_codes} ETF"),
        _kpi("Risk / Safe", f"{int(by_risk.get('risk',0))} / {int(by_risk.get('safe',0))}"),
        _kpi("Last fetch (UTC)", last),
        _kpi("Fetch OK / Fail", f"{n_ok} / {n_fail}"),
    ])

    # Recent 1-month return ranking
    rec_html = ""
    if not prices.empty and len(prices) > 21:
        recent = (prices.iloc[-1] / prices.iloc[-21] - 1).dropna().sort_values(ascending=False)
        top = recent.head(10).to_frame("1M_return")
        bot = recent.tail(10).to_frame("1M_return")
        top.index.name = "code"
        bot.index.name = "code"
        rec_html = (
            "<div class='card'><h3>최근 1개월 수익률 — TOP 10</h3>"
            + _df_to_html(top.reset_index(), {"1M_return": lambda x: f"{x:+.2%}"})
            + "</div><div class='card'><h3>최근 1개월 수익률 — BOTTOM 10</h3>"
            + _df_to_html(bot.reset_index(), {"1M_return": lambda x: f"{x:+.2%}"})
            + "</div>"
        )

    class_chart = P.bar_exposure(by_class, "Universe by asset_class")

    body = f"""
    <div class="grid">{kpis}</div>
    <div class="card"><h3>구성</h3>{class_chart}</div>
    {rec_html}
    """
    return body


def page_universe() -> str:
    uni = load_universe().df.copy()
    table = _df_to_html(
        uni[["code", "name", "risk_type", "asset_class", "sub_category", "aum_eok"]],
        {"aum_eok": lambda x: f"{int(x):,}"},
    )
    return f"<div class='card'><h3>188 ETF Universe</h3>{table}</div>"


def page_backtest(prices: pd.DataFrame, strategy_names: list[str]) -> str:
    if prices.empty:
        return "<div class='card'>No price data yet. Run the data pipeline first.</div>"

    cfg_base = BacktestConfig.from_settings()
    cards = []
    summary_rows = []

    px = prices.dropna(how="all", axis=1)
    # Keep only assets with enough history
    valid = px.columns[px.notna().sum() > cfg_base.lookback_days + 30]
    px = px[valid].dropna(how="all")

    for sname in strategy_names:
        try:
            strat = get_strategy(sname)
            cfg = BacktestConfig(
                rebalance=cfg_base.rebalance,
                lookback_days=cfg_base.lookback_days,
                cost_bps=cfg_base.cost_bps,
                slippage_bps=cfg_base.slippage_bps,
                initial_capital=cfg_base.initial_capital,
                risk_free_rate=cfg_base.risk_free_rate,
                name=sname,
            )
            res = run_backtest(strat, px, cfg)
            summary_rows.append({"strategy": sname, **res.metrics})

            eq = res.equity / res.equity.iloc[0]
            rolling_df = rolling_3m_backtest(strat, px, cfg)
            rolling_html = _df_to_html(
                rolling_df,
                {
                    "cagr": lambda x: f"{x:+.2%}" if pd.notna(x) else "",
                    "vol": lambda x: f"{x:.2%}" if pd.notna(x) else "",
                    "sharpe": lambda x: f"{x:.2f}" if pd.notna(x) else "",
                    "sortino": lambda x: f"{x:.2f}" if pd.notna(x) else "",
                    "max_drawdown": lambda x: f"{x:.2%}" if pd.notna(x) else "",
                    "calmar": lambda x: f"{x:.2f}" if pd.notna(x) else "",
                    "hit_ratio": lambda x: f"{x:.2%}" if pd.notna(x) else "",
                },
            )
            metric_rows = "".join([
                _kpi("CAGR", f"{res.metrics['cagr']:+.2%}"),
                _kpi("Vol", f"{res.metrics['vol']:.2%}"),
                _kpi("Sharpe", f"{res.metrics['sharpe']:.2f}"),
                _kpi("Max DD", f"{res.metrics['max_drawdown']:.2%}"),
            ])
            cards.append(
                f"<div class='card'><h3>{sname} · rebalance={cfg.rebalance}</h3>"
                f"<div class='grid'>{metric_rows}</div>"
                f"{P.line_equity(eq, f'{sname} equity')}"
                f"<h3>Rolling 3-month windows</h3>{rolling_html}</div>"
            )
        except Exception as e:
            LOG.error("Backtest %s failed: %s", sname, e)
            cards.append(f"<div class='card'>Backtest <code>{sname}</code> failed: {e}</div>")

    summary_html = ""
    if summary_rows:
        sdf = pd.DataFrame(summary_rows).set_index("strategy")
        summary_html = (
            "<div class='card'><h3>전략 비교</h3>"
            + _df_to_html(
                sdf.reset_index(),
                {
                    "cagr": lambda x: f"{x:+.2%}",
                    "vol": lambda x: f"{x:.2%}",
                    "sharpe": lambda x: f"{x:.2f}",
                    "sortino": lambda x: f"{x:.2f}",
                    "max_drawdown": lambda x: f"{x:.2%}",
                    "calmar": lambda x: f"{x:.2f}",
                    "hit_ratio": lambda x: f"{x:.2%}",
                    "n_days": lambda x: f"{int(x)}",
                },
            )
            + "</div>"
        )

    return summary_html + "".join(cards)


def page_risk(prices: pd.DataFrame, returns: pd.DataFrame, weights: pd.Series | None) -> str:
    if returns.empty:
        return "<div class='card'>No returns data yet.</div>"
    settings = load_settings()
    s = settings["risk"]

    if weights is None or weights.sum() == 0:
        # Default: equal weight across columns with enough history
        valid = returns.columns[returns.notna().sum() > s["vol_lookback"]]
        weights = pd.Series(1.0 / len(valid), index=valid)

    var_df = var_table(
        returns,
        weights,
        confidence_levels=tuple(s["confidence_levels"]),
        horizon_days=tuple(s["horizon_days"]),
        methods=tuple(s["methods"]),
        mc_paths=s["mc_paths"],
        mc_seed=s["mc_seed"],
    )
    es_df = es_table(
        returns,
        weights,
        confidence_levels=tuple(s["confidence_levels"]),
        horizon_days=tuple(s["horizon_days"]),
        methods=tuple(s["methods"]),
        mc_paths=s["mc_paths"],
        mc_seed=s["mc_seed"],
    )

    paths = simulate_portfolio_paths(
        returns,
        weights,
        horizon_days=s["mc_horizon_days"],
        n_paths=min(5000, s["mc_paths"]),
        seed=s["mc_seed"],
    )
    mcs = mc_summary(paths)

    # Correlation (top 30 by current weight to keep heatmap legible)
    top_w = weights.sort_values(ascending=False).head(30).index
    corr = correlation_matrix(returns[top_w], lookback=s["corr_lookback"])

    expo = asset_class_exposure(weights, by="asset_class")

    body = f"""
    <div class='card'><h3>Exposure (asset_class)</h3>{P.bar_exposure(expo, 'Portfolio exposure')}</div>
    <div class='card'><h3>VaR Grid</h3>{_df_to_html(var_df, {'VaR': lambda x: f'{x:.2%}', 'alpha': lambda x: f'{x:.0%}'})}</div>
    <div class='card'><h3>Expected Shortfall (CVaR) Grid</h3>{_df_to_html(es_df, {'ES': lambda x: f'{x:.2%}', 'alpha': lambda x: f'{x:.0%}'})}</div>
    <div class='card'><h3>Monte Carlo — {s['mc_horizon_days']}일 누적</h3>{P.mc_fan_chart(paths, 'MC fan chart')}<br>{_df_to_html(mcs, {c: lambda x: f'{x:+.2%}' for c in mcs.columns})}</div>
    <div class='card'><h3>Correlation (top {len(top_w)} by weight)</h3>{P.heatmap_corr(corr)}</div>
    """
    return body


def page_portfolio(prices: pd.DataFrame, weights: pd.Series | None) -> str:
    uni = load_universe().df.set_index("code")
    if weights is None or weights.sum() == 0:
        return ("<div class='card'>전략이 확정되지 않아 현재 포트폴리오가 없습니다. "
                "전략을 등록하면 이 페이지에 보유종목/비중/팔로업 지표가 표시됩니다.</div>")
    w = weights[weights > 0].sort_values(ascending=False)
    df = pd.DataFrame({"code": w.index, "weight": w.values})
    df = df.merge(uni[["name", "asset_class", "sub_category", "risk_type"]].reset_index(), on="code", how="left")
    if not prices.empty:
        last_px = prices.iloc[-1]
        chg_1d = prices.iloc[-1] / prices.iloc[-2] - 1 if len(prices) >= 2 else None
        df["last_price"] = df["code"].map(last_px)
        if chg_1d is not None:
            df["1d_change"] = df["code"].map(chg_1d)

    fmt = {"weight": lambda x: f"{x:.2%}"}
    if "last_price" in df.columns:
        fmt["last_price"] = lambda x: f"{x:,.0f}" if pd.notna(x) else ""
    if "1d_change" in df.columns:
        fmt["1d_change"] = lambda x: f"{x:+.2%}" if pd.notna(x) else ""
    expo_html = P.bar_exposure(asset_class_exposure(weights), "Asset-class exposure")
    return (
        f"<div class='card'><h3>Current portfolio ({len(df)} positions)</h3>"
        f"{_df_to_html(df, fmt)}</div>"
        f"<div class='card'>{expo_html}</div>"
    )


# --------------------------- driver ---------------------------


def build_site(
    strategies: list[str] | None = None,
    output_dir: str | Path | None = None,
    current_weights: pd.Series | None = None,
) -> Path:
    settings = load_settings()
    out = Path(output_dir) if output_dir else resolve_path(settings["report"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    prices, returns = _load_processed()
    env = _env()

    if strategies is None:
        active = settings.get("strategies", {}).get("active") or []
        strategies = active if active else available_strategies()

    # Pages
    pages = {
        "dashboard": ("Dashboard", page_dashboard(prices, returns)),
        "universe": ("Universe", page_universe()),
        "backtest": ("Backtest", page_backtest(prices, strategies)),
        "risk": ("Risk", page_risk(prices, returns, current_weights)),
        "portfolio": ("Portfolio", page_portfolio(prices, current_weights)),
    }

    file_map = {
        "dashboard": "index.html",
        "universe": "universe.html",
        "backtest": "backtest.html",
        "risk": "risk.html",
        "portfolio": "portfolio.html",
    }

    for key, (title, body) in pages.items():
        html = _render(env, current=key, title=title, body=body)
        (out / file_map[key]).write_text(html, encoding="utf-8")
        LOG.info("Wrote %s", out / file_map[key])

    # CNAME-friendly noop / disable Jekyll
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out
