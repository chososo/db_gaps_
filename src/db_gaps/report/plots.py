"""Plotly chart helpers - return inline HTML fragments."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

_PLOTLY_KW = dict(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})


def _fig_html(fig: go.Figure) -> str:
    return to_html(fig, **_PLOTLY_KW)


def line_equity(equity: pd.Series, title: str = "Equity Curve") -> str:
    fig = px.line(equity, title=title, labels={"value": "Equity", "index": "Date"})
    fig.update_layout(showlegend=False, height=400)
    return _fig_html(fig)


def line_multi(df: pd.DataFrame, title: str = "") -> str:
    fig = px.line(df, title=title)
    fig.update_layout(height=420)
    return _fig_html(fig)


def heatmap_corr(corr: pd.DataFrame, title: str = "Correlation") -> str:
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=title, aspect="auto")
    fig.update_layout(height=min(900, 60 + 14 * len(corr)))
    return _fig_html(fig)


def bar_exposure(series: pd.Series, title: str = "Exposure") -> str:
    fig = px.bar(series.sort_values(ascending=True), orientation="h", title=title)
    fig.update_layout(showlegend=False, height=max(300, 18 * len(series)))
    return _fig_html(fig)


def mc_fan_chart(paths: pd.DataFrame, title: str = "Monte Carlo Fan") -> str:
    """paths: rows=days, cols=path_*. Plot median + quantile fan."""
    qs = {q: paths.quantile(q, axis=1) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    days = pd.Index(range(1, len(paths) + 1), name="day")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=qs[0.95], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=days, y=qs[0.05], fill="tonexty", line=dict(width=0),
                   fillcolor="rgba(0,128,255,0.15)", name="5-95%")
    )
    fig.add_trace(go.Scatter(x=days, y=qs[0.75], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=days, y=qs[0.25], fill="tonexty", line=dict(width=0),
                   fillcolor="rgba(0,128,255,0.30)", name="25-75%")
    )
    fig.add_trace(go.Scatter(x=days, y=qs[0.5], line=dict(color="navy", width=2), name="Median"))
    fig.update_layout(title=title, height=420, xaxis_title="Trading days ahead", yaxis_title="Portfolio level")
    return _fig_html(fig)


def histogram_returns(rets: pd.Series, title: str = "Return Distribution") -> str:
    fig = px.histogram(rets, nbins=60, title=title)
    fig.update_layout(showlegend=False, height=350)
    return _fig_html(fig)
