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
    {"key": "explorer", "label": "Explorer", "href": "explorer.html"},
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


def _write_series_json(prices: pd.DataFrame, out_dir: Path) -> dict:
    """Export per-ETF JSON files + an index for the Explorer page.

    Returns a small summary dict (n_codes, n_with_data).
    """
    uni = load_universe().df.copy()
    uni["code"] = uni["code"].astype(str)

    series_dir = out_dir / "data" / "series"
    series_dir.mkdir(parents=True, exist_ok=True)

    n_with_data = 0
    index_rows: list[dict] = []

    for row in uni.itertuples(index=False):
        code = str(row.code)
        meta_row = {
            "code": code,
            "name": row.name,
            "risk_type": row.risk_type,
            "asset_class": row.asset_class,
            "sub_category": row.sub_category,
            "aum_eok": int(row.aum_eok) if pd.notna(row.aum_eok) else 0,
            "start": None,
            "end": None,
            "n_obs": 0,
        }
        if code in prices.columns:
            s = prices[code].dropna()
            if len(s) > 0:
                dates = s.index.strftime("%Y-%m-%d").tolist()
                closes = [round(float(v), 4) for v in s.values]
                payload = {"code": code, "name": row.name, "d": dates, "c": closes}
                (series_dir / f"{code}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                meta_row["start"] = dates[0]
                meta_row["end"] = dates[-1]
                meta_row["n_obs"] = len(dates)
                n_with_data += 1
        index_rows.append(meta_row)

    index_path = out_dir / "data" / "series_index.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "n_codes": len(index_rows),
                "n_with_data": n_with_data,
                "series": index_rows,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    LOG.info("Series JSON written: %d codes (%d with data) -> %s", len(index_rows), n_with_data, series_dir)
    return {"n_codes": len(index_rows), "n_with_data": n_with_data}


def page_explorer() -> str:
    """Interactive FRED-style time-series explorer (client-side JS + Plotly)."""
    # Plotly CDN is loaded here (other pages embed plotlyjs="cdn" per figure already)
    # All UI logic is plain JS that fetches /data/series_index.json + /data/series/{code}.json
    return r"""
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

<div class="card">
  <h3>Asset Explorer</h3>
  <p style="color:var(--muted); margin-top:-4px; font-size:13px;">
    대분류 → 세부분류 → 종목을 골라 차트에 추가하세요. 여러 시리즈를 겹쳐서 비교할 수 있습니다 (FRED 스타일).
  </p>

  <div style="display:grid; grid-template-columns: 1.1fr 1.4fr 2fr auto; gap:10px; align-items:end; margin-top:10px;">
    <div>
      <label class="lbl">대분류 (asset_class)</label>
      <select id="sel-class"></select>
    </div>
    <div>
      <label class="lbl">세부분류 (sub_category)</label>
      <select id="sel-sub"></select>
    </div>
    <div>
      <label class="lbl">종목 (code · name)</label>
      <select id="sel-code"></select>
    </div>
    <div>
      <button id="btn-add" class="btn primary">＋ Add to chart</button>
    </div>
  </div>

  <div id="info-panel" style="margin-top:14px;"></div>
</div>

<div class="card">
  <div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between;">
    <div>
      <strong>기간</strong>
      <span id="range-buttons" style="margin-left:8px;">
        <button data-range="1M" class="range">1M</button>
        <button data-range="3M" class="range">3M</button>
        <button data-range="6M" class="range">6M</button>
        <button data-range="YTD" class="range">YTD</button>
        <button data-range="1Y" class="range active">1Y</button>
        <button data-range="3Y" class="range">3Y</button>
        <button data-range="5Y" class="range">5Y</button>
        <button data-range="MAX" class="range">MAX</button>
      </span>
      <span style="margin-left:14px;">
        <label class="lbl-inline">From</label>
        <input type="date" id="date-from">
        <label class="lbl-inline">To</label>
        <input type="date" id="date-to">
        <button id="btn-apply-range" class="btn">Apply</button>
      </span>
    </div>
    <div>
      <label class="lbl-inline"><input type="checkbox" id="opt-rebase"> Rebase to 100</label>
      &nbsp;
      <label class="lbl-inline"><input type="checkbox" id="opt-log"> Log scale</label>
    </div>
  </div>

  <div id="chart" style="height:520px; margin-top:14px;"></div>
  <div id="series-list" style="margin-top:8px;"></div>
</div>

<style>
  .lbl { display:block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
  .lbl-inline { color: var(--muted); font-size: 12px; margin-right: 4px; }
  select, input[type=date] { padding:6px 8px; border:1px solid var(--line); border-radius:6px; font-size:13px; background:white; }
  select { width: 100%; }
  .btn { padding:7px 12px; border:1px solid var(--line); background:white; border-radius:6px; cursor:pointer; font-size:13px; }
  .btn:hover { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: var(--accent); color: white; border-color: var(--accent); }
  .btn.primary:hover { color: white; opacity: 0.92; }
  .range { padding:4px 10px; border:1px solid var(--line); background:white; border-radius:14px; cursor:pointer; font-size:12px; margin-right:4px; }
  .range.active { background: var(--accent); color: white; border-color: var(--accent); }
  .range:hover { border-color: var(--accent); }
  .chip { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border:1px solid var(--line); border-radius:16px; margin:4px 6px 0 0; font-size:12px; background:#f7f9fc; }
  .chip .swatch { width:10px; height:10px; border-radius:50%; }
  .chip .x { cursor:pointer; color: var(--muted); margin-left:4px; }
  .chip .x:hover { color: #c8341e; }
  #info-panel .infogrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:10px; }
  #info-panel .item { background:#f7f9fc; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  #info-panel .item .k { color: var(--muted); font-size: 11px; }
  #info-panel .item .v { font-size: 14px; font-weight: 600; }
</style>

<script>
(function(){
  const COLORS = ["#234bdf","#c8341e","#198042","#a06b00","#7e3ab3","#0a8a8a","#bf3475","#3a5e8c","#8c5e3a","#5e8c3a"];
  const state = {
    index: null,         // series_index.json content
    cache: {},           // code -> {d:[],c:[]}
    series: [],          // [{code,name,color}]
    range: "1Y",         // 1M/3M/6M/YTD/1Y/3Y/5Y/MAX/CUSTOM
    customFrom: null,
    customTo: null,
    rebase: false,
    log: false,
  };

  const fmtPct = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : (x>=0?"+":"") + (x*100).toFixed(2) + "%";
  const fmtNum = (x) => (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toLocaleString(undefined,{maximumFractionDigits:2});

  function $(id){ return document.getElementById(id); }

  function init(){
    fetch("data/series_index.json").then(r=>r.json()).then(idx => {
      state.index = idx;
      populateClassDropdown();
      attachEvents();
      // Try a sensible default selection: largest-AUM with data
      const withData = idx.series.filter(s => s.n_obs > 0).sort((a,b)=>b.aum_eok-a.aum_eok);
      if (withData.length){
        const def = withData[0];
        $("sel-class").value = def.asset_class;
        onClassChange();
        $("sel-sub").value = def.sub_category;
        onSubChange();
        $("sel-code").value = def.code;
        addSeries(def.code);
      } else {
        renderInfoPanel(null);
        renderChart();
      }
    }).catch(err => {
      $("chart").innerHTML = "<div style='padding:30px;color:#c8341e;text-align:center;'>" +
        "데이터가 아직 없습니다. <code>scripts/fetch_daily.py</code> 실행 후 다시 빌드하세요.<br><small>"+err+"</small></div>";
    });
  }

  function populateClassDropdown(){
    const classes = [...new Set(state.index.series.map(s=>s.asset_class))].sort();
    const sel = $("sel-class");
    sel.innerHTML = classes.map(c=>`<option value="${c}">${c}</option>`).join("");
    onClassChange();
  }

  function onClassChange(){
    const cls = $("sel-class").value;
    const subs = [...new Set(state.index.series.filter(s=>s.asset_class===cls).map(s=>s.sub_category))].sort();
    const sel = $("sel-sub");
    sel.innerHTML = subs.map(c=>`<option value="${c}">${c}</option>`).join("");
    onSubChange();
  }

  function onSubChange(){
    const cls = $("sel-class").value;
    const sub = $("sel-sub").value;
    const codes = state.index.series.filter(s=>s.asset_class===cls && s.sub_category===sub)
      .sort((a,b)=>b.aum_eok-a.aum_eok);
    const sel = $("sel-code");
    sel.innerHTML = codes.map(s=>{
      const tag = s.n_obs > 0 ? "" : " ⚠️ no data";
      return `<option value="${s.code}">${s.code} · ${s.name} (AUM ${s.aum_eok.toLocaleString()}억)${tag}</option>`;
    }).join("");
    onCodeChange();
  }

  function onCodeChange(){
    const code = $("sel-code").value;
    if(!code){ renderInfoPanel(null); return; }
    const meta = state.index.series.find(s=>s.code===code);
    if (meta.n_obs > 0) {
      loadCode(code).then(()=>renderInfoPanel(code));
    } else {
      renderInfoPanel(code);
    }
  }

  function loadCode(code){
    if (state.cache[code]) return Promise.resolve(state.cache[code]);
    return fetch(`data/series/${code}.json`).then(r=>r.json()).then(j=>{ state.cache[code]=j; return j; });
  }

  function addSeries(code){
    if (state.series.find(s=>s.code===code)) return;  // dedupe
    if (state.series.length >= COLORS.length){ alert("최대 "+COLORS.length+"개 시리즈까지 비교할 수 있어요."); return; }
    const meta = state.index.series.find(s=>s.code===code);
    if (!meta || meta.n_obs === 0) { alert("이 종목은 아직 데이터가 없습니다."); return; }
    const color = COLORS[state.series.length];
    state.series.push({code, name: meta.name, color});
    loadCode(code).then(()=>{ renderChart(); renderSeriesList(); });
  }

  function removeSeries(code){
    state.series = state.series.filter(s=>s.code!==code);
    // re-color
    state.series.forEach((s,i)=>{ s.color = COLORS[i]; });
    renderChart();
    renderSeriesList();
  }

  function rangeBounds(){
    if (state.range === "CUSTOM" && state.customFrom && state.customTo){
      return [state.customFrom, state.customTo];
    }
    // Compute relative to the latest date across loaded series (or today)
    let latest = new Date();
    state.series.forEach(s=>{
      const d = state.cache[s.code];
      if (d && d.d.length){
        const lastD = new Date(d.d[d.d.length-1]);
        if (lastD > latest) latest = lastD;
      }
    });
    const days = {"1M":30,"3M":91,"6M":182,"1Y":365,"3Y":365*3,"5Y":365*5};
    let from;
    if (state.range === "YTD"){
      from = new Date(latest.getFullYear(),0,1);
    } else if (state.range === "MAX"){
      from = new Date(1970,0,1);
    } else {
      const d = days[state.range] || 365;
      from = new Date(latest.getTime() - d*86400000);
    }
    const to = latest;
    const iso = (x)=>x.toISOString().slice(0,10);
    return [iso(from), iso(to)];
  }

  function sliceSeries(d, c, fromIso, toIso){
    let i0=0, i1=d.length-1;
    while (i0 < d.length && d[i0] < fromIso) i0++;
    while (i1 >= 0 && d[i1] > toIso) i1--;
    if (i0 > i1) return {d:[],c:[]};
    return {d: d.slice(i0, i1+1), c: c.slice(i0, i1+1)};
  }

  function renderChart(){
    const [fromIso, toIso] = rangeBounds();
    const traces = state.series.map(s=>{
      const raw = state.cache[s.code];
      if (!raw) return null;
      const sl = sliceSeries(raw.d, raw.c, fromIso, toIso);
      let y = sl.c;
      if (state.rebase && y.length){
        const base = y[0];
        y = y.map(v => v / base * 100);
      }
      return {
        type:"scattergl", mode:"lines", name: s.code+" "+s.name,
        x: sl.d, y, line:{color: s.color, width: 1.6}
      };
    }).filter(Boolean);
    const layout = {
      margin:{l:54,r:18,t:18,b:38},
      xaxis:{ range:[fromIso,toIso], showgrid:true },
      yaxis:{ title: state.rebase ? "Rebased = 100" : "Price (KRW)", type: state.log ? "log" : "linear" },
      hovermode: "x unified",
      legend: {orientation:"h", y:-0.2},
      paper_bgcolor:"white",
    };
    Plotly.react("chart", traces, layout, {responsive:true, displaylogo:false});
  }

  function renderSeriesList(){
    const el = $("series-list");
    if (!state.series.length){ el.innerHTML = "<span style='color:var(--muted); font-size:12px;'>선택된 시리즈가 없습니다.</span>"; return; }
    el.innerHTML = state.series.map(s=>`
      <span class="chip">
        <span class="swatch" style="background:${s.color}"></span>
        <span>${s.code} · ${s.name}</span>
        <span class="x" data-code="${s.code}">✕</span>
      </span>`).join("");
    el.querySelectorAll(".x").forEach(x=>{
      x.addEventListener("click", ()=>removeSeries(x.dataset.code));
    });
  }

  function returnsFor(d, c){
    if (!d || !d.length) return {};
    const last = c[c.length-1];
    const latestDate = new Date(d[d.length-1]);
    const findBack = (days)=>{
      const target = new Date(latestDate.getTime() - days*86400000).toISOString().slice(0,10);
      // find first idx >= target
      let lo=0,hi=d.length-1,ans=-1;
      while(lo<=hi){const m=(lo+hi)>>1; if (d[m]>=target){ans=m;hi=m-1;} else lo=m+1;}
      return ans>=0 ? c[ans] : null;
    };
    const ytdTarget = new Date(latestDate.getFullYear(),0,1).toISOString().slice(0,10);
    let ytdBase = null;
    for (let i=0;i<d.length;i++){ if (d[i] >= ytdTarget){ ytdBase = c[i]; break; } }
    const r = (base) => (base===null||base===undefined||base===0) ? null : (last/base - 1);
    return {
      last,
      "1D": r(findBack(1)),
      "1W": r(findBack(7)),
      "1M": r(findBack(30)),
      "3M": r(findBack(91)),
      "6M": r(findBack(182)),
      "YTD": r(ytdBase),
      "1Y": r(findBack(365)),
      "3Y": r(findBack(365*3)),
    };
  }

  function annualizedVol(c){
    if (c.length < 21) return null;
    let mean=0; const rets=[];
    for(let i=1;i<c.length;i++){ const r = Math.log(c[i]/c[i-1]); rets.push(r); mean+=r; }
    mean /= rets.length;
    let s=0; for (const r of rets) s += (r-mean)*(r-mean);
    return Math.sqrt(s/(rets.length-1)) * Math.sqrt(252);
  }

  function maxDD(c){
    if (!c.length) return null;
    let peak = c[0], maxdd = 0;
    for (const v of c){ if (v>peak) peak=v; const dd = v/peak - 1; if (dd<maxdd) maxdd=dd; }
    return maxdd;
  }

  function renderInfoPanel(code){
    const el = $("info-panel");
    if (!code){ el.innerHTML = ""; return; }
    const meta = state.index.series.find(s=>s.code===code);
    if (!meta){ el.innerHTML = ""; return; }
    const data = state.cache[code];
    let returnsHtml = "";
    let extra = "";
    if (data){
      const r = returnsFor(data.d, data.c);
      // For vol/MDD use last 1Y
      const [fIso, tIso] = (function(){
        const last = new Date(data.d[data.d.length-1]);
        const from = new Date(last.getTime() - 365*86400000).toISOString().slice(0,10);
        return [from, data.d[data.d.length-1]];
      })();
      const sl = sliceSeries(data.d, data.c, fIso, tIso);
      const vol = annualizedVol(sl.c);
      const mdd = maxDD(sl.c);
      const items = [
        ["Last price", fmtNum(r.last)],
        ["1D", fmtPct(r["1D"])], ["1W", fmtPct(r["1W"])],
        ["1M", fmtPct(r["1M"])], ["3M", fmtPct(r["3M"])],
        ["6M", fmtPct(r["6M"])], ["YTD", fmtPct(r.YTD)],
        ["1Y", fmtPct(r["1Y"])], ["3Y", fmtPct(r["3Y"])],
        ["Vol (1Y)", fmtPct(vol)], ["Max DD (1Y)", fmtPct(mdd)],
      ];
      returnsHtml = "<div class='infogrid'>" + items.map(([k,v])=>{
        const cls = (typeof v === "string" && v.startsWith("+")) ? "pos" : (typeof v === "string" && v.startsWith("-")) ? "neg" : "";
        return `<div class="item"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
      }).join("") + "</div>";
      extra = `<div style='color:var(--muted); font-size:12px; margin-top:6px;'>데이터 기간: ${meta.start} ~ ${meta.end} (${meta.n_obs}일)</div>`;
    } else if (meta.n_obs === 0) {
      extra = "<div style='color:#c8341e; font-size:12px; margin-top:6px;'>이 종목은 아직 데이터가 없습니다.</div>";
    }
    el.innerHTML = `
      <div style='display:flex; justify-content:space-between; align-items:baseline; gap:10px;'>
        <div>
          <strong style='font-size:16px;'>${meta.code} · ${meta.name}</strong>
          <span style='color:var(--muted); font-size:12px; margin-left:8px;'>
            ${meta.risk_type} · ${meta.asset_class} · ${meta.sub_category} · AUM ${meta.aum_eok.toLocaleString()}억
          </span>
        </div>
      </div>
      ${returnsHtml}
      ${extra}
    `;
  }

  function attachEvents(){
    $("sel-class").addEventListener("change", onClassChange);
    $("sel-sub").addEventListener("change", onSubChange);
    $("sel-code").addEventListener("change", onCodeChange);
    $("btn-add").addEventListener("click", ()=> {
      const code = $("sel-code").value;
      if (code) addSeries(code);
    });
    document.querySelectorAll("#range-buttons .range").forEach(b=>{
      b.addEventListener("click", ()=>{
        document.querySelectorAll("#range-buttons .range").forEach(x=>x.classList.remove("active"));
        b.classList.add("active");
        state.range = b.dataset.range;
        state.customFrom = state.customTo = null;
        renderChart();
      });
    });
    $("btn-apply-range").addEventListener("click", ()=>{
      const f = $("date-from").value, t = $("date-to").value;
      if (!f || !t || f > t){ alert("From/To 날짜를 확인하세요."); return; }
      state.range = "CUSTOM"; state.customFrom = f; state.customTo = t;
      document.querySelectorAll("#range-buttons .range").forEach(x=>x.classList.remove("active"));
      renderChart();
    });
    $("opt-rebase").addEventListener("change", e=>{ state.rebase = e.target.checked; renderChart(); });
    $("opt-log").addEventListener("change", e=>{ state.log = e.target.checked; renderChart(); });
  }

  init();
})();
</script>
"""


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

    # Write per-series JSON for the Explorer page
    _write_series_json(prices, out)

    # Pages
    pages = {
        "dashboard": ("Dashboard", page_dashboard(prices, returns)),
        "explorer": ("Explorer", page_explorer()),
        "universe": ("Universe", page_universe()),
        "backtest": ("Backtest", page_backtest(prices, strategies)),
        "risk": ("Risk", page_risk(prices, returns, current_weights)),
        "portfolio": ("Portfolio", page_portfolio(prices, current_weights)),
    }

    file_map = {
        "dashboard": "index.html",
        "explorer": "explorer.html",
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
