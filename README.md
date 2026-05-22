# GAPS Asset Allocation (2026-06 ~ 2026-09)

3-4개월(2026년 6월~9월) 자산배분 운용을 위한 데이터·백테스트·리스크·리포트 자동화 파이프라인.

- **Universe**: 한국 상장 ETF 188종 (위험자산 138 + 안전자산 50)
- **Data**: `pykrx` 1차 / `yfinance` 폴백 — 가용한 최대 과거 데이터
- **Auto-fetch**: GitHub Actions cron (매 영업일 KST 18:30)
- **Backtest**: 3개월 rolling walk-forward, 리밸런싱 주기 파라미터화 (weekly/monthly/quarterly)
- **Risk**: VaR (Historical/Parametric/Monte Carlo), Expected Shortfall (CVaR), MC fan chart, exposure, correlation
- **Report**: 멀티 페이지 HTML → GitHub Pages 배포

## 디렉토리 구조

```
config/
  universe.csv          # 188 ETF 분류표
  settings.yml          # 전역 설정 (백테스트/리스크/리포트)
data/
  raw/                  # 종목별 parquet (pykrx 결과)
  processed/            # prices_close.parquet, returns_simple.parquet, returns_log.parquet
  meta/last_run.json    # 마지막 패치 요약
src/db_gaps/
  data/                 # universe / fetcher / pipeline
  strategy/             # base, registry, builtin (equal_weight, risk_parity_naive, momentum_topn, min_variance)
  backtest/             # engine, rebalancer, metrics
  risk/                 # var, es, monte_carlo, exposure
  report/               # html_builder, plots, templates/
  utils/                # config, logging
scripts/
  fetch_daily.py        # 데이터 패치
  run_backtest.py       # 단일 백테스트 (CLI)
  build_report.py       # 멀티페이지 HTML 빌드
  full_pipeline.py      # fetch + report (one shot)
docs/                   # GitHub Pages 산출물
.github/workflows/
  daily_fetch.yml       # 일일 cron: fetch → report → commit → Pages 배포
```

## 빠른 시작

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 데이터 패치 (최초 실행 시 188개 × 가용 최대 기간)
python scripts/fetch_daily.py

# 3) 백테스트 (예: 동일가중, 월간 리밸런싱)
python scripts/run_backtest.py --strategy equal_weight --rebalance monthly

# 4) 3개월 rolling walk-forward
python scripts/run_backtest.py --strategy risk_parity_naive --rolling-3m

# 5) HTML 리포트 생성 → docs/
python scripts/build_report.py

# 6) 전체 한 번에
python scripts/full_pipeline.py
```

## 전략 추가하기

`src/db_gaps/strategy/` 안에 새 파일을 만들고 등록만 하면 됩니다.

```python
from db_gaps.strategy import register
from db_gaps.strategy.base import Strategy, constrain_weights

@register("my_strategy")
class MyStrategy(Strategy):
    name: str = "my_strategy"
    def decide(self, returns, prices, asof):
        # returns/prices = 룩백 윈도우 데이터, asof = 리밸런싱 시점
        ...
        return constrain_weights(weights, max_weight=0.20)
```

`config/settings.yml` 의 `strategies.active` 에 이름을 추가하면 리포트에 포함됩니다.

## 설정 (`config/settings.yml`)

| 키 | 의미 |
| --- | --- |
| `backtest.rebalance_frequency` | weekly / monthly / quarterly |
| `backtest.test_window_days` | rolling 3M 윈도우 (기본 63 영업일) |
| `backtest.train_lookback_days` | 결정 시 룩백 (기본 252) |
| `backtest.cost_bps` / `slippage_bps` | 거래비용/슬리피지 (bp) |
| `risk.confidence_levels` | [0.95, 0.99] |
| `risk.horizon_days` | [1, 5, 21] |
| `risk.mc_paths` / `mc_horizon_days` | MC 경로 수 / 시뮬 일수 |

## 자동화

`.github/workflows/daily_fetch.yml` 가 매 영업일 KST 18:30 에 실행되어
1) 188개 ETF 데이터 패치 → 2) processed 행렬 갱신 → 3) HTML 리포트 빌드 →
4) `claude/loving-einstein-3rG26` 브랜치로 커밋 → 5) GitHub Pages 배포 까지 처리합니다.

수동 실행: GitHub Actions 탭 → "Daily Data Fetch + Report" → Run workflow.

## 테스트

```bash
pip install -e ".[dev]"
pytest -q
```
