# B3 Systematic Equity

A multifactor stock-selection and asset-allocation system for Brazilian equities: it ranks ~97 IBrX stocks, builds a risk-weighted top-5 portfolio, decides how much capital goes to equities versus CDI, US equities and inflation-linked bonds, and tracks the result against CDI and the Ibovespa. It runs unattended on GitHub Actions and reports through Telegram.

## Methodology

### Stock selection

- **Universe:** ~97 liquid names from the IBrX, collected from brapi.dev with yfinance as fallback.
- **Pillars and weights:**

  | Pillar | Weight | Factors |
  |---|---|---|
  | Fundamental | 45% | earnings yield, P/B, ROE, ROIC, net debt/EBITDA, dividend yield, size, 3y revenue/earnings growth, asset growth |
  | Momentum | 30% | 3/6/12-month return vs. Ibovespa (12-1 convention), residual momentum, post-earnings drift, analyst target revisions |
  | Quality / risk | 25% | 180-day Yang-Zhang volatility, beta, 30-day average traded value |

- **Sector-adaptive normalization:** sector z-score when the sector has 8 or more names, sector percentile for 4 to 7, global z-score below 4. This keeps banks from being ranked against miners on P/B.
- **Hard filters:** average traded value below R$5M/day, net debt/EBITDA above 5x (non-financials), value traps (negative P/E and ROE below -5%), fewer than 3 available fundamentals. Names under R$10M/day get a square-root liquidity penalty on the total score.
- Regime-dependent pillar weights exist in the code but are switched off until they can be validated.

### Portfolio construction

- Hierarchical Risk Parity on the top 5, with a Ledoit-Wolf shrunk covariance over 126 trading days; inverse-volatility fallback.
- Position limits of 5% to 30% per name.
- Volatility targeting at 14% a.a. (EWMA, lambda 0.94), with exposure between 50% and 100%. Cash released by the target goes to the CDI sleeve.
- Diversification cascade (company, sub-sector, sector, macro theme) and a turnover band that gives incumbents a 5-point edge to avoid churn.

### Allocation layer

Capital is split across four sleeves: B3 equities (the top-5 portfolio), CDI, IVVB11 (S&P 500, unhedged) and IMAB11 (NTN-B real yield).

- **Live rule:** a static 40/30/15/15 mix.
- **Tested alternative (`ALLOCATION_MODE = "dynamic"`):** a two-state Gaussian HMM on Ibovespa returns (calm vs. high volatility) sets a regime-dependent base (60% / 40% / 20% in equities). Two signals then tilt equities by up to ±10pp each: the implied equity risk premium (portfolio earnings yield minus Selic) and 12-1 time-series momentum of the Ibovespa vs. CDI.
- The dynamic rule lost to the static mix in every backtest window and start date (see Results), so its signals are still computed and reported but no longer move capital.

## Results

Backtest of the allocation layer (`src/allocation_backtest.py`, `data/allocation_backtest.json`), 2019-05-20 to 2026-10-01: the period in which every sleeve has a traded ETF. Rebalancing every 21 trading days at 10 bps per side, with signals computed on data up to day t and traded on t+1, and weights drifting between rebalances. The equity sleeve is BOVA11, so this tests the allocation rules, not stock selection.

| Strategy | Ann. return | Ann. vol | Sharpe vs CDI | Max drawdown |
|---|---|---|---|---|
| Static mix 40/30/15/15 | 12.5% | 10.7% | 0.28 | -23.8% |
| Dynamic allocation (regime + tilts) | 9.5% | 12.3% | 0.04 | -28.5% |
| Buy and hold BOVA11 | 10.5% | 23.4% | 0.15 | -46.9% |
| 100% CDI | 9.8% | 0.3% | — | 0.0% |

The static mix beat the dynamic rules in every window and start date tested, including the longer 2016–2026 run with CDI standing in for IMAB11 before it listed (Sharpe 0.34 vs. 0.15). Neither is significantly better than CDI: the probabilistic Sharpe ratio of the static mix is 0.78, and the deflated Sharpe stays below 0.4 for any plausible number of strategies tried. What the layer delivers is diversification, with equity-like returns at half the volatility and drawdown of the index. The regime timing has not added value.

## Validation

Every scheduled run writes its artifacts to `data/` and commits them back, so the validation sample grows over time:

- **Factor IC:** Spearman rank IC per factor and horizon over the full scored universe, with Newey-West t-stats for overlapping horizons and a Benjamini-Hochberg correction across factors (`data/factor_ic.json`, `src/factor_analysis.py`). It is a monitor, not a trigger for weight changes.
- **Walk-forward:** out-of-sample returns of each past recommendation over 1, 4 and 12 weeks, with portfolio and Ibovespa on the same window and Sharpe in excess of CDI (`data/walk_forward.json`).
- **Backtest and attribution:** each weekly or monthly run measures the previous recommendation on a total-return basis (dividends and JCP from B3) against the Ibovespa and CDI, charging costs on actual turnover, with a Brinson-style attribution by sector. The Ibovespa is measured from its level at the moment of entry and cross-checked against BOVA11; runs made during the session are flagged as intraday.
- **Equity curve:** a daily NAV of both the full allocated portfolio and the equity sleeve (`data/equity_curve.json`).

Live tracking started in May 2026. With about 17 weekly observations, none of the live metrics is statistically significant: no factor survives the multiple-testing correction, and weekly alpha vs. the Ibovespa has a t-stat below 1. Detecting an IC of 0.05 would take more than a year of weekly data, so factor validation has to come from long point-in-time history rather than from the live sample.

## Architecture

```
.
├── main.py              # weekly/monthly pipeline: collect → score → allocate → backtest → report
├── main_daily.py        # morning and closing reports, equity curve, stop monitoring
├── main_scanner.py      # intraday technical scanner on high-scoring names
├── src/
│   ├── data_collector.py      # brapi + yfinance, disk cache
│   ├── scoring_engine.py      # factors, normalization, filters, diversification
│   ├── snapshot_manager.py    # HRP weights, vol targeting, persistence
│   ├── allocator.py           # HMM regime + sleeve allocation
│   ├── allocation_backtest.py # 10-year backtest of the allocation layer
│   ├── risk_model.py          # simplified factor risk model
│   ├── backtester.py          # previous recommendation vs. benchmarks
│   ├── benchmark.py           # Ibovespa (yfinance), Selic/CDI (BCB SGS)
│   ├── factor_analysis.py     # IC / IR / decay
│   ├── walk_forward.py        # out-of-sample evaluation
│   ├── equity_curve.py        # daily NAV
│   └── ...                    # report builders, charts, Telegram client
├── data/
│   ├── universe.csv     # investable universe
│   ├── history/         # dated snapshots, recommendations and backtests
│   └── *.json           # IC, walk-forward, equity curve, allocation backtest
├── tests/
└── tools/               # one-off data download scripts
```

GitHub Actions workflows (times in BRT):

| Workflow | Schedule | Output |
|---|---|---|
| `weekly_report.yml` | Mondays 08:00 | recommendation, allocation, backtest, report |
| `monthly_report.yml` | 1st of the month 08:00 | same, monthly horizon |
| `daily_morning_report.yml` | weekdays 08:30 | pre-market report |
| `opportunity_scanner.yml` | weekdays 10:30 | technical alerts |
| `daily_closing_report.yml` | weekdays 18:00 | equity curve update, stop alerts |

## How to run

Python 3.11.

```bash
pip install -r requirements.txt
cp .env.example .env        # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, optional BRAPI_TOKEN

python main.py --mode weekly --dry-run            # full pipeline, no Telegram
python main.py --mode weekly --dry-run --capital 50000   # plus an order sheet for R$50k
python main.py --mode monthly --send
python main_daily.py --mode closing --dry-run
python -m src.allocation_backtest --years 10
python -m src.factor_analysis
python -m src.walk_forward

python -m pytest tests -q
```

For scheduled runs, set the same variables as repository secrets and enable the workflows.

## Limitations

- Free data only: fundamentals are latest values, not point-in-time, and the universe is today's list, so long backtests of stock selection carry survivorship bias.
- The allocation backtest uses a volatility proxy for the regime instead of the live HMM, and BOVA11 instead of the stock portfolio.
- The live sample is too short to separate skill from noise.
- Historical index membership and analyst estimates would require a terminal (Bloomberg or Refinitiv). A download script is in `tools/`.

Not investment advice.
