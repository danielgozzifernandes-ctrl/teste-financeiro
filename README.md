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

- A two-state Gaussian HMM (calm vs. high-volatility) is fitted on Ibovespa returns; the posterior probability of the high-volatility state maps to risk-on, mean-reverting (no clear state) or bear. A simple Ibovespa/VIX rule is the fallback.
- Each regime has a base split across four sleeves: B3 equities (the top-5 portfolio), CDI, IVVB11 (S&P 500, unhedged) and IMAB11 (NTN-B real yield). Equities get 60% / 40% / 20% of capital by regime.
- Two signals tilt equities by up to ±10pp each: the implied equity risk premium (portfolio earnings yield minus Selic) and 12-1 time-series momentum of the Ibovespa vs. CDI. The equity sleeve is bounded to 10%–70%.

## Results

Ten-year backtest of the allocation layer (`src/allocation_backtest.py`, 2016-05-12 to 2026-06-11). Rebalancing every 21 trading days, 10 bps per side. The equity sleeve is BOVA11, so this tests the allocation rules, not stock selection.

| Strategy | Ann. return | Ann. vol | Sharpe vs CDI | Max drawdown |
|---|---|---|---|---|
| Static mix 40/30/15/15 | 12.4% | 10.4% | 0.34 | -25.5% |
| Dynamic allocation (regime + tilts) | 11.0% | 11.7% | 0.21 | -22.9% |
| Buy and hold BOVA11 | 12.8% | 23.2% | 0.26 | -46.9% |
| 100% CDI | 9.1% | 0.3% | — | 0.0% |

The static mix beat the dynamic rules on return and risk-adjusted return. The value of this layer so far comes from diversification: roughly equity-like returns at half the volatility and drawdown. The regime timing has not added value.

## Validation

Every scheduled run writes its artifacts to `data/` and commits them back, so the validation sample grows over time:

- **Factor IC:** Spearman rank IC per factor and horizon, with a data-sufficiency flag (`data/factor_ic.json`, `src/factor_analysis.py`).
- **Walk-forward:** out-of-sample returns of each past recommendation over 1, 4 and 12 weeks (`data/walk_forward.json`).
- **Backtest and attribution:** each weekly or monthly run measures the previous recommendation against the Ibovespa and CDI, with liquidity-adjusted transaction costs and a Brinson-style attribution by sector. The benchmark return is cross-checked against BOVA11 over the same window.
- **Equity curve:** a daily NAV of both the full allocated portfolio and the equity sleeve (`data/equity_curve.json`).

Live tracking started in May 2026. With about 20 weekly observations, none of the live metrics (IC, alpha, hit rate) is statistically significant yet, and they should be read as noise until the sample is much larger.

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
- Live portfolio returns are price-only, while the Ibovespa is a total-return index, which understates the portfolio by the dividend yield.
- The allocation backtest uses a volatility proxy for the regime instead of the live HMM, and BOVA11 instead of the stock portfolio.
- The live sample is too short to separate skill from noise.
- Historical index membership and analyst estimates would require a terminal (Bloomberg or Refinitiv). A download script is in `tools/`.

Not investment advice.
