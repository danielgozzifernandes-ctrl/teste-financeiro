# B3 Stock Recommender

Sistema automatizado de recomendação de ações da B3 (universo ~97 tickers do IBrX) com relatórios semanais/mensais e diários via Telegram. Pipeline systematic-equity multifator com detecção de regime, ponderação de risco e arcabouço de validação retroativa.

## O que faz

- **Decide a alocação de capital** ("investidor absoluto"): split entre 4 sleeves — carteira B3 / CDI / IVVB11 (S&P+dólar) / IMAB11 (juro real) — por regime HMM + ERP implícito + time-series momentum 12-1. Com Selic alta, *quanto* estar em bolsa importa mais que *qual* ação (Brinson 1986).
- **Coleta** fundamentos + OHLC de ~97 tickers via [brapi.dev](https://brapi.dev) + yfinance (fallback), com cache diário e suporte point-in-time (`universe_at(date)`).
- **Pontua** cada ação (0–100) com normalização adaptativa por setor (z-score setorial / percentil setorial / z-score global) sobre três pilares e sub-fatores estilo Fama-French/QMJ. Momentum em convenção 12-1 (skip-month).
- **Detecta o regime de mercado** (risk_on / mean_rev / bear) por HMM 2-estados — o regime alimenta a camada de alocação; os pesos por regime no *scoring* estão congelados até validação por IC (n≥8).
- **Constrói a carteira** do top-5 com HRP (linkage ward, Ledoit-Wolf), **cap/floor de peso por posição (30%/5%)** e *volatility targeting* (≤100%, caixa liberado migra pro CDI), com diversificação por empresa/sub-setor/setor/tema macro e *turnover band* anti-churn.
- **Executa como PF**: folha de ordens em quantidades fracionárias para um capital real (`--capital`), nota de IR (isenção R$20k/mês), e stops do trade advisor **monitorados todo fechamento** com alerta de rompimento.
- **Mede a performance** em três camadas: equity curve diária (NAV da carteira completa E do sleeve bolsa vs **CDI** e IBOV, com banda de ruído do alpha), backtest semanal rec-vs-preços (ADV-friction + Brinson) e backtest da camada de alocação com 10 anos de ETFs reais.
- **Reporta** via Telegram com gráfico de performance e alertas de concentração e de qualidade de dados.
- Roda 100% no **GitHub Actions** (semanal, mensal, diário manhã/fechamento, scanner de oportunidades), commitando o histórico de volta ao repo.

> **Honestidade de maturidade:** o motor de scoring é robusto e auditável (cada recomendação traz `why` + `norm_details` rastreáveis). As métricas de validação (IC, walk-forward, track record) só ganham significância com **histórico real acumulado** — veja [Validação e limitações](#validação-e-limitações). Não é "elite hedge fund": é buy-side institucional simplificado.

## Estrutura

```
.
├── main.py                    # Orquestrador semanal/mensal
├── main_daily.py              # Relatórios diários (manhã/fechamento)
├── main_scanner.py            # Scanner de oportunidades event-driven
├── data/
│   ├── universe.csv           # ~97 tickers + inclusion/exclusion_date (PIT)
│   ├── history/               # snapshots, recomendações, backtests datados
│   ├── factor_ic.json         # Information Coefficient por fator
│   └── walk_forward.json      # Backtest out-of-sample rolante
├── src/
│   ├── config.py              # Pesos, flags de fatores e thresholds
│   ├── data_collector.py      # brapi + yfinance + cache + Yang-Zhang vol
│   ├── scoring_engine.py      # Normalização adaptativa, fatores, filtros, diversificação
│   ├── risk_model.py          # Risk model Barra-simplificado (Gram-Schmidt)
│   ├── snapshot_manager.py    # Persistência atômica + HRP + vol targeting
│   ├── backtester.py          # Performance da carteira anterior + Brinson
│   ├── factor_analysis.py     # Framework de IC / IR / hit-rate / decay
│   ├── walk_forward.py        # Backtest out-of-sample por janela
│   ├── benchmark.py           # IBOVESPA (yfinance) + SELIC/CDI (BCB)
│   ├── technical_analyzer.py  # RSI/MACD/Bollinger/ATR (diários)
│   ├── trade_advisor.py       # Entry/target/stop/R-R do top 5
│   ├── report_builder.py      # Mensagem MarkdownV2 (semanal/mensal)
│   ├── chart_generator.py     # Gráfico matplotlib (modo Agg)
│   └── telegram_sender.py     # Envio via Bot API (retry exponencial)
├── tests/
│   ├── test_scoring.py        # Casos do scoring engine (normalização/filtros/pesos)
│   └── test_backtester.py     # Backtester (anti-self-backtest) + gating estatístico
└── .github/workflows/         # weekly / monthly / daily_* / opportunity_scanner
```

## Configuração

### 1. Criar bot no Telegram

1. Abra o [@BotFather](https://t.me/BotFather) no Telegram
2. Envie `/newbot` e siga as instruções
3. Salve o **token** exibido (formato: `123456:ABC-DEF...`)

### 2. Obter o Chat ID

**Para um canal:**
1. Crie um canal e adicione o bot como administrador
2. Envie uma mensagem no canal e acesse:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Procure `"chat":{"id":` — esse número negativo é o Chat ID

**Para uso pessoal:**
1. Envie `/start` para o seu bot
2. Acesse a URL acima e copie o `id` do campo `"from"`

### 3. Configurar GitHub Secrets

No repositório: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Valor |
|--------|-------|
| `TELEGRAM_BOT_TOKEN` | Token do @BotFather |
| `TELEGRAM_CHAT_ID` | ID do canal/grupo (negativo para grupos) |
| `BRAPI_TOKEN` | Token brapi.dev (opcional, aumenta rate limit) |

### 4. Ativar GitHub Actions

1. Faça push do código para `main`
2. Vá em **Actions → Enable workflows**
3. Para testar: **Actions → Weekly Report → Run workflow → dry_run: true**

## Execução local

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com seu token e chat_id

# Dry run (não envia ao Telegram)
python main.py --mode weekly --dry-run

# Envio real
python main.py --mode weekly --send

# Relatório mensal
python main.py --mode monthly --send

# Data específica
python main.py --mode weekly --date 2025-01-06 --dry-run

# Validar token
python main.py --validate-token

# Debug verbose
python main.py --mode weekly --dry-run --debug

# Folha de ordens executável para um capital real (qty fracionária + IR)
python main.py --mode weekly --dry-run --capital 50000

# Backtest da camada de alocação (10 anos de ETFs reais)
python -m src.allocation_backtest --years 10
```

## Modelo de scoring

Pesos-base dos pilares (regime `mean_rev`; variam por regime — ver abaixo):

| Pilar | Peso base | Fatores |
|-------|-----------|---------|
| Fundamentalista | 45% | Earnings Yield, P/VP, ROE, ROIC, Dívida/EBITDA, DY **+ sub-fatores** Size (log mkt cap), Growth (receita/lucro 3y), Investment (asset growth) |
| Momentum | 30% | Alpha 3m/6m/12m vs IBOV, momentum idiossincrático (resíduo OLS), PEAD, revisões/preço-alvo de analistas |
| Qualidade/Risco | 25% | Volatilidade 180d (Yang-Zhang), Beta, Volume médio 30d |

**Pesos por regime no scoring: CONGELADOS** (`ENABLE_REGIME_ADAPTIVE_WEIGHTS=False`). Os três conjuntos de pesos por regime nunca foram validados (IC com n≈2); o regime detectado pelo HMM alimenta apenas a **camada de alocação de capital**, onde a regra é backtestável com 10 anos de dados de ETF (`python -m src.allocation_backtest`).

**Camada de alocação (investidor absoluto):** base por regime (risk_on 60% / mean_rev 40% / bear 20% em bolsa) com tilts de ±10pp por ERP implícito (EY da carteira − Selic) e TSMOM 12-1 do IBOV vs CDI; bolsa limitada a [10%, 70%]. Resultado honesto do backtest 2016–2026: o mix **estático** 40/30/15/15 teve Sharpe melhor que as regras dinâmicas (0,34 vs 0,21) — o ganho robusto da camada é a *diversificação em si* (≈ retorno da bolsa pura com metade da vol/drawdown), não o timing.

**Normalização adaptativa por setor:** N ≥ 8 → z-score setorial · N ∈ [4,7] → percentil setorial · N < 4 → z-score global.

**Filtros duros:** liquidez < R$5M/dia · D/EBITDA > 5x (ex-financeiro) · value trap (P/L<0 & ROE<−5%) · cobertura mínima de fundamentos (≥3 não-NaN). Penalidade de liquidez `sqrt(ADV/threshold)` no score total.

**Construção de carteira:** HRP (Ledoit-Wolf) → *volatility targeting* (14% a.a., EWMA λ=0.94) → diversificação em cascata (empresa → sub-setor → setor → tema macro) → *turnover band* (+5 pts a incumbentes).

## Validação e limitações

O sistema acumula, a cada execução, três artefatos de validação em `data/`:

- **`backtester`** — performance da carteira **anterior** (estritamente anterior à data atual) vs IBOV/CDI/SELIC, com fricção ajustada por ADV e atribuição Brinson. O backtester nunca compara uma recomendação contra os preços do próprio dia (`period_days > 0` garantido).
- **`factor_ic.json`** — Information Coefficient (Spearman) por fator/janela, com `data_sufficiency` indicando se há observações suficientes.
- **`walk_forward.json`** — backtest out-of-sample rolante por janela (1w/4w/12w), também com `data_sufficiency`.

**Gating estatístico (importante):** IC/IR/hit-rate/Sharpe só são confiáveis com amostra suficiente. Abaixo de `MIN_OBS_FOR_SIGNIFICANCE` (8 obs/fator) ou `MIN_PERIODS_WALK_FORWARD` (6 janelas), os JSONs marcam `"is_significant": false` e exibem um aviso — **trate esses números como ruído, não sinal**, até acumular ~3 meses de snapshots semanais reais.

**Limitações conhecidas:**
- Dados gratuitos: sem consenso EPS estruturado, sem fundamentos PIT com *vintage*, sem composição setorial histórica do IBOV. Beta de fonte é validado por bounds de sanidade e preterido pelo beta calculado dos preços.
- Cobertura: nem todos os ~97 tickers sobrevivem à coleta + filtros a cada run. A cobertura efetiva (declarado → coletado → pontuado) é logada e persistida em `execution_metadata.data_quality`; abaixo de `MIN_UNIVERSE_COVERAGE` (60%) o relatório emite alerta.
- HMM detecta regime contemporâneo com lag de ~10 dias (limite teórico).

```bash
# Análises retroativas standalone
python -m src.factor_analysis      # regenera data/factor_ic.json
python -m src.walk_forward         # regenera data/walk_forward.json
```

## 🔴 Próximo passo crítico: dados Bloomberg (Insper)

O maior limitador do sistema hoje é **dado, não modelo**: universo com survivorship bias, sem consenso de EPS real, sem fundamentos point-in-time. Tudo isso se resolve com **1 ida ao lab do Insper** com Bloomberg Terminal.

**→ Passo a passo completo em [`docs/BLOOMBERG_DOWNLOAD.md`](docs/BLOOMBERG_DOWNLOAD.md)** (script pronto em `tools/bloomberg_download.py`).

Prioridade dos downloads (bang/buck):
1. **Composição histórica do IBX/IBrA** — mata o survivorship bias do universo (destrava walk-forward honesto)
2. **BEst EPS + Reported EPS** — SUE real para o fator PEAD
3. **BEst Target Price (mean/high/low + nº analistas)** — upside com dispersão real
4. Short interest, PIT fundamentals, recommendation distribution

## Disclaimer

> Este sistema é uma ferramenta de análise quantitativa automatizada. **Não constitui recomendação de investimento.** Faça sua própria análise antes de tomar qualquer decisão financeira. O autor não se responsabiliza por perdas decorrentes do uso deste sistema.
