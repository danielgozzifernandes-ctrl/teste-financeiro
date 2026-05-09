# B3 Stock Recommender

Sistema automatizado de recomendação de ações da B3 com relatórios semanais e mensais via Telegram.

## O que faz

- Coleta dados fundamentalistas de 83 tickers via [brapi.dev](https://brapi.dev) + yfinance (fallback)
- Pontua cada ação com normalização Z-Score setorial adaptativa (5 fatores)
- Compara desempenho da carteira anterior vs IBOVESPA, CDI e SELIC
- Gera gráfico de performance acumulada (1200×800px, fundo escuro)
- Envia relatório semanal (toda segunda) e mensal (dia 1) para o Telegram
- Roda 100% no GitHub Actions (gratuito)

## Estrutura

```
.
├── main.py                    # Orquestrador
├── requirements.txt
├── data/
│   ├── universe.csv           # 83 tickers do IBrX-100
│   └── snapshots/             # Histórico de preços e recomendações
├── src/
│   ├── config.py              # Parâmetros e pesos do modelo
│   ├── data_collector.py      # brapi.dev + yfinance + cache
│   ├── scoring_engine.py      # Normalização adaptativa + scoring
│   ├── benchmark.py           # IBOVESPA (yfinance) + SELIC/CDI (BCB)
│   ├── snapshot_manager.py    # Persistência atômica de snapshots
│   ├── chart_generator.py     # Gráfico matplotlib (modo Agg)
│   ├── report_builder.py      # Mensagem MarkdownV2 para Telegram
│   ├── backtester.py          # Backtesting da carteira anterior
│   └── telegram_sender.py     # Envio via Bot API (retry exponencial)
├── tests/
│   └── test_scoring.py        # 13 casos de teste do scoring engine
└── .github/workflows/
    ├── weekly_report.yml      # Cron: segunda 08h Brasília
    └── monthly_report.yml     # Cron: dia 1 08h Brasília
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
```

## Modelo de scoring

| Categoria | Peso | Fatores |
|-----------|------|---------|
| Fundamentalista | 45% | Earnings Yield, P/VP, ROE, ROIC, DY |
| Momentum | 30% | Alpha 3m, 6m, 12m vs IBOVESPA |
| Qualidade | 25% | Margem Líquida, Dívida/EBITDA |

**Normalização adaptativa por setor:**
- N ≥ 8 tickers → Z-Score setorial
- N ∈ [4, 7] → Percentil setorial
- N < 4 → Z-Score global (fallback)

## Disclaimer

> Este sistema é uma ferramenta de análise quantitativa automatizada. **Não constitui recomendação de investimento.** Faça sua própria análise antes de tomar qualquer decisão financeira. O autor não se responsabiliza por perdas decorrentes do uso deste sistema.
