"""
Configurações centralizadas do Recomendador B3.
Todos os parâmetros, pesos, thresholds e paths ficam aqui.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
HISTORY_DIR   = DATA_DIR / "history"
CACHE_DIR     = ROOT_DIR / "cache"
CACHE_DIR_PATH = CACHE_DIR  # alias para compatibilidade
SNAPSHOT_DIR  = DATA_DIR / "snapshots"
OUTPUT_DIR    = ROOT_DIR / "output"
UNIVERSE_FILE = DATA_DIR / "universe.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Pesos principais do score composto
# Contexto: Selic 15% penaliza empresas alavancadas → fundamental pesa mais
# ---------------------------------------------------------------------------
WEIGHTS = {
    "fundamental": 0.45,
    "momentum":    0.30,
    "quality":     0.25,
}

# ---------------------------------------------------------------------------
# Fatores fundamentalistas
# direction: lower_is_better = valores menores recebem score maior
# ---------------------------------------------------------------------------
FUNDAMENTAL_FACTORS = {
    "pl": {
        "weight":    0.20,
        "direction": "lower_is_better",
        "label":     "P/L",
        "norm":      "sectoral",   # normalização setorial adaptativa
    },
    "pvp": {
        "weight":    0.15,
        "direction": "lower_is_better",
        "label":     "P/VP",
        "norm":      "sectoral",
    },
    "roe": {
        "weight":    0.25,
        "direction": "higher_is_better",
        "label":     "ROE",
        "norm":      "sectoral",
    },
    "roic": {
        "weight":    0.20,
        "direction": "higher_is_better",
        "label":     "ROIC",
        "norm":      "sectoral",
    },
    "divida_ebitda": {
        "weight":    0.10,
        "direction": "lower_is_better",
        "label":     "Dívida/EBITDA",
        "norm":      "sectoral",
    },
    "dividend_yield": {
        "weight":    0.10,
        "direction": "higher_is_better",
        "label":     "Dividend Yield",
        "norm":      "sectoral",
    },
}

# ---------------------------------------------------------------------------
# Fatores de momentum
# Todos globais: retorno de preço é comparável entre setores
# ---------------------------------------------------------------------------
MOMENTUM_FACTORS = {
    "ret_3m":  {"weight": 0.30, "label": "Retorno 3m",  "norm": "global"},
    "ret_6m":  {"weight": 0.40, "label": "Retorno 6m",  "norm": "global"},
    "ret_12m": {"weight": 0.30, "label": "Retorno 12m", "norm": "global"},
}

# ---------------------------------------------------------------------------
# Fatores de qualidade/risco
# Todos globais: risco e liquidez são universais
# ---------------------------------------------------------------------------
QUALITY_FACTORS = {
    "volatility_180d": {
        "weight":    0.40,
        "direction": "lower_is_better",
        "label":     "Volatilidade 180d",
        "norm":      "global",
    },
    "beta": {
        "weight":    0.35,
        "direction": "lower_is_better",
        "label":     "Beta vs IBOV",
        "norm":      "global",
    },
    "avg_volume_30d": {
        "weight":    0.25,
        "direction": "higher_is_better",
        "label":     "Volume Médio 30d",
        "norm":      "global",
    },
}

# ---------------------------------------------------------------------------
# Thresholds e filtros
# ---------------------------------------------------------------------------
MIN_DAILY_VOLUME_BRL  = 5_000_000   # R$ 5M/dia — filtro de liquidez mínima
MAX_DIVIDA_EBITDA     = 5.0         # filtro hard: acima disso, ação é excluída
MAX_PL                = 80.0        # filtro hard: P/L > 80 é distorção (prejuízo)
MIN_ROE               = -0.50       # filtro hard: ROE < -50% indica destruição de valor

# ---------------------------------------------------------------------------
# Filtros de sanidade de dados
# ---------------------------------------------------------------------------
# Acima de 20% costuma indicar provento extraordinário/amortização de capital
# (ex.: SBSP3 retornou DY=55% após distribuição especial em 2024 — não é
# dividend yield recorrente). Setamos para NaN para não contaminar o ranking.
MAX_PLAUSIBLE_DY      = 0.20
# Mínimo de fundamentos não-NaN para que um ticker seja elegível a ranking.
# Empresas com 0-2 fundamentos válidos têm score dominado por momentum/qualidade
# e tendem a entrar artificialmente no top 5 por dados ausentes.
MIN_FUNDAMENTALS_REQUIRED = 3
# Score mínimo para entrar no top 5. Universo pequeno + score baixo = forçar
# recomendação medíocre. Melhor recomendar 3 boas que 5 marginais.
MIN_SCORE_THRESHOLD   = 50.0

# Normalização adaptativa
MIN_SECTOR_ZSCORE     = 8           # N mínimo para Z-Score setorial (estatisticamente válido)
MIN_SECTOR_PERCENTILE = 4           # N mínimo para Percentil setorial
TOP_N_RECOMMENDATIONS = 5           # Top 5 na carteira recomendada
EQUAL_WEIGHT          = 1.0 / TOP_N_RECOMMENDATIONS  # 20% cada posição

# ---------------------------------------------------------------------------
# Diversificação do portfólio
# ---------------------------------------------------------------------------
MAX_PER_SECTOR        = 2           # máx 2 ações do mesmo setor B3
MAX_PER_SUBSECTOR     = 1           # máx 1 ação por sub-setor (evita 2 bancos)
MAX_PER_MACRO_THEME   = 3           # máx 3 ações exposição a mesmo tema macro

# Mapeamento setor B3 → tema macroeconômico
# Tema captura sensibilidade dominante: commodity vs doméstico vs juros etc.
# É grosseiro de propósito — quebrar concentração macro, não fina classificação.
MACRO_THEME_MAP: dict[str, str] = {
    "Petróleo Gás e Biocombustíveis": "commodity_export",
    "Materiais Básicos":              "commodity_export",
    "Energia Elétrica":               "defensive_utilities",
    "Utilidade Pública":              "defensive_utilities",
    "Saúde":                          "rate_sensitive_growth",
    "Tecnologia da Informação":       "rate_sensitive_growth",
    "Comunicações":                   "rate_sensitive_growth",
    "Consumo Cíclico":                "domestic_consumer",
    "Consumo Não Cíclico":            "domestic_consumer",
    "Financeiro e Outros":            "financials",
    "Bens Industriais":               "industrials",
}

# Janelas de momentum (dias úteis aproximados)
MOMENTUM_WINDOWS = {
    "ret_3m":  63,
    "ret_6m":  126,
    "ret_12m": 252,
}

VOLATILITY_WINDOW = 180  # dias para cálculo de volatilidade histórica
VOLUME_WINDOW     = 30   # dias para média de volume

# ---------------------------------------------------------------------------
# Tributação (Lei 15.270/2025 — em vigor a partir de 2026)
# ---------------------------------------------------------------------------
# IRRF 10% sobre dividendos para PF residente quando soma de proventos no mês
# ultrapassa R$ 50k (única empresa) ou em pagamento intra-grupo. Para PF típico
# de carteira diversificada, aplica-se em DY mensal expressivo.
# JCP segue com 15% como sempre.
DIVIDEND_TAX_RATE_PF = 0.10
JCP_TAX_RATE_PF      = 0.15

# ---------------------------------------------------------------------------
# Portfólio: alocação e rotação
# ---------------------------------------------------------------------------
# HRP (Hierarchical Risk Parity, López de Prado 2016) é mais robusto que
# inverse-vol em portfólios pequenos: respeita correlações via clustering.
# Quando False, mantém inverse-vol legado.
USE_HRP_WEIGHTS  = True
HRP_LOOKBACK_DAYS = 126   # 6 meses de retornos diários para estimar covariância

# Turnover band: só rotaciona uma posição existente se o ticker candidato tem
# score MIN_SCORE_GAP_FOR_ROTATION pontos acima do incumbente. Reduz fricção
# real (e ruído de medição) — diferenças <5 pontos não são estatisticamente
# significativas em score multi-fator.
TURNOVER_BAND_PTS = 5.0

# ---------------------------------------------------------------------------
# Novos fatores fundamentalistas (QMJ-style: Growth, Investment, Size)
# ---------------------------------------------------------------------------
# Size (SMB): log(market_cap) — small caps brasileiras têm prêmio documentado
# (NEFIN-USP). Direção: lower_is_better (menor cap = maior score).
# Peso baixo (~5%) para não dominar o pilar.
ENABLE_SIZE_FACTOR = True
SIZE_WEIGHT       = 0.05

# Growth: ROE/margem média 3y e tendência. Peso modesto para evitar
# survivorship bias (empresas em queda têm growth ruim → score baixo →
# pode penalizar value plays legítimos em recuperação).
ENABLE_GROWTH_FACTOR = True
GROWTH_WEIGHT     = 0.08

# Investment / CMA: asset growth YoY. Direção lower_is_better
# (empresas que investem muito under-perform — literatura Fama-French).
ENABLE_INVESTMENT_FACTOR = True
INVESTMENT_WEIGHT = 0.05

# FCF Payout: dividendos pagos / FCF. Direção lower_is_better
# (acima de 1.0 = pagando mais que gera de caixa = insustentável).
# Substitui parcialmente o filtro EY/DY heurístico atual.
ENABLE_FCF_PAYOUT_CHECK = True
FCF_PAYOUT_UNSUSTAINABLE = 1.2  # >120% do FCF → DY zerado no score

# ---------------------------------------------------------------------------
# Earnings revisions (proxy via yfinance analyst recommendations)
# ---------------------------------------------------------------------------
# Tendência recente de revisões de analistas como sub-fator de momentum.
# Direção: higher_is_better (mais upgrades nas últimas semanas = bom sinal).
ENABLE_ANALYST_REVISIONS = True
ANALYST_REVISIONS_WEIGHT = 0.10  # dentro do pilar momentum

# ---------------------------------------------------------------------------
# Análise quantitativa avançada (Pacote profissional)
# ---------------------------------------------------------------------------

# Covariance estimation method no HRP. Opções (Riskfolio-Lib):
#   "hist"   — covariância amostral (default antigo)
#   "ledoit" — Ledoit-Wolf shrinkage para identidade (Ledoit-Wolf 2004b)
#   "oas"    — Oracle Approximating Shrinkage (Chen 2010) — melhor p/ N pequeno
# Para portfólios de 5 ativos com 126d de dados (p/N=0.04), shrinkage ganha
# pouco vs sample; mas "oas" tem boa propriedade Gaussian-assintótica.
HRP_COVARIANCE_METHOD = "ledoit"

# EWMA de fundamentais trimestrais
# Half-life em trimestres. 6Q = 1.5 anos: smoothing significativo mas ainda
# responsivo a mudanças estruturais. Aplica-se SÓ a métricas de qualidade
# (ROE, ROIC, margens) — NÃO a múltiplos (P/L, P/VP) ou growth.
ENABLE_EWMA_FUNDAMENTALS = True
EWMA_HALFLIFE_QUARTERS = 6

# PEAD (Post-Earnings Announcement Drift)
# Janela de surpresa: ±1 dia úteis ao redor do anúncio (CAR vs IBOV).
# Holding window: 5-60 dias úteis pós-anúncio. Tickers com EAR positivo
# nesse intervalo ganham bônus em momentum.
ENABLE_PEAD_FACTOR = True
PEAD_SURPRISE_WINDOW_DAYS = 1     # CAR(-1, +1) ao redor da data
PEAD_DRIFT_HOLDING_DAYS = 60      # quantos dias úteis após o anúncio o sinal vale
PEAD_DRIFT_ENTRY_DAYS = 5         # ignorar primeiros N dias para evitar reversão imediata
PEAD_WEIGHT = 0.10                # peso dentro do pilar momentum

# Analyst price targets (yfinance)
ENABLE_ANALYST_TARGET = True
ANALYST_TARGET_WEIGHT = 0.05      # peso baixo: sinal tem viés EM otimista crônico
ANALYST_TARGET_MAX_UPSIDE = 1.50  # cap em +150% (data error guard)

# BRL exposure factor (correlação log-returns com USDBRL)
# Janela 90d é compromisso entre tactical (60d) e strategic (252d).
ENABLE_BRL_FACTOR = True
BRL_CORRELATION_WINDOW = 90       # dias úteis para correlação rolling
BCB_USDBRL_SERIES = 1             # série BCB SGS — PTAX venda diária

# HMM regime detection
# 2-state é o padrão profissional (bull/bear ≈ low-vol/high-vol). 3-state
# é popular em research mas frequentemente tem um estado quase vazio fora
# de períodos extremos. Mantemos fallback ao detector binário se hmmlearn
# falhar ou histórico insuficiente.
USE_HMM_REGIME = True
HMM_N_STATES = 2
HMM_MIN_HISTORY_DAYS = 200        # ~10 meses de dados mínimo para ajustar HMM
HMM_RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Análise de Factor IC (Information Coefficient)
# ---------------------------------------------------------------------------
# Janelas de retorno forward para medir poder preditivo dos fatores.
IC_FORWARD_WINDOWS = {
    "1w":  5,
    "4w":  20,
    "12w": 60,
}
IC_OUTPUT_PATH = DATA_DIR / "factor_ic.json"

# ---------------------------------------------------------------------------
# Data Sources
# ---------------------------------------------------------------------------
BRAPI_BASE_URL    = "https://brapi.dev/api"
BRAPI_TIMEOUT     = 15          # segundos por request
YFINANCE_TIMEOUT  = 20
CACHE_TTL_HOURS   = 24          # TTL do cache local em horas
BRAPI_RATE_LIMIT  = 0.5         # segundos entre requests (respeitar rate limit free tier)

# Séries do Banco Central (python-bcb)
BCB_SELIC_SERIES = 11   # Taxa SELIC diária
BCB_CDI_SERIES   = 12   # Taxa CDI diária

# ---------------------------------------------------------------------------
# Graham Number
# ---------------------------------------------------------------------------
# Graham's constant 22.5 = P/L 15 × P/VP 1.5, calibrated for ~4% risk-free rate.
# Brazil's higher rate environment lowers the fair-value multiple.
# Formula: GRAHAM_CONSTANT = 22.5 × (GRAHAM_RF_BASE / current_selic)
# GRAHAM_CONSTANT_FALLBACK is used when live SELIC is unavailable.
GRAHAM_RF_BASE          = 0.04    # Graham's original US rf assumption (~4% a.a.)
GRAHAM_SELIC_FALLBACK   = 0.1375  # update when SELIC changes significantly
GRAHAM_CONSTANT_FALLBACK = round(22.5 * (GRAHAM_RF_BASE / GRAHAM_SELIC_FALLBACK), 4)  # ≈ 6.55

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "ibovespa": {"ticker": "^BVSP", "label": "IBOVESPA", "source": "yfinance"},
    "selic":    {"series": BCB_SELIC_SERIES, "label": "SELIC",    "source": "bcb"},
    "cdi":      {"series": BCB_CDI_SERIES,   "label": "CDI",      "source": "bcb"},
}

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_PARSE_MODE = "Markdown"

# ---------------------------------------------------------------------------
# Modos de execução
# ---------------------------------------------------------------------------
VALID_MODES = ["weekly", "monthly"]

CHART_LOOKBACK_WEEKS = {
    "weekly":  4,   # últimas 4 semanas no gráfico semanal
    "monthly": 26,  # últimos 6 meses no gráfico mensal
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CHART_OUTPUT_PATH = ROOT_DIR / "output_chart.png"
REPORT_OUTPUT_PATH = ROOT_DIR / "output_report.md"
