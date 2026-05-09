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

# Normalização adaptativa
MIN_SECTOR_ZSCORE     = 8           # N mínimo para Z-Score setorial (estatisticamente válido)
MIN_SECTOR_PERCENTILE = 4           # N mínimo para Percentil setorial
TOP_N_RECOMMENDATIONS = 5           # Top 5 na carteira recomendada
EQUAL_WEIGHT          = 1.0 / TOP_N_RECOMMENDATIONS  # 20% cada posição

# Janelas de momentum (dias úteis aproximados)
MOMENTUM_WINDOWS = {
    "ret_3m":  63,
    "ret_6m":  126,
    "ret_12m": 252,
}

VOLATILITY_WINDOW = 180  # dias para cálculo de volatilidade histórica
VOLUME_WINDOW     = 30   # dias para média de volume

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
