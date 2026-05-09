"""
snapshot_manager.py — Módulo Eixo

Responsável pela persistência auditável de cada execução do sistema.
Sem esse módulo, não há backtesting — não sabemos quais preços
valiam quando a recomendação foi feita.

Três artefatos por execução:
  1. snapshot_YYYY-MM-DD.json  — preços de fechamento de todos os tickers
                                  naquele pregão (entrada do backtester)
  2. recommendations_YYYY-MM-DD_MODE.json — Top 10 scores completos
  3. universe_YYYY-MM-DD.json  — estado do universe.csv naquele momento

Estrutura de diretórios esperada:
  data/history/
    snapshot_2025-01-06.json
    recommendations_2025-01-06_weekly.json
    universe_2025-01-06.json
    ...

Por que snapshot separado das recomendações?
  As recomendações mudam semanalmente, mas o backtester precisa dos preços
  de TODOS os tickers (não só top 5) para recalcular hipóteses alternativas.
  Manter os dois artefatos separados permite análises post-hoc.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import HISTORY_DIR, UNIVERSE_FILE, WEIGHTS

logger = logging.getLogger(__name__)


# ─── Tipos de retorno estruturados ───────────────────────────────────────────

@dataclass
class TickerRecommendation:
    """Recomendação individual — espelha a estrutura do briefing."""
    ticker:               str
    nome:                 str
    score:                float
    score_breakdown:      dict[str, float]      # {fundamental, momentum, quality}
    normalization_method: str
    sector:               str
    sector_peers_count:   int
    why:                  str
    metrics:              dict[str, Any]        # pl, pvp, roe, roic, dy, divida_ebitda
    entry_price:          Optional[float]       # preço de fechamento na data da rec.
    norm_details:         Optional[dict] = field(default=None, repr=False)


@dataclass
class RecommendationRecord:
    """Registro completo de uma execução do sistema."""
    date:                     str
    mode:                     str                           # "weekly" | "monthly"
    universe_snapshot_path:   str
    top5:                     list[TickerRecommendation]
    top10:                    list[TickerRecommendation]
    weights:                  dict[str, float]
    entry_prices:             dict[str, float]             # todos os tickers do top-10
    execution_metadata:       dict[str, Any] = field(default_factory=dict)


# ─── Helpers de serialização ──────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    """Converte para float, retorna None em caso de NaN/inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _to_json_safe(obj: Any) -> Any:
    """Torna qualquer objeto serializável em JSON (recursivo)."""
    if isinstance(obj, float):
        return _safe_float(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return _safe_float(float(obj))
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)[:10]
    return obj


# ─── SnapshotManager ─────────────────────────────────────────────────────────

class SnapshotManager:
    """
    Gerencia a persistência de snapshots e recomendações em data/history/.

    Convenção de nomes:
      snapshot_YYYY-MM-DD.json          — preços de fechamento (todos tickers)
      universe_YYYY-MM-DD.json          — estado do universe.csv
      recommendations_YYYY-MM-DD_MODE.json — scores e top-10

    Design:
      - Todos os arquivos são JSON UTF-8 (legíveis por humanos e por scripts)
      - Escritas atômicas: escreve para .tmp depois renomeia → sem arquivos corrompidos
      - IDs de data são strings ISO 8601 ("2025-01-06"), não timestamps Unix
    """

    def __init__(self, history_dir: Path = HISTORY_DIR):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Escrita
    # ═══════════════════════════════════════════════════════════════════════

    def save_price_snapshot(
        self,
        df_prices: pd.DataFrame,
        run_date: Optional[str | date] = None,
    ) -> Path:
        """
        Salva os preços de fechamento mais recentes de todos os tickers.

        Armazena apenas o último preço disponível no df_prices
        (preço de fechamento do pregão da recomendação).

        Returns:
            Path do arquivo salvo.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"snapshot_{run_date_str}.json"

        if df_prices.empty:
            logger.warning("df_prices vazio — snapshot de preços não salvo.")
            return path

        # Último preço de fechamento disponível para cada ticker
        last_prices: dict[str, Optional[float]] = {}
        for ticker in df_prices.columns:
            series = df_prices[ticker].dropna()
            last_prices[ticker] = _safe_float(series.iloc[-1]) if not series.empty else None

        payload = {
            "date":        run_date_str,
            "snapshot_type": "prices",
            "pregao_date": str(df_prices.index[-1].date()) if not df_prices.empty else None,
            "tickers_count": len(last_prices),
            "prices":      last_prices,
        }

        _write_atomic(path, payload)
        logger.info("Snapshot de preços salvo: %s (%d tickers)", path.name, len(last_prices))
        return path

    def save_universe_snapshot(
        self,
        run_date: Optional[str | date] = None,
        universe_file: Path = UNIVERSE_FILE,
    ) -> Path:
        """
        Persiste o estado atual do universe.csv em JSON.

        Permite auditar se o universo mudou entre execuções
        (ex: tickers adicionados/removidos manualmente).
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"universe_{run_date_str}.json"

        try:
            df = pd.read_csv(universe_file)
            payload = {
                "date":          run_date_str,
                "snapshot_type": "universe",
                "tickers_count": len(df),
                "sectors":       df["setor"].value_counts().to_dict(),
                "tickers":       df.to_dict(orient="records"),
            }
            _write_atomic(path, payload)
            logger.info("Universe snapshot salvo: %s (%d tickers)", path.name, len(df))
        except Exception as exc:
            logger.error("Falha ao salvar universe snapshot: %s", exc)

        return path

    def save_recommendation(
        self,
        df_scored: pd.DataFrame,
        df_prices: pd.DataFrame,
        mode: str,
        run_date: Optional[str | date] = None,
        top_n: int = 10,
    ) -> Path:
        """
        Salva o output do ScoringEngine em JSON estruturado.

        Constrói obrigatoriamente:
          - top5: as 5 recomendações da semana (para Telegram + backtesting)
          - top10: ranking completo até 10 para referência
          - entry_prices: preços de entrada de todos os top-10 (para backtester)
          - universe_snapshot_path: referência cruzada com o snapshot de universo

        Args:
            df_scored:  Output do ScoringEngine.score(), ordenado por total_score DESC.
            df_prices:  DataFrame wide de preços (para capturar entry_price).
            mode:       "weekly" ou "monthly".
            run_date:   Data de referência; default = hoje.
            top_n:      Quantos tickers salvar além do top-5 (máx 10 recomendado).

        Returns:
            Path do arquivo JSON salvo.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"recommendations_{run_date_str}_{mode}.json"

        universe_snap_path = str(
            self.history_dir / f"universe_{run_date_str}.json"
        )

        # Capturar preços de entrada dos top-N tickers
        entry_prices = _extract_entry_prices(df_scored, df_prices, top_n)

        # Montar lista de recomendações
        top_recs: list[dict] = []
        for _, row in df_scored.head(top_n).iterrows():
            ticker = row.get("ticker", "")
            rec = _row_to_recommendation(row, entry_prices.get(ticker))
            top_recs.append(rec)

        top5  = top_recs[:5]
        top10 = top_recs[:top_n]

        payload = {
            "date":                   run_date_str,
            "mode":                   mode,
            "universe_snapshot":      universe_snap_path,
            "top5":                   top5,
            "top10":                  top10,
            "weights":                WEIGHTS,
            "entry_prices":           entry_prices,
            "execution_metadata": {
                "universe_size":      len(df_scored),
                "tickers_scored":     int(df_scored["total_score"].notna().sum()),
                "generated_at":       datetime.now().isoformat(),
                "score_range": {
                    "max": _safe_float(df_scored["total_score"].max()),
                    "min": _safe_float(df_scored["total_score"].min()),
                    "p50": _safe_float(df_scored["total_score"].median()),
                },
            },
        }

        _write_atomic(path, _to_json_safe(payload))
        logger.info(
            "Recomendação salva: %s (top5: %s)",
            path.name,
            [r["ticker"] for r in top5],
        )
        return path

    # ═══════════════════════════════════════════════════════════════════════
    # Leitura — usada pelo backtester.py
    # ═══════════════════════════════════════════════════════════════════════

    def load_recommendation(
        self,
        run_date: str | date,
        mode: str = "weekly",
    ) -> Optional[dict]:
        """
        Carrega o JSON de recomendação de uma data específica.

        Returns:
            dict com a estrutura completa, ou None se o arquivo não existir.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"recommendations_{run_date_str}_{mode}.json"
        return _read_json(path)

    def load_latest_recommendation(self, mode: str = "weekly") -> Optional[dict]:
        """
        Carrega o JSON de recomendação mais recente disponível.

        Itera em ordem reversa pelos arquivos recommendations_*_{mode}.json
        e retorna o primeiro encontrado.

        Returns:
            dict com a recomendação mais recente, ou None se não houver nenhuma.
        """
        pattern = f"recommendations_*_{mode}.json"
        files = sorted(self.history_dir.glob(pattern), reverse=True)
        if not files:
            logger.info("Nenhuma recomendação %s encontrada em %s", mode, self.history_dir)
            return None
        return _read_json(files[0])

    def load_price_snapshot(self, run_date: str | date) -> Optional[dict[str, float]]:
        """
        Carrega o snapshot de preços de uma data.

        Returns:
            dict {ticker: price} ou None se o arquivo não existir.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"snapshot_{run_date_str}.json"
        data = _read_json(path)
        return data.get("prices") if data else None

    def list_recommendations(self, mode: str = "weekly") -> list[dict]:
        """
        Lista todas as recomendações disponíveis em ordem cronológica.

        Returns:
            Lista de dicts com campos: date, mode, path, top5_tickers.
        """
        pattern = f"recommendations_*_{mode}.json"
        files = sorted(self.history_dir.glob(pattern))
        result = []
        for f in files:
            data = _read_json(f)
            if data:
                result.append({
                    "date":         data.get("date"),
                    "mode":         data.get("mode"),
                    "path":         str(f),
                    "top5_tickers": [r.get("ticker") for r in data.get("top5", [])],
                    "top5_scores":  [r.get("score") for r in data.get("top5", [])],
                })
        return result

    def get_entry_prices_for_backtesting(
        self,
        recommendation: dict,
    ) -> dict[str, float]:
        """
        Extrai os preços de entrada a partir de uma recomendação carregada.

        O backtester.py usa esses preços para calcular o retorno da carteira
        da semana/mês anterior comparado ao preço atual.

        Args:
            recommendation: dict retornado por load_recommendation()

        Returns:
            {ticker: entry_price} para os tickers do top-5.
        """
        entry_prices = recommendation.get("entry_prices", {})
        top5_tickers = [r["ticker"] for r in recommendation.get("top5", [])]
        return {t: entry_prices[t] for t in top5_tickers if t in entry_prices}


# ─── Helpers privados ─────────────────────────────────────────────────────────

def _date_str(d: Optional[str | date | datetime]) -> str:
    """Normaliza para string ISO 8601 'YYYY-MM-DD'."""
    if d is None:
        return date.today().isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _write_atomic(path: Path, payload: dict) -> None:
    """
    Escrita atômica: escreve em .tmp e renomeia.
    Previne arquivos corrompidos se o processo for interrompido.
    """
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)   # rename atômico no mesmo filesystem
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao salvar {path}: {exc}") from exc


def _read_json(path: Path) -> Optional[dict]:
    """Lê um JSON com tratamento de erros."""
    if not path.exists():
        logger.debug("Arquivo não encontrado: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Falha ao ler %s: %s", path, exc)
        return None


def _extract_entry_prices(
    df_scored: pd.DataFrame,
    df_prices: pd.DataFrame,
    top_n: int,
) -> dict[str, float]:
    """Captura o último preço disponível para os top-N tickers."""
    prices: dict[str, float] = {}
    for _, row in df_scored.head(top_n).iterrows():
        ticker = row.get("ticker", "")
        # Primeiro: usar current_price do DataFrame de fundamentos
        cp = _safe_float(row.get("current_price"))
        if cp is not None:
            prices[ticker] = cp
            continue
        # Fallback: último preço no df_prices
        if ticker in df_prices.columns:
            series = df_prices[ticker].dropna()
            if not series.empty:
                prices[ticker] = round(float(series.iloc[-1]), 2)
    return prices


def _row_to_recommendation(row: pd.Series, entry_price: Optional[float]) -> dict:
    """Converte uma linha do df_scored para o schema de recomendação."""
    ticker = str(row.get("ticker", ""))

    # Capturar peers_count do norm_details se disponível
    norm_details = row.get("norm_details", {})
    if isinstance(norm_details, str):
        try:
            import ast
            norm_details = ast.literal_eval(norm_details)
        except Exception:
            norm_details = {}

    # Estimar sector_peers_count: pegar de qualquer fator fundamental
    sector_peers_count = 0
    for factor_detail in (norm_details or {}).values():
        if isinstance(factor_detail, dict) and "n_peers" in factor_detail:
            sector_peers_count = int(factor_detail["n_peers"])
            break

    metrics = {
        "pl":            _safe_float(row.get("pl")),
        "earnings_yield":_safe_float(row.get("earnings_yield")),
        "pvp":           _safe_float(row.get("pvp")),
        "roe":           _safe_float(row.get("roe")),
        "roic":          _safe_float(row.get("roic")),
        "divida_ebitda": _safe_float(row.get("divida_ebitda")),
        "dy":            _safe_float(row.get("dividend_yield")),
        "alpha_6m":      _safe_float(row.get("alpha_6m")),
        "volatility":    _safe_float(row.get("volatility_180d")),
        "beta":          _safe_float(row.get("beta")),
    }

    fund_score = _safe_float(row.get("fundamental_score")) or 0.0
    mom_score  = _safe_float(row.get("momentum_score"))    or 0.0
    qual_score = _safe_float(row.get("quality_score"))     or 0.0

    return {
        "ticker":               ticker,
        "nome":                 str(row.get("nome", "")),
        "score":                _safe_float(row.get("total_score")),
        "score_breakdown": {
            "fundamental": round(WEIGHTS["fundamental"] * fund_score, 2),
            "momentum":    round(WEIGHTS["momentum"]    * mom_score,  2),
            "quality":     round(WEIGHTS["quality"]     * qual_score, 2),
        },
        "normalization_method": str(row.get("normalization_method", "")),
        "sector":               str(row.get("setor", "")),
        "sector_peers_count":   sector_peers_count,
        "why":                  str(row.get("why", "")),
        "metrics":              metrics,
        "entry_price":          entry_price,
        "norm_details":         norm_details if norm_details else None,
    }
