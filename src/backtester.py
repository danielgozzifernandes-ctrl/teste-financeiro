"""
backtester.py — Módulo 7

Audita o desempenho das recomendações anteriores comparando o preço
de entrada registrado no snapshot com o preço atual de mercado.

Fórmula do retorno da carteira (briefing, pesos iguais 20%):
    R_total = Σ 0.20 × (P_atual_i / P_entrada_i − 1)  para i=1..N

Por que essa fórmula?
  É equivalente à média aritmética dos retornos individuais com pesos iguais.
  Corresponde a uma carteira rebalanceada diariamente (simplificação conservadora
  — subestima ligeiramente o retorno real, que seria geométrico com rebalanceamento
  mensal). Para janelas curtas (1-4 semanas) a diferença é desprezível (<0.1pp).

Tratamento de "primeira execução":
  Se não há recomendação anterior gravada → retorna status "no_history"
  sem lançar exceção. O main.py trata esse caso graciosamente.

Tratamento de ticker indisponível (delisting, erro de dado):
  Se um ticker da recomendação anterior não tem preço atual → excluído
  do cálculo e os pesos são redistribuídos proporcionalmente.
  Ex: 4 de 5 tickers disponíveis → peso efetivo de 25% cada.

Output salvo em data/history/backtest_YYYY-MM-DD_{mode}.json.
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import HISTORY_DIR
from src.snapshot_manager import SnapshotManager, _write_atomic, _date_str

logger = logging.getLogger(__name__)

# 13 bps per leg (entry + exit) = 26 bps round-trip.
# Covers: corretagem ~3 bps, emolumentos ~3 bps, slippage ~7 bps.
_FRICTION_BPS = 13
_FRICTION = _FRICTION_BPS / 10_000


# ─── Tipos de resultado ───────────────────────────────────────────────────────

class BacktestStatus:
    SUCCESS      = "success"
    NO_HISTORY   = "no_history"       # primeira execução
    PARTIAL_DATA = "partial_data"     # alguns tickers sem preço atual
    ERROR        = "error"


# ─── Backtester ───────────────────────────────────────────────────────────────

class Backtester:
    """
    Calcula e persiste o resultado do backtesting da carteira anterior.

    Uso:
        bt = Backtester()
        result = bt.run(
            mode="weekly",
            current_prices={"PETR4": 41.55, "VALE3": 72.10, ...},
            benchmark_period_returns={"ibovespa": 0.051, "cdi": 0.016, "selic": 0.016},
        )
    """

    def __init__(
        self,
        snapshot_manager: Optional[SnapshotManager] = None,
        history_dir: Path = HISTORY_DIR,
    ):
        self.snap    = snapshot_manager or SnapshotManager()
        self.history = Path(history_dir)
        self.history.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # API pública
    # ═══════════════════════════════════════════════════════════════════════

    def run(
        self,
        mode: str,
        current_prices: dict[str, float],
        benchmark_period_returns: dict[str, float],
        run_date: Optional[str | date] = None,
    ) -> dict:
        """
        Executa o backtest comparando a recomendação anterior com os preços atuais.

        Args:
            mode:                      "weekly" ou "monthly".
            current_prices:            {ticker: preço_atual} — de df_prices.iloc[-1]
                                       ou df_scored["current_price"].
            benchmark_period_returns:  {"ibovespa": 0.051, "cdi": 0.016, "selic": 0.016}
                                       retornos acumulados do período (de BenchmarkManager).
            run_date:                  Data de referência; default = hoje.

        Returns:
            dict com o resultado completo (veja _build_result).
            Sempre retorna — nunca lança exceção para o caller.
        """
        run_date_str = _date_str(run_date)

        try:
            # Carregar recomendação anterior
            previous = self.snap.load_latest_recommendation(mode=mode)
            if previous is None:
                logger.info("Backtester: nenhuma recomendação anterior encontrada (%s).", mode)
                result = self._first_run_result(run_date_str, mode)
                self._save(result, run_date_str, mode)
                return result

            rec_date = previous.get("date", "")
            logger.info(
                "Backtester: comparando recomendação de %s com preços de %s",
                rec_date, run_date_str,
            )

            # Calcular retorno da carteira
            port_return, holdings, status_detail = self._calc_portfolio_return(
                previous, current_prices
            )

            # Calcular métricas de alpha
            ibov = benchmark_period_returns.get("ibovespa")
            cdi  = benchmark_period_returns.get("cdi")
            selic = benchmark_period_returns.get("selic")

            alpha_ibov = (port_return - ibov) if ibov is not None and port_return is not None else None
            alpha_cdi  = (port_return - cdi)  if cdi  is not None and port_return is not None else None

            # Calcular dias do período
            period_days = self._calc_period_days(rec_date, run_date_str)

            result = self._build_result(
                status=status_detail,
                backtest_date=run_date_str,
                recommendation_date=rec_date,
                mode=mode,
                portfolio_return=port_return,
                benchmark_returns={"ibovespa": ibov, "cdi": cdi, "selic": selic},
                alpha_vs_ibov=alpha_ibov,
                alpha_vs_cdi=alpha_cdi,
                holdings=holdings,
                period_days=period_days,
                previous_top5=[r.get("ticker") for r in previous.get("top5", [])],
            )

            self._save(result, run_date_str, mode)
            logger.info(
                "Backtest concluído: carteira=%+.2f%% | IBOV=%+.2f%% | Alpha=%+.2f pp",
                (port_return or 0) * 100,
                (ibov or 0) * 100,
                (alpha_ibov or 0) * 100,
            )
            return result

        except Exception as exc:
            logger.error("Backtester: erro inesperado: %s", exc, exc_info=True)
            result = {
                "status":  BacktestStatus.ERROR,
                "message": str(exc),
                "backtest_date": run_date_str,
                "mode":    mode,
            }
            self._save(result, run_date_str, mode)
            return result

    def run_from_df(
        self,
        mode: str,
        df_scored: pd.DataFrame,
        df_prices: pd.DataFrame,
        benchmark_manager,
        run_date: Optional[str | date] = None,
    ) -> dict:
        """
        Conveniência: extrai preços e retornos de benchmark automaticamente.

        Args:
            df_scored:         Output do ScoringEngine (para preços atuais).
            df_prices:         DataFrame wide de preços ajustados.
            benchmark_manager: Instância de BenchmarkManager.
            mode, run_date:    idem ao método run().

        Returns:
            dict com resultado do backtest.
        """
        # Preços atuais: prefer current_price do scored, fallback para df_prices
        current_prices = self._extract_current_prices(df_scored, df_prices)

        # Período para benchmark: da recomendação anterior até hoje
        previous = self.snap.load_latest_recommendation(mode=mode)
        bench_returns: dict[str, float] = {}

        if previous:
            rec_date = previous.get("date")
            if rec_date:
                try:
                    from src.benchmark import BenchmarkManager  # import lazy
                    if not isinstance(benchmark_manager, BenchmarkManager):
                        benchmark_manager = BenchmarkManager()
                    period_ret = benchmark_manager.get_period_return(
                        start_date=rec_date,
                        end_date=run_date or date.today(),
                    )
                    bench_returns = period_ret.to_dict()
                except Exception as exc:
                    logger.warning("Falha ao buscar retornos do benchmark para backtest: %s", exc)

        return self.run(
            mode=mode,
            current_prices=current_prices,
            benchmark_period_returns=bench_returns,
            run_date=run_date,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Cálculo de retorno
    # ═══════════════════════════════════════════════════════════════════════

    def _calc_portfolio_return(
        self,
        previous: dict,
        current_prices: dict[str, float],
    ) -> tuple[Optional[float], list[dict], str]:
        """
        Calcula o retorno da carteira usando a fórmula de pesos iguais.

        Fórmula: R_total = Σ w_i × (P_atual_i / P_entrada_i − 1)
        onde w_i = 1/N para os N tickers com dados disponíveis.

        Tratamento de dados ausentes:
          - Ticker sem preço atual → excluído, peso redistribuído
          - Ticker sem preço de entrada → excluído, peso redistribuído
          - Se < 3 tickers disponíveis → status PARTIAL_DATA

        Returns:
            (portfolio_return, holdings_list, status)
        """
        entry_prices = previous.get("entry_prices", {})
        top5         = previous.get("top5", [])

        if not top5:
            return None, [], BacktestStatus.ERROR

        # Stored inverse-volatility weights from SnapshotManager (may be absent in old snapshots)
        stored_weights: dict[str, float] = previous.get("portfolio_weights", {})

        holdings: list[dict]  = []
        excluded: list[str]   = []

        for stock in top5:
            ticker        = stock.get("ticker", "")
            entry_price   = entry_prices.get(ticker) or stock.get("entry_price")
            current_price = current_prices.get(ticker)

            if entry_price is None or entry_price <= 0:
                logger.warning("Backtester: preço de entrada inválido para %s", ticker)
                excluded.append(ticker)
                continue

            if current_price is None or current_price <= 0:
                logger.warning(
                    "Backtester: preço atual não disponível para %s — excluído do cálculo",
                    ticker,
                )
                excluded.append(ticker)
                continue

            # Frictional cost: 13 bps per leg (buy + sell)
            stock_return = (
                (current_price * (1 - _FRICTION)) / (entry_price * (1 + _FRICTION))
            ) - 1

            holdings.append({
                "ticker":        ticker,
                "entry_price":   round(float(entry_price), 4),
                "current_price": round(float(current_price), 4),
                "return":        round(float(stock_return), 6),
                "return_pct":    round(float(stock_return * 100), 4),
                "weight":        stored_weights.get(ticker),
                "included":      True,
            })

        # Marcar excluídos
        for ticker in excluded:
            holdings.append({
                "ticker":   ticker,
                "included": False,
                "reason":   "preço ausente",
            })

        n_valid = sum(1 for h in holdings if h.get("included"))

        if n_valid == 0:
            logger.error("Backtester: nenhum ticker válido para cálculo.")
            return None, holdings, BacktestStatus.ERROR

        # Portfolio weights: use inverse-volatility weights when available,
        # redistributing excluded tickers' weights proportionally.
        valid_tickers = [h["ticker"] for h in holdings if h.get("included")]
        if stored_weights and all(t in stored_weights for t in valid_tickers):
            raw_w = {t: stored_weights[t] for t in valid_tickers}
            total_w = sum(raw_w.values())
            eff_weights = {t: w / total_w for t, w in raw_w.items()} if total_w > 0 else {}
        else:
            eq = 1.0 / n_valid
            eff_weights = {t: eq for t in valid_tickers}

        portfolio_return = sum(
            eff_weights[h["ticker"]] * h["return"]
            for h in holdings
            if h.get("included")
        )

        status = (
            BacktestStatus.SUCCESS if n_valid == len(top5)
            else BacktestStatus.PARTIAL_DATA
        )

        if status == BacktestStatus.PARTIAL_DATA:
            logger.warning(
                "Backtester: %d/%d tickers disponíveis. Pesos redistribuídos.",
                n_valid, len(top5),
            )

        return float(portfolio_return), holdings, status

    # ═══════════════════════════════════════════════════════════════════════
    # Persistência
    # ═══════════════════════════════════════════════════════════════════════

    def _save(self, result: dict, run_date_str: str, mode: str) -> Path:
        """Salva o resultado do backtest em JSON atômico."""
        path = self.history / f"backtest_{run_date_str}_{mode}.json"
        try:
            _write_atomic(path, _safe_serialize(result))
            logger.debug("Backtest salvo: %s", path.name)
        except Exception as exc:
            logger.error("Falha ao salvar backtest: %s", exc)
        return path

    def load_backtest(
        self,
        run_date: str | date,
        mode: str = "weekly",
    ) -> Optional[dict]:
        """Carrega o resultado de um backtest específico."""
        date_str = _date_str(run_date)
        path = self.history / f"backtest_{date_str}_{mode}.json"
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Falha ao ler backtest %s: %s", path, exc)
            return None

    def load_all_backtests(self, mode: str = "weekly") -> list[dict]:
        """Carrega todos os backtests disponíveis em ordem cronológica."""
        files = sorted(self.history.glob(f"backtest_*_{mode}.json"))
        results = []
        for f in files:
            try:
                with open(f, encoding="utf-8") as fp:
                    results.append(json.load(fp))
            except Exception:
                pass
        return results

    def compute_track_record(self, mode: str = "weekly") -> dict:
        """
        Calcula o histórico acumulado de alpha vs IBOV e CDI.

        Usado no relatório para mostrar: "Sistema vs IBOV nos últimos 6 meses: +X pp"

        Returns:
            dict com métricas agregadas de todos os backtests:
            {
                "n_periods": int,
                "cumulative_portfolio": float,
                "cumulative_ibov": float,
                "cumulative_cdi": float,
                "cumulative_alpha_ibov": float,
                "win_rate_vs_ibov": float,   # % de períodos em que bateu o IBOV
                "avg_alpha_ibov": float,
                "last_6_periods": list[dict],
            }
        """
        backtests = [
            b for b in self.load_all_backtests(mode)
            if b.get("status") == BacktestStatus.SUCCESS
        ]

        if not backtests:
            return {"n_periods": 0, "message": "Sem histórico suficiente."}

        portfolio_rets = [b["portfolio_return"] for b in backtests if b.get("portfolio_return") is not None]
        ibov_rets      = [b["benchmark_returns"].get("ibovespa", 0) for b in backtests]
        cdi_rets       = [b["benchmark_returns"].get("cdi", 0) for b in backtests]
        alphas_ibov    = [b.get("alpha_vs_ibov", 0) for b in backtests if b.get("alpha_vs_ibov") is not None]

        # Retorno acumulado composto
        def cumulative(rets: list[float]) -> float:
            result = 1.0
            for r in rets:
                result *= (1 + r)
            return result - 1

        win_rate = sum(1 for a in alphas_ibov if a > 0) / len(alphas_ibov) if alphas_ibov else 0

        return {
            "n_periods":              len(backtests),
            "cumulative_portfolio":   cumulative(portfolio_rets),
            "cumulative_ibov":        cumulative(ibov_rets),
            "cumulative_cdi":         cumulative(cdi_rets),
            "cumulative_alpha_ibov":  cumulative(portfolio_rets) - cumulative(ibov_rets),
            "win_rate_vs_ibov":       win_rate,
            "avg_alpha_ibov":         sum(alphas_ibov) / len(alphas_ibov) if alphas_ibov else None,
            "last_6_periods":         backtests[-6:],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _first_run_result(run_date_str: str, mode: str) -> dict:
        """Resultado para a primeira execução (sem histórico)."""
        return {
            "status":       BacktestStatus.NO_HISTORY,
            "message":      "Primeira execução — sem recomendação anterior para backtesting.",
            "backtest_date": run_date_str,
            "mode":         mode,
        }

    @staticmethod
    def _build_result(
        status: str,
        backtest_date: str,
        recommendation_date: str,
        mode: str,
        portfolio_return: Optional[float],
        benchmark_returns: dict,
        alpha_vs_ibov: Optional[float],
        alpha_vs_cdi: Optional[float],
        holdings: list[dict],
        period_days: Optional[int],
        previous_top5: list[str],
    ) -> dict:
        """Constrói o dict de resultado normalizado."""
        return {
            "status":              status,
            "backtest_date":       backtest_date,
            "recommendation_date": recommendation_date,
            "mode":                mode,
            "period_days":         period_days,
            "portfolio_return":    portfolio_return,
            "portfolio_return_pct": round(float(portfolio_return or 0) * 100, 4),
            "benchmark_returns":   {
                k: (round(float(v) * 100, 4) / 100 if v is not None else None)
                for k, v in benchmark_returns.items()
            },
            "alpha_vs_ibov":       alpha_vs_ibov,
            "alpha_vs_cdi":        alpha_vs_cdi,
            "alpha_vs_ibov_pp":    round(float(alpha_vs_ibov or 0) * 100, 4),
            "alpha_vs_cdi_pp":     round(float(alpha_vs_cdi or 0) * 100, 4),
            "holdings":            holdings,
            "previous_top5":       previous_top5,
            "generated_at":        datetime.now().isoformat(),
        }

    @staticmethod
    def _extract_current_prices(
        df_scored: pd.DataFrame,
        df_prices: pd.DataFrame,
    ) -> dict[str, float]:
        """Extrai preços atuais priorizando df_scored, fallback df_prices."""
        prices: dict[str, float] = {}

        # Fonte 1: current_price no df_scored (mais recente, da API)
        if "current_price" in df_scored.columns and "ticker" in df_scored.columns:
            for _, row in df_scored.iterrows():
                ticker = str(row.get("ticker", ""))
                cp = row.get("current_price")
                if cp is not None and not (isinstance(cp, float) and np.isnan(cp)):
                    prices[ticker] = float(cp)

        # Fonte 2: df_prices (último pregão disponível)
        if not df_prices.empty:
            last_row = df_prices.iloc[-1]
            for ticker in df_prices.columns:
                if ticker not in prices and not pd.isna(last_row.get(ticker)):
                    prices[ticker] = float(last_row[ticker])

        return prices

    @staticmethod
    def _calc_period_days(start_date_str: str, end_date_str: str) -> Optional[int]:
        """Calcula dias corridos entre duas datas ISO 8601."""
        try:
            start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end   = datetime.strptime(end_date_str,   "%Y-%m-%d").date()
            return (end - start).days
        except Exception:
            return None


# ─── Helpers de serialização ─────────────────────────────────────────────────

def _safe_serialize(obj):
    """Garante que todos os floats NaN/inf sejam None antes de serializar."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    return obj


# ─── Função de conveniência ───────────────────────────────────────────────────

def run_backtest(
    mode: str,
    df_scored: pd.DataFrame,
    df_prices: pd.DataFrame,
    benchmark_manager,
    run_date: Optional[str | date] = None,
) -> dict:
    """Ponto de entrada simplificado para main.py."""
    return Backtester().run_from_df(
        mode=mode,
        df_scored=df_scored,
        df_prices=df_prices,
        benchmark_manager=benchmark_manager,
        run_date=run_date,
    )
