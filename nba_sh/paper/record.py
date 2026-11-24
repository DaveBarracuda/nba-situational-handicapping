"""Record del paper-trading: historial apuesta a apuesta y acumulado.

El reporte de CLV (:mod:`nba_sh.paper.report`) responde «¿tiene esto edge?».
El record responde «¿qué ha pasado?»: cada apuesta con su resultado, el balance
corrido y el ROI.

Son preguntas distintas y la segunda engaña más. Un 5-0 con cinco apuestas sale
por azar el 3% de las veces si cada una fuera una moneda, así que el record
**no puede** leerse como evidencia mientras la muestra sea pequeña; para eso
está la CLV, que necesita ~100 veces menos muestra. Por eso cada resumen lleva
la probabilidad de ese récord bajo la hipótesis nula: para que el número no se
lea solo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import pandas as pd
from scipy.stats import binom

from nba_sh.paper.ledger import PaperLedger

LOGGER = logging.getLogger(__name__)

RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_PUSH = "push"
# Lo que hay que acertar a -110 para no perder dinero.
BREAK_EVEN_RATE = 110.0 / 210.0
# Por debajo de esto, un récord no dice nada que merezca mirarse.
MIN_DECIDED_FOR_SIGNAL = 30
NAN = float("nan")
LEDGER_COLUMNS: tuple[str, ...] = (
    "game_date",
    "home_team",
    "away_team",
    "market",
    "side",
    "line",
    "american",
    "stake",
    "result",
    "pnl",
    "clv",
    "clv_points",
    "closing_line",
)


@dataclass(frozen=True)
class TrackRecord:
    """Lo que ha pasado con las apuestas de papel."""

    n_bets: int
    wins: int
    losses: int
    pushes: int
    pending: int
    staked: float
    pnl: float
    roi: float
    hit_rate: float
    # Probabilidad de un récord al menos tan bueno si cada apuesta fuera una
    # moneda: la vacuna contra leer una racha como habilidad.
    p_value_coin: float
    mean_clv: float
    mean_clv_points: float
    n_with_clv: int
    by_market: dict[str, dict[str, float]] = field(default_factory=dict)
    verdict: str = ""

    def as_dict(self) -> dict[str, object]:
        """Vista serializable para artefactos."""
        return {
            "n_bets": self.n_bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "pending": self.pending,
            "staked": self.staked,
            "pnl": self.pnl,
            "roi": self.roi,
            "hit_rate": self.hit_rate,
            "p_value_coin": self.p_value_coin,
            "mean_clv": self.mean_clv,
            "mean_clv_points": self.mean_clv_points,
            "n_with_clv": self.n_with_clv,
            "by_market": self.by_market,
            "verdict": self.verdict,
        }


def coin_flip_p_value(wins: int, decided: int) -> float:
    """Probabilidad de ganar al menos ``wins`` de ``decided`` a cara o cruz."""
    if decided <= 0:
        return NAN
    return float(binom.sf(wins - 1, decided, 0.5))


def _market_breakdown(settled: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Resumen por mercado: no todos son igual de blandos."""
    breakdown: dict[str, dict[str, float]] = {}
    for market, group in settled.groupby("market"):
        results = group["result"].astype(str)
        stake = pd.to_numeric(group["stake"], errors="coerce").fillna(0.0)
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        breakdown[str(market)] = {
            "n": int(len(group)),
            "wins": int((results == RESULT_WIN).sum()),
            "losses": int((results == RESULT_LOSS).sum()),
            "pnl": float(pnl.sum()),
            "roi": float(pnl.sum() / stake.sum()) if stake.sum() > 0 else NAN,
        }
    return breakdown


def _verdict(record: TrackRecord) -> str:
    """Lo que el record permite y no permite concluir."""
    if record.n_bets == 0:
        return "Sin apuestas liquidadas todavía."
    decided = record.wins + record.losses
    if decided == 0:
        return f"{record.n_bets} apuestas liquidadas, todas push."
    probability = record.p_value_coin
    base = (
        f"{record.wins}-{record.losses}"
        + (f"-{record.pushes}" if record.pushes else "")
        + f" ({100 * record.hit_rate:.1f}%), ROI {100 * record.roi:+.1f}%."
    )
    if decided < MIN_DECIDED_FOR_SIGNAL:
        return (
            f"{base} Con {decided} apuestas decididas esto no es evidencia de "
            f"nada: un récord así o mejor sale por azar el "
            f"{100 * probability:.1f}% de las veces. Mira la CLV."
        )
    if probability < 0.05 and record.hit_rate > BREAK_EVEN_RATE:
        return (
            f"{base} Supera el break-even del −110 y el azar lo explicaría solo "
            f"el {100 * probability:.1f}% de las veces, pero el ROI sigue siendo "
            "ruidoso: la CLV decide."
        )
    return (
        f"{base} El azar explicaría un récord así o mejor el "
        f"{100 * probability:.1f}% de las veces."
    )


def track_record(settled: pd.DataFrame, *, pending: int = 0) -> TrackRecord:
    """Construye el record desde las apuestas liquidadas.

    Args:
        settled: apuestas liquidadas del libro.
        pending: apuestas registradas que esperan resultado.

    Returns:
        :class:`TrackRecord` con el acumulado y su lectura honesta.
    """
    if settled.empty:
        empty = TrackRecord(
            n_bets=0,
            wins=0,
            losses=0,
            pushes=0,
            pending=pending,
            staked=0.0,
            pnl=0.0,
            roi=NAN,
            hit_rate=NAN,
            p_value_coin=NAN,
            mean_clv=NAN,
            mean_clv_points=NAN,
            n_with_clv=0,
        )
        return replace(empty, verdict=_verdict(empty))

    results = settled["result"].astype(str)
    stake = pd.to_numeric(settled["stake"], errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(settled["pnl"], errors="coerce").fillna(0.0)
    clv = pd.to_numeric(settled.get("clv"), errors="coerce").dropna()
    points = pd.to_numeric(settled.get("clv_points"), errors="coerce").dropna()

    wins = int((results == RESULT_WIN).sum())
    losses = int((results == RESULT_LOSS).sum())
    decided = wins + losses
    record = TrackRecord(
        n_bets=int(len(settled)),
        wins=wins,
        losses=losses,
        pushes=int((results == RESULT_PUSH).sum()),
        pending=pending,
        staked=float(stake.sum()),
        pnl=float(pnl.sum()),
        roi=float(pnl.sum() / stake.sum()) if stake.sum() > 0 else NAN,
        hit_rate=wins / decided if decided else NAN,
        p_value_coin=coin_flip_p_value(wins, decided),
        mean_clv=float(clv.mean()) if clv.size else NAN,
        mean_clv_points=float(points.mean()) if points.size else NAN,
        n_with_clv=int(clv.size),
        by_market=_market_breakdown(settled),
    )
    return replace(record, verdict=_verdict(record))


def bet_log(settled: pd.DataFrame) -> pd.DataFrame:
    """Historial apuesta a apuesta, en orden cronológico, con balance corrido."""
    if settled.empty:
        return pd.DataFrame(columns=[*LEDGER_COLUMNS, "running_pnl"])
    columns = [name for name in LEDGER_COLUMNS if name in settled.columns]
    log = settled[columns].copy()
    if "game_date" in log.columns:
        log = log.sort_values("game_date")
    log["running_pnl"] = pd.to_numeric(log["pnl"], errors="coerce").fillna(0.0).cumsum()
    return log.reset_index(drop=True)


def record_from_ledger(*, ledger: PaperLedger | None = None) -> TrackRecord:
    """Record leyendo el libro del disco."""
    book = ledger or PaperLedger()
    return track_record(book.settled(), pending=len(book.pending_bets()))


__all__ = [
    "BREAK_EVEN_RATE",
    "LEDGER_COLUMNS",
    "MIN_DECIDED_FOR_SIGNAL",
    "TrackRecord",
    "bet_log",
    "coin_flip_p_value",
    "record_from_ledger",
    "track_record",
]
