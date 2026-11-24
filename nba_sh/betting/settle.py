"""Liquidación de apuestas y cálculo de CLV (guía 10.2 paso 6, y 13).

Cada apuesta se resuelve con el resultado real del partido (incluye prórroga,
como el mercado) y se le calcula la CLV contra la línea de cierre.

La CLV tiene dos componentes, porque en spread y totales lo que se mueve es
la **línea**, no el precio (ambos lados suelen quedarse en -110):

    CLV = [prob_fair(cierre) - prob_fair(apuesta)]      <- componente precio
        + [Phi(puntos_ganados / sigma) - 0.5]           <- componente línea

Las dos probabilidades de precio se calculan **sin vig**, de modo que la
métrica no arrastre el juice y mida solo el movimiento del mercado a favor o
en contra de la apuesta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from nba_sh.betting.edge import SIDE_AWAY, SIDE_HOME, SIDE_OVER, SIDE_UNDER
from nba_sh.odds_utils import american_to_decimal, fair_two_way
from nba_sh.schemas import MARKET_ML, MARKET_SPREAD, MARKET_TOTAL

RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_PUSH = "push"
RESULT_UNKNOWN = "unknown"

SETTLEMENT_COLUMNS = (
    "result",
    "pnl",
    "clv",
    "clv_points",
    "closing_american",
    "closing_line",
)

MARGIN_LINE_SIGMA = 12.0
TOTAL_LINE_SIGMA = 17.5

_PRICE_COLUMNS = {
    (MARKET_ML, SIDE_HOME): ("ml_home", "ml_away"),
    (MARKET_ML, SIDE_AWAY): ("ml_away", "ml_home"),
    (MARKET_SPREAD, SIDE_HOME): ("spread_home_price", "spread_away_price"),
    (MARKET_SPREAD, SIDE_AWAY): ("spread_away_price", "spread_home_price"),
    (MARKET_TOTAL, SIDE_OVER): ("over_price", "under_price"),
    (MARKET_TOTAL, SIDE_UNDER): ("under_price", "over_price"),
}


def _outcome(market: str, side: str, line: float, margin: float, total: float) -> str:
    """Resuelve una apuesta con el marcador final."""
    if np.isnan(margin) or np.isnan(total):
        return RESULT_UNKNOWN
    if market == MARKET_ML:
        if margin == 0.0:
            return RESULT_PUSH
        return RESULT_WIN if (margin > 0.0) == (side == SIDE_HOME) else RESULT_LOSS
    if market == MARKET_SPREAD:
        signed_margin = margin if side == SIDE_HOME else -margin
        result = signed_margin + line
        if result == 0.0:
            return RESULT_PUSH
        return RESULT_WIN if result > 0.0 else RESULT_LOSS
    if market == MARKET_TOTAL:
        if total == line:
            return RESULT_PUSH
        return RESULT_WIN if (total > line) == (side == SIDE_OVER) else RESULT_LOSS
    raise ValueError(f"unknown market: {market!r}")


def _pnl(result: str, stake: float, american: float) -> float:
    """P&L de una apuesta liquidada."""
    if result == RESULT_WIN:
        return float(stake * (american_to_decimal(american) - 1.0))
    if result == RESULT_LOSS:
        return float(-stake)
    return 0.0


def _side_line(market: str, side: str, odds_row: pd.Series) -> float:
    """Línea de una fila de cuotas desde la perspectiva del lado apostado."""
    if market == MARKET_SPREAD:
        spread_home = float(odds_row["spread_home"])
        return spread_home if side == SIDE_HOME else -spread_home
    if market == MARKET_TOTAL:
        return float(odds_row["total_line"])
    return float("nan")


def _fair_probability(market: str, side: str, odds_row: pd.Series) -> float:
    """Probabilidad sin vig del lado apostado en una fila de cuotas."""
    key = (market, side)
    if key not in _PRICE_COLUMNS:
        return float("nan")
    own_column, other_column = _PRICE_COLUMNS[key]
    fair, _ = fair_two_way(
        float(odds_row[own_column]), float(odds_row[other_column])
    )
    return float(fair)


def _line_points_gained(
    market: str, side: str, taken_line: float, closing_line: float
) -> float:
    """Puntos ganados (o perdidos) por el movimiento de la línea."""
    if np.isnan(closing_line) or np.isnan(taken_line):
        return 0.0
    if market == MARKET_SPREAD:
        return float(taken_line - closing_line)
    if market == MARKET_TOTAL:
        return float(
            closing_line - taken_line if side == SIDE_OVER else taken_line - closing_line
        )
    return 0.0


def _line_probability(market: str, points: float) -> float:
    """Convierte puntos de ventaja de línea en puntos de probabilidad."""
    if points == 0.0:
        return 0.0
    sigma = MARGIN_LINE_SIGMA if market == MARKET_SPREAD else TOTAL_LINE_SIGMA
    return float(norm.cdf(points / sigma) - 0.5)


def settle_bets(
    bets: pd.DataFrame,
    games: pd.DataFrame,
    *,
    closing_odds: pd.DataFrame | None = None,
    bet_odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Liquida las apuestas y añade resultado, P&L y CLV.

    Args:
        bets: apuestas aceptadas (con ``stake``).
        games: juegos con ``actual_margin`` y ``actual_total``.
        closing_odds: líneas de cierre indexadas por ``game_id``.
        bet_odds: líneas vigentes al apostar, indexadas por ``game_id``
            (necesarias para desvigorizar el precio tomado).

    Returns:
        Copia de ``bets`` con las columnas de ``SETTLEMENT_COLUMNS``.
    """
    if bets.empty:
        settled = bets.copy()
        for column in SETTLEMENT_COLUMNS:
            settled[column] = pd.Series(dtype="float64")
        return settled

    results_lookup = games.set_index("game_id")[["actual_margin", "actual_total"]]
    settled = bets.copy()
    columns: dict[str, list[object]] = {name: [] for name in SETTLEMENT_COLUMNS}

    for record in settled.to_dict(orient="records"):
        game_id = record["game_id"]
        market, side = str(record["market"]), str(record["side"])
        if game_id not in results_lookup.index:
            columns["result"].append(RESULT_UNKNOWN)
            columns["pnl"].append(0.0)
            columns["clv"].append(np.nan)
            columns["clv_points"].append(np.nan)
            columns["closing_american"].append(np.nan)
            columns["closing_line"].append(np.nan)
            continue

        margin = float(results_lookup.loc[game_id, "actual_margin"])
        total = float(results_lookup.loc[game_id, "actual_total"])
        result = _outcome(market, side, float(record["line"]), margin, total)
        columns["result"].append(result)
        columns["pnl"].append(
            _pnl(result, float(record["stake"]), float(record["american"]))
        )

        has_close = (
            closing_odds is not None
            and not closing_odds.empty
            and game_id in closing_odds.index
        )
        if not has_close:
            columns["clv"].append(np.nan)
            columns["clv_points"].append(np.nan)
            columns["closing_american"].append(np.nan)
            columns["closing_line"].append(np.nan)
            continue

        closing_row = closing_odds.loc[game_id]
        own_column, _ = _PRICE_COLUMNS[(market, side)]
        closing_line = _side_line(market, side, closing_row)
        points = _line_points_gained(
            market, side, float(record["line"]), closing_line
        )
        fair_close = _fair_probability(market, side, closing_row)
        fair_bet = (
            _fair_probability(market, side, bet_odds.loc[game_id])
            if bet_odds is not None
            and not bet_odds.empty
            and game_id in bet_odds.index
            else np.nan
        )
        price_component = (
            float(fair_close - fair_bet)
            if np.isfinite(fair_close) and np.isfinite(fair_bet)
            else 0.0
        )
        columns["clv"].append(price_component + _line_probability(market, points))
        columns["clv_points"].append(points)
        columns["closing_american"].append(float(closing_row[own_column]))
        columns["closing_line"].append(closing_line)

    for name, values in columns.items():
        settled[name] = values
    return settled


__all__ = [
    "MARGIN_LINE_SIGMA",
    "RESULT_LOSS",
    "RESULT_PUSH",
    "RESULT_UNKNOWN",
    "RESULT_WIN",
    "SETTLEMENT_COLUMNS",
    "TOTAL_LINE_SIGMA",
    "settle_bets",
]
