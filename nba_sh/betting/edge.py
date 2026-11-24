"""Detección de valor: modelo vs mercado (guía 10.1, 10.2, 7.4 y 8.3).

Para cada juego se evalúan las seis salidas posibles (ML local/visitante,
spread local/visitante, over/under) y se registran **todas**, apostables o no:
sin registro de las apuestas no tomadas el backtest queda sesgado (error
común #7 de la guía).

Umbrales (guía 10.2):
    ML      -> edge de probabilidad >= 3% y valor esperado >= 3%
    Spread  -> >= 1.5 pts de diferencia de línea
    Totales -> >= 3.5 pts de diferencia de línea y 3% de probabilidad
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.betting.kelly import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    KELLY_MAIN,
    KELLY_SITUATIONAL,
    kelly_fraction,
)
from nba_sh.config import CONFIG, Config
from nba_sh.odds_utils import (
    american_to_decimal,
    american_to_probability,
    expected_value,
    fair_two_way,
)
from nba_sh.schemas import MARKET_ML, MARKET_SPREAD, MARKET_TOTAL

SIDE_HOME = "home"
SIDE_AWAY = "away"
SIDE_OVER = "over"
SIDE_UNDER = "under"

BET_COLUMNS: tuple[str, ...] = (
    "bet_id",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "market",
    "side",
    "line",
    "american",
    "decimal",
    "model_probability",
    "market_fair_probability",
    "prob_edge",
    "points_edge",
    "expected_value",
    "kelly_full",
    "kelly_bucket",
    "confidence",
    "line_move_confirms",
    "situational_points",
    "is_candidate",
    "rejection_reason",
)

STRONG_SIGNAL_MULTIPLIER = 2.0
SITUATIONAL_SIGNAL_POINTS = 1.5
REASON_OK = ""
REASON_NO_EDGE = "edge por debajo del umbral"
REASON_ODDS_RANGE = "cuota fuera del rango liquido"
REASON_NO_LINE = "sin linea de mercado"
REASON_NO_VALUE = "valor esperado negativo"
REASON_LINE_MOVE = "movimiento de linea no confirma"


def _confidence(
    *,
    prob_edge: float,
    prob_threshold: float,
    points_edge: float,
    points_threshold: float,
    line_move_confirms: bool,
    situational_points: float,
    injury_driven: bool,
) -> str:
    """Nivel de confianza por número de señales independientes (guía 11.3)."""
    if injury_driven:
        return CONFIDENCE_LOW
    signals = 0
    if (
        points_threshold > 0.0
        and abs(points_edge) >= STRONG_SIGNAL_MULTIPLIER * points_threshold
    ):
        signals += 1
    if prob_edge >= STRONG_SIGNAL_MULTIPLIER * prob_threshold:
        signals += 1
    if line_move_confirms:
        signals += 1
    if abs(situational_points) >= SITUATIONAL_SIGNAL_POINTS:
        signals += 1
    if signals >= 2:
        return CONFIDENCE_HIGH
    if signals == 1:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _row(
    *,
    game: pd.Series,
    market: str,
    side: str,
    line: float,
    american: float,
    model_probability: float,
    market_fair_probability: float,
    points_edge: float,
    prob_threshold: float,
    points_threshold: float,
    line_move_confirms: bool,
    situational_points: float,
    injury_driven: bool,
    config: Config,
) -> dict[str, object]:
    """Construye una fila de oportunidad evaluada (apostable o no)."""
    value = expected_value(model_probability, american)
    prob_edge = float(model_probability - market_fair_probability)
    odds_in_range = (
        config.market.ml_odds_min <= american <= config.market.ml_odds_max
        if market == MARKET_ML
        else True
    )

    reason = REASON_OK
    if not odds_in_range:
        reason = REASON_ODDS_RANGE
    elif prob_edge < prob_threshold or (
        points_threshold > 0.0 and points_edge < points_threshold
    ):
        reason = REASON_NO_EDGE
    elif market == MARKET_ML and value < config.market.ml_min_value or value <= 0.0:
        reason = REASON_NO_VALUE
    elif config.market.require_line_move_confirmation and not line_move_confirms:
        reason = REASON_LINE_MOVE

    return {
        "bet_id": f"{game['game_id']}-{market}-{side}",
        "game_id": game["game_id"],
        "game_date": game["game_date"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "market": market,
        "side": side,
        "line": float(line),
        "american": float(american),
        "decimal": american_to_decimal(american),
        "model_probability": float(model_probability),
        "market_fair_probability": float(market_fair_probability),
        "prob_edge": prob_edge,
        "points_edge": float(points_edge),
        "expected_value": float(value),
        "kelly_full": kelly_fraction(model_probability, american),
        "kelly_bucket": KELLY_SITUATIONAL if injury_driven else KELLY_MAIN,
        "confidence": _confidence(
            prob_edge=prob_edge,
            prob_threshold=prob_threshold,
            points_edge=points_edge,
            points_threshold=points_threshold,
            line_move_confirms=line_move_confirms,
            situational_points=situational_points,
            injury_driven=injury_driven,
        ),
        "line_move_confirms": bool(line_move_confirms),
        "situational_points": float(situational_points),
        "is_candidate": reason == REASON_OK,
        "rejection_reason": reason,
    }


def _ml_line_move_confirms(game: pd.Series, lines: pd.Series, side: str) -> bool:
    """True si la línea se movió hacia el lado apostado desde la apertura."""
    opening = game.get("open_ml_home")
    if opening is None or pd.isna(opening):
        return False
    opening_probability = american_to_probability(float(opening))
    current_probability = american_to_probability(float(lines["ml_home"]))
    if side == SIDE_HOME:
        return current_probability > opening_probability
    return current_probability < opening_probability


def _moneyline_rows(
    game: pd.Series, lines: pd.Series, *, config: Config
) -> list[dict[str, object]]:
    """Oportunidades de moneyline para un juego."""
    fair_home, fair_away = fair_two_way(lines["ml_home"], lines["ml_away"])
    probability_home = float(game["p_home_ml"])
    return [
        _row(
            game=game,
            market=MARKET_ML,
            side=side,
            line=float(american),
            american=float(american),
            model_probability=probability,
            market_fair_probability=fair,
            points_edge=0.0,
            prob_threshold=config.market.ml_min_prob_edge,
            points_threshold=0.0,
            line_move_confirms=_ml_line_move_confirms(game, lines, side),
            situational_points=float(game.get("situational_margin_adjustment", 0.0)),
            injury_driven=bool(game.get("injury_driven", False)),
            config=config,
        )
        for side, probability, american, fair in (
            (SIDE_HOME, probability_home, lines["ml_home"], fair_home),
            (SIDE_AWAY, 1.0 - probability_home, lines["ml_away"], fair_away),
        )
    ]


def _spread_rows(
    game: pd.Series, lines: pd.Series, *, config: Config
) -> list[dict[str, object]]:
    """Oportunidades de spread para un juego."""
    spread_home = float(lines["spread_home"])
    market_margin = -spread_home
    model_margin = float(game["model_margin"])
    fair_home, fair_away = fair_two_way(
        lines["spread_home_price"], lines["spread_away_price"]
    )
    probability_cover = float(game["p_home_cover"])
    push = float(game.get("p_push_spread", 0.0) or 0.0)
    probability_away = max(1.0 - probability_cover - push, 0.0)
    opening_spread = game.get("open_spread_home")
    moved = (
        0.0
        if opening_spread is None or pd.isna(opening_spread)
        else float(opening_spread) - spread_home
    )
    return [
        _row(
            game=game,
            market=MARKET_SPREAD,
            side=side,
            line=line,
            american=float(american),
            model_probability=probability,
            market_fair_probability=fair,
            points_edge=points_edge,
            prob_threshold=config.market.spread_min_prob_edge,
            points_threshold=config.market.spread_min_points,
            line_move_confirms=confirms,
            situational_points=float(game.get("situational_margin_adjustment", 0.0)),
            injury_driven=bool(game.get("injury_driven", False)),
            config=config,
        )
        for side, probability, american, fair, points_edge, line, confirms in (
            (
                SIDE_HOME,
                probability_cover,
                lines["spread_home_price"],
                fair_home,
                model_margin - market_margin,
                spread_home,
                moved > 0.0,
            ),
            (
                SIDE_AWAY,
                probability_away,
                lines["spread_away_price"],
                fair_away,
                market_margin - model_margin,
                -spread_home,
                moved < 0.0,
            ),
        )
    ]


def _total_rows(
    game: pd.Series, lines: pd.Series, *, config: Config
) -> list[dict[str, object]]:
    """Oportunidades de total para un juego."""
    total_line = float(lines["total_line"])
    model_total = float(game["model_total"])
    fair_over, fair_under = fair_two_way(lines["over_price"], lines["under_price"])
    probability_over = float(game["p_over"])
    push = float(game.get("p_push_total", 0.0) or 0.0)
    probability_under = max(1.0 - probability_over - push, 0.0)
    opening_total = game.get("open_total_line")
    moved = (
        0.0
        if opening_total is None or pd.isna(opening_total)
        else total_line - float(opening_total)
    )
    return [
        _row(
            game=game,
            market=MARKET_TOTAL,
            side=side,
            line=total_line,
            american=float(american),
            model_probability=probability,
            market_fair_probability=fair,
            points_edge=points_edge,
            prob_threshold=config.market.total_min_prob_edge,
            points_threshold=config.market.total_min_points,
            line_move_confirms=confirms,
            situational_points=float(game.get("situational_total_adjustment", 0.0)),
            injury_driven=bool(game.get("injury_driven", False)),
            config=config,
        )
        for side, probability, american, fair, points_edge, confirms in (
            (
                SIDE_OVER,
                probability_over,
                lines["over_price"],
                fair_over,
                model_total - total_line,
                moved > 0.0,
            ),
            (
                SIDE_UNDER,
                probability_under,
                lines["under_price"],
                fair_under,
                total_line - model_total,
                moved < 0.0,
            ),
        )
    ]


def compute_candidates(
    predictions: pd.DataFrame, lines: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Evalúa todas las salidas de cada juego contra el mercado.

    Args:
        predictions: predicciones del modelo (ver ``PREDICTION_COLUMNS``), con
            las columnas de contexto situacional ya mezcladas.
        lines: líneas del mercado indexadas por ``game_id``.
        config: configuración del modelo.

    Returns:
        DataFrame con el esquema ``BET_COLUMNS``; ``is_candidate`` indica si la
        oportunidad pasó todos los filtros.
    """
    if predictions.empty:
        return pd.DataFrame(columns=list(BET_COLUMNS))

    rows: list[dict[str, object]] = []
    for _, game in predictions.iterrows():
        game_id = game["game_id"]
        if game_id not in lines.index:
            rows.append(
                {
                    **dict.fromkeys(BET_COLUMNS, np.nan),
                    "bet_id": f"{game_id}-none-none",
                    "game_id": game_id,
                    "game_date": game["game_date"],
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "is_candidate": False,
                    "rejection_reason": REASON_NO_LINE,
                }
            )
            continue
        market_lines = lines.loc[game_id]
        rows.extend(_moneyline_rows(game, market_lines, config=config))
        rows.extend(_spread_rows(game, market_lines, config=config))
        rows.extend(_total_rows(game, market_lines, config=config))

    return pd.DataFrame(rows, columns=list(BET_COLUMNS))


__all__ = [
    "BET_COLUMNS",
    "REASON_LINE_MOVE",
    "REASON_NO_EDGE",
    "REASON_NO_LINE",
    "REASON_NO_VALUE",
    "REASON_ODDS_RANGE",
    "REASON_OK",
    "SIDE_AWAY",
    "SIDE_HOME",
    "SIDE_OVER",
    "SIDE_UNDER",
    "SITUATIONAL_SIGNAL_POINTS",
    "STRONG_SIGNAL_MULTIPLIER",
    "compute_candidates",
]
