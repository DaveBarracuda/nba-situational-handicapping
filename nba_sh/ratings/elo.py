"""Elo dinámico adaptado a NBA (guía 6.1).

    E_A = 1 / (1 + 10^(-(R_A - R_B + HFA)/400))
    K   = K_base * (1 + margen/40)
    HFA = 55 puntos Elo, ajustado por altitud y fatiga del visitante
    Carryover entre temporadas: R_new = 0.75*R_old + 0.25*1505

El recorrido es estrictamente cronológico y cada juego se evalúa con los
ratings **previos** al tip-off: no hay look-ahead por construcción.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.teams import VALID_TEAMS, conference

# Conversión Elo -> puntos de margen (≈25 pts Elo por punto real).
ELO_POINTS_PER_POINT = 25.0
CONTEXT_COLUMNS = ("is_b2b_away", "altitude_game", "away_road_trip_4plus")


@dataclass(frozen=True)
class EloResult:
    """Salida del recorrido Elo."""

    game_ratings: pd.DataFrame
    daily_ratings: pd.DataFrame
    final_ratings: Mapping[str, float]

    def ranks_by_date(self) -> pd.DataFrame:
        """Rankings de liga y conferencia por fecha.

        Las filas de una fecha describen el estado **al terminar** ese día; el
        feature store debe consultar la última fecha anterior al juego.
        """
        frame = self.daily_ratings.copy()
        if frame.empty:
            return frame.assign(league_rank=[], conference=[], conference_rank=[])
        frame["league_rank"] = frame.groupby("date")["rating"].rank(
            ascending=False, method="min"
        )
        frame["conference"] = frame["team"].map(conference)
        frame["conference_rank"] = frame.groupby(["date", "conference"])["rating"].rank(
            ascending=False, method="min"
        )
        return frame


def expected_home_probability(
    elo_home: float, elo_away: float, *, hfa: float, config: Config = CONFIG
) -> float:
    """Probabilidad Elo de victoria local."""
    exponent = -(elo_home - elo_away + hfa) / config.elo.elo_scale
    return float(1.0 / (1.0 + 10.0**exponent))


def dynamic_hfa(
    *,
    is_b2b_away: bool = False,
    altitude_game: bool = False,
    away_road_trip_4plus: bool = False,
    config: Config = CONFIG,
) -> float:
    """HFA dinámico en puntos Elo (guía 6.1).

    El HFA sube cuando el visitante llega fatigado o juega en altitud.
    """
    elo = config.elo
    hfa = elo.hfa_elo
    if is_b2b_away:
        hfa += elo.hfa_away_b2b_bonus
    if altitude_game:
        hfa += elo.hfa_altitude_bonus
    if away_road_trip_4plus:
        hfa += elo.hfa_away_long_trip_bonus
    return float(hfa)


def _margin_multiplier(margin: float, *, config: Config = CONFIG) -> float:
    """Factor de K por margen de victoria (las palizas mueven más)."""
    capped = min(abs(float(margin)), config.elo.margin_cap)
    return float(1.0 + capped / config.elo.margin_divisor)


def _apply_carryover(
    ratings: Mapping[str, float], *, config: Config = CONFIG
) -> dict[str, float]:
    """Regresión a la media entre temporadas."""
    elo = config.elo
    return {
        team: elo.carryover * rating + (1.0 - elo.carryover) * elo.carryover_target
        for team, rating in ratings.items()
    }


def run_elo(
    games: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    config: Config = CONFIG,
) -> EloResult:
    """Recorre los juegos en orden y devuelve ratings pre-juego y diarios.

    Args:
        games: tabla de juegos (requiere ``game_date``, ``season``, equipos y
            marcadores), que se ordena internamente por fecha.
        context: opcional, indexado por ``game_id``, con las columnas de
            ``CONTEXT_COLUMNS`` para el HFA dinámico.
        config: configuración del modelo.

    Returns:
        :class:`EloResult` con una fila por juego y el histórico diario.
    """
    ratings: dict[str, float] = dict.fromkeys(sorted(VALID_TEAMS), config.elo.base_rating)
    ordered = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    has_context = context is not None and not context.empty
    context_columns = [
        name for name in CONTEXT_COLUMNS if has_context and name in context.columns
    ]

    rows: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    current_season: str | None = None
    current_date: pd.Timestamp | None = None

    for record in ordered.to_dict(orient="records"):
        season = str(record["season"])
        game_date = pd.Timestamp(record["game_date"])
        if current_season is not None and season != current_season:
            ratings = _apply_carryover(ratings, config=config)
        if current_date is not None and game_date != current_date:
            daily.extend(
                {"date": current_date, "team": team, "rating": rating}
                for team, rating in ratings.items()
            )
        current_season, current_date = season, game_date

        home, away = str(record["home_team"]), str(record["away_team"])
        elo_home, elo_away = ratings[home], ratings[away]

        game_id = str(record["game_id"])
        flags = dict.fromkeys(CONTEXT_COLUMNS, False)
        if has_context and game_id in context.index:
            row = context.loc[game_id]
            for name in context_columns:
                flags[name] = bool(row[name])
        hfa = dynamic_hfa(
            is_b2b_away=flags["is_b2b_away"],
            altitude_game=flags["altitude_game"],
            away_road_trip_4plus=flags["away_road_trip_4plus"],
            config=config,
        )
        probability = expected_home_probability(
            elo_home, elo_away, hfa=hfa, config=config
        )

        rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "elo_home_pre": elo_home,
                "elo_away_pre": elo_away,
                "elo_hfa": hfa,
                "elo_diff": elo_home - elo_away + hfa,
                "elo_margin": (elo_home - elo_away + hfa) / ELO_POINTS_PER_POINT,
                "p_elo_home": probability,
            }
        )

        home_score, away_score = record.get("home_score"), record.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        margin = float(home_score) - float(away_score)
        outcome = 1.0 if margin > 0 else 0.0
        k_factor = config.elo.k_base * _margin_multiplier(margin, config=config)
        delta = k_factor * (outcome - probability)
        ratings[home] = elo_home + delta
        ratings[away] = elo_away - delta

    if current_date is not None:
        daily.extend(
            {"date": current_date, "team": team, "rating": rating}
            for team, rating in ratings.items()
        )

    return EloResult(
        game_ratings=pd.DataFrame(rows),
        daily_ratings=pd.DataFrame(daily),
        final_ratings=dict(ratings),
    )


__all__ = [
    "CONTEXT_COLUMNS",
    "ELO_POINTS_PER_POINT",
    "EloResult",
    "dynamic_hfa",
    "expected_home_probability",
    "run_elo",
]
