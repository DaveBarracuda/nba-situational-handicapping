"""Ratings de potencia ofensiva/defensiva y ritmo (guía 6.2).

    Ataque_adj(t)  = media móvil ponderada (decay 0.94/semana) del OffRtg
                     ajustado por oponente
    Defensa_adj(t) = ídem con DefRtg
    Ritmo(t)       = ídem con las posesiones por 48 min

El ajuste por oponente usa el rating **previo** del rival, el decay es por días
reales transcurridos, y los juegos con lesiones clave pesan menos. Todo se
calcula online, juego a juego: los valores de un partido nunca entran en sus
propias medias (guía 5.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_sh.config import CONFIG, POSSESSIONS_SCALE, Config
from nba_sh.teams import VALID_TEAMS

DAYS_PER_WEEK = 7.0
INJURY_WEIGHT_DIVISOR = 10.0


@dataclass
class _TeamRatingState:
    """Estado online de los ratings de un equipo."""

    off_value: float
    def_value: float
    pace_value: float
    weight: float = 0.0
    last_date: pd.Timestamp | None = None
    games: int = 0

    def decayed_weight(self, day: pd.Timestamp, *, weekly_decay: float) -> float:
        """Peso efectivo tras aplicar el decay temporal."""
        if self.last_date is None:
            return self.weight
        days = max(float((day - self.last_date).days), 0.0)
        return self.weight * weekly_decay ** (days / DAYS_PER_WEEK)

    def update(
        self,
        day: pd.Timestamp,
        *,
        off_adjusted: float,
        def_adjusted: float,
        pace: float,
        sample_weight: float,
        weekly_decay: float,
    ) -> None:
        """Incorpora un juego al estado (media exponencial ponderada)."""
        decayed = self.decayed_weight(day, weekly_decay=weekly_decay)
        total = decayed + sample_weight
        self.off_value = (
            self.off_value * decayed + off_adjusted * sample_weight
        ) / total
        self.def_value = (
            self.def_value * decayed + def_adjusted * sample_weight
        ) / total
        self.pace_value = (self.pace_value * decayed + pace * sample_weight) / total
        self.weight = total
        self.last_date = day
        self.games += 1


@dataclass
class _LeagueMeans:
    """Medias de liga acumuladas (sin look-ahead)."""

    prior_off_mean: float
    prior_pace_mean: float
    rating_sum: float = field(default=0.0)
    pace_sum: float = field(default=0.0)
    count: int = field(default=0)

    def current(self) -> tuple[float, float]:
        """Medias vigentes (prior si aún no hay juegos)."""
        if self.count == 0:
            return self.prior_off_mean, self.prior_pace_mean
        return self.rating_sum / self.count, self.pace_sum / self.count

    def observe(self, *, ratings: tuple[float, float], pace: float) -> None:
        """Registra los ratings crudos de un juego."""
        self.rating_sum += sum(ratings)
        self.pace_sum += 2.0 * pace
        self.count += 2


def _shrunk(value: float, prior: float, weight: float, prior_weight: float) -> float:
    """Media ponderada entre el estado observado y el prior de liga."""
    return float((prior * prior_weight + value * weight) / (prior_weight + weight))


def _sample_weight(injury_points: float, *, config: Config) -> float:
    """Peso del juego: menor si el equipo jugó sin piezas clave."""
    penalty = abs(float(injury_points)) / INJURY_WEIGHT_DIVISOR
    return float(max(1.0 - penalty, config.power.injury_weight_floor))


def _apply_season_reset(state: _TeamRatingState, *, config: Config) -> None:
    """Regresión de los ratings hacia el prior al cambiar de temporada."""
    power = config.power
    carry = power.season_carryover
    state.off_value = carry * state.off_value + (1.0 - carry) * power.prior_off_rating
    state.def_value = carry * state.def_value + (1.0 - carry) * power.prior_def_rating
    state.pace_value = carry * state.pace_value + (1.0 - carry) * power.prior_pace
    state.weight = min(state.weight, power.prior_weight_games)
    state.games = 0


def run_power_ratings(
    games: pd.DataFrame,
    *,
    injury_impact: pd.DataFrame | None = None,
    config: Config = CONFIG,
) -> pd.DataFrame:
    """Calcula ratings de potencia pre-juego para cada partido.

    Args:
        games: tabla de juegos con ``possessions`` y marcadores.
        injury_impact: opcional, columnas ``game_id``, ``team``, ``impact_points``;
            reduce el peso de los juegos con bajas relevantes.
        config: configuración del modelo.

    Returns:
        DataFrame con una fila por juego: ratings y ritmo previos de ambos
        equipos, medias de liga y juegos disputados.
    """
    power = config.power
    states = {
        team: _TeamRatingState(
            off_value=power.prior_off_rating,
            def_value=power.prior_def_rating,
            pace_value=power.prior_pace,
        )
        for team in sorted(VALID_TEAMS)
    }
    league = _LeagueMeans(
        prior_off_mean=power.prior_off_rating, prior_pace_mean=power.prior_pace
    )

    impact_lookup: dict[tuple[str, str], float] = {}
    if injury_impact is not None and not injury_impact.empty:
        impact_lookup = {
            (str(row["game_id"]), str(row["team"])): float(row["impact_points"])
            for row in injury_impact.to_dict(orient="records")
        }

    ordered = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    current_season: str | None = None

    for record in ordered.to_dict(orient="records"):
        season = str(record["season"])
        day = pd.Timestamp(record["game_date"])
        if current_season is not None and season != current_season:
            for state in states.values():
                _apply_season_reset(state, config=config)
        current_season = season

        home, away = str(record["home_team"]), str(record["away_team"])
        league_off_mean, league_pace_mean = league.current()
        pre: dict[str, dict[str, float]] = {}
        for team in (home, away):
            state = states[team]
            weight = state.decayed_weight(day, weekly_decay=power.weekly_decay)
            pre[team] = {
                "off": _shrunk(
                    state.off_value,
                    power.prior_off_rating,
                    weight,
                    power.prior_weight_games,
                ),
                "def": _shrunk(
                    state.def_value,
                    power.prior_def_rating,
                    weight,
                    power.prior_weight_games,
                ),
                "pace": _shrunk(
                    state.pace_value, power.prior_pace, weight, power.prior_weight_games
                ),
                "games": float(state.games),
            }

        rows.append(
            {
                "game_id": str(record["game_id"]),
                "off_home_pre": pre[home]["off"],
                "def_home_pre": pre[home]["def"],
                "pace_home_pre": pre[home]["pace"],
                "games_played_home": pre[home]["games"],
                "off_away_pre": pre[away]["off"],
                "def_away_pre": pre[away]["def"],
                "pace_away_pre": pre[away]["pace"],
                "games_played_away": pre[away]["games"],
                "league_off_mean": league_off_mean,
                "league_pace_mean": league_pace_mean,
            }
        )

        possessions = record.get("possessions")
        home_score, away_score = record.get("home_score"), record.get("away_score")
        if pd.isna(possessions) or pd.isna(home_score) or pd.isna(away_score):
            continue
        possessions = float(possessions)
        if possessions <= 0.0:
            continue
        off_raw_home = float(home_score) / possessions * POSSESSIONS_SCALE
        off_raw_away = float(away_score) / possessions * POSSESSIONS_SCALE
        league.observe(ratings=(off_raw_home, off_raw_away), pace=possessions)

        game_id = str(record["game_id"])
        for team, opponent, own_raw, opponent_raw in (
            (home, away, off_raw_home, off_raw_away),
            (away, home, off_raw_away, off_raw_home),
        ):
            off_adjusted = own_raw - (pre[opponent]["def"] - league_off_mean)
            def_adjusted = opponent_raw - (pre[opponent]["off"] - league_off_mean)
            states[team].update(
                day,
                off_adjusted=off_adjusted,
                def_adjusted=def_adjusted,
                pace=possessions,
                sample_weight=_sample_weight(
                    impact_lookup.get((game_id, team), 0.0), config=config
                ),
                weekly_decay=power.weekly_decay,
            )

    return pd.DataFrame(rows)


__all__ = ["DAYS_PER_WEEK", "INJURY_WEIGHT_DIVISOR", "run_power_ratings"]
