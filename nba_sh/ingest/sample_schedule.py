"""Generador de calendario sintético con patrones NBA realistas.

Produce back-to-backs, road trips largos, descansos dispares y congestión
semanal: la materia prima de los factores situacionales (guía 5.1, 5.2).
El calendario es determinista dado el ``rng``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_sh.schemas import SEASON_TYPE_REGULAR
from nba_sh.teams import VALID_TEAMS

TIPOFF_HOURS_ET = (19.0, 19.5, 20.0, 22.5)
ET_TO_UTC_OFFSET_HOURS = 5.0
MAX_H2H_MEETINGS = 4
# Peso que se añade a la prioridad de un equipo que jugó ayer. El calendario
# real de la NBA evita los días consecutivos más de lo que haría un sorteo
# aleatorio, así que el peso es negativo; está calibrado para dar ~15
# back-to-backs por equipo y temporada (~18% de los juegos), que es la tasa
# real y el factor situacional #1 de la guía (5.1).
BACK_TO_BACK_SELECTION_WEIGHT = -1.5


@dataclass(frozen=True)
class ScheduleConfig:
    """Parámetros del calendario sintético."""

    games_per_team: int = 82
    season_start_month: int = 10
    season_start_day: int = 22
    season_days: int = 168
    max_games_per_day: int = 12
    road_trip_continue_prob: float = 0.55
    max_away_streak: int = 6


DEFAULT_SCHEDULE_CONFIG = ScheduleConfig()


@dataclass
class _TeamState:
    """Estado mutable interno del scheduler (no sale del módulo)."""

    games: int = 0
    home_games: int = 0
    away_streak: int = 0
    last_dates: tuple[pd.Timestamp, ...] = ()

    def played_on(self, day: pd.Timestamp) -> bool:
        """True si el equipo jugó ese día."""
        return day in self.last_dates

    def played_last_two_days(self, day: pd.Timestamp) -> bool:
        """True si jugó ayer y antes de ayer (bloquea 3 juegos en 3 días)."""
        yesterday = day - pd.Timedelta(days=1)
        before = day - pd.Timedelta(days=2)
        return self.played_on(yesterday) and self.played_on(before)

    def record(self, day: pd.Timestamp, *, is_home: bool) -> None:
        """Registra un juego jugado."""
        self.games += 1
        self.home_games += int(is_home)
        self.away_streak = 0 if is_home else self.away_streak + 1
        self.last_dates = (*self.last_dates[-4:], day)


def season_label(start_year: int) -> str:
    """Etiqueta de temporada estilo NBA: ``2023-24``."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _season_dates(start_year: int, config: ScheduleConfig) -> list[pd.Timestamp]:
    """Fechas candidatas de la temporada regular."""
    start = pd.Timestamp(
        year=start_year, month=config.season_start_month, day=config.season_start_day
    )
    return [start + pd.Timedelta(days=offset) for offset in range(config.season_days)]


def _games_today(
    rng: np.random.Generator,
    *,
    eligible: int,
    remaining_games: int,
    remaining_days: int,
    config: ScheduleConfig,
) -> int:
    """Cuántos juegos programar hoy para terminar la temporada a tiempo."""
    if eligible < 2 or remaining_games <= 0 or remaining_days <= 0:
        return 0
    target = remaining_games / max(remaining_days, 1)
    count = int(round(target + rng.normal(0.0, 1.6)))
    return int(np.clip(count, 1, min(config.max_games_per_day, eligible // 2)))


def _pick_teams(
    rng: np.random.Generator,
    states: Mapping[str, _TeamState],
    day: pd.Timestamp,
    *,
    n_games: int,
    config: ScheduleConfig,
) -> list[str]:
    """Selecciona 2*n_games equipos priorizando a los más atrasados."""
    eligible = [
        team
        for team, state in states.items()
        if state.games < config.games_per_team
        and not state.played_on(day)
        and not state.played_last_two_days(day)
    ]
    if len(eligible) < 2 * n_games:
        n_games = len(eligible) // 2
    if n_games <= 0:
        return []
    yesterday = day - pd.Timedelta(days=1)
    scores = np.array(
        [
            config.games_per_team
            - states[team].games
            + (
                BACK_TO_BACK_SELECTION_WEIGHT
                if states[team].played_on(yesterday)
                else 0.0
            )
            for team in eligible
        ],
        dtype=float,
    )
    order = np.argsort(-(scores + rng.normal(0.0, 1.0, size=scores.size)))
    return [eligible[index] for index in order[: 2 * n_games]]


def _pair_teams(
    rng: np.random.Generator,
    selected: list[str],
    meetings: dict[frozenset[str], int],
) -> list[tuple[str, str]]:
    """Empareja equipos evitando repetir demasiado el mismo matchup."""
    pool = list(selected)
    rng.shuffle(pool)
    pairs: list[tuple[str, str]] = []
    while len(pool) >= 2:
        first = pool.pop(0)
        choice_index = 0
        for index, candidate in enumerate(pool):
            if meetings.get(frozenset((first, candidate)), 0) < MAX_H2H_MEETINGS:
                choice_index = index
                break
        second = pool.pop(choice_index)
        key = frozenset((first, second))
        meetings[key] = meetings.get(key, 0) + 1
        pairs.append((first, second))
    return pairs


def _assign_home_away(
    rng: np.random.Generator,
    pair: tuple[str, str],
    states: Mapping[str, _TeamState],
    config: ScheduleConfig,
) -> tuple[str, str]:
    """Decide local y visitante, prolongando road trips a veces."""
    first, second = pair
    for team, other in ((first, second), (second, first)):
        state = states[team]
        if (
            0 < state.away_streak < config.max_away_streak
            and rng.random() < config.road_trip_continue_prob
        ):
            return other, team
    if states[first].home_games <= states[second].home_games:
        return first, second
    return second, first


def build_schedule(
    start_year: int,
    rng: np.random.Generator,
    *,
    config: ScheduleConfig = DEFAULT_SCHEDULE_CONFIG,
) -> pd.DataFrame:
    """Construye el calendario de una temporada.

    Returns:
        DataFrame con ``game_id``, ``season``, ``season_type``, ``game_date``,
        ``tipoff_utc``, ``home_team`` y ``away_team``, ordenado por fecha.
    """
    states: dict[str, _TeamState] = {team: _TeamState() for team in sorted(VALID_TEAMS)}
    meetings: dict[frozenset[str], int] = {}
    season = season_label(start_year)
    dates = _season_dates(start_year, config)
    rows: list[dict[str, object]] = []
    sequence = 0

    for day_index, day in enumerate(dates):
        remaining_games = (
            sum(max(config.games_per_team - state.games, 0) for state in states.values())
            // 2
        )
        remaining_days = len(dates) - day_index
        eligible_count = sum(
            1
            for state in states.values()
            if state.games < config.games_per_team
            and not state.played_last_two_days(day)
        )
        n_games = _games_today(
            rng,
            eligible=eligible_count,
            remaining_games=remaining_games,
            remaining_days=remaining_days,
            config=config,
        )
        if n_games == 0:
            continue
        selected = _pick_teams(rng, states, day, n_games=n_games, config=config)
        for pair in _pair_teams(rng, selected, meetings):
            home, away = _assign_home_away(rng, pair, states, config)
            sequence += 1
            tipoff_hour = float(rng.choice(TIPOFF_HOURS_ET))
            rows.append(
                {
                    "game_id": f"00{str(start_year)[-2:]}2{sequence:05d}",
                    "season": season,
                    "season_type": SEASON_TYPE_REGULAR,
                    "game_date": day,
                    "tipoff_utc": day
                    + pd.Timedelta(hours=tipoff_hour + ET_TO_UTC_OFFSET_HOURS),
                    "home_team": home,
                    "away_team": away,
                }
            )
            states[home].record(day, is_home=True)
            states[away].record(day, is_home=False)

    schedule = pd.DataFrame(rows)
    return schedule.sort_values(["game_date", "game_id"]).reset_index(drop=True)


__all__ = [
    "BACK_TO_BACK_SELECTION_WEIGHT",
    "DEFAULT_SCHEDULE_CONFIG",
    "ScheduleConfig",
    "build_schedule",
    "season_label",
]
