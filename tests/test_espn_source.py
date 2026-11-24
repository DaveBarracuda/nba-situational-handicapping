"""Tests del ingestor de ESPN (sin red: payload sintético con su formato)."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sh.ingest.espn_source import (
    FREE_THROW_POSSESSION_FACTOR,
    LEAGUE_AVERAGE_TURNOVERS,
    OFFENSIVE_REBOUND_RATE,
    estimate_possessions_from_shooting,
    events_to_games,
    season_months,
)
from nba_sh.schemas import GAME_COLUMNS, SEASON_TYPE_REGULAR
from nba_sh.validation import validate_game_row

SEASON = "2023-24"
MONTHS_PER_SEASON = 9


def _competitor(
    *,
    abbreviation: str,
    home_away: str,
    score: int,
    fga: float = 88.0,
    fgm: float = 42.0,
    fta: float = 20.0,
    linescores: int = 4,
) -> dict[str, object]:
    """Competidor con el formato que devuelve ESPN."""
    return {
        "homeAway": home_away,
        "score": str(score),
        "team": {"abbreviation": abbreviation, "displayName": abbreviation},
        "linescores": [{"value": 25.0} for _ in range(linescores)],
        "statistics": [
            {"name": "fieldGoalsAttempted", "displayValue": str(fga)},
            {"name": "fieldGoalsMade", "displayValue": str(fgm)},
            {"name": "freeThrowsAttempted", "displayValue": str(fta)},
            {"name": "rebounds", "displayValue": "44"},
        ],
    }


def _event(
    *,
    event_id: str = "401585183",
    date: str = "2024-01-16T00:30Z",
    status: str = "STATUS_FINAL",
    season_type: int = 2,
    periods: int = 4,
    home: str = "BOS",
    away: str = "NYK",
    home_score: int = 112,
    away_score: int = 104,
) -> dict[str, object]:
    """Evento completo con el formato que devuelve ESPN."""
    return {
        "id": event_id,
        "date": date,
        "season": {"year": 2024, "type": season_type},
        "competitions": [
            {
                "date": date,
                "status": {"period": periods, "type": {"name": status}},
                "competitors": [
                    _competitor(
                        abbreviation=home,
                        home_away="home",
                        score=home_score,
                        linescores=periods,
                    ),
                    _competitor(
                        abbreviation=away,
                        home_away="away",
                        score=away_score,
                        linescores=periods,
                    ),
                ],
            }
        ],
    }


@pytest.mark.unit
def test_season_months_cover_october_to_june() -> None:
    """Una temporada son 9 meses, de octubre a junio."""
    months = season_months(SEASON)
    assert len(months) == MONTHS_PER_SEASON
    assert months[0] == "202310"
    assert months[-1] == "202406"


@pytest.mark.unit
def test_possession_estimate_uses_the_documented_substitutions() -> None:
    """La estimación sustituye OREB por su tasa y TOV por la media de liga."""
    estimated = estimate_possessions_from_shooting(88.0, 42.0, 20.0)
    expected = (
        88.0
        + FREE_THROW_POSSESSION_FACTOR * 20.0
        - OFFENSIVE_REBOUND_RATE * (88.0 - 42.0)
        + LEAGUE_AVERAGE_TURNOVERS
    )
    assert estimated == pytest.approx(expected)
    assert 85.0 < estimated < 115.0


@pytest.mark.unit
def test_events_to_games_builds_a_valid_canonical_row() -> None:
    """Un evento final se convierte en una fila que pasa la validación."""
    games = events_to_games([_event()], season=SEASON)
    assert list(games.columns) == list(GAME_COLUMNS)
    assert len(games) == 1
    row = games.iloc[0]
    assert row["home_team"] == "BOS"
    assert row["away_team"] == "NYK"
    assert row["home_score"] == 112
    assert row["season"] == SEASON
    assert row["season_type"] == SEASON_TYPE_REGULAR
    assert row["periods"] == 4
    assert row["crew_id"] is None
    assert all(
        validate_game_row(row.to_dict(), today=pd.Timestamp("2025-01-01")).values()
    )


@pytest.mark.unit
def test_game_date_is_eastern_not_utc() -> None:
    """Un partido de las 20:30 ET se guarda con la fecha ET, no la UTC."""
    games = events_to_games([_event(date="2024-01-16T01:30Z")], season=SEASON)
    assert games.loc[0, "game_date"] == pd.Timestamp("2024-01-15")


@pytest.mark.unit
def test_overtime_periods_are_preserved() -> None:
    """Las prórrogas llegan al esquema canónico."""
    games = events_to_games([_event(periods=6)], season=SEASON)
    assert games.loc[0, "periods"] == 6


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "STATUS_SCHEDULED"},
        {"status": "STATUS_IN_PROGRESS"},
        {"season_type": 1},
        {"season_type": 3},
        {"home": "SHQ", "away": "CAN"},
    ],
)
def test_non_regular_or_unfinished_events_are_skipped(
    overrides: dict[str, object],
) -> None:
    """Pretemporada, playoffs, partidos sin acabar y el All-Star se descartan."""
    assert events_to_games([_event(**overrides)], season=SEASON).empty


@pytest.mark.unit
def test_events_without_shooting_statistics_are_skipped() -> None:
    """Sin datos de tiro no se puede estimar el ritmo: se descarta la fila."""
    event = _event()
    event["competitions"][0]["competitors"][0]["statistics"] = []
    assert events_to_games([event], season=SEASON).empty


@pytest.mark.unit
def test_duplicate_events_are_deduplicated() -> None:
    """El mismo partido en dos meses solapados cuenta una vez."""
    assert len(events_to_games([_event(), _event()], season=SEASON)) == 1


@pytest.mark.unit
def test_empty_payload_returns_the_canonical_schema() -> None:
    """Un mes sin partidos devuelve el esquema vacío, no un error."""
    games = events_to_games([], season=SEASON)
    assert games.empty
    assert list(games.columns) == list(GAME_COLUMNS)
