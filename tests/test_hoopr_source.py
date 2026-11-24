"""Tests de los box scores de hoopR: posesiones exactas y disponibilidad."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_sh.ingest.hoopr_source import (
    FREE_THROW_POSSESSION_FACTOR,
    ON_OFF_CLIP_PER_100,
    RECONSTRUCTED_HOURS_BEFORE_TIP,
    ROTATION_MIN_APPEARANCES,
    ROTATION_MIN_MINUTES,
    ROTATION_WINDOW_GAMES,
    apply_exact_possessions,
    build_availability,
    exact_possessions,
    player_game_impact,
    season_label_to_hoopr,
)
from nba_sh.schemas import INJURY_COLUMNS, STATUS_OUT, TEAM_BOX_COLUMNS

HOME, AWAY = "BOS", "NYK"
STAR, ROLE, BENCH = "star", "role", "bench"
SEASON = "2023-24"
PLAYER_BOX_SCHEMA = (
    "game_id",
    "team",
    "player_id",
    "player_name",
    "minutes",
    "plus_minus",
    "did_not_play",
)


def _team_box(**overrides: float) -> pd.DataFrame:
    """Box score de equipo de un partido (dos filas)."""
    base = {
        "field_goals_attempted": 88.0,
        "free_throws_attempted": 20.0,
        "offensive_rebounds": 10.0,
        "turnovers": 14.0,
    }
    base.update(overrides)
    possessions = (
        base["field_goals_attempted"]
        + FREE_THROW_POSSESSION_FACTOR * base["free_throws_attempted"]
        - base["offensive_rebounds"]
        + base["turnovers"]
    )
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team": team,
                "home_away": home_away,
                "score": 110.0,
                "opponent_score": 104.0,
                **base,
                "possessions": possessions,
            }
            for team, home_away in ((HOME, "home"), (AWAY, "away"))
        ]
    )


def _games(n_games: int = 1, *, periods: int = 4) -> pd.DataFrame:
    """Calendario mínimo con el esquema que usan las funciones."""
    return pd.DataFrame(
        [
            {
                "game_id": f"g{index + 1}",
                "season": SEASON,
                "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=2 * index),
                "home_team": HOME,
                "away_team": AWAY,
                "home_score": 110,
                "away_score": 104,
                "periods": periods,
                "possessions": 100.0,
            }
            for index in range(n_games)
        ]
    )


def _player_rows(
    game_id: str, players: dict[str, tuple[float, float]], *, team: str = HOME
) -> list[dict[str, object]]:
    """Filas de box score de jugador: ``{player: (minutos, plus_minus)}``."""
    return [
        {
            "game_id": game_id,
            "team": team,
            "player_id": player_id,
            "player_name": player_id,
            "minutes": minutes,
            "plus_minus": plus_minus,
            "did_not_play": minutes == 0.0,
        }
        for player_id, (minutes, plus_minus) in players.items()
    ]


@pytest.mark.unit
def test_season_label_maps_to_hoopr_year() -> None:
    """hoopR nombra los ficheros por el año en que acaba la temporada."""
    assert season_label_to_hoopr("2023-24") == 2024
    assert season_label_to_hoopr("2022-23") == 2023


@pytest.mark.unit
def test_exact_possessions_apply_the_guide_formula() -> None:
    """Poss = FGA + 0.44*FTA - OREB + TOV, promediando ambos equipos."""
    assert exact_possessions(_team_box()).loc["g1"] == pytest.approx(
        88.0 + 8.8 - 10.0 + 14.0
    )


@pytest.mark.unit
def test_apply_exact_possessions_replaces_only_known_games() -> None:
    """Los juegos sin box score conservan su estimación previa."""
    games = _games(2)
    games.loc[1, "possessions"] = 95.0
    updated, replaced = apply_exact_possessions(games, _team_box())
    assert replaced == 1
    assert updated.loc[0, "possessions"] == pytest.approx(100.8)
    assert updated.loc[1, "possessions"] == pytest.approx(95.0)


@pytest.mark.unit
def test_player_game_impact_matches_the_hand_computed_on_off() -> None:
    """net_on - net_off, escalado a 100 posesiones (caso bajo el clip)."""
    impact = player_game_impact(
        _games(1), pd.DataFrame(_player_rows("g1", {STAR: (30.0, 6.0)}))
    )
    assert len(impact) == 1
    net_on = 6.0 / 30.0 * 48.0
    net_off = (6.0 - 6.0) / (48.0 - 30.0) * 48.0
    expected = net_on - net_off
    assert abs(expected) < ON_OFF_CLIP_PER_100
    assert impact.loc[0, "on_off_per_100"] == pytest.approx(expected)


@pytest.mark.unit
def test_player_game_impact_is_clipped() -> None:
    """Un partido extremo no puede dominar la media del on/off."""
    impact = player_game_impact(
        _games(1), pd.DataFrame(_player_rows("g1", {STAR: (1.0, 20.0)}))
    )
    assert abs(impact.loc[0, "on_off_per_100"]) <= ON_OFF_CLIP_PER_100


@pytest.mark.unit
def test_player_game_impact_ignores_players_without_minutes() -> None:
    """Quien no jugó no tiene on/off en ese partido."""
    assert player_game_impact(
        _games(1), pd.DataFrame(_player_rows("g1", {STAR: (0.0, 0.0)}))
    ).empty


@pytest.mark.unit
def test_availability_flags_the_absence_of_a_rotation_player() -> None:
    """Tras varias apariciones, faltar un partido se registra como baja."""
    n_games = ROTATION_MIN_APPEARANCES + 1
    rows: list[dict[str, object]] = []
    for index in range(n_games):
        minutes = 0.0 if index == n_games - 1 else 30.0
        rows.extend(_player_rows(f"g{index + 1}", {STAR: (minutes, 4.0)}))
        rows.extend(_player_rows(f"g{index + 1}", {ROLE: (25.0, 2.0)}))
    availability = build_availability(_games(n_games), pd.DataFrame(rows))

    assert list(availability.columns) == list(INJURY_COLUMNS)
    assert len(availability) == 1
    row = availability.iloc[0]
    assert row["player_id"] == STAR
    assert row["game_id"] == f"g{n_games}"
    assert row["team"] == HOME
    assert row["status"] == STATUS_OUT
    assert row["hours_before_tip"] == RECONSTRUCTED_HOURS_BEFORE_TIP
    assert row["minutes_pg"] == pytest.approx(30.0)


@pytest.mark.unit
def test_availability_weights_use_only_previous_games() -> None:
    """Los minutos del ausente son los de ANTES, no los de después."""
    rows: list[dict[str, object]] = []
    for index, minutes in enumerate([20.0, 20.0, 20.0, 0.0, 40.0, 40.0]):
        rows.extend(_player_rows(f"g{index + 1}", {STAR: (minutes, 3.0)}))
        rows.extend(_player_rows(f"g{index + 1}", {ROLE: (25.0, 1.0)}))
    availability = build_availability(_games(6), pd.DataFrame(rows))
    absence = availability[availability["player_id"] == STAR].iloc[0]
    assert absence["game_id"] == "g4"
    assert absence["minutes_pg"] == pytest.approx(20.0)


@pytest.mark.unit
def test_availability_ignores_players_below_the_minutes_threshold() -> None:
    """Un jugador de minutos residuales no cuenta como baja de rotación."""
    n_games = ROTATION_MIN_APPEARANCES + 1
    rows: list[dict[str, object]] = []
    for index in range(n_games):
        minutes = 0.0 if index == n_games - 1 else ROTATION_MIN_MINUTES - 5.0
        rows.extend(_player_rows(f"g{index + 1}", {BENCH: (minutes, 0.0)}))
    assert build_availability(_games(n_games), pd.DataFrame(rows)).empty


@pytest.mark.unit
def test_availability_needs_a_minimum_number_of_appearances() -> None:
    """Con una sola aparición previa no se puede estimar el peso."""
    rows = _player_rows("g1", {STAR: (30.0, 4.0)}) + _player_rows(
        "g2", {ROLE: (30.0, 1.0)}
    )
    assert build_availability(_games(2), pd.DataFrame(rows)).empty


@pytest.mark.unit
def test_availability_forgets_players_outside_the_window() -> None:
    """Quien no aparece en muchos partidos deja de ser rotación (traspasos)."""
    n_games = ROTATION_WINDOW_GAMES + 6
    rows: list[dict[str, object]] = []
    for index in range(n_games):
        if index < ROTATION_MIN_APPEARANCES:
            rows.extend(_player_rows(f"g{index + 1}", {STAR: (30.0, 4.0)}))
        rows.extend(_player_rows(f"g{index + 1}", {ROLE: (25.0, 1.0)}))
    flagged = set(build_availability(_games(n_games), pd.DataFrame(rows))["game_id"])
    assert f"g{ROTATION_MIN_APPEARANCES + 1}" in flagged
    assert f"g{n_games}" not in flagged


@pytest.mark.unit
def test_availability_resets_between_seasons() -> None:
    """Las plantillas cambian: el estado no cruza de temporada."""
    games = _games(4)
    games.loc[3, "season"] = "2024-25"
    rows: list[dict[str, object]] = []
    for index in range(4):
        minutes = 0.0 if index == 3 else 30.0
        rows.extend(_player_rows(f"g{index + 1}", {STAR: (minutes, 4.0)}))
    assert build_availability(games, pd.DataFrame(rows)).empty


@pytest.mark.unit
def test_availability_handles_empty_input() -> None:
    """Sin box score de jugador no hay bajas, pero sí esquema."""
    availability = build_availability(
        _games(1), pd.DataFrame(columns=list(PLAYER_BOX_SCHEMA))
    )
    assert availability.empty
    assert list(availability.columns) == list(INJURY_COLUMNS)


def _raw_team_box(path: Path) -> Path:
    """Parquet de box score de equipo con el formato crudo de hoopR."""
    rows = [
        {
            "game_id": "g1",
            "season_type": 2,
            "team_abbreviation": team,
            "team_home_away": home_away,
            "team_score": 110,
            "opponent_team_score": 104,
            "field_goals_made": 42,
            "field_goals_attempted": 88,
            "three_point_field_goals_made": 12,
            "three_point_field_goals_attempted": 35,
            "free_throws_attempted": 20,
            "offensive_rebounds": 10,
            "total_turnovers": 14,
            "turnovers": 14,
            "fouls": 20,
            "points_in_paint": 44,
            "fast_break_points": 12,
        }
        for team, home_away in ((HOME, "home"), (AWAY, "away"))
    ]
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.mark.unit
def test_box_scores_expose_the_canonical_team_table(tmp_path: Path) -> None:
    """``canonical_team`` trae el detalle de tiro; ``team``, solo posesiones.

    Regresión de diseño: pasar ``box.team`` al ``Dataset`` deja las features de
    tiro a 0 sin que nada falle, y pasó. La tabla correcta viene ahora en el
    mismo objeto, para que no haya que rederivar rutas y elegir mal.
    """
    from nba_sh.ingest.hoopr_source import canonical_team_box, load_team_box

    path = _raw_team_box(tmp_path / "team_box.parquet")
    team = load_team_box([path])
    canonical = canonical_team_box([path], team)
    assert "field_goals_made" not in team.columns
    assert "field_goals_made" in canonical.columns
    assert canonical.loc[0, "possessions"] == pytest.approx(
        team.loc[0, "possessions"]
    )
    assert len(canonical) == len(team)
    assert list(canonical.columns) == list(TEAM_BOX_COLUMNS)
