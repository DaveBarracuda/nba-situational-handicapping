"""Tests del perfil de tiro (guía 5.4): cálculo, causalidad y ausencia de box."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.features.long_frame import ORDER_COLUMN
from nba_sh.features.shooting import (
    EFG_THREE_WEIGHT,
    MIN_PERIODS,
    SHOOTING_COLUMNS,
    add_shooting_features,
    team_shooting_profile,
)

FGA = 80.0
FGM = 40.0
THREES_MADE = 10.0
THREES_ATTEMPTED = 30.0
FTA = 20.0
SCORE = 100.0
PAINT = 40.0
FOULS = 20.0
FAST_BREAK = 12.0
GAMES = MIN_PERIODS + 2
RIVAL = "NYK"
TEAM = "BOS"


def _box_row(game_id: str, team: str, *, made: float = FGM) -> dict[str, object]:
    """Fila de box score de equipo con el detalle de tiro que usa el perfil."""
    return {
        "game_id": game_id,
        "team": team,
        "score": SCORE,
        "field_goals_attempted": FGA,
        "field_goals_made": made,
        "three_point_field_goals_made": THREES_MADE,
        "three_point_field_goals_attempted": THREES_ATTEMPTED,
        "free_throws_attempted": FTA,
        "points_in_paint": PAINT,
        "fouls": FOULS,
        "fast_break_points": FAST_BREAK,
    }


def _long_frame(count: int) -> pd.DataFrame:
    """Frame largo mínimo: un partido por noche, siempre los mismos dos equipos."""
    rows = [
        {
            "game_id": f"g{index}",
            "game_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
            "team": team,
            "opponent": RIVAL if team == TEAM else TEAM,
            ORDER_COLUMN: 2 * index + offset,
        }
        for index in range(count)
        for offset, team in enumerate((TEAM, RIVAL))
    ]
    return pd.DataFrame(rows)


def _team_box(count: int, *, made: float = FGM) -> pd.DataFrame:
    """Box score de los dos equipos; el local puede acertar distinto."""
    rows = [
        _box_row(f"g{index}", team, made=made if team == TEAM else FGM)
        for index in range(count)
        for team in (TEAM, RIVAL)
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_profile_computes_the_documented_ratios() -> None:
    """eFG, 3PA rate, FT rate y pintura salen de sus definiciones."""
    profile = team_shooting_profile(_team_box(1)).iloc[0]
    assert profile["efg"] == pytest.approx(
        (FGM + EFG_THREE_WEIGHT * THREES_MADE) / FGA
    )
    assert profile["three_rate"] == pytest.approx(THREES_ATTEMPTED / FGA)
    assert profile["ft_rate"] == pytest.approx(FTA / FGA)
    assert profile["paint_share"] == pytest.approx(PAINT / SCORE)
    assert profile["fouls"] == pytest.approx(FOULS)
    assert profile["fast_break_points"] == pytest.approx(FAST_BREAK)


@pytest.mark.unit
def test_profile_skips_games_without_attempts() -> None:
    """Sin intentos de campo no hay eFG que calcular: la fila se descarta."""
    box = _team_box(1)
    box.loc[0, "field_goals_attempted"] = 0.0
    profile = team_shooting_profile(box)
    assert len(profile) == 1
    assert profile.iloc[0]["team"] == RIVAL


@pytest.mark.unit
def test_profile_parses_numbers_stored_as_text() -> None:
    """hoopR devuelve las columnas como texto: se convierten, no se rompen."""
    box = _team_box(1).astype({"field_goals_attempted": str, "field_goals_made": str})
    assert team_shooting_profile(box).iloc[0]["efg"] == pytest.approx(
        (FGM + EFG_THREE_WEIGHT * THREES_MADE) / FGA
    )


@pytest.mark.unit
def test_empty_box_leaves_the_columns_at_zero() -> None:
    """La liga sintética no tiene box score: las columnas quedan a 0."""
    enriched = add_shooting_features(_long_frame(3), pd.DataFrame())
    assert list(SHOOTING_COLUMNS) == [
        column for column in SHOOTING_COLUMNS if column in enriched.columns
    ]
    assert (enriched[list(SHOOTING_COLUMNS)] == 0.0).all().all()


@pytest.mark.unit
def test_rolling_profile_excludes_the_current_game() -> None:
    """La media de hoy son los partidos anteriores, nunca el de hoy."""
    enriched = add_shooting_features(_long_frame(GAMES), _team_box(GAMES))
    home = enriched[enriched.team == TEAM].reset_index(drop=True)
    # Sin MIN_PERIODS partidos previos no hay media: queda a 0.
    assert (home.loc[: MIN_PERIODS - 1, "efg_last_10"] == 0.0).all()
    assert home.loc[MIN_PERIODS, "efg_last_10"] == pytest.approx(
        (FGM + EFG_THREE_WEIGHT * THREES_MADE) / FGA
    )


@pytest.mark.unit
def test_allowed_profile_is_the_opponents_shooting() -> None:
    """Lo concedido es el tiro del rival, no el propio."""
    sharp = FGM + 20.0
    enriched = add_shooting_features(_long_frame(GAMES), _team_box(GAMES, made=sharp))
    rival = enriched[enriched.team == RIVAL].reset_index(drop=True)
    assert rival.loc[MIN_PERIODS, "efg_allowed_last_10"] == pytest.approx(
        (sharp + EFG_THREE_WEIGHT * THREES_MADE) / FGA
    )
    assert rival.loc[MIN_PERIODS, "efg_last_10"] == pytest.approx(
        (FGM + EFG_THREE_WEIGHT * THREES_MADE) / FGA
    )


@pytest.mark.unit
def test_games_without_box_score_keep_a_zero_profile() -> None:
    """Un partido sin detalle de tiro no inventa perfil ni propaga NaN."""
    enriched = add_shooting_features(_long_frame(GAMES), _team_box(GAMES - 1))
    assert not enriched[list(SHOOTING_COLUMNS)].isna().to_numpy().any()
    assert np.isfinite(enriched[list(SHOOTING_COLUMNS)].to_numpy()).all()


@pytest.mark.unit
def test_the_original_row_order_is_preserved() -> None:
    """El frame vuelve en su orden cronológico, listo para el pivote."""
    frame = _long_frame(GAMES)
    enriched = add_shooting_features(frame, _team_box(GAMES))
    assert enriched[ORDER_COLUMN].is_monotonic_increasing
    assert len(enriched) == len(frame)
