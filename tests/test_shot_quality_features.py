"""Tests de las features de calidad de tiro: causalidad y ausencia de pbp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.features.long_frame import ORDER_COLUMN
from nba_sh.features.shot_quality import (
    MIN_PERIODS,
    SHOT_QUALITY_COLUMNS,
    add_shot_quality_features,
)

TEAM, RIVAL = "BOS", "NYK"
GAMES = MIN_PERIODS + 2
HOME_QUALITY = 1.10
AWAY_QUALITY = 1.02
HOME_MAKING = 0.05
AWAY_MAKING = -0.03
RIM_RATE = 0.30
MID_RATE = 0.25


def _long_frame(count: int) -> pd.DataFrame:
    """Frame largo mínimo: un partido por noche entre los dos mismos equipos."""
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


def _shot_quality(count: int) -> pd.DataFrame:
    """Calidad constante por equipo, para poder comprobar las medias a mano."""
    rows = [
        {
            "game_id": f"g{index}",
            "team": team,
            "attempts": 88,
            "shot_quality": HOME_QUALITY if team == TEAM else AWAY_QUALITY,
            "shot_making": HOME_MAKING if team == TEAM else AWAY_MAKING,
            "rim_rate": RIM_RATE,
            "mid_rate": MID_RATE,
        }
        for index in range(count)
        for team in (TEAM, RIVAL)
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_without_play_by_play_the_columns_stay_at_zero() -> None:
    """La liga sintética no tiene pbp: las columnas quedan a 0, no a NaN."""
    enriched = add_shot_quality_features(_long_frame(3), pd.DataFrame())
    assert all(column in enriched.columns for column in SHOT_QUALITY_COLUMNS)
    assert (enriched[list(SHOT_QUALITY_COLUMNS)] == 0.0).all().all()


@pytest.mark.unit
def test_the_rolling_mean_excludes_the_current_game() -> None:
    """La media de hoy son los partidos anteriores, nunca el de hoy."""
    enriched = add_shot_quality_features(_long_frame(GAMES), _shot_quality(GAMES))
    home = enriched[enriched.team == TEAM].reset_index(drop=True)
    assert (home.loc[: MIN_PERIODS - 1, "shot_quality_last_10"] == 0.0).all()
    assert home.loc[MIN_PERIODS, "shot_quality_last_10"] == pytest.approx(HOME_QUALITY)
    assert home.loc[MIN_PERIODS, "shot_making_last_10"] == pytest.approx(HOME_MAKING)


@pytest.mark.unit
def test_the_allowed_profile_is_the_opponents() -> None:
    """Lo concedido es la calidad del rival, no la propia."""
    enriched = add_shot_quality_features(_long_frame(GAMES), _shot_quality(GAMES))
    home = enriched[enriched.team == TEAM].reset_index(drop=True)
    assert home.loc[MIN_PERIODS, "shot_quality_allowed_last_10"] == pytest.approx(
        AWAY_QUALITY
    )
    assert home.loc[MIN_PERIODS, "shot_making_allowed_last_10"] == pytest.approx(
        AWAY_MAKING
    )


@pytest.mark.unit
def test_games_without_shot_data_do_not_propagate_nan() -> None:
    """Un partido sin pbp no inventa calidad ni contamina la media."""
    enriched = add_shot_quality_features(_long_frame(GAMES), _shot_quality(GAMES - 1))
    values = enriched[list(SHOT_QUALITY_COLUMNS)].to_numpy()
    assert not np.isnan(values).any()
    assert np.isfinite(values).all()


@pytest.mark.unit
def test_the_chronological_order_survives() -> None:
    """El frame vuelve en su orden original, listo para el pivote."""
    frame = _long_frame(GAMES)
    enriched = add_shot_quality_features(frame, _shot_quality(GAMES))
    assert enriched[ORDER_COLUMN].is_monotonic_increasing
    assert len(enriched) == len(frame)
