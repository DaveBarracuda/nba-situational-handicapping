"""Tests de las ventanas móviles con peso por temporada."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.features.windows import AGGREGATE_SUM, season_weighted_rolling

WINDOW = 10
MIN_PERIODS = 4
TEAMS = ("BOS", "NYK")
PRIOR, CURRENT = "2023-24", "2024-25"
GAMES_PER_SEASON = 12
PRIOR_VALUE = 100.0
CURRENT_VALUE = 120.0


def _frame(
    *, prior_value: float = PRIOR_VALUE, current_value: float = CURRENT_VALUE
) -> pd.DataFrame:
    """Dos equipos, dos temporadas, valor constante dentro de cada una."""
    rows = []
    for team in TEAMS:
        for season, value in ((PRIOR, prior_value), (CURRENT, current_value)):
            for _index in range(GAMES_PER_SEASON):
                rows.append(
                    {
                        "team": team,
                        "season": season,
                        "game_date": pd.Timestamp("2023-10-01")
                        + pd.Timedelta(days=len(rows)),
                        "x": value,
                    }
                )
    return pd.DataFrame(rows).sort_values(["team", "game_date"]).reset_index(drop=True)


def _legacy(frame: pd.DataFrame, *, aggregate: str = "mean") -> pd.Series:
    """La media móvil que el proyecto usaba antes: por equipo, sin temporada."""
    grouped = frame.groupby("team", sort=False)["x"]
    return grouped.transform(
        lambda series: getattr(
            series.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS), aggregate
        )()
    )


@pytest.mark.unit
def test_weight_one_reproduces_the_previous_behaviour() -> None:
    """Con peso 1 el resultado es el de siempre, valor a valor."""
    frame = _frame()
    expected = _legacy(frame)
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=1.0
    )
    pd.testing.assert_series_equal(
        actual.rename(None), expected.rename(None), check_names=False
    )


@pytest.mark.unit
def test_weight_one_reproduces_the_previous_sum() -> None:
    """Y la suma móvil también."""
    frame = _frame()
    actual = season_weighted_rolling(
        frame,
        "x",
        window=WINDOW,
        min_periods=MIN_PERIODS,
        carryover_weight=1.0,
        aggregate=AGGREGATE_SUM,
    )
    expected = _legacy(frame, aggregate="sum")
    assert np.allclose(actual.dropna(), expected.dropna())


@pytest.mark.unit
def test_zero_weight_ignores_the_previous_season() -> None:
    """Con peso 0 solo cuenta la temporada en curso."""
    frame = _frame()
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=0.0
    )
    current = frame.season == CURRENT
    # Los partidos 5 a 12 de la temporada nueva ya tienen MIN_PERIODS propios.
    deep = current & (frame.groupby(["team", "season"]).cumcount() >= MIN_PERIODS)
    assert np.allclose(actual[deep], CURRENT_VALUE)


@pytest.mark.unit
def test_the_first_game_of_a_season_has_no_value_without_carryover() -> None:
    """Sin carryover, el estreno de temporada se queda sin media: no hay de qué."""
    frame = _frame()
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=0.0
    )
    openers = frame.groupby(["team", "season"]).head(1).index
    current_openers = [i for i in openers if frame.loc[i, "season"] == CURRENT]
    assert actual.loc[current_openers].isna().all()
    # Con carryover 1 ese mismo partido hereda la forma del año anterior.
    inherited = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=1.0
    )
    assert np.allclose(inherited.loc[current_openers], PRIOR_VALUE)


@pytest.mark.unit
def test_the_blend_weights_the_previous_season_as_documented() -> None:
    """3 partidos nuevos y 7 viejos con peso w dan la media ponderada exacta."""
    frame = _frame()
    weight = 0.5
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=weight
    )
    team = frame[frame.team == TEAMS[0]].reset_index(drop=True)
    position = team.index[team.season == CURRENT][3]
    new_games, old_games = 3, WINDOW - 3
    expected = (new_games * CURRENT_VALUE + weight * old_games * PRIOR_VALUE) / (
        new_games + weight * old_games
    )
    single = season_weighted_rolling(
        team, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=weight
    )
    assert single.iloc[position] == pytest.approx(expected)
    assert actual.notna().sum() > 0


@pytest.mark.unit
@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75, 1.0])
def test_a_partial_weight_still_inherits_at_the_season_opener(weight: float) -> None:
    """Regresión: con 0 < w < 1 el estreno heredaba NaN en vez de la forma.

    La suma móvil dentro de la temporada sale NaN —no 0— cuando no hay
    partidos previos, y eso anulaba toda la mezcla. Con valor constante en la
    temporada anterior, el estreno debe valer exactamente ese valor para
    cualquier peso distinto de 0.
    """
    frame = _frame()
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=weight
    )
    openers = [
        index
        for index in frame.groupby(["team", "season"]).head(1).index
        if frame.loc[index, "season"] == CURRENT
    ]
    assert len(openers) == len(TEAMS)
    assert np.allclose(actual.loc[openers], PRIOR_VALUE)


@pytest.mark.unit
@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_min_periods_is_measured_on_the_unweighted_count(weight: float) -> None:
    """El mínimo de partidos se mide sobre el total, igual que siempre."""
    frame = _frame()
    actual = season_weighted_rolling(
        frame, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=weight
    )
    assert actual.isna().sum() == _legacy(frame).isna().sum()


@pytest.mark.unit
def test_an_empty_frame_returns_an_empty_series() -> None:
    """Sin filas no hay error."""
    empty = pd.DataFrame(columns=["team", "season", "x"])
    assert season_weighted_rolling(
        empty, "x", window=WINDOW, min_periods=MIN_PERIODS, carryover_weight=0.5
    ).empty
