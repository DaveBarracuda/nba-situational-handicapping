"""Tests de la calidad de tiro desde play-by-play (sin red: parquet sintético)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_sh.ingest.pbp_source import (
    CORNER_THREE_DISTANCE,
    HOOP_X,
    MIN_CELL_ATTEMPTS,
    QUALITY_COLUMNS,
    THREE_POINT_ARC,
    attach_team_labels,
    classify_family,
    distance_band,
    is_three_point_shot,
    load_shots,
    shot_distance,
    shot_value_table,
    team_shot_quality,
)

HOME_ID, AWAY_ID = 2, 17
HOME, AWAY = "BOS", "NYK"
ALL_STAR_ID, ALL_STAR = 99, "LEB"
REGULAR, PLAYOFF = 2, 3
PRIOR_SEASON, SEASON = 2024, 2025
RIM_X = HOOP_X - 1.0
CORNER_Y = 23.0
MID_X = HOOP_X - 15.0
BROKEN_COORDINATE = -2147483647.0


def _play(
    *,
    game_id: int = 401,
    season: int = SEASON,
    season_type: int = REGULAR,
    team_id: int = HOME_ID,
    shooting_play: bool = True,
    scoring_play: bool = False,
    score_value: int = 0,
    type_text: str = "Jump Shot",
    coordinate_x: float = MID_X,
    coordinate_y: float = 0.0,
) -> dict[str, object]:
    """Una fila de play-by-play con el formato que publica hoopR."""
    return {
        "game_id": game_id,
        "season": season,
        "season_type": season_type,
        "team_id": team_id,
        "shooting_play": shooting_play,
        "scoring_play": scoring_play,
        "score_value": score_value,
        "type_text": type_text,
        "coordinate_x": coordinate_x,
        "coordinate_y": coordinate_y,
    }


def _write_pbp(path: Path, plays: list[dict[str, object]]) -> Path:
    """Escribe un parquet de play-by-play con las columnas que se leen."""
    frame = pd.DataFrame(plays)
    frame["home_team_id"] = HOME_ID
    frame["away_team_id"] = AWAY_ID
    frame["home_team_abbrev"] = HOME
    frame["away_team_abbrev"] = AWAY
    frame.to_parquet(path)
    return path


@pytest.mark.unit
def test_distance_is_measured_from_the_hoop() -> None:
    """Un tiro bajo el aro está a 0 pies, en cualquiera de los dos aros."""
    distances = shot_distance(
        pd.Series([HOOP_X, -HOOP_X, HOOP_X]), pd.Series([0.0, 0.0, 10.0])
    )
    assert distances.iloc[0] == pytest.approx(0.0)
    assert distances.iloc[1] == pytest.approx(0.0)
    assert distances.iloc[2] == pytest.approx(10.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Driving Floating Bank Jump Shot", "floater"),
        ("Turnaround Fade Away Jump Shot", "fadeaway"),
        ("Running Pullup Jump Shot", "pullup"),
        ("Alley Oop Dunk Shot", "dunk"),
        ("Alley Oop Layup Shot", "layup"),
        ("Driving Finger Roll Layup", "layup"),
        ("Tip Dunk Shot", "dunk"),
        ("Turnaround Hook Shot", "hook"),
        ("Step Back Jump Shot", "stepback"),
        ("Jump Shot", "jumper"),
    ],
)
def test_family_follows_the_documented_priority(text: str, expected: str) -> None:
    """El primer patrón de la lista manda, no el último del texto."""
    assert classify_family(pd.Series([text])).iloc[0] == expected


@pytest.mark.unit
def test_three_pointers_use_the_score_when_the_shot_went_in() -> None:
    """Si entró, ``score_value`` lo dice exactamente, sin geometría."""
    flags = is_three_point_shot(
        pd.Series([25.0, 25.0]),
        pd.Series([0.0, 0.0]),
        pd.Series([2, 3]),
        pd.Series([True, True]),
    )
    assert not bool(flags.iloc[0])
    assert bool(flags.iloc[1])


@pytest.mark.unit
def test_missed_three_pointers_come_from_the_line_geometry() -> None:
    """En los errados manda la línea: 23,75 de arco y 22 en la esquina."""
    flags = is_three_point_shot(
        pd.Series([THREE_POINT_ARC - 0.5, THREE_POINT_ARC, CORNER_THREE_DISTANCE]),
        pd.Series([0.0, 0.0, CORNER_Y]),
        pd.Series([0, 0, 0]),
        pd.Series([False, False, False]),
    )
    assert not bool(flags.iloc[0])
    assert bool(flags.iloc[1])
    assert bool(flags.iloc[2])


@pytest.mark.unit
def test_load_shots_keeps_only_regular_season_field_goals(tmp_path: Path) -> None:
    """Fuera tiros libres, playoffs, jugadas que no son tiro y coordenadas rotas."""
    path = _write_pbp(
        tmp_path / "pbp.parquet",
        [
            _play(type_text="Jump Shot"),
            _play(type_text="Free Throw - 1 of 2"),
            _play(type_text="Defensive Rebound", shooting_play=False),
            _play(season_type=PLAYOFF),
            _play(coordinate_x=BROKEN_COORDINATE),
        ],
    )
    shots = load_shots([path])
    assert len(shots) == 1
    assert shots.loc[0, "family"] == "jumper"
    assert shots.loc[0, "game_id"] == "401"


@pytest.mark.unit
def test_points_are_zero_on_a_miss_and_the_shot_value_on_a_make(
    tmp_path: Path,
) -> None:
    """Los puntos de la jugada salen del valor del tiro, no de ``score_value``."""
    path = _write_pbp(
        tmp_path / "pbp.parquet",
        [
            _play(coordinate_x=RIM_X, scoring_play=True, score_value=2),
            _play(coordinate_x=RIM_X, scoring_play=False),
            _play(coordinate_x=0.0, scoring_play=True, score_value=3),
            _play(coordinate_x=0.0, scoring_play=False),
        ],
    )
    shots = load_shots([path])
    assert shots["points"].tolist() == [2, 0, 3, 0]
    assert shots["is_three"].tolist() == [False, False, True, True]


@pytest.mark.unit
def test_scarce_cells_inherit_their_distance_band() -> None:
    """Una familia sin muestra propia usa el valor de su banda."""
    rim_distance = 2.0
    band = distance_band(pd.Series([rim_distance])).iloc[0]
    common = pd.DataFrame(
        {
            "band": [band] * MIN_CELL_ATTEMPTS,
            "family": ["layup"] * MIN_CELL_ATTEMPTS,
            "points": [2] * MIN_CELL_ATTEMPTS,
            "distance": [rim_distance] * MIN_CELL_ATTEMPTS,
        }
    )
    rare = common.head(1).assign(family="tip", points=0)
    table = shot_value_table(pd.concat([common, rare], ignore_index=True))
    by_family = table.set_index("family")["xpps"]
    # La banda vale (200*2 + 0)/201; la celda rara hereda eso, no su propio 0.
    band_value = (MIN_CELL_ATTEMPTS * 2) / (MIN_CELL_ATTEMPTS + 1)
    assert by_family["tip"] == pytest.approx(band_value)
    assert by_family["layup"] == pytest.approx(2.0)


def _two_season_shots(*, made_in_target: int) -> pd.DataFrame:
    """Dos temporadas idénticas salvo el acierto de la segunda."""
    size = MIN_CELL_ATTEMPTS
    base = {
        "family": ["layup"] * size,
        "distance": [2.0] * size,
        "band": [0] * size,
        "is_three": [False] * size,
        "team_id": [HOME_ID] * size,
        "game_id": ["401"] * size,
    }
    prior = pd.DataFrame({**base, "season": [PRIOR_SEASON] * size})
    prior["points"] = [2] * (size // 2) + [0] * (size - size // 2)
    target = pd.DataFrame({**base, "season": [SEASON] * size})
    target["points"] = [2] * made_in_target + [0] * (size - made_in_target)
    return pd.concat([prior, target], ignore_index=True)


@pytest.mark.unit
def test_the_first_season_has_no_reference_and_is_omitted() -> None:
    """Sin temporada anterior no hay referencia: esa temporada no se usa."""
    quality = team_shot_quality(
        _two_season_shots(made_in_target=MIN_CELL_ATTEMPTS // 2)
    )
    assert len(quality) == 1
    assert list(quality.columns) == list(QUALITY_COLUMNS)


@pytest.mark.unit
def test_quality_plus_making_rebuilds_the_actual_points_per_shot() -> None:
    """La descomposición es exacta: calidad + acierto = puntos por intento."""
    made = int(MIN_CELL_ATTEMPTS * 0.75)
    row = team_shot_quality(_two_season_shots(made_in_target=made)).iloc[0]
    actual = 2.0 * made / MIN_CELL_ATTEMPTS
    assert row["shot_quality"] + row["shot_making"] == pytest.approx(actual)
    assert row["shot_quality"] == pytest.approx(1.0)  # la referencia: 50% de 2
    assert row["rim_rate"] == pytest.approx(1.0)
    assert row["attempts"] == MIN_CELL_ATTEMPTS


@pytest.mark.unit
def test_all_star_teams_are_dropped(tmp_path: Path) -> None:
    """Los equipos del All-Star no son de la liga: fuera, sin reventar."""
    path = tmp_path / "pbp.parquet"
    frame = pd.DataFrame([_play()])
    frame["home_team_id"] = HOME_ID
    frame["away_team_id"] = ALL_STAR_ID
    frame["home_team_abbrev"] = HOME
    frame["away_team_abbrev"] = ALL_STAR
    frame.to_parquet(path)
    quality = pd.DataFrame(
        {
            "game_id": ["401", "401"],
            "team_id": [HOME_ID, ALL_STAR_ID],
            "attempts": [80, 80],
            "shot_quality": [1.08, 1.08],
            "shot_making": [0.01, 0.01],
            "rim_rate": [0.3, 0.3],
            "mid_rate": [0.3, 0.3],
        }
    )
    labelled = attach_team_labels(quality, [path])
    assert labelled["team"].tolist() == [HOME]
    assert "team_id" not in labelled.columns


@pytest.mark.unit
def test_empty_input_returns_the_declared_schema() -> None:
    """Sin tiros no hay error: esquemas vacíos y NaN que no contaminan."""
    empty = pd.DataFrame(columns=["season", "band", "family", "points", "distance"])
    assert team_shot_quality(empty).empty
    assert shot_value_table(empty).empty
    assert np.isnan(shot_distance(pd.Series([np.nan]), pd.Series([np.nan])).iloc[0])
