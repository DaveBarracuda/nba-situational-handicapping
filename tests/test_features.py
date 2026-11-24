"""Tests del feature store, incluida la prueba de ausencia de look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_sh.config import CONFIG, Config
from nba_sh.features.build import FEATURE_COLUMNS, build_features
from nba_sh.features.injuries import (
    TEAM_MINUTES_PER_GAME,
    aggregate_injuries,
    missing_weight,
)
from nba_sh.features.rolling import WINDOW_LONG
from nba_sh.features.season_stage import (
    STAGE_EARLY,
    STAGE_GOLDEN,
    STAGE_LATE,
    STAGE_MID,
    stage_for,
    stage_multipliers,
)
from nba_sh.ingest.sample_data import SampleDataset
from nba_sh.schemas import STATUS_DOUBTFUL, STATUS_OUT, STATUS_QUESTIONABLE, Dataset

PREFIX_GAMES = 800
COMPARED_GAMES = 400
MIN_HISTORY_FOR_ROLLING = 12
MAX_REST_DAYS = 10


@pytest.mark.integration
def test_feature_matrix_is_complete(games: pd.DataFrame) -> None:
    """La matriz de diseño existe y no tiene huecos."""
    assert set(FEATURE_COLUMNS).issubset(games.columns)
    assert int(games[list(FEATURE_COLUMNS)].isna().sum().sum()) == 0


@pytest.mark.integration
def test_rest_days_match_the_schedule(team_games: pd.DataFrame) -> None:
    """``rest_days`` es la diferencia real con el juego anterior del equipo."""
    team = team_games["team"].iloc[0]
    rows = team_games[team_games["team"] == team].sort_values("game_date")
    gaps = rows["game_date"].diff().dt.days
    for expected, actual, opener in zip(
        gaps.iloc[1:], rows["rest_days"].iloc[1:], rows["is_season_opener"].iloc[1:]
    , strict=True):
        if not bool(opener):
            assert actual == pytest.approx(min(expected - 1, MAX_REST_DAYS))


@pytest.mark.integration
def test_b2b_flags_are_consistent(team_games: pd.DataFrame) -> None:
    """``is_b2b`` equivale a 0 días de descanso y existe en la muestra."""
    assert (team_games["is_b2b"] == (team_games["rest_days"] == 0)).all()
    assert team_games["is_b2b_away"].sum() > 0
    assert not team_games.loc[team_games["is_b2b_away"], "is_home"].any()


@pytest.mark.integration
def test_rolling_features_exclude_the_current_game(team_games: pd.DataFrame) -> None:
    """Regla de oro: el juego de hoy no entra en sus propias medias."""
    team = team_games["team"].iloc[0]
    rows = (
        team_games[team_games["team"] == team]
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    index = MIN_HISTORY_FOR_ROLLING
    expected = rows["off_rtg"].iloc[max(index - WINDOW_LONG, 0) : index].mean()
    assert rows["off_rtg_last_10"].iloc[index] == pytest.approx(expected, abs=1e-6)


@pytest.mark.integration
def test_travel_miles_are_non_negative_and_present_on_the_road(
    team_games: pd.DataFrame,
) -> None:
    """Las millas nunca son negativas y los visitantes acumulan viaje."""
    assert (team_games["travel_miles_7d"] >= 0.0).all()
    away = team_games[~team_games["is_home"].astype(bool)]
    assert away["travel_miles_7d"].max() > 0.0


@pytest.mark.integration
def test_road_trip_counter_increases_on_the_road(team_games: pd.DataFrame) -> None:
    """El contador de gira sube en visitante y se resetea en casa."""
    assert (team_games.loc[team_games["is_home"], "road_trip_game_number"] == 0).all()
    assert team_games["road_trip_game_number"].max() >= 3
    assert (
        team_games.loc[team_games["is_road_trip_4plus"], "road_trip_game_number"] >= 4
    ).all()


@pytest.mark.integration
def test_altitude_flag_only_for_visitors(team_games: pd.DataFrame) -> None:
    """La altitud penaliza al visitante, no al local de Denver o Utah."""
    flagged = team_games[team_games["altitude_game"]]
    assert not flagged.empty
    assert not flagged["is_home"].any()
    assert set(flagged["venue"].unique()).issubset({"DEN", "UTA"})


@pytest.mark.integration
def test_spots_require_a_weak_opponent(team_games: pd.DataFrame) -> None:
    """Lookahead, letdown y sandwich solo aparecen contra rivales bottom-10."""
    for column in ("spot_lookahead", "spot_letdown", "spot_sandwich"):
        flagged = team_games[team_games[column]]
        if flagged.empty:
            continue
        assert flagged["opponent_weak"].all()


@pytest.mark.integration
def test_lookahead_needs_a_marquee_game_within_two_days(
    team_games: pd.DataFrame, config: Config
) -> None:
    """El lookahead exige un juego grande en <= 2 días."""
    flagged = team_games[team_games["spot_lookahead"]]
    assert not flagged.empty
    assert flagged["next_opponent_marquee"].all()
    assert (flagged["days_to_next_game"] <= config.spots.lookahead_max_days_ahead).all()


@pytest.mark.integration
def test_situational_points_respect_the_cap(
    team_games: pd.DataFrame, config: Config
) -> None:
    """El cap total de ±4 puntos se cumple en toda la muestra."""
    assert (
        team_games["situational_points"].abs().max()
        <= config.spots.total_situational_cap + 1e-9
    )


@pytest.mark.integration
def test_momentum_only_applies_when_extreme(
    team_games: pd.DataFrame, config: Config
) -> None:
    """Con momentum dentro de banda, el ajuste por momentum es exactamente 0."""
    inside_band = team_games[
        team_games["momentum_score"].abs() <= config.spots.momentum_sigma_gate
    ]
    assert not inside_band.empty
    assert inside_band["momentum_points"].abs().max() == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("game_number", "stage"),
    [
        (1, STAGE_EARLY),
        (20, STAGE_EARLY),
        (30, STAGE_MID),
        (60, STAGE_GOLDEN),
        (80, STAGE_LATE),
    ],
)
def test_stage_boundaries(game_number: int, stage: str) -> None:
    """Las etapas de temporada siguen los tramos de la guía."""
    assert stage_for(game_number) == stage


@pytest.mark.unit
def test_stage_multipliers_follow_the_guide() -> None:
    """Octubre no da peso al momentum; la ventana dorada sí."""
    assert stage_multipliers(STAGE_EARLY)["momentum"] == 0.0
    assert (
        stage_multipliers(STAGE_EARLY)["spot"] < stage_multipliers(STAGE_GOLDEN)["spot"]
    )
    assert stage_multipliers(STAGE_LATE)["motivation"] > 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "weight"),
    [(STATUS_OUT, 1.0), (STATUS_DOUBTFUL, 0.75), (STATUS_QUESTIONABLE, 0.5)],
)
def test_injury_missing_weight(status: str, weight: float) -> None:
    """Los estados dudosos se ponderan por probabilidad de jugar."""
    assert missing_weight(status) == pytest.approx(weight)


@pytest.mark.unit
def test_injury_impact_uses_minutes_times_on_off() -> None:
    """El impacto es minutos * on/off * coeficiente, no 'estrella = -X'."""
    injuries = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team": "BOS",
                "player_id": "BOS-P01",
                "minutes_pg": 36.0,
                "on_off_net": 6.0,
                "status": STATUS_OUT,
                "hours_before_tip": 20.0,
            }
        ]
    )
    aggregated = aggregate_injuries(injuries, config=CONFIG)
    expected = 36.0 * 6.0 * CONFIG.injury.points_per_minute_net_rating
    assert aggregated.loc[0, "impact_points"] == pytest.approx(expected)
    assert aggregated.loc[0, "late_scratch_flag"] == 0.0
    assert pytest.approx(240.0) == TEAM_MINUTES_PER_GAME


@pytest.mark.unit
def test_late_scratch_is_flagged_and_amplified() -> None:
    """Una baja a menos de 3h del tip pesa más y se marca."""
    base = {
        "game_id": "g1",
        "team": "BOS",
        "player_id": "BOS-P01",
        "minutes_pg": 30.0,
        "on_off_net": 5.0,
        "status": STATUS_OUT,
    }
    normal = aggregate_injuries(pd.DataFrame([{**base, "hours_before_tip": 10.0}]))
    late = aggregate_injuries(pd.DataFrame([{**base, "hours_before_tip": 1.0}]))
    assert late.loc[0, "late_scratch_flag"] == 1.0
    assert late.loc[0, "impact_points"] > normal.loc[0, "impact_points"]


@pytest.mark.integration
def test_past_features_do_not_change_when_future_games_are_removed(
    sample: SampleDataset, config: Config
) -> None:
    """Prueba fuerte de causalidad: truncar el futuro no altera el pasado.

    Se recalculan las features con solo los primeros ``PREFIX_GAMES`` juegos y
    se comparan las de los primeros ``COMPARED_GAMES``: si alguna feature
    mirara al futuro, aquí cambiaría.
    """
    dataset = sample.dataset
    prefix_ids = set(
        dataset.games.sort_values(["game_date", "game_id"]).head(PREFIX_GAMES)["game_id"]
    )
    truncated = Dataset(
        games=dataset.games[dataset.games["game_id"].isin(prefix_ids)].copy(),
        odds=dataset.odds[dataset.odds["game_id"].isin(prefix_ids)].copy(),
        injuries=dataset.injuries[dataset.injuries["game_id"].isin(prefix_ids)].copy(),
        referees=dataset.referees.copy(),
    )
    full = build_features(dataset, config=config).games
    partial = build_features(truncated, config=config).games

    columns = list(FEATURE_COLUMNS)
    full_subset = full.head(COMPARED_GAMES).set_index("game_id")[columns]
    partial_subset = partial.head(COMPARED_GAMES).set_index("game_id")[columns]
    assert list(full_subset.index) == list(partial_subset.index)
    np.testing.assert_allclose(
        full_subset.to_numpy(dtype=float),
        partial_subset.to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )
