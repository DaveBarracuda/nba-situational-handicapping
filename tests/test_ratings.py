"""Tests de los ratings base: Elo, potencia y puntos esperados (guía 6)."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sh.config import CONFIG, Config
from nba_sh.ingest.sample_data import SampleDataset
from nba_sh.ratings.base_points import expected_base_points, home_factor
from nba_sh.ratings.elo import (
    ELO_POINTS_PER_POINT,
    dynamic_hfa,
    expected_home_probability,
    run_elo,
)
from nba_sh.ratings.power import run_power_ratings
from nba_sh.teams import VALID_TEAMS

BASE_RATING = 1500.0
ELO_TOLERANCE = 1e-9


@pytest.mark.unit
def test_equal_ratings_give_home_edge() -> None:
    """Con ratings iguales, el HFA deja al local por encima del 50%."""
    assert expected_home_probability(BASE_RATING, BASE_RATING, hfa=55.0) > 0.5
    assert expected_home_probability(
        BASE_RATING, BASE_RATING, hfa=0.0
    ) == pytest.approx(0.5)


@pytest.mark.unit
def test_elo_probability_is_antisymmetric() -> None:
    """Sin HFA, invertir los equipos invierte la probabilidad."""
    first = expected_home_probability(1600.0, 1450.0, hfa=0.0)
    second = expected_home_probability(1450.0, 1600.0, hfa=0.0)
    assert first + second == pytest.approx(1.0, abs=ELO_TOLERANCE)


@pytest.mark.unit
def test_dynamic_hfa_grows_with_visitor_fatigue() -> None:
    """El HFA sube con B2B visitante, altitud y gira larga."""
    base = dynamic_hfa()
    assert dynamic_hfa(is_b2b_away=True) > base
    assert dynamic_hfa(altitude_game=True) > base
    assert dynamic_hfa(away_road_trip_4plus=True) > base
    assert dynamic_hfa(
        is_b2b_away=True, altitude_game=True, away_road_trip_4plus=True
    ) > dynamic_hfa(is_b2b_away=True)


@pytest.mark.integration
def test_elo_is_zero_sum_and_covers_every_game(sample: SampleDataset) -> None:
    """El Elo redistribuye puntos sin crearlos y cubre todos los juegos."""
    games = sample.dataset.games
    result = run_elo(games)
    assert len(result.game_ratings) == len(games)
    mean_rating = sum(result.final_ratings.values()) / len(result.final_ratings)
    assert mean_rating == pytest.approx(BASE_RATING, abs=1e-6)
    assert set(result.final_ratings) == set(VALID_TEAMS)


@pytest.mark.integration
def test_elo_daily_history_has_every_team(sample: SampleDataset) -> None:
    """El histórico diario permite rankear la liga en cualquier fecha."""
    result = run_elo(sample.dataset.games)
    counts = result.daily_ratings.groupby("date")["team"].nunique()
    assert (counts == len(VALID_TEAMS)).all()
    ranks = result.ranks_by_date()
    assert ranks["league_rank"].min() == 1
    assert ranks["league_rank"].max() == len(VALID_TEAMS)
    assert ranks["conference_rank"].max() == len(VALID_TEAMS) / 2


@pytest.mark.integration
def test_elo_carryover_keeps_season_opener_near_the_mean(
    two_season_sample: SampleDataset, config: Config
) -> None:
    """El carryover acerca los ratings a la media al cambiar de temporada."""
    games = two_season_sample.dataset.games
    seasons = sorted(games["season"].unique())
    first_season = games[games["season"] == seasons[0]]
    final_first = run_elo(first_season, config=config).final_ratings
    spread_before = max(final_first.values()) - min(final_first.values())

    full = run_elo(games, config=config).game_ratings.merge(
        games[["game_id", "season"]], on="game_id"
    )
    opener = full[full["season"] == seasons[1]].iloc[0]
    assert abs(opener["elo_home_pre"] - config.elo.carryover_target) <= spread_before


@pytest.mark.integration
def test_elo_margin_is_in_points(sample: SampleDataset) -> None:
    """``elo_margin`` convierte puntos Elo a puntos de margen."""
    row = run_elo(sample.dataset.games).game_ratings.iloc[0]
    assert row["elo_margin"] == pytest.approx(row["elo_diff"] / ELO_POINTS_PER_POINT)


@pytest.mark.integration
def test_power_ratings_start_at_the_prior_and_stay_sane(
    sample: SampleDataset, config: Config
) -> None:
    """El primer juego usa el prior y los ratings se mantienen en rango NBA."""
    power = run_power_ratings(sample.dataset.games, config=config)
    assert len(power) == len(sample.dataset.games)
    first = power.iloc[0]
    assert first["off_home_pre"] == pytest.approx(config.power.prior_off_rating)
    assert first["games_played_home"] == 0.0
    assert power["off_home_pre"].between(95.0, 135.0).all()
    assert power["pace_home_pre"].between(88.0, 112.0).all()


@pytest.mark.integration
def test_power_ratings_separate_good_and_bad_teams(
    sample: SampleDataset, config: Config
) -> None:
    """Al final de la temporada los ratings distinguen equipos."""
    tail = run_power_ratings(sample.dataset.games, config=config).tail(200)
    assert (tail["off_home_pre"] - tail["def_home_pre"]).std() > 1.0


@pytest.mark.unit
def test_home_factor_is_slightly_above_one() -> None:
    """El factor de casa equivale a poco más de un punto por equipo."""
    assert 1.0 < home_factor() < 1.02


@pytest.mark.unit
def test_expected_base_points_reacts_to_ratings_and_adjustments() -> None:
    """Mejor ataque local sube su marcador; el ajuste situacional lo mueve."""
    frame = pd.DataFrame(
        {
            "off_home_pre": [120.0],
            "def_home_pre": [110.0],
            "pace_home_pre": [100.0],
            "off_away_pre": [108.0],
            "def_away_pre": [118.0],
            "pace_away_pre": [100.0],
            "league_off_mean": [114.0],
            "league_pace_mean": [100.0],
        }
    )
    base = expected_base_points(frame, config=CONFIG)
    assert base.loc[0, "margin_base"] > 0.0
    assert base.loc[0, "total_base"] == pytest.approx(
        base.loc[0, "pts_base_home"] + base.loc[0, "pts_base_away"]
    )

    adjusted = expected_base_points(
        frame,
        margin_adjustment=pd.Series([-3.0]),
        total_adjustment=pd.Series([4.0]),
        config=CONFIG,
    )
    assert adjusted.loc[0, "margin_expected"] == pytest.approx(
        base.loc[0, "margin_base"] - 3.0
    )
    assert adjusted.loc[0, "total_expected"] == pytest.approx(
        base.loc[0, "total_base"] + 4.0
    )


@pytest.mark.unit
def test_expected_base_points_requires_rating_columns() -> None:
    """Faltar columnas de ratings es un error explícito."""
    with pytest.raises(ValueError):
        expected_base_points(pd.DataFrame({"off_home_pre": [110.0]}))
