"""Tests del orquestador, la CLI, el almacén y las fuentes de datos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_sh.cli import build_parser, main
from nba_sh.config import Config
from nba_sh.ingest.injury_source import empty_injuries, load_injuries
from nba_sh.ingest.nba_api_source import (
    box_scores_to_games,
    estimate_possessions,
    infer_periods,
)
from nba_sh.ingest.odds_source import ODDS_API_KEY_ENV, api_key, payload_to_odds
from nba_sh.ingest.sample_data import SampleDataset, generate_sample_dataset
from nba_sh.pipeline import PreparedData, model_quality, run_daily
from nba_sh.scheduler import DAILY_PLAN, build_scheduler, describe_plan
from nba_sh.schemas import (
    INJURY_COLUMNS,
    ODDS_COLUMNS,
    TABLE_NAMES,
    TIMING_PREGAME,
    Dataset,
    odds_at,
)
from nba_sh.storage import Storage

DEMO_SIMULATIONS = "150"
EXPECTED_GAMES_PER_SEASON = 1_230
MIN_GAMES_PER_TEAM = 80
TIMINGS_PER_GAME = 3
OUTCOMES_PER_GAME = 6
FIXED_TIME_JOBS = 5
TOTAL_JOBS = 8


@pytest.mark.integration
def test_sample_dataset_shape_is_a_real_season(sample: SampleDataset) -> None:
    """La temporada sintética tiene el tamaño de una temporada NBA."""
    games = sample.dataset.games
    assert len(games) == pytest.approx(EXPECTED_GAMES_PER_SEASON, rel=0.01)
    appearances = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    assert appearances.min() >= MIN_GAMES_PER_TEAM
    assert len(sample.dataset.odds) == TIMINGS_PER_GAME * len(games)
    assert not sample.dataset.injuries.empty
    assert not sample.dataset.referees.empty


@pytest.mark.integration
def test_sample_dataset_is_deterministic(config: Config) -> None:
    """La misma semilla produce exactamente el mismo dataset."""
    first = generate_sample_dataset(1, seed=99, config=config)
    second = generate_sample_dataset(1, seed=99, config=config)
    pd.testing.assert_frame_equal(first.dataset.games, second.dataset.games)


@pytest.mark.unit
def test_sample_dataset_rejects_zero_seasons(config: Config) -> None:
    """Pedir cero temporadas es un error explícito."""
    with pytest.raises(ValueError):
        generate_sample_dataset(0, config=config)


@pytest.mark.integration
def test_odds_cover_opening_and_closing(sample: SampleDataset) -> None:
    """Se guarda apertura y cierre de cada línea (requisito de la CLV)."""
    odds = sample.dataset.odds
    assert set(odds["timing"]) == {"open", "t_minus_60", "close"}
    assert len(odds_at(odds, TIMING_PREGAME)) == len(sample.dataset.games)


@pytest.mark.integration
def test_run_daily_produces_a_playable_slate(
    prepared: PreparedData, config: Config
) -> None:
    """El pipeline diario entrena con el pasado y evalúa la jornada."""
    slate = run_daily(prepared, config=config)
    assert not slate.predictions.empty
    assert slate.bundle.is_fitted
    assert len(slate.evaluated) == OUTCOMES_PER_GAME * len(slate.predictions)
    if slate.n_bets:
        assert (slate.bets["stake"] > 0.0).all()


@pytest.mark.integration
def test_run_daily_rejects_dates_without_history(
    prepared: PreparedData, config: Config
) -> None:
    """Sin histórico previo no se puede modelar una fecha."""
    first_day = prepared.games["game_date"].min()
    with pytest.raises(ValueError):
        run_daily(prepared, game_date=first_day, config=config)


@pytest.mark.integration
def test_model_quality_needs_two_seasons(prepared: PreparedData, config: Config) -> None:
    """Las métricas fuera de muestra exigen más de una temporada."""
    with pytest.raises(ValueError):
        model_quality(prepared, config=config)


@pytest.mark.integration
def test_storage_snapshot_roundtrip(tmp_path: Path, sample: SampleDataset) -> None:
    """Los snapshots son inmutables y se releen idénticos."""
    storage = Storage(root=tmp_path)
    snapshot = storage.write_snapshot(sample.dataset, label="test")
    assert snapshot.exists()
    assert storage.read_snapshot(snapshot).n_games == sample.dataset.n_games
    assert storage.latest_snapshot(label="test") == snapshot


@pytest.mark.integration
def test_storage_writes_tables_and_artifacts(
    tmp_path: Path, sample: SampleDataset
) -> None:
    """SQLite y los artefactos versionados se escriben y se consultan."""
    storage = Storage(root=tmp_path)
    storage.write_tables(sample.dataset)
    counted = storage.query("SELECT COUNT(*) AS n FROM games")
    assert int(counted.loc[0, "n"]) == sample.dataset.n_games
    assert storage.write_artifact({"roi": 0.03}, name="report", version="v1").exists()
    assert storage.write_table_artifact(
        sample.dataset.games.head(5), name="games", version="v1"
    ).exists()


@pytest.mark.unit
def test_possessions_and_periods_from_box_scores() -> None:
    """Las posesiones y los periodos se derivan del box score."""
    assert estimate_possessions(90.0, 20.0, 10.0, 14.0) == pytest.approx(102.8)
    assert infer_periods(240.0) == 4
    assert infer_periods(265.0) == 5
    assert infer_periods(290.0) == 6


@pytest.mark.unit
def test_box_scores_to_games_builds_the_canonical_schema() -> None:
    """Dos filas de box score se convierten en un juego canónico."""
    box_scores = pd.DataFrame(
        [
            {
                "GAME_ID": "0022300001",
                "SEASON_ID": "22023",
                "GAME_DATE": "2023-10-24",
                "TEAM_ABBREVIATION": "BOS",
                "MATCHUP": "BOS vs. NYK",
                "PTS": 108,
                "FGA": 88.0,
                "FTA": 18.0,
                "OREB": 9.0,
                "TOV": 13.0,
                "MIN": 240.0,
            },
            {
                "GAME_ID": "0022300001",
                "SEASON_ID": "22023",
                "GAME_DATE": "2023-10-24",
                "TEAM_ABBREVIATION": "NYK",
                "MATCHUP": "NYK @ BOS",
                "PTS": 104,
                "FGA": 86.0,
                "FTA": 20.0,
                "OREB": 11.0,
                "TOV": 12.0,
                "MIN": 240.0,
            },
        ]
    )
    games = box_scores_to_games(box_scores)
    assert len(games) == 1
    row = games.iloc[0]
    assert row["home_team"] == "BOS"
    assert row["away_team"] == "NYK"
    assert row["home_score"] == 108
    assert row["periods"] == 4
    assert 90.0 < row["possessions"] < 110.0


@pytest.mark.unit
def test_box_scores_to_games_handles_empty_input() -> None:
    """Sin datos devuelve el esquema vacío, no un error."""
    assert box_scores_to_games(pd.DataFrame()).empty


@pytest.mark.unit
def test_odds_api_key_is_read_from_the_environment(monkeypatch) -> None:
    """La clave viene del entorno y su ausencia es un error claro."""
    monkeypatch.delenv(ODDS_API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError):
        api_key()
    monkeypatch.setenv(ODDS_API_KEY_ENV, "test-key")
    assert api_key() == "test-key"
    assert api_key("explicit") == "explicit"


@pytest.mark.unit
def test_payload_to_odds_maps_the_three_markets() -> None:
    """La respuesta de The Odds API se traduce al esquema canónico."""
    payload = [
        {
            "home_team": "Boston Celtics",
            "away_team": "New York Knicks",
            "commence_time": "2023-10-24T23:30:00Z",
            "bookmakers": [
                {
                    "key": "testbook",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -180},
                                {"name": "New York Knicks", "price": 155},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {
                                    "name": "Boston Celtics",
                                    "price": -110,
                                    "point": -4.5,
                                },
                                {
                                    "name": "New York Knicks",
                                    "price": -110,
                                    "point": 4.5,
                                },
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 222.5},
                                {"name": "Under", "price": -110, "point": 222.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    odds = payload_to_odds(payload, timing=TIMING_PREGAME)
    assert list(odds.columns) == list(ODDS_COLUMNS)
    row = odds.iloc[0]
    assert row["ml_home"] == -180.0
    assert row["spread_home"] == -4.5
    assert row["total_line"] == 222.5
    assert row["timing"] == TIMING_PREGAME


@pytest.mark.unit
def test_payload_to_odds_skips_unknown_teams() -> None:
    """Un evento con equipos irreconocibles se descarta con aviso."""
    assert payload_to_odds([{"home_team": "Madrid", "away_team": "Barcelona"}]).empty


@pytest.mark.unit
def test_injury_loader_validates_the_schema(tmp_path: Path) -> None:
    """El fichero de lesiones se valida antes de entrar al pipeline."""
    assert list(empty_injuries().columns) == list(INJURY_COLUMNS)
    path = tmp_path / "injuries.csv"
    pd.DataFrame(
        [
            {
                "game_id": "g1",
                "team": "BOS",
                "player_id": "BOS-P01",
                "minutes_pg": 30.0,
                "on_off_net": 4.0,
                "status": "out",
                "hours_before_tip": 6.0,
            }
        ]
    ).to_csv(path, index=False)
    assert len(load_injuries(path)) == 1

    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"game_id": "g1"}]).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_injuries(bad)
    with pytest.raises(FileNotFoundError):
        load_injuries(tmp_path / "missing.csv")


@pytest.mark.unit
def test_daily_plan_covers_the_production_schedule() -> None:
    """El plan diario tiene las ocho tareas de la guía."""
    assert len(DAILY_PLAN) == TOTAL_JOBS
    assert "06:00" in describe_plan()
    fixed = [job for job in DAILY_PLAN if job.cron_fields is not None]
    assert len(fixed) == FIXED_TIME_JOBS
    assert any(job.is_relative_to_tipoff for job in DAILY_PLAN)


@pytest.mark.unit
def test_scheduler_requires_a_handler_per_fixed_job() -> None:
    """Montar el scheduler sin handlers es un error explícito."""
    with pytest.raises(KeyError):
        build_scheduler({})


@pytest.mark.unit
def test_cli_parser_exposes_every_command() -> None:
    """La CLI declara todos los comandos documentados."""
    parser = build_parser()
    for command in (
        "demo",
        "backtest",
        "validate",
        "metrics",
        "monitor",
        "spots",
        "plan",
    ):
        assert parser.parse_args([command]).command == command


@pytest.mark.unit
def test_cli_plan_command_runs() -> None:
    """``python -m nba_sh plan`` termina correctamente."""
    assert main(["plan"]) == 0


@pytest.mark.integration
def test_cli_demo_command_runs_end_to_end() -> None:
    """``demo`` ejecuta el pipeline completo sin errores."""
    exit_code = main(
        ["demo", "--seasons", "1", "--sims", DEMO_SIMULATIONS, "--seed", "7"]
    )
    assert exit_code == 0


@pytest.mark.integration
def test_cli_validate_and_spots_commands_run() -> None:
    """``validate`` y ``spots`` funcionan sobre la data sintética."""
    assert main(["validate", "--seasons", "1", "--sims", DEMO_SIMULATIONS]) == 0
    assert main(["spots", "--seasons", "1", "--sims", DEMO_SIMULATIONS]) == 0


@pytest.mark.unit
def test_dataset_helpers_filter_and_slice() -> None:
    """El contenedor de datos filtra por juego y por fecha sin mutar."""
    dataset = Dataset.empty()
    assert dataset.n_games == 0
    assert dataset.seasons == ()
    assert dataset.filtered([]).n_games == 0
    assert dataset.before(pd.Timestamp("2023-01-01")).n_games == 0
    assert set(dataset.tables()) == set(TABLE_NAMES)
