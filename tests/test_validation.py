"""Tests de la capa de limpieza y validación (guía 4)."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sh.config import CONFIG
from nba_sh.pipeline import PreparedData
from nba_sh.schemas import Dataset
from nba_sh.validation import (
    cross_source_score_check,
    juice_report,
    normalize_games,
    validate_dataset,
    validate_game_row,
    validate_odds_row,
)

VALID_GAME = {
    "game_id": "0022200001",
    "season": "2022-23",
    "season_type": "regular",
    "game_date": pd.Timestamp("2022-10-22"),
    "tipoff_utc": pd.Timestamp("2022-10-23 00:00:00"),
    "home_team": "BOS",
    "away_team": "MIA",
    "home_score": 112,
    "away_score": 104,
    "periods": 4,
    "possessions": 99.0,
    "crew_id": "CREW01",
}
VALID_ODDS = {
    "game_id": "0022200001",
    "book": "test",
    "timing": "close",
    "ml_home": -180.0,
    "ml_away": 155.0,
    "spread_home": -4.5,
    "spread_home_price": -110.0,
    "spread_away_price": -110.0,
    "total_line": 222.5,
    "over_price": -110.0,
    "under_price": -110.0,
}
REFERENCE_DAY = pd.Timestamp("2023-01-01")


@pytest.mark.unit
def test_valid_game_passes_every_check() -> None:
    """Una fila correcta pasa todos los checks de la guía."""
    assert all(validate_game_row(VALID_GAME, today=REFERENCE_DAY).values())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "failing_check"),
    [
        ("home_score", 40, "score_consistent"),
        ("periods", 3, "quarters_ok"),
        ("away_team", "BOS", "teams_distinct"),
        ("home_team", "XXX", "teams_valid"),
        ("possessions", 10.0, "possessions_sane"),
    ],
)
def test_broken_game_rows_fail_the_right_check(
    field: str, value: object, failing_check: str
) -> None:
    """Cada regla detecta exactamente su problema."""
    checks = validate_game_row({**VALID_GAME, field: value}, today=REFERENCE_DAY)
    assert not checks[failing_check]


@pytest.mark.unit
def test_future_game_fails_date_check() -> None:
    """Un juego con fecha futura no puede tener resultado."""
    checks = validate_game_row(VALID_GAME, today=pd.Timestamp("2020-01-01"))
    assert not checks["date_ok"]


@pytest.mark.unit
def test_valid_odds_row_passes() -> None:
    """Una fila de cuotas correcta pasa los checks."""
    assert all(validate_odds_row(VALID_ODDS).values())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "failing_check"),
    [
        ("ml_home", 0.0, "ml_sane"),
        ("ml_home", 50.0, "ml_sane"),
        ("spread_home", -4.25, "spread_sane"),
        ("total_line", 90.0, "total_sane"),
        ("timing", "whenever", "timing_ok"),
        ("over_price", 10.0, "prices_sane"),
    ],
)
def test_broken_odds_rows_fail_the_right_check(
    field: str, value: object, failing_check: str
) -> None:
    """Las cuotas imposibles se detectan una por una."""
    assert not validate_odds_row({**VALID_ODDS, field: value})[failing_check]


@pytest.mark.unit
def test_normalize_games_dedupes_keeping_last() -> None:
    """El dedupe por game_id conserva la última versión."""
    duplicated = pd.DataFrame([VALID_GAME, {**VALID_GAME, "home_score": 120}])
    normalized = normalize_games(duplicated)
    assert len(normalized) == 1
    assert int(normalized.loc[0, "home_score"]) == 120


@pytest.mark.unit
def test_normalize_games_maps_historical_franchises() -> None:
    """Los equipos históricos se normalizan al código actual."""
    frame = pd.DataFrame([{**VALID_GAME, "home_team": "SEA"}])
    assert normalize_games(frame).loc[0, "home_team"] == "OKC"


@pytest.mark.unit
def test_validate_dataset_drops_invalid_rows_and_orphan_odds() -> None:
    """Las filas inválidas y las cuotas huérfanas se descartan con reporte."""
    games = pd.DataFrame([VALID_GAME, {**VALID_GAME, "game_id": "bad", "periods": 2}])
    odds = pd.DataFrame([VALID_ODDS, {**VALID_ODDS, "game_id": "ghost"}])
    dataset = Dataset(
        games=games,
        odds=odds,
        injuries=pd.DataFrame(columns=["game_id", "team"]),
        referees=pd.DataFrame(columns=["crew_id"]),
    )
    clean, reports = validate_dataset(dataset, today=REFERENCE_DAY)
    assert clean.n_games == 1
    assert len(clean.odds) == 1
    assert any(report.n_invalid > 0 for report in reports)


@pytest.mark.unit
def test_validate_dataset_carries_the_tables_it_does_not_validate() -> None:
    """Regresión: ``team_box`` y ``shot_quality`` tienen que sobrevivir.

    Validar reconstruía el ``Dataset`` con el constructor, y las tablas que
    este módulo no valida llegaban vacías al feature store por sus valores por
    defecto: el perfil de tiro se quedó en 0 sin que nada fallara.
    """
    team_box = pd.DataFrame([{"game_id": VALID_GAME["game_id"], "team": "BOS"}])
    shot_quality = pd.DataFrame(
        [{"game_id": VALID_GAME["game_id"], "team": "BOS", "shot_quality": 1.08}]
    )
    dataset = Dataset(
        games=pd.DataFrame([VALID_GAME]),
        odds=pd.DataFrame([VALID_ODDS]),
        injuries=pd.DataFrame(columns=["game_id", "team"]),
        referees=pd.DataFrame(columns=["crew_id"]),
        team_box=team_box,
        shot_quality=shot_quality,
    )
    clean, _ = validate_dataset(dataset, today=REFERENCE_DAY)
    assert len(clean.team_box) == 1
    assert len(clean.shot_quality) == 1
    assert clean.shot_quality.loc[0, "shot_quality"] == pytest.approx(1.08)


@pytest.mark.unit
def test_cross_source_check_flags_divergences() -> None:
    """El cruce entre fuentes marca diferencias por encima de la tolerancia."""
    primary = pd.DataFrame([{"game_id": "1", "home_score": 110, "away_score": 100}])
    secondary = pd.DataFrame([{"game_id": "1", "home_score": 118, "away_score": 100}])
    flagged = cross_source_score_check(primary, secondary, config=CONFIG)
    assert len(flagged) == 1
    assert flagged.loc[0, "home_delta"] == 8


@pytest.mark.unit
def test_juice_report_computes_hold() -> None:
    """El hold de un mercado -110/-110 es ~4.8%."""
    report = juice_report(pd.DataFrame([VALID_ODDS]))
    assert report.loc[0, "total_hold"] == pytest.approx(0.0476, abs=1e-3)


@pytest.mark.integration
def test_sample_dataset_is_fully_valid(prepared: PreparedData) -> None:
    """La data sintética pasa la validación sin filas rechazadas."""
    assert all(report.is_clean for report in prepared.validation)
