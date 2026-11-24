"""Capa 3: limpieza y validación de data (guía sección 4).

Implementa las reglas mínimas de validación, el dedupe, la normalización de
equipos y el cruce entre fuentes. La validación nunca "arregla" silenciosamente:
devuelve un reporte con el motivo de cada fila rechazada.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.odds_utils import american_to_probability
from nba_sh.schemas import (
    GAME_COLUMNS,
    INJURY_STATUSES,
    ODDS_COLUMNS,
    TIMINGS,
    Dataset,
)
from nba_sh.teams import is_valid_team, normalize_team

LOGGER = logging.getLogger(__name__)

GAME_CHECKS: tuple[str, ...] = (
    "score_consistent",
    "quarters_ok",
    "date_ok",
    "teams_valid",
    "teams_distinct",
    "possessions_sane",
)
ODDS_CHECKS: tuple[str, ...] = (
    "timing_ok",
    "ml_sane",
    "spread_sane",
    "total_sane",
    "prices_sane",
)

# El invariante es posesiones por 48 minutos: una prórroga añade 5 minutos de
# juego y, con ellos, posesiones legítimas (un doble OT real llega a 132).
MIN_POSSESSIONS_PER_48 = 70.0
MAX_POSSESSIONS_PER_48 = 130.0
REGULATION_MINUTES = 48.0
OVERTIME_MINUTES = 5.0
REGULATION_PERIODS = 4
MIN_ABS_AMERICAN = 100.0


def possessions_scale(periods: float) -> float:
    """Proporción de minutos jugados respecto a los 48 de reglamento."""
    overtimes = max(float(periods) - REGULATION_PERIODS, 0.0)
    return (REGULATION_MINUTES + OVERTIME_MINUTES * overtimes) / REGULATION_MINUTES


@dataclass(frozen=True)
class ValidationReport:
    """Resultado de validar una tabla."""

    table: str
    n_rows: int
    n_valid: int
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)
    messages: tuple[str, ...] = ()

    @property
    def n_invalid(self) -> int:
        """Filas rechazadas."""
        return self.n_rows - self.n_valid

    @property
    def is_clean(self) -> bool:
        """True si no hay filas rechazadas."""
        return self.n_invalid == 0

    def summary(self) -> str:
        """Resumen legible de una línea."""
        return (
            f"{self.table}: {self.n_valid}/{self.n_rows} válidas"
            f" ({self.n_invalid} rechazadas)"
        )


def validate_game_row(
    row: Mapping[str, object],
    *,
    today: pd.Timestamp | None = None,
    config: Config = CONFIG,
) -> dict[str, bool]:
    """Checks de una fila de juego (guía 4).

    Devuelve el dict de checks para poder auditar *qué* falló, no solo que
    falló.
    """
    rules = config.validation
    reference_day = (
        pd.Timestamp(today)
        if today is not None
        else pd.Timestamp.now(tz="UTC").tz_localize(None)
    )
    home_score = float(row.get("home_score") or 0.0)
    away_score = float(row.get("away_score") or 0.0)
    total = home_score + away_score
    periods = int(row.get("periods") or 0)
    possessions = float(row.get("possessions") or 0.0)
    scale = possessions_scale(periods)
    game_date = pd.Timestamp(row.get("game_date"))
    return {
        "score_consistent": rules.min_game_total <= total <= rules.max_game_total,
        "quarters_ok": periods in rules.valid_periods,
        "date_ok": bool(game_date <= reference_day.normalize() + pd.Timedelta(days=1)),
        "teams_valid": is_valid_team(str(row.get("home_team")))
        and is_valid_team(str(row.get("away_team"))),
        "teams_distinct": str(row.get("home_team")) != str(row.get("away_team")),
        "possessions_sane": (
            MIN_POSSESSIONS_PER_48 * scale
            <= possessions
            <= MAX_POSSESSIONS_PER_48 * scale
        ),
    }


def validate_odds_row(
    row: Mapping[str, object], *, config: Config = CONFIG
) -> dict[str, bool]:
    """Checks de una fila de cuotas (guía 4: odds_sane, juice coherente)."""
    rules = config.validation
    ml_home = float(row.get("ml_home") or 0.0)
    ml_away = float(row.get("ml_away") or 0.0)
    spread_home = float(row.get("spread_home") or 0.0)
    total_line = float(row.get("total_line") or 0.0)
    prices = [
        float(row.get(name) or 0.0)
        for name in (
            "spread_home_price",
            "spread_away_price",
            "over_price",
            "under_price",
        )
    ]
    ml_in_range = all(
        rules.min_american_odds < value < rules.max_american_odds
        and abs(value) >= MIN_ABS_AMERICAN
        for value in (ml_home, ml_away)
    )
    return {
        "timing_ok": str(row.get("timing")) in TIMINGS,
        "ml_sane": ml_in_range,
        "spread_sane": abs(spread_home) <= rules.max_abs_spread
        and float(spread_home * 2.0).is_integer(),
        "total_sane": rules.min_total_line <= total_line <= rules.max_total_line,
        "prices_sane": all(
            rules.min_american_odds < value < rules.max_american_odds
            and abs(value) >= MIN_ABS_AMERICAN
            for value in prices
        ),
    }


def _apply_row_checks(
    frame: pd.DataFrame,
    checker: Callable[[Mapping[str, object]], dict[str, bool]],
    *,
    table: str,
    check_names: Sequence[str],
) -> tuple[pd.DataFrame, ValidationReport]:
    """Aplica un validador fila a fila y separa válidas de rechazadas."""
    if frame.empty:
        return frame.copy(), ValidationReport(table=table, n_rows=0, n_valid=0)
    results = [checker(row) for row in frame.to_dict(orient="records")]
    checks = pd.DataFrame(results, index=frame.index).reindex(
        columns=list(check_names), fill_value=True
    )
    valid_mask = checks.all(axis=1)
    failures = frame.loc[~valid_mask].copy()
    if not failures.empty:
        failures["failed_checks"] = [
            ",".join(sorted(checks.columns[~checks.loc[index]]))
            for index in failures.index
        ]
    report = ValidationReport(
        table=table,
        n_rows=int(len(frame)),
        n_valid=int(valid_mask.sum()),
        failures=failures,
    )
    return frame.loc[valid_mask].copy(), report


def normalize_games(games: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos, equipos y fechas de la tabla de juegos.

    - códigos de equipo canónicos (franquicias mudadas/renombradas)
    - ``game_date`` como fecha (ET) sin hora, ``tipoff_utc`` en UTC
    - dedupe por ``game_id`` conservando la última versión
    """
    if games.empty:
        return games.copy()
    frame = games.copy()
    missing = [column for column in GAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"games table missing columns: {missing}")
    for column in ("home_team", "away_team"):
        frame[column] = frame[column].map(
            lambda value: normalize_team(value) if is_valid_team(value) else value
        )
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame["tipoff_utc"] = pd.to_datetime(frame["tipoff_utc"], utc=True, errors="coerce")
    for column in ("home_score", "away_score", "periods"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["possessions"] = pd.to_numeric(frame["possessions"], errors="coerce")
    frame["game_id"] = frame["game_id"].astype(str)
    frame = frame.drop_duplicates(subset=["game_id"], keep="last")
    return frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def normalize_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la tabla de cuotas y deduplica por juego/libro/timing."""
    if odds.empty:
        return odds.copy()
    frame = odds.copy()
    missing = [column for column in ODDS_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"odds table missing columns: {missing}")
    frame["game_id"] = frame["game_id"].astype(str)
    numeric = (
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_home_price",
        "spread_away_price",
        "total_line",
        "over_price",
        "under_price",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates(
        subset=["game_id", "book", "timing"], keep="last"
    ).reset_index(drop=True)


def normalize_injuries(injuries: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la tabla de lesiones (equipos, estados, numéricos)."""
    if injuries.empty:
        return injuries.copy()
    frame = injuries.copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["team"] = frame["team"].map(
        lambda value: normalize_team(value) if is_valid_team(value) else value
    )
    frame["status"] = frame["status"].astype(str).str.lower()
    frame = frame[frame["status"].isin(INJURY_STATUSES)]
    for column in ("minutes_pg", "on_off_net", "hours_before_tip"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.reset_index(drop=True)


def validate_dataset(
    dataset: Dataset, *, today: pd.Timestamp | None = None, config: Config = CONFIG
) -> tuple[Dataset, tuple[ValidationReport, ...]]:
    """Normaliza y valida el dataset completo.

    Returns:
        El dataset limpio (solo filas válidas) y los reportes por tabla.
    """
    duplicated = (
        int(dataset.games["game_id"].duplicated().sum()) if not dataset.games.empty else 0
    )
    games = normalize_games(dataset.games)
    odds = normalize_odds(dataset.odds)
    injuries = normalize_injuries(dataset.injuries)

    valid_games, games_report = _apply_row_checks(
        games,
        lambda row: validate_game_row(row, today=today, config=config),
        table="games",
        check_names=GAME_CHECKS,
    )
    valid_odds, odds_report = _apply_row_checks(
        odds,
        lambda row: validate_odds_row(row, config=config),
        table="odds",
        check_names=ODDS_CHECKS,
    )

    known_games = set(valid_games["game_id"]) if not valid_games.empty else set()
    orphan_odds = (
        int((~valid_odds["game_id"].isin(known_games)).sum())
        if not valid_odds.empty
        else 0
    )
    if not valid_odds.empty:
        valid_odds = valid_odds[valid_odds["game_id"].isin(known_games)].copy()
    if not injuries.empty:
        injuries = injuries[injuries["game_id"].isin(known_games)].copy()

    messages: list[str] = []
    if duplicated:
        messages.append(f"{duplicated} game_id duplicados eliminados")
    if orphan_odds:
        messages.append(f"{orphan_odds} filas de odds sin juego válido descartadas")
    for report in (games_report, odds_report):
        if report.n_invalid:
            LOGGER.warning("validación: %s", report.summary())

    # Con ``replace`` las tablas que no valida este módulo (box score,
    # calidad de tiro) pasan intactas. Con el constructor se perdían en
    # silencio en cuanto el Dataset crecía: ``team_box`` llegó vacío al
    # feature store durante varias mediciones.
    clean = replace(
        dataset,
        games=valid_games,
        odds=valid_odds,
        injuries=injuries,
        referees=dataset.referees.copy(),
    )
    reports = (
        games_report,
        odds_report,
        ValidationReport(
            table="dataset",
            n_rows=dataset.n_games,
            n_valid=clean.n_games,
            messages=tuple(messages),
        ),
    )
    return clean, reports


def cross_source_score_check(
    primary: pd.DataFrame, secondary: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Cruce de box scores entre dos fuentes (guía 4, problema 4).

    Devuelve los juegos cuya diferencia de puntos supera la tolerancia.
    """
    merged = primary.merge(
        secondary, on="game_id", suffixes=("_primary", "_secondary"), how="inner"
    )
    if merged.empty:
        return merged
    tolerance = config.validation.cross_source_points_tolerance
    home_delta = (merged["home_score_primary"] - merged["home_score_secondary"]).abs()
    away_delta = (merged["away_score_primary"] - merged["away_score_secondary"]).abs()
    merged = merged.assign(home_delta=home_delta, away_delta=away_delta)
    return merged[(home_delta > tolerance) | (away_delta > tolerance)].reset_index(
        drop=True
    )


def juice_report(odds: pd.DataFrame) -> pd.DataFrame:
    """Vig implícito por juego/timing (detecta juice inconsistente)."""
    columns = ["game_id", "timing", "ml_hold", "total_hold"]
    if odds.empty:
        return pd.DataFrame(columns=columns)
    frame = odds.copy()
    frame["ml_hold"] = [
        american_to_probability(home) + american_to_probability(away) - 1.0
        for home, away in zip(frame["ml_home"], frame["ml_away"], strict=True)
    ]
    frame["total_hold"] = [
        american_to_probability(over) + american_to_probability(under) - 1.0
        for over, under in zip(frame["over_price"], frame["under_price"], strict=True)
    ]
    return frame[columns].reset_index(drop=True)


__all__ = [
    "GAME_CHECKS",
    "ODDS_CHECKS",
    "ValidationReport",
    "cross_source_score_check",
    "juice_report",
    "normalize_games",
    "normalize_injuries",
    "normalize_odds",
    "possessions_scale",
    "validate_dataset",
    "validate_game_row",
    "validate_odds_row",
]
