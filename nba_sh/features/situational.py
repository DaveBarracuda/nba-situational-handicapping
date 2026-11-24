"""Combinación de los ajustes situacionales en puntos (guía 5.5A.3).

Convierte las features en **puntos** aplicando las cinco reglas de peso:

    (a) spots del mismo signo: max + 0.4 * segundo (no suma lineal)
    (b) momentum solo si |z| > 1.5σ (el rolling ya absorbió el resto)
    (c) spots * 0.5 si el equipo está fatigado (B2B o 1 día de descanso)
    (d) los spots se validan por backtest antes de entrar (ver backtest.report)
    (e) |Σ ajustes situacionales| <= 4.0 pts

Los ajustes de lesiones van aparte (tienen su propio cap) y el de árbitros
afecta **solo** al total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.season_stage import stage_multipliers
from nba_sh.situational_rules import (
    apply_fatigue_multiplier,
    cap_total,
    momentum_points,
)

SITUATIONAL_COLUMNS = (
    "fatigue_points",
    "spot_points",
    "momentum_points",
    "motivation_points",
    "situational_points",
    "injury_points",
    "team_adjustment",
)
FATIGUED_REST_DAYS = 1


def _fatigue_points(frame: pd.DataFrame, *, config: Config) -> np.ndarray:
    """Puntos por fatiga de calendario y viajes (guía 5.1, 5.2)."""
    rest = config.rest
    rest_days = frame["rest_days"].to_numpy(dtype=float)
    is_home = frame["is_home"].to_numpy(dtype=bool)
    is_b2b = rest_days == 0.0

    points = np.zeros(len(frame), dtype=float)
    points -= np.where(
        is_b2b, np.where(is_home, rest.b2b_home_points, rest.b2b_away_points), 0.0
    )
    points += (
        rest.rest_diff_points_per_day
        * (np.minimum(rest_days, rest.rest_diff_cap_days) - 1.0)
        / 2.0
    )
    points -= np.where(
        frame["is_road_trip_4plus"].to_numpy(dtype=bool),
        rest.road_trip_4plus_points,
        0.0,
    )
    crossings = np.minimum(
        frame["timezone_crossings_7d"].to_numpy(dtype=float),
        rest.timezone_crossing_cap,
    )
    points -= rest.timezone_crossing_points * crossings
    miles = np.minimum(
        frame["travel_miles_7d"].to_numpy(dtype=float), rest.travel_miles_cap_7d
    )
    points -= rest.travel_points_per_1000_miles_7d * miles / 1000.0
    points -= np.where(
        frame["altitude_game"].to_numpy(dtype=bool), rest.altitude_home_points, 0.0
    )
    congestion = np.maximum(
        frame["games_last_7"].to_numpy(dtype=float) - rest.congestion_games_threshold,
        0.0,
    )
    points -= rest.congestion_points_per_extra_game_7d * congestion
    return points


def _motivation_points(frame: pd.DataFrame, *, config: Config) -> np.ndarray:
    """Puntos por motivación: must-win, tanking y equipos ya clasificados."""
    motivation = config.motivation
    points = np.where(
        frame["must_win_flag"].to_numpy(dtype=bool), motivation.must_win_points, 0.0
    )
    points += frame["tanking_score"].to_numpy(dtype=float) * motivation.tanking_points
    points += np.where(
        frame["clinched_no_stakes_flag"].to_numpy(dtype=bool),
        motivation.clinched_no_stakes_points,
        0.0,
    )
    return points


def add_situational_points(
    long_frame: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Calcula los puntos situacionales de cada equipo-juego."""
    if long_frame.empty:
        empty = long_frame.copy()
        for column in SITUATIONAL_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    frame = long_frame.copy()
    multipliers = pd.DataFrame(
        [stage_multipliers(stage, config=config) for stage in frame["stage"]],
        index=frame.index,
    )

    frame["fatigue_points"] = (
        _fatigue_points(frame, config=config) * multipliers["rest"].to_numpy()
    )

    fatigued = frame["rest_days"].to_numpy(dtype=float) <= FATIGUED_REST_DAYS
    frame["spot_points"] = [
        apply_fatigue_multiplier(
            value * multiplier,
            is_fatigued=bool(flag),
            multiplier=config.spots.fatigued_spot_multiplier,
        )
        for value, multiplier, flag in zip(
            frame["spot_points_raw"].to_numpy(dtype=float),
            multipliers["spot"].to_numpy(),
            fatigued,
            strict=True,
        )
    ]
    frame["momentum_points"] = [
        momentum_points(
            float(score),
            sigma_gate=config.spots.momentum_sigma_gate,
            points_per_sigma=config.spots.momentum_points_per_sigma,
            cap=config.spots.momentum_points_cap,
        )
        * multiplier
        for score, multiplier in zip(
            frame["momentum_score"].fillna(0.0).to_numpy(dtype=float),
            multipliers["momentum"].to_numpy(),
            strict=True,
        )
    ]
    frame["motivation_points"] = (
        _motivation_points(frame, config=config) * multipliers["motivation"].to_numpy()
    )

    raw_sum = (
        frame["fatigue_points"]
        + frame["spot_points"]
        + frame["momentum_points"]
        + frame["motivation_points"]
    )
    frame["situational_points"] = [
        cap_total(value, cap=config.spots.total_situational_cap) for value in raw_sum
    ]
    frame["injury_points"] = (
        frame["injury_impact_points"].to_numpy(dtype=float)
        * multipliers["injury"].to_numpy()
    )
    frame["team_adjustment"] = frame["situational_points"] - frame["injury_points"]
    return frame


__all__ = ["FATIGUED_REST_DAYS", "SITUATIONAL_COLUMNS", "add_situational_points"]
