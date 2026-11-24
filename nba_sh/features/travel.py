"""Viajes y fatiga (guía 5.2).

    travel_miles_last_7d / _14d  = millas voladas en la ventana
    timezone_crossings_7d        = zonas horarias cruzadas
    road_trip_game_number        = juego consecutivo fuera de casa
    is_road_trip_4plus           = 4to+ juego de gira
    altitude_game                = visitar Denver / Utah
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from nba_sh.features.long_frame import ORDER_COLUMN, restore_order, sorted_by_team
from nba_sh.features.rest import days_since_epoch
from nba_sh.geo import venue_distance_miles
from nba_sh.teams import arena, timezone_crossings

TRAVEL_COLUMNS = (
    "travel_miles_7d",
    "travel_miles_14d",
    "timezone_crossings_7d",
    "road_trip_game_number",
    "is_road_trip_4plus",
    "altitude_game",
)
ROAD_TRIP_THRESHOLD = 4


@cache
def _distance(from_team: str, to_team: str) -> float:
    """Distancia memoizada entre sedes."""
    return venue_distance_miles(from_team, to_team)


@cache
def _crossings(from_team: str, to_team: str) -> int:
    """Cruces de zona horaria memoizados."""
    return timezone_crossings(from_team, to_team)


@cache
def _is_altitude(team: str) -> bool:
    """True si la sede está en altitud."""
    return arena(team).is_altitude_venue


def _window_sums(days: np.ndarray, legs: np.ndarray, window: int) -> np.ndarray:
    """Suma de tramos cuya llegada cae dentro de la ventana (incluye hoy)."""
    totals = np.zeros(days.size, dtype=float)
    for index in range(days.size):
        start = days[index] - window
        left = int(np.searchsorted(days[: index + 1], start, side="right"))
        totals[index] = float(legs[left : index + 1].sum())
    return totals


def _road_trip_numbers(is_home: np.ndarray) -> np.ndarray:
    """Número de juego consecutivo fuera de casa (0 si es local)."""
    numbers = np.zeros(is_home.size, dtype=float)
    streak = 0
    for index, home_game in enumerate(is_home):
        streak = 0 if home_game else streak + 1
        numbers[index] = streak
    return numbers


def add_travel_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Añade millas, husos horarios, giras y altitud al frame largo."""
    if long_frame.empty:
        empty = long_frame.copy()
        for column in TRAVEL_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    frame = sorted_by_team(long_frame)
    collected: dict[str, list[np.ndarray]] = {column: [] for column in TRAVEL_COLUMNS}
    order: list[np.ndarray] = []

    for team, group in frame.groupby("team", sort=False):
        days = days_since_epoch(group["game_date"])
        venues = group["venue"].astype(str).to_numpy()
        is_home = group["is_home"].to_numpy(dtype=bool)
        origins = np.concatenate(([str(team)], venues[:-1]))

        leg_miles = np.array(
            [
                _distance(origin, venue)
                for origin, venue in zip(origins, venues, strict=True)
            ],
            dtype=float,
        )
        leg_crossings = np.array(
            [
                float(_crossings(origin, venue))
                for origin, venue in zip(origins, venues, strict=True)
            ],
            dtype=float,
        )
        trip_number = _road_trip_numbers(is_home)

        collected["travel_miles_7d"].append(_window_sums(days, leg_miles, 7))
        collected["travel_miles_14d"].append(_window_sums(days, leg_miles, 14))
        collected["timezone_crossings_7d"].append(_window_sums(days, leg_crossings, 7))
        collected["road_trip_game_number"].append(trip_number)
        collected["is_road_trip_4plus"].append(
            (trip_number >= ROAD_TRIP_THRESHOLD).astype(float)
        )
        collected["altitude_game"].append(
            np.array(
                [
                    float((not home_game) and _is_altitude(venue))
                    for home_game, venue in zip(is_home, venues, strict=True)
                ],
                dtype=float,
            )
        )
        order.append(group[ORDER_COLUMN].to_numpy())

    enriched = frame.set_index(ORDER_COLUMN)
    index = np.concatenate(order)
    for column, chunks in collected.items():
        enriched.loc[index, column] = np.concatenate(chunks)
    enriched = enriched.reset_index()
    enriched["is_road_trip_4plus"] = enriched["is_road_trip_4plus"].astype(bool)
    enriched["altitude_game"] = enriched["altitude_game"].astype(bool)
    return restore_order(enriched)


__all__ = ["ROAD_TRIP_THRESHOLD", "TRAVEL_COLUMNS", "add_travel_features"]
