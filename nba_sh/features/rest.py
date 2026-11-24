"""Rest y calendario: el factor situacional #1 en NBA (guía 5.1).

    rest_days   = días *de descanso* desde el último juego del equipo, es
                  decir, días de calendario transcurridos menos uno: jugar en
                  días consecutivos son 0 días de descanso (back-to-back)
    is_b2b      = rest_days == 0
    is_b2b_away = B2B siendo visitante  <- el peor escenario
    games_in_last_7_days / _14_days

Todas las cuentas usan exclusivamente juegos anteriores al actual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.features.long_frame import ORDER_COLUMN, restore_order, sorted_by_team

SEASON_OPENER_REST_DAYS = 3
MAX_REST_DAYS = 10
REST_COLUMNS = (
    "rest_days",
    "is_b2b",
    "is_b2b_away",
    "games_last_7",
    "games_last_14",
    "is_season_opener",
)
EPOCH = pd.Timestamp("1970-01-01")


def days_since_epoch(dates: pd.Series) -> np.ndarray:
    """Fechas como enteros (días) para aritmética de ventanas."""
    return (pd.to_datetime(dates) - EPOCH).dt.days.to_numpy(dtype=np.int64)


def _games_in_window(days: np.ndarray, window: int) -> np.ndarray:
    """Juegos previos dentro de la ventana, para cada posición."""
    counts = np.zeros(days.size, dtype=np.int64)
    for index in range(days.size):
        start = days[index] - window
        left = int(np.searchsorted(days[:index], start, side="right"))
        counts[index] = index - left
    return counts


def add_rest_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Añade descanso, B2B y congestión de calendario al frame largo."""
    if long_frame.empty:
        empty = long_frame.copy()
        for column in REST_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    frame = sorted_by_team(long_frame)
    rest_days: list[np.ndarray] = []
    games_7: list[np.ndarray] = []
    games_14: list[np.ndarray] = []
    openers: list[np.ndarray] = []
    order: list[np.ndarray] = []

    for _, group in frame.groupby("team", sort=False):
        days = days_since_epoch(group["game_date"])
        previous = np.empty(days.size, dtype=np.int64)
        previous[0] = days[0] - SEASON_OPENER_REST_DAYS
        previous[1:] = days[:-1]
        # Días de descanso = días de calendario - 1 (B2B => 0 días).
        gaps = np.clip(days - previous - 1, 0, MAX_REST_DAYS)
        seasons = group["season"].to_numpy()
        is_opener = np.zeros(days.size, dtype=bool)
        is_opener[0] = True
        if days.size > 1:
            is_opener[1:] = seasons[1:] != seasons[:-1]
        gaps = np.where(is_opener, SEASON_OPENER_REST_DAYS, gaps)
        rest_days.append(gaps.astype(float))
        games_7.append(_games_in_window(days, 7).astype(float))
        games_14.append(_games_in_window(days, 14).astype(float))
        openers.append(is_opener.astype(float))
        order.append(group[ORDER_COLUMN].to_numpy())

    enriched = frame.set_index(ORDER_COLUMN)
    index = np.concatenate(order)
    enriched.loc[index, "rest_days"] = np.concatenate(rest_days)
    enriched.loc[index, "games_last_7"] = np.concatenate(games_7)
    enriched.loc[index, "games_last_14"] = np.concatenate(games_14)
    enriched.loc[index, "is_season_opener"] = np.concatenate(openers)
    enriched = enriched.reset_index()
    enriched["is_season_opener"] = enriched["is_season_opener"].astype(bool)
    enriched["is_b2b"] = enriched["rest_days"] == 0
    enriched["is_b2b_away"] = enriched["is_b2b"] & ~enriched["is_home"]
    return restore_order(enriched)


__all__ = [
    "MAX_REST_DAYS",
    "REST_COLUMNS",
    "SEASON_OPENER_REST_DAYS",
    "add_rest_features",
    "days_since_epoch",
]
