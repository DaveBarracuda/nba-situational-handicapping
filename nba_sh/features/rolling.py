"""Ritmo, estilo, momentum real y racha de suerte (guía 5.4 y 5.5A.2).

Separa las dos cosas que el público confunde:

    MOMENTUM_REAL   -> diferencial de puntos y net rating rolling
    RACHA_DE_SUERTE -> récord en juegos cerrados fuera de banda binomial

Todas las medias móviles llevan ``shift(1)``: el juego de hoy nunca entra en
sus propias medias (guía 5.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from nba_sh.config import CONFIG, Config
from nba_sh.features.long_frame import ORDER_COLUMN, restore_order, sorted_by_team
from nba_sh.features.windows import season_weighted_rolling

WINDOW_SHORT = 5
WINDOW_LONG = 10
MIN_PERIODS_SHORT = 3
MIN_PERIODS_LONG = 4
# Escalas fijas de liga para estandarizar (evitan cualquier look-ahead).
PT_DIFF_SHORT_SCALE = 7.5
NET_RATING_SHORT_SCALE = 7.0
CLOSE_GAME_MIN_SAMPLE = 4
FAIR_COIN = 0.5

ROLLING_COLUMNS = (
    "off_rtg_last_10",
    "def_rtg_last_10",
    "pace_last_10",
    "net_rating_last_5",
    "net_rating_last_10",
    "pt_diff_last_5",
    "pt_diff_last_10",
    "win_pct_last_10",
    "win_streak_current",
    "close_games_last_10",
    "close_wins_last_10",
    "luck_flag",
    "momentum_score",
)


def _shifted_rolling_mean(
    series: pd.Series, window: int, min_periods: int
) -> pd.Series:
    """Media móvil de los juegos **anteriores**."""
    return series.shift(1).rolling(window, min_periods=min_periods).mean()


def _shifted_rolling_sum(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Suma móvil de los juegos **anteriores**."""
    return series.shift(1).rolling(window, min_periods=min_periods).sum()


def _win_streak(wins: np.ndarray) -> np.ndarray:
    """Racha de victorias consecutivas previa a cada juego (0 si perdió)."""
    streaks = np.zeros(wins.size, dtype=float)
    streak = 0
    for index in range(wins.size):
        streaks[index] = streak
        streak = streak + 1 if wins[index] else 0
    return streaks


def _luck_flag(
    close_games: pd.Series, close_wins: pd.Series, *, threshold: float
) -> pd.Series:
    """+1 racha inflada por suerte, -1 castigada, 0 dentro de banda.

    Usa la cola binomial del récord en juegos decididos por <= 5 puntos.
    """
    games = close_games.fillna(0.0).to_numpy(dtype=float)
    wins = close_wins.fillna(0.0).to_numpy(dtype=float)
    flags = np.zeros(games.size, dtype=float)
    enough = games >= CLOSE_GAME_MIN_SAMPLE
    if not enough.any():
        return pd.Series(flags, index=close_games.index)
    upper = binom.cdf(wins[enough] - 1.0, games[enough], FAIR_COIN)
    lower = binom.cdf(wins[enough], games[enough], FAIR_COIN)
    subset = np.zeros(int(enough.sum()), dtype=float)
    subset[upper > threshold] = 1.0
    subset[lower < 1.0 - threshold] = -1.0
    flags[enough] = subset
    return pd.Series(flags, index=close_games.index)


def add_rolling_features(
    long_frame: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Añade rolling de eficiencia, ritmo, momentum y flag de suerte."""
    if long_frame.empty:
        empty = long_frame.copy()
        for column in ROLLING_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    frame = sorted_by_team(long_frame)
    frame["net_rating_game"] = frame["off_rtg"] - frame["def_rtg"]
    frame["is_close_game"] = (
        frame["margin"].abs() <= config.spots.close_game_margin
    ).astype(float)
    frame["is_close_win"] = frame["is_close_game"] * frame["won"].astype(float)

    carryover = config.form.season_carryover_weight
    grouped = frame.groupby("team", sort=False)
    rolling_means = {
        "off_rtg_last_10": ("off_rtg", WINDOW_LONG),
        "def_rtg_last_10": ("def_rtg", WINDOW_LONG),
        "pace_last_10": ("possessions", WINDOW_LONG),
        "net_rating_last_5": ("net_rating_game", WINDOW_SHORT),
        "net_rating_last_10": ("net_rating_game", WINDOW_LONG),
        "pt_diff_last_5": ("margin", WINDOW_SHORT),
        "pt_diff_last_10": ("margin", WINDOW_LONG),
    }
    for target, (source, window) in rolling_means.items():
        frame[target] = season_weighted_rolling(
            frame,
            source,
            window=window,
            min_periods=MIN_PERIODS_SHORT,
            carryover_weight=carryover,
        )
    frame["won_float"] = frame["won"].astype(float)
    frame["win_pct_last_10"] = season_weighted_rolling(
        frame,
        "won_float",
        window=WINDOW_LONG,
        min_periods=MIN_PERIODS_LONG,
        carryover_weight=carryover,
    )
    # Los recuentos de partidos ajustados son la muestra del test binomial de
    # suerte: ahí un partido es un partido, sin ponderar por temporada.
    frame["close_games_last_10"] = grouped["is_close_game"].transform(
        lambda series: _shifted_rolling_sum(series, WINDOW_LONG, 1)
    )
    frame["close_wins_last_10"] = grouped["is_close_win"].transform(
        lambda series: _shifted_rolling_sum(series, WINDOW_LONG, 1)
    )
    frame = frame.drop(columns=["won_float"])

    streaks: list[np.ndarray] = []
    order: list[np.ndarray] = []
    for _, group in frame.groupby("team", sort=False):
        # Una jornada por jugar no tiene resultado: cuenta como no-victoria.
        # Solo afecta a la racha de filas posteriores, y el slate es la
        # última fila de cada equipo.
        played = group["won"].fillna(False).to_numpy(dtype=bool)
        streaks.append(_win_streak(played))
        order.append(group[ORDER_COLUMN].to_numpy())
    indexed = frame.set_index(ORDER_COLUMN)
    indexed.loc[np.concatenate(order), "win_streak_current"] = np.concatenate(streaks)
    frame = indexed.reset_index()

    frame["luck_flag"] = _luck_flag(
        frame["close_games_last_10"],
        frame["close_wins_last_10"],
        threshold=config.spots.luck_binomial_pvalue,
    )
    frame["momentum_score"] = (
        config.spots.momentum_pt_diff_weight
        * frame["pt_diff_last_5"].fillna(0.0)
        / PT_DIFF_SHORT_SCALE
        + config.spots.momentum_net_rating_weight
        * frame["net_rating_last_5"].fillna(0.0)
        / NET_RATING_SHORT_SCALE
    )
    frame = frame.drop(columns=["net_rating_game", "is_close_game", "is_close_win"])
    return restore_order(frame)


__all__ = [
    "CLOSE_GAME_MIN_SAMPLE",
    "NET_RATING_SHORT_SCALE",
    "PT_DIFF_SHORT_SCALE",
    "ROLLING_COLUMNS",
    "WINDOW_LONG",
    "WINDOW_SHORT",
    "add_rolling_features",
]
