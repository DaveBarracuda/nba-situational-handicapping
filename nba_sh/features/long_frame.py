"""Vista "equipo-juego" (formato largo): dos filas por partido.

Trabajar en formato largo hace que rest, viajes, rolling y spots sean
cálculos naturales por equipo; el pivote a formato ancho (una fila por
partido, local vs visitante) se hace al final en :mod:`nba_sh.features.build`.
"""

from __future__ import annotations

import pandas as pd

from nba_sh.config import POSSESSIONS_SCALE

ORDER_COLUMN = "_order"
TEAM_GAME_COLUMNS = (
    "game_id",
    "season",
    "game_date",
    "team",
    "opponent",
    "is_home",
    "venue",
    "team_score",
    "opponent_score",
    "margin",
    "won",
    "possessions",
    "off_rtg",
    "def_rtg",
    "crew_id",
    "overtime",
)


def build_team_game_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Convierte la tabla de juegos en formato equipo-juego.

    Args:
        games: tabla de juegos normalizada.

    Returns:
        DataFrame con dos filas por partido (local y visitante), ordenado por
        fecha, con ``off_rtg``/``def_rtg`` por 100 posesiones y el número de
        juego de cada equipo dentro de la temporada.
    """
    if games.empty:
        return pd.DataFrame(columns=[*TEAM_GAME_COLUMNS, "team_game_number"])

    base = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    sides: list[pd.DataFrame] = []
    for is_home in (True, False):
        team_column = "home_team" if is_home else "away_team"
        opponent_column = "away_team" if is_home else "home_team"
        score_column = "home_score" if is_home else "away_score"
        opponent_score_column = "away_score" if is_home else "home_score"
        sides.append(
            pd.DataFrame(
                {
                    "game_id": base["game_id"],
                    "season": base["season"],
                    "game_date": base["game_date"],
                    "team": base[team_column],
                    "opponent": base[opponent_column],
                    "is_home": is_home,
                    "venue": base["home_team"],
                    "team_score": pd.to_numeric(base[score_column], errors="coerce"),
                    "opponent_score": pd.to_numeric(
                        base[opponent_score_column], errors="coerce"
                    ),
                    "possessions": pd.to_numeric(base["possessions"], errors="coerce"),
                    "crew_id": base["crew_id"],
                    "overtime": pd.to_numeric(base["periods"], errors="coerce") > 4,
                }
            )
        )

    long_frame = pd.concat(sides, ignore_index=True)
    long_frame["margin"] = long_frame["team_score"] - long_frame["opponent_score"]
    long_frame["won"] = long_frame["margin"] > 0
    possessions = long_frame["possessions"].replace(0.0, float("nan"))
    long_frame["off_rtg"] = long_frame["team_score"] / possessions * POSSESSIONS_SCALE
    long_frame["def_rtg"] = (
        long_frame["opponent_score"] / possessions * POSSESSIONS_SCALE
    )
    long_frame = long_frame.sort_values(
        ["game_date", "game_id", "is_home"]
    ).reset_index(drop=True)
    long_frame["team_game_number"] = long_frame.groupby(["season", "team"]).cumcount() + 1
    long_frame[ORDER_COLUMN] = range(len(long_frame))
    return long_frame


def sorted_by_team(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Copia ordenada por equipo y fecha (para rolling/shift por equipo)."""
    return long_frame.sort_values(["team", "game_date", "game_id"]).copy()


def restore_order(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el frame al orden cronológico original."""
    return long_frame.sort_values(ORDER_COLUMN).reset_index(drop=True)


__all__ = [
    "ORDER_COLUMN",
    "TEAM_GAME_COLUMNS",
    "build_team_game_frame",
    "restore_order",
    "sorted_by_team",
]
