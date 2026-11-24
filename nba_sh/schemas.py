"""Esquemas canónicos de la data (capa 2 y 3 de la arquitectura).

Define nombres de columnas y el contenedor inmutable :class:`Dataset` que
atraviesa todo el pipeline. Todas las fechas de juego son *fechas ET*
(``datetime64[ns]`` normalizado a medianoche) y los timestamps de tip-off se
guardan en UTC, según la regla de timezones de la guía (sección 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import pandas as pd

# --------------------------------------------------------------------------
# Columnas
# --------------------------------------------------------------------------
GAME_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "season_type",
    "game_date",
    "tipoff_utc",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "periods",
    "possessions",
    "crew_id",
)

ODDS_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "book",
    "timing",
    "ml_home",
    "ml_away",
    "spread_home",
    "spread_home_price",
    "spread_away_price",
    "total_line",
    "over_price",
    "under_price",
)

INJURY_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "team",
    "player_id",
    "minutes_pg",
    "on_off_net",
    "status",
    "hours_before_tip",
)

REFEREE_COLUMNS: Final[tuple[str, ...]] = (
    "crew_id",
    "crew_foul_rate",
    "crew_fta_rate",
)

# Detalle de tiro por equipo y partido (guía 5.4: ritmo y estilo).
TEAM_BOX_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "team",
    "score",
    "opponent_score",
    "possessions",
    "field_goals_made",
    "field_goals_attempted",
    "three_point_field_goals_made",
    "three_point_field_goals_attempted",
    "free_throws_attempted",
    "offensive_rebounds",
    "turnovers",
    "fouls",
    "points_in_paint",
    "fast_break_points",
)

TABLE_NAMES: Final[tuple[str, ...]] = (
    "games",
    "odds",
    "injuries",
    "referees",
    "team_box",
    "shot_quality",
)

# Calidad de tiro por equipo y partido (guía 5.4: shot quality del pbp).
SHOT_QUALITY_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "team",
    "attempts",
    "shot_quality",
    "shot_making",
    "rim_rate",
    "mid_rate",
)

# Momentos en los que se captura una línea (guía 3.2: guardar apertura y cierre).
TIMING_OPEN: Final[str] = "open"
TIMING_PREGAME: Final[str] = "t_minus_60"
TIMING_CLOSE: Final[str] = "close"
TIMINGS: Final[tuple[str, ...]] = (TIMING_OPEN, TIMING_PREGAME, TIMING_CLOSE)

MARKET_ML: Final[str] = "ml"
MARKET_SPREAD: Final[str] = "spread"
MARKET_TOTAL: Final[str] = "total"
MARKETS: Final[tuple[str, ...]] = (MARKET_ML, MARKET_SPREAD, MARKET_TOTAL)

STATUS_OUT: Final[str] = "out"
STATUS_DOUBTFUL: Final[str] = "doubtful"
STATUS_QUESTIONABLE: Final[str] = "questionable"
INJURY_STATUSES: Final[tuple[str, ...]] = (
    STATUS_OUT,
    STATUS_DOUBTFUL,
    STATUS_QUESTIONABLE,
)

SEASON_TYPE_REGULAR: Final[str] = "regular"
SEASON_TYPE_PLAYOFF: Final[str] = "playoff"


def empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    """DataFrame vacío con las columnas indicadas."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


@dataclass(frozen=True)
class Dataset:
    """Conjunto de datos crudos ya normalizado.

    Inmutable: las transformaciones devuelven copias nuevas (``with_games``,
    ``filtered``), nunca mutan los DataFrames originales.
    """

    games: pd.DataFrame
    odds: pd.DataFrame
    injuries: pd.DataFrame
    referees: pd.DataFrame
    team_box: pd.DataFrame = field(
        default_factory=lambda: empty_frame(TEAM_BOX_COLUMNS)
    )
    shot_quality: pd.DataFrame = field(
        default_factory=lambda: empty_frame(SHOT_QUALITY_COLUMNS)
    )

    @staticmethod
    def empty() -> Dataset:
        """Dataset vacío con el esquema correcto."""
        return Dataset(
            games=empty_frame(GAME_COLUMNS),
            odds=empty_frame(ODDS_COLUMNS),
            injuries=empty_frame(INJURY_COLUMNS),
            referees=empty_frame(REFEREE_COLUMNS),
            team_box=empty_frame(TEAM_BOX_COLUMNS),
            shot_quality=empty_frame(SHOT_QUALITY_COLUMNS),
        )

    @property
    def seasons(self) -> tuple[str, ...]:
        """Temporadas presentes, ordenadas."""
        if self.games.empty:
            return ()
        return tuple(sorted(self.games["season"].unique()))

    @property
    def n_games(self) -> int:
        """Número de juegos."""
        return int(len(self.games))

    def with_games(self, games: pd.DataFrame) -> Dataset:
        """Copia del dataset con otra tabla de juegos."""
        return replace(self, games=games.copy())

    def filtered(self, game_ids: Sequence[str]) -> Dataset:
        """Sub-dataset restringido a los ``game_ids`` indicados."""
        keep = set(game_ids)
        by_game = {
            name: frame[frame["game_id"].isin(keep)].copy()
            for name, frame in (
                ("games", self.games),
                ("odds", self.odds),
                ("injuries", self.injuries),
                ("team_box", self.team_box),
                ("shot_quality", self.shot_quality),
            )
            if not frame.empty
        }
        return Dataset(
            games=by_game.get("games", self.games.copy()),
            odds=by_game.get("odds", self.odds.copy()),
            injuries=by_game.get("injuries", self.injuries.copy()),
            referees=self.referees.copy(),
            team_box=by_game.get("team_box", self.team_box.copy()),
            shot_quality=by_game.get("shot_quality", self.shot_quality.copy()),
        )

    def before(self, cutoff: pd.Timestamp) -> Dataset:
        """Sub-dataset con los juegos estrictamente anteriores a ``cutoff``."""
        mask = self.games["game_date"] < pd.Timestamp(cutoff)
        return self.filtered(self.games.loc[mask, "game_id"].tolist())

    def tables(self) -> Mapping[str, pd.DataFrame]:
        """Mapa nombre -> DataFrame (para persistir o auditar)."""
        return {
            "games": self.games,
            "odds": self.odds,
            "injuries": self.injuries,
            "referees": self.referees,
            "team_box": self.team_box,
            "shot_quality": self.shot_quality,
        }


def odds_at(odds: pd.DataFrame, timing: str) -> pd.DataFrame:
    """Líneas de un momento concreto, indexadas por ``game_id``.

    Si el timing pedido no existe para un juego, ese juego no aparece: el
    motor de apuestas debe tratar la ausencia de línea como "no apostable".
    """
    if timing not in TIMINGS:
        raise ValueError(f"unknown timing: {timing!r}")
    subset = odds[odds["timing"] == timing].copy()
    return subset.drop_duplicates(subset=["game_id"], keep="last").set_index("game_id")


__all__ = [
    "Dataset",
    "GAME_COLUMNS",
    "INJURY_COLUMNS",
    "INJURY_STATUSES",
    "MARKETS",
    "MARKET_ML",
    "MARKET_SPREAD",
    "MARKET_TOTAL",
    "ODDS_COLUMNS",
    "REFEREE_COLUMNS",
    "SHOT_QUALITY_COLUMNS",
    "TABLE_NAMES",
    "TEAM_BOX_COLUMNS",
    "SEASON_TYPE_PLAYOFF",
    "SEASON_TYPE_REGULAR",
    "STATUS_DOUBTFUL",
    "STATUS_OUT",
    "STATUS_QUESTIONABLE",
    "TIMINGS",
    "TIMING_CLOSE",
    "TIMING_OPEN",
    "TIMING_PREGAME",
    "empty_frame",
    "odds_at",
]
