"""Módulo de spots: lookahead, letdown, sandwich y revenge (guía 5.5A.1).

Todos los spots son detectables automáticamente desde el calendario + los
resultados ya conocidos, sin scraping de narrativa. Los rankings Elo se
consultan **as of** el día anterior al partido (``allow_exact_matches=False``),
de modo que ninguna clasificación incluye el juego que se está evaluando.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.long_frame import restore_order, sorted_by_team
from nba_sh.situational_rules import combine_same_sign
from nba_sh.teams import VALID_TEAMS, is_rivalry

SPOT_COLUMNS = (
    "opponent_weak",
    "opponent_marquee",
    "next_opponent_marquee",
    "prev_opponent_marquee",
    "days_to_next_game",
    "spot_lookahead",
    "spot_letdown",
    "spot_sandwich",
    "spot_revenge",
    "letdown_strength",
    "spot_points_raw",
)
BLOWOUT_OPPONENT_RANK = 5
FAR_FUTURE_DAYS = 99.0
UNRANKED = 99.0


def _ranks_as_of(
    dates: pd.Series, teams: pd.Series, ranks: pd.DataFrame
) -> pd.DataFrame:
    """Rankings Elo del equipo indicado, vigentes antes de la fecha dada."""
    size = len(dates)
    result = pd.DataFrame(
        {
            "league_rank": np.full(size, np.nan),
            "conference_rank": np.full(size, np.nan),
        }
    )
    if ranks.empty or size == 0:
        return result
    left = pd.DataFrame(
        {
            "game_date": pd.to_datetime(pd.Series(dates).to_numpy()),
            "team": pd.Series(teams).to_numpy(),
            "_row": np.arange(size),
        }
    )
    left = left[left["team"].isin(VALID_TEAMS)]
    if left.empty:
        return result
    left = left.sort_values("game_date")
    right = (
        ranks[["date", "team", "league_rank", "conference_rank"]]
        .dropna(subset=["team"])
        .sort_values("date")
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="game_date",
        right_on="date",
        by="team",
        allow_exact_matches=False,
    )
    rows = merged["_row"].to_numpy()
    result.loc[rows, "league_rank"] = merged["league_rank"].to_numpy()
    result.loc[rows, "conference_rank"] = merged["conference_rank"].to_numpy()
    return result


def _boolean(series: pd.Series) -> np.ndarray:
    """Columna booleana con los ausentes como False, sin avisos de pandas."""
    return series.astype("boolean").fillna(False).to_numpy(dtype=bool)


def _marquee(
    teams: pd.Series,
    opponents: pd.Series,
    conference_ranks: pd.Series,
    *,
    config: Config,
) -> np.ndarray:
    """Oponente de alto perfil: top-6 de conferencia por Elo o rivalidad."""
    ranks = conference_ranks.to_numpy(dtype=float)
    rank_flag = np.where(
        np.isnan(ranks), False, ranks <= config.spots.marquee_elo_rank_top
    )
    rivalry_flag = np.array(
        [
            opponent in VALID_TEAMS and is_rivalry(team, opponent)
            for team, opponent in zip(teams.to_numpy(), opponents.to_numpy(), strict=True)
        ],
        dtype=bool,
    )
    return rank_flag | rivalry_flag


def _revenge_flags(frame: pd.DataFrame, *, config: Config) -> np.ndarray:
    """Derrota por 25+ en el H2H más reciente (temporada actual o previa)."""
    threshold = config.spots.revenge_margin
    flags = np.zeros(len(frame), dtype=bool)
    last_margin: dict[tuple[str, str], float] = {}
    chronological = frame.sort_values(["game_date", "game_id"])
    position = {label: index for index, label in enumerate(frame.index)}
    for _, game_rows in chronological.groupby("game_id", sort=False):
        records = game_rows.to_dict(orient="index")
        for label, record in records.items():
            key = (str(record["team"]), str(record["opponent"]))
            previous = last_margin.get(key)
            if previous is not None and previous <= -threshold:
                flags[position[label]] = True
        for record in records.values():
            margin = record.get("margin")
            if margin is None or pd.isna(margin):
                continue
            last_margin[(str(record["team"]), str(record["opponent"]))] = float(margin)
    return flags


def detect_spots(
    long_frame: pd.DataFrame, elo_ranks: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Marca los spots de calendario de cada equipo-juego.

    Args:
        long_frame: frame equipo-juego con ``rest_days`` ya calculado.
        elo_ranks: salida de :meth:`nba_sh.ratings.elo.EloResult.ranks_by_date`.
        config: configuración del modelo.

    Returns:
        El frame con las columnas de ``SPOT_COLUMNS``.
    """
    if long_frame.empty:
        empty = long_frame.copy()
        for column in SPOT_COLUMNS:
            empty[column] = pd.Series(dtype="float64")
        return empty

    frame = sorted_by_team(long_frame)
    grouped = frame.groupby("team", sort=False)
    frame["next_opponent"] = grouped["opponent"].shift(-1)
    frame["next_game_date"] = grouped["game_date"].shift(-1)
    frame["next_season"] = grouped["season"].shift(-1)
    frame["prev_opponent"] = grouped["opponent"].shift(1)
    frame["prev_season"] = grouped["season"].shift(1)
    frame["prev_margin"] = grouped["margin"].shift(1)
    frame["prev_overtime"] = grouped["overtime"].shift(1)
    frame["prev2_margin"] = grouped["margin"].shift(2)
    frame["prev2_overtime"] = grouped["overtime"].shift(2)
    frame["prev2_season"] = grouped["season"].shift(2)

    not_same_next = frame["next_season"] != frame["season"]
    not_same_prev = frame["prev_season"] != frame["season"]
    not_same_prev2 = frame["prev2_season"] != frame["season"]
    frame.loc[not_same_next, "next_opponent"] = np.nan
    frame.loc[not_same_next, "next_game_date"] = pd.NaT
    frame.loc[not_same_prev, "prev_opponent"] = np.nan
    frame.loc[not_same_prev, "prev_margin"] = np.nan
    frame.loc[not_same_prev2, "prev2_margin"] = np.nan

    opponent_ranks = _ranks_as_of(frame["game_date"], frame["opponent"], elo_ranks)
    next_ranks = _ranks_as_of(
        frame["game_date"], frame["next_opponent"].fillna(""), elo_ranks
    )
    prev_ranks = _ranks_as_of(
        frame["game_date"], frame["prev_opponent"].fillna(""), elo_ranks
    )

    spots = config.spots
    weak_threshold = len(VALID_TEAMS) - spots.weak_opponent_elo_rank_bottom
    frame["opponent_weak"] = (
        opponent_ranks["league_rank"].fillna(0.0).to_numpy(dtype=float) > weak_threshold
    )
    frame["opponent_marquee"] = _marquee(
        frame["team"],
        frame["opponent"],
        opponent_ranks["conference_rank"],
        config=config,
    )
    frame["next_opponent_marquee"] = _marquee(
        frame["team"],
        frame["next_opponent"].fillna(""),
        next_ranks["conference_rank"],
        config=config,
    )
    frame["prev_opponent_marquee"] = _marquee(
        frame["team"],
        frame["prev_opponent"].fillna(""),
        prev_ranks["conference_rank"],
        config=config,
    )
    frame["days_to_next_game"] = (
        pd.to_datetime(frame["next_game_date"]) - pd.to_datetime(frame["game_date"])
    ).dt.days

    has_next = frame["next_opponent"].notna().to_numpy()
    has_prev = frame["prev_opponent"].notna().to_numpy()
    opponent_weak = frame["opponent_weak"].to_numpy(dtype=bool)

    frame["spot_lookahead"] = (
        opponent_weak
        & has_next
        & frame["next_opponent_marquee"]
        & (
            frame["days_to_next_game"].fillna(FAR_FUTURE_DAYS).to_numpy(dtype=float)
            <= spots.lookahead_max_days_ahead
        )
    )

    prev_blowout_win = (
        frame["prev_margin"].fillna(0.0).to_numpy(dtype=float)
        >= spots.blowout_margin_for_emotional_win
    ) & (
        prev_ranks["league_rank"].fillna(UNRANKED).to_numpy(dtype=float)
        <= BLOWOUT_OPPONENT_RANK
    )
    prev_overtime_win = _boolean(frame["prev_overtime"]) & (
        frame["prev_margin"].fillna(0.0).to_numpy(dtype=float) > 0.0
    )
    emotional_prev = (prev_blowout_win | prev_overtime_win) & has_prev
    emotional_prev2 = _boolean(frame["prev2_overtime"]) & (
        frame["prev2_margin"].fillna(0.0).to_numpy(dtype=float) > 0.0
    )
    eligible_letdown = opponent_weak & (
        frame["rest_days"].to_numpy(dtype=float) >= spots.letdown_min_rest
    )
    frame["letdown_strength"] = np.where(
        eligible_letdown & emotional_prev,
        1.0,
        np.where(
            eligible_letdown & emotional_prev2, spots.letdown_decay_second_game, 0.0
        ),
    )
    frame["spot_letdown"] = frame["letdown_strength"] > 0.0
    frame["spot_sandwich"] = (
        opponent_weak
        & has_prev
        & has_next
        & frame["prev_opponent_marquee"]
        & frame["next_opponent_marquee"]
    )
    frame["spot_revenge"] = _revenge_flags(frame, config=config)

    components = np.vstack(
        [
            np.where(frame["spot_lookahead"], spots.lookahead_points, 0.0),
            frame["letdown_strength"].to_numpy(dtype=float) * spots.letdown_points,
            np.where(frame["spot_sandwich"], spots.sandwich_points, 0.0),
            np.where(frame["spot_revenge"], spots.revenge_points, 0.0),
        ]
    ).T
    frame["spot_points_raw"] = [
        combine_same_sign(row, secondary_weight=spots.secondary_spot_weight)
        for row in components
    ]

    frame = frame.drop(
        columns=[
            "next_season",
            "prev_season",
            "prev2_season",
            "prev_overtime",
            "prev2_overtime",
            "prev2_margin",
        ]
    )
    return restore_order(frame)


__all__ = ["BLOWOUT_OPPONENT_RANK", "SPOT_COLUMNS", "detect_spots"]
