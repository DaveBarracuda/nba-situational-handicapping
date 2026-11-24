"""Ensamblado del feature store (capa 4 completa).

Orden del pipeline (importa, porque cada paso depende del anterior):

    1. frame equipo-juego           -> long_frame
    2. rest / viajes / rolling / lesiones / etapa de temporada
    3. Elo con HFA dinámico         (necesita las flags de fatiga del paso 2)
    4. spots                        (necesitan los rankings Elo del paso 3)
    5. puntos situacionales         (reglas de peso de la guía 5.5A.3)
    6. ratings de potencia          (ponderados por lesiones del paso 2)
    7. pivote a formato ancho + puntos base esperados

Cada fila resultante usa **solo** información anterior al tip-off, por lo que
el frame completo puede construirse una vez y recortarse por fecha en el
backtest sin introducir look-ahead (guía 5.7 y 12.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nba_sh.config import CONFIG, Config
from nba_sh.features.injuries import add_injury_features
from nba_sh.features.long_frame import build_team_game_frame
from nba_sh.features.referees import add_referee_features
from nba_sh.features.rest import add_rest_features
from nba_sh.features.rolling import add_rolling_features
from nba_sh.features.season_stage import add_stage_features
from nba_sh.features.shooting import SHOOTING_COLUMNS, add_shooting_features
from nba_sh.features.shot_quality import (
    SHOT_QUALITY_COLUMNS,
    add_shot_quality_features,
)
from nba_sh.features.situational import add_situational_points
from nba_sh.features.spots import detect_spots
from nba_sh.features.travel import add_travel_features
from nba_sh.ratings.base_points import expected_base_points
from nba_sh.ratings.elo import EloResult, run_elo
from nba_sh.ratings.power import run_power_ratings
from nba_sh.schemas import Dataset

MILES_SCALE = 1000.0

# Features por equipo que se pivotan a _home / _away.
TEAM_FEATURES: tuple[str, ...] = (
    "rest_days",
    "is_b2b",
    "games_last_7",
    "games_last_14",
    "travel_miles_7d",
    "travel_miles_14d",
    "timezone_crossings_7d",
    "road_trip_game_number",
    "is_road_trip_4plus",
    "altitude_game",
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
    "injury_impact_points",
    "injury_n_out",
    "injury_pct_minutes_out",
    "injury_pct_minutes_out_7d",
    "late_scratch_flag",
    "injury_pace_impact",
    "stage",
    "team_game_number",
    "conference_rank",
    "win_pct_to_date",
    "must_win_flag",
    "tanking_score",
    "clinched_no_stakes_flag",
    "opponent_weak",
    "opponent_marquee",
    "next_opponent_marquee",
    "prev_opponent_marquee",
    "spot_lookahead",
    "spot_letdown",
    "spot_sandwich",
    "spot_revenge",
    "letdown_strength",
    "spot_points_raw",
    "fatigue_points",
    "spot_points",
    "momentum_points",
    "motivation_points",
    "situational_points",
    "injury_points",
    "team_adjustment",
    *SHOOTING_COLUMNS,
    *SHOT_QUALITY_COLUMNS,
)

# La calidad de tiro propia NO está aquí, y es a propósito: medida sobre
# 2024-25, la suma de la calidad de los dos equipos correlaciona +0.006 con el
# total, mientras que su acierto sobre lo esperado correlaciona +0.197. Elegir
# bien los tiros decide quién gana, no cuántos puntos se anotan; la calidad sí
# está en la matriz de margen (``shot_quality_diff``).
#
# Matriz de diseño del modelo de TOTALES. Lo que importa aquí son las SUMAS de
# ambos equipos (ritmo y eficiencia conjuntos), no las diferencias.
TOTAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "total_base",
    "situational_total_adjustment",
    "expected_possessions",
    "referee_total_points",
    "pace_sum",
    "off_rtg_sum",
    "def_rtg_sum",
    "rest_sum",
    "b2b_count",
    "injury_total",
    "efg_sum",
    "efg_allowed_sum",
    "three_rate_sum",
    "ft_rate_sum",
    "fouls_sum",
    "fast_break_sum",
    "paint_share_sum",
    "pace_gap",
    "efg_gap",
    "shot_making_sum",
    "shot_making_allowed_sum",
    "rim_rate_sum",
    "mid_rate_sum",
)

# Matriz de diseño de los modelos de margen (logit de ML y ridge de spread).
#
# Deliberadamente NO incluye ``situational_margin_adjustment``: ese agregado es
# la suma (con reglas) de las features situacionales que sí están aquí una por
# una. Meter el agregado y sus partes a la vez crea colinealidad severa y los
# coeficientes cambian de signo. El agregado vive en la vía estructural
# (``margin_expected``), y aquí el modelo lineal aprende sus propios pesos:
# son los dos enfoques complementarios de la guía (8.1 y 9.2).
FEATURE_COLUMNS: tuple[str, ...] = (
    "margin_base",
    "elo_margin",
    "rest_diff",
    "is_b2b_home",
    "is_b2b_away",
    "road_trip_away",
    "travel_diff",
    "timezone_crossings_away",
    "altitude_game",
    "injury_margin",
    "late_scratch_diff",
    "spot_margin",
    "momentum_diff",
    "luck_diff",
    "win_pct_diff",
    "net_rating_diff_10",
    "games_last_7_diff",
    "motivation_diff",
    "pace_sum",
    "season_maturity",
    "shot_quality_diff",
    "shot_making_diff",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "actual_margin",
    "actual_total",
    "home_win",
    "overtime",
)


@dataclass(frozen=True)
class FeatureStore:
    """Feature store de una ventana de datos."""

    games: pd.DataFrame
    team_games: pd.DataFrame
    elo: EloResult
    power: pd.DataFrame

    @property
    def n_games(self) -> int:
        """Juegos con features calculadas."""
        return int(len(self.games))

    def before(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Juegos estrictamente anteriores a una fecha (para entrenar)."""
        return self.games[self.games["game_date"] < pd.Timestamp(cutoff)].copy()

    def on_date(self, day: pd.Timestamp) -> pd.DataFrame:
        """Juegos de una fecha concreta (para predecir)."""
        return self.games[self.games["game_date"] == pd.Timestamp(day)].copy()


def _elo_context(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Flags del visitante que alimentan el HFA dinámico del Elo."""
    away = long_frame[~long_frame["is_home"].astype(bool)]
    return pd.DataFrame(
        {
            "is_b2b_away": away["is_b2b"].to_numpy(dtype=bool),
            "altitude_game": away["altitude_game"].to_numpy(dtype=bool),
            "away_road_trip_4plus": away["is_road_trip_4plus"].to_numpy(dtype=bool),
        },
        index=away["game_id"].to_numpy(),
    )


def _pivot_team_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Pasa las features por equipo a una fila por juego (_home / _away)."""
    available = [column for column in TEAM_FEATURES if column in long_frame.columns]
    sides = []
    for is_home, suffix in ((True, "_home"), (False, "_away")):
        side = long_frame[long_frame["is_home"].astype(bool) == is_home]
        sides.append(
            side.set_index("game_id")[available].add_suffix(suffix).sort_index()
        )
    return pd.concat(sides, axis=1)


def _derive_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construye la matriz de diseño a partir de las columnas _home/_away."""
    derived = pd.DataFrame(index=frame.index)
    derived["margin_base"] = frame["margin_base"]
    derived["situational_margin_adjustment"] = frame["situational_margin_adjustment"]
    derived["elo_margin"] = frame["elo_margin"]
    derived["rest_diff"] = frame["rest_days_home"] - frame["rest_days_away"]
    derived["is_b2b_home"] = frame["is_b2b_home"].astype(float)
    derived["is_b2b_away"] = frame["is_b2b_away"].astype(float)
    derived["road_trip_away"] = frame["is_road_trip_4plus_away"].astype(float)
    derived["travel_diff"] = (
        frame["travel_miles_7d_away"] - frame["travel_miles_7d_home"]
    ) / MILES_SCALE
    derived["timezone_crossings_away"] = frame["timezone_crossings_7d_away"]
    derived["altitude_game"] = frame["altitude_game_away"].astype(float)
    derived["injury_margin"] = frame["injury_points_away"] - frame["injury_points_home"]
    derived["late_scratch_diff"] = (
        frame["late_scratch_flag_away"] - frame["late_scratch_flag_home"]
    )
    derived["spot_margin"] = frame["spot_points_home"] - frame["spot_points_away"]
    derived["momentum_diff"] = frame["momentum_score_home"] - frame["momentum_score_away"]
    derived["luck_diff"] = frame["luck_flag_home"] - frame["luck_flag_away"]
    derived["win_pct_diff"] = (
        frame["win_pct_last_10_home"] - frame["win_pct_last_10_away"]
    )
    derived["net_rating_diff_10"] = (
        frame["net_rating_last_10_home"] - frame["net_rating_last_10_away"]
    )
    derived["games_last_7_diff"] = (
        frame["games_last_7_home"] - frame["games_last_7_away"]
    )
    derived["motivation_diff"] = (
        frame["motivation_points_home"] - frame["motivation_points_away"]
    )
    derived["pace_sum"] = frame["pace_last_10_home"].fillna(frame["pace_home_pre"]) + (
        frame["pace_last_10_away"].fillna(frame["pace_away_pre"])
    )
    # Calidad de tiro neta (lo que valen mis tiros menos lo que valen los que
    # concedo) y acierto neto, cada uno como diferencia local-visitante.
    for target, metric in (
        ("shot_quality_diff", "shot_quality"),
        ("shot_making_diff", "shot_making"),
    ):
        derived[target] = (
            frame[f"{metric}_last_10_home"] - frame[f"{metric}_allowed_last_10_home"]
        ) - (frame[f"{metric}_last_10_away"] - frame[f"{metric}_allowed_last_10_away"])
    derived["season_maturity"] = frame[
        ["team_game_number_home", "team_game_number_away"]
    ].min(axis=1)
    return derived.fillna(0.0)


def _derive_total_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Columnas agregadas (sumas de ambos equipos) para el modelo de totales."""
    derived = pd.DataFrame(index=frame.index)
    # ``pace_sum`` ya lo define la matriz de margen con la misma fórmula; aquí
    # solo se añade lo que es propio de totales para no duplicar columnas.
    pace_home = frame["pace_last_10_home"].fillna(frame["pace_home_pre"])
    pace_away = frame["pace_last_10_away"].fillna(frame["pace_away_pre"])
    derived["pace_gap"] = (pace_home - pace_away).abs()
    derived["off_rtg_sum"] = frame["off_home_pre"] + frame["off_away_pre"]
    derived["def_rtg_sum"] = frame["def_home_pre"] + frame["def_away_pre"]
    derived["rest_sum"] = frame["rest_days_home"] + frame["rest_days_away"]
    derived["b2b_count"] = frame["is_b2b_home"].astype(float) + frame[
        "is_b2b_away"
    ].astype(float)
    derived["injury_total"] = frame["injury_points_home"] + frame["injury_points_away"]
    for target, source in (
        ("efg_sum", "efg_last_10"),
        ("efg_allowed_sum", "efg_allowed_last_10"),
        ("three_rate_sum", "three_rate_last_10"),
        ("ft_rate_sum", "ft_rate_last_10"),
        ("fouls_sum", "fouls_last_10"),
        ("fast_break_sum", "fast_break_points_last_10"),
        ("paint_share_sum", "paint_share_last_10"),
        ("shot_making_sum", "shot_making_last_10"),
        ("shot_making_allowed_sum", "shot_making_allowed_last_10"),
        ("rim_rate_sum", "rim_rate_last_10"),
        ("mid_rate_sum", "mid_rate_last_10"),
    ):
        derived[target] = frame[f"{source}_home"] + frame[f"{source}_away"]
    derived["efg_gap"] = (
        frame["efg_last_10_home"] - frame["efg_last_10_away"]
    ).abs()
    return derived.fillna(0.0)


def build_features(dataset: Dataset, *, config: Config = CONFIG) -> FeatureStore:
    """Construye el feature store completo de un dataset ya validado."""
    games = dataset.games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    if games.empty:
        return FeatureStore(
            games=games,
            team_games=pd.DataFrame(),
            elo=run_elo(games, config=config),
            power=pd.DataFrame(),
        )

    long_frame = build_team_game_frame(games)
    long_frame = add_rest_features(long_frame)
    long_frame = add_travel_features(long_frame)
    long_frame = add_rolling_features(long_frame, config=config)
    long_frame = add_injury_features(long_frame, dataset.injuries, config=config)
    long_frame = add_shooting_features(long_frame, dataset.team_box, config=config)
    long_frame = add_shot_quality_features(
        long_frame, dataset.shot_quality, config=config
    )
    long_frame = add_stage_features(long_frame, config=config)

    elo = run_elo(games, context=_elo_context(long_frame), config=config)
    long_frame = detect_spots(long_frame, elo.ranks_by_date(), config=config)
    long_frame = add_situational_points(long_frame, config=config)

    injury_impact = long_frame[["game_id", "team", "injury_impact_points"]].rename(
        columns={"injury_impact_points": "impact_points"}
    )
    power = run_power_ratings(games, injury_impact=injury_impact, config=config)

    wide = games.set_index("game_id").sort_index()
    wide = wide.join(_pivot_team_features(long_frame))
    wide = wide.join(power.set_index("game_id").sort_index())
    wide = wide.join(
        elo.game_ratings.set_index("game_id").drop(columns=["game_date"]).sort_index()
    )
    wide = add_referee_features(wide.reset_index(), dataset.referees, config=config)

    # El margen se lleva el ajuste completo (fatiga, spots, motivación y bajas).
    wide["situational_margin_adjustment"] = (
        wide["team_adjustment_home"] - wide["team_adjustment_away"]
    )
    # El total NO se lleva el impacto de las bajas: el on/off mide margen, no
    # puntos anotados (ver ``InjuryConfig.injury_total_share``).
    injury_share = config.injury.injury_total_share
    wide["situational_total_adjustment"] = (
        wide["situational_points_home"]
        + wide["situational_points_away"]
        - injury_share * (wide["injury_points_home"] + wide["injury_points_away"])
        + wide["referee_total_points"]
    )

    base = expected_base_points(
        wide,
        margin_adjustment=wide["situational_margin_adjustment"],
        total_adjustment=wide["situational_total_adjustment"],
        config=config,
    )
    wide = pd.concat([wide, base], axis=1)

    home_score = pd.to_numeric(wide["home_score"], errors="coerce")
    away_score = pd.to_numeric(wide["away_score"], errors="coerce")
    wide["actual_margin"] = home_score - away_score
    wide["actual_total"] = home_score + away_score
    wide["home_win"] = (wide["actual_margin"] > 0).astype(float)
    wide["overtime"] = pd.to_numeric(wide["periods"], errors="coerce") > 4

    model_matrix = pd.concat(
        [_derive_model_features(wide), _derive_total_features(wide)], axis=1
    )
    wide = pd.concat(
        [wide.drop(columns=list(model_matrix.columns), errors="ignore"), model_matrix],
        axis=1,
    )
    wide = wide.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return FeatureStore(games=wide, team_games=long_frame, elo=elo, power=power)


__all__ = [
    "FEATURE_COLUMNS",
    "FeatureStore",
    "MILES_SCALE",
    "TARGET_COLUMNS",
    "TEAM_FEATURES",
    "TOTAL_FEATURE_COLUMNS",
    "build_features",
]
