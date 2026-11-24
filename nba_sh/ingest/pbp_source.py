"""Calidad de tiro desde el play-by-play (guía 5.4, "shot quality").

El box score dice **si entró**; el play-by-play dice **de dónde se tiró**. La
diferencia importa porque las dos cosas se comportan distinto:

    pps_real = calidad_de_tiro + acierto_sobre_lo_esperado
               (dónde tiras)     (si lo metes)

La calidad es casi todo selección —y la selección es estable—, mientras que el
acierto sobre lo esperado es en buena parte varianza. Separarlas permite
quedarse con la parte repetible.

Los puntos esperados de un tiro (``xpps``) salen de una tabla de referencia
``(banda de distancia × familia de tiro) -> puntos por intento``, estimada
**con la temporada anterior**: la tabla de 2023-24 se calcula con 2022-23. Así
ningún partido contribuye a su propia referencia.

Fuente: ``sportsdataverse/hoopR-nba-data`` (`pbp/parquet`), derivado de ESPN,
con los mismos ``game_id`` que el resto del proyecto. ~21 MB por temporada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sh.ingest.hoopr_source import (
    HOOPR_BASE,
    USER_AGENT,
    season_label_to_hoopr,
)

LOGGER = logging.getLogger(__name__)

PBP_KIND = "pbp"
PBP_PREFIX = "play_by_play"
DEFAULT_CACHE_DIR = Path("data/raw/hoopr")
DEFAULT_TIMEOUT_SECONDS = 300
REGULAR_SEASON_TYPE = 2

# Geometría de la pista ESPN: x de -47 a 47 (94 pies de largo), y de -25 a 25
# (50 de ancho). Los aros están en x = ±41.75, y = 0.
HOOP_X = 41.75
MAX_COORDINATE_X = 50.0
MAX_COORDINATE_Y = 30.0
MAX_SHOT_DISTANCE = 94.0
# Línea de tres NBA: arco de 23.75 pies, recortado a 22 en las esquinas (donde
# la línea corre paralela a la banda). Verificado contra la verdad de 2024-25:
# clasifica bien el 99,88% de los tiros.
THREE_POINT_ARC = 23.75
CORNER_THREE_DISTANCE = 22.0
CORNER_Y = 22.0
MIN_THREE_POINT_DISTANCE = 21.0
# Los tiros libres son jugadas de tiro, pero su aportación ya entra por
# ``ft_rate``: se reconocen por el texto, no por un campo numérico.
FREE_THROW_PREFIX = "Free Throw"

# Bandas de distancia en pies: finas cerca del aro, donde el valor cambia de
# golpe, y anchas fuera, donde apenas cambia.
DISTANCE_EDGES = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 23.75, 27.0, 35.0, MAX_SHOT_DISTANCE)
# Mínimo de intentos en una celda antes de fiarse de ella; por debajo se usa
# el valor de su banda de distancia.
MIN_CELL_ATTEMPTS = 200

# Familias de tiro, por prioridad: el primer patrón que aparece en el texto
# manda ("Driving Floating Bank Jump Shot" es un floater, no un jump shot).
SHOT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dunk", ("dunk",)),
    ("tip", ("tip",)),
    ("layup", ("layup", "finger roll")),
    ("hook", ("hook",)),
    ("floater", ("floating", "floater")),
    ("stepback", ("step back", "stepback")),
    ("fadeaway", ("fade away", "fadeaway", "turnaround")),
    ("pullup", ("pullup", "pull up")),
)
DEFAULT_FAMILY = "jumper"

VALUE_TABLE_COLUMNS = ("band", "family", "xpps", "attempts")
QUALITY_COLUMNS = (
    "game_id",
    "team_id",
    "attempts",
    "shot_quality",
    "shot_making",
    "rim_rate",
    "mid_rate",
)

PBP_READ_COLUMNS = (
    "game_id",
    "season",
    "season_type",
    "team_id",
    "shooting_play",
    "scoring_play",
    "score_value",
    "type_text",
    "coordinate_x",
    "coordinate_y",
)
ROSTER_READ_COLUMNS = (
    "game_id",
    "home_team_id",
    "away_team_id",
    "home_team_abbrev",
    "away_team_abbrev",
)

RIM_BAND_LIMIT = 4.0
MID_BAND_LOWER = 8.0
MID_BAND_UPPER = 23.75


def download_play_by_play(
    seasons: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> dict[int, Path]:
    """Descarga (y cachea) el play-by-play de cada temporada.

    Descarga también la temporada **anterior** a la primera pedida: es la que
    da la tabla de referencia de la primera, sin que ningún partido entre en
    su propia referencia.

    Args:
        seasons: temporadas en formato NBA (``"2023-24"``).
        cache_dir: directorio de caché; es data cruda, no se sobrescribe.
        timeout: timeout por petición (los ficheros son de ~21 MB).
        force: vuelve a descargar aunque exista el fichero.

    Returns:
        Mapa ``año_hoopr -> ruta local``.
    """
    import requests  # import diferido: el paquete funciona sin red

    cache_dir.mkdir(parents=True, exist_ok=True)
    requested = sorted({season_label_to_hoopr(season) for season in seasons})
    years = sorted({*requested, min(requested) - 1})
    paths: dict[int, Path] = {}
    for year in years:
        path = cache_dir / f"{PBP_PREFIX}_{year}.parquet"
        if force or not path.exists():
            url = f"{HOOPR_BASE}/{PBP_KIND}/parquet/{PBP_PREFIX}_{year}.parquet"
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
            response.raise_for_status()
            path.write_bytes(response.content)
            LOGGER.info("pbp: %s -> %s (%d bytes)", url, path, len(response.content))
        paths[year] = path
    return paths


def classify_family(type_text: pd.Series) -> pd.Series:
    """Familia de tiro a partir del texto de la jugada."""
    lowered = type_text.fillna("").str.lower()
    family = pd.Series(DEFAULT_FAMILY, index=lowered.index, dtype="object")
    assigned = pd.Series(False, index=lowered.index)
    for name, patterns in SHOT_FAMILIES:
        matches = lowered.str.contains("|".join(patterns), regex=True, na=False)
        family = family.mask(matches & ~assigned, name)
        assigned = assigned | matches
    return family


def shot_distance(coordinate_x: pd.Series, coordinate_y: pd.Series) -> pd.Series:
    """Distancia al aro en pies, desde las coordenadas de ESPN."""
    along = pd.to_numeric(coordinate_x, errors="coerce").abs() - HOOP_X
    across = pd.to_numeric(coordinate_y, errors="coerce")
    return pd.Series(np.hypot(along, across), index=coordinate_x.index)


def distance_band(distance: pd.Series) -> pd.Series:
    """Índice de banda de distancia (entero, no un intervalo)."""
    return pd.Series(
        np.clip(
            np.searchsorted(DISTANCE_EDGES, distance.to_numpy(), side="right") - 1,
            0,
            len(DISTANCE_EDGES) - 2,
        ),
        index=distance.index,
    )


def is_three_point_shot(
    distance: pd.Series, coordinate_y: pd.Series, score_value: pd.Series, made: pd.Series
) -> pd.Series:
    """Marca los triples.

    En los tiros anotados ``score_value`` lo dice exactamente. En los errados
    no hay campo que lo diga —``points_attempted`` solo está poblado en la
    última temporada— y se deduce de la geometría de la línea.
    """
    across = pd.to_numeric(coordinate_y, errors="coerce").abs()
    threshold = np.where(across >= CORNER_Y, CORNER_THREE_DISTANCE, THREE_POINT_ARC)
    geometric = distance.to_numpy() >= threshold
    scored_three = pd.to_numeric(score_value, errors="coerce").to_numpy() == 3
    return pd.Series(
        np.where(made.to_numpy(), scored_three, geometric), index=distance.index
    )


def load_shots(paths: Sequence[Path]) -> pd.DataFrame:
    """Tiros de campo de la temporada regular, limpios y geolocalizados.

    Descarta los tiros libres (su aportación ya entra por ``ft_rate``) y los
    tiros con coordenadas imposibles o geometría contradictoria (un triple
    anotado a 10 pies es un error de la fuente).
    """
    frames = [pd.read_parquet(path, columns=list(PBP_READ_COLUMNS)) for path in paths]
    pbp = pd.concat(frames, ignore_index=True)
    free_throws = pbp["type_text"].fillna("").str.startswith(FREE_THROW_PREFIX)
    shots = pbp[
        (pbp["season_type"] == REGULAR_SEASON_TYPE)
        & pbp["shooting_play"].fillna(False)
        & ~free_throws
    ].copy()

    on_court = (
        pd.to_numeric(shots["coordinate_x"], errors="coerce").abs() <= MAX_COORDINATE_X
    ) & (pd.to_numeric(shots["coordinate_y"], errors="coerce").abs() <= MAX_COORDINATE_Y)
    shots = shots[on_court]

    shots["distance"] = shot_distance(shots["coordinate_x"], shots["coordinate_y"])
    made = shots["scoring_play"].fillna(False)
    is_three = is_three_point_shot(
        shots["distance"], shots["coordinate_y"], shots["score_value"], made
    )
    plausible = (
        shots["distance"].notna()
        & (shots["distance"] <= MAX_SHOT_DISTANCE)
        & ~(is_three & (shots["distance"] < MIN_THREE_POINT_DISTANCE))
    )
    dropped = int((~plausible).sum())
    if dropped:
        LOGGER.info("pbp: %d tiros descartados por geometría imposible", dropped)

    shots, is_three, made = shots[plausible], is_three[plausible], made[plausible]
    value = np.where(is_three.to_numpy(), 3, 2)
    clean = pd.DataFrame(
        {
            "game_id": shots["game_id"].astype(str).to_numpy(),
            "season": shots["season"].astype(int).to_numpy(),
            "team_id": shots["team_id"].astype(int).to_numpy(),
            "is_three": is_three.to_numpy(),
            "distance": shots["distance"].to_numpy(),
            "family": classify_family(shots["type_text"]).to_numpy(),
            "band": distance_band(shots["distance"]).to_numpy(),
            "points": np.where(made.to_numpy(), value, 0),
        }
    )
    return clean.reset_index(drop=True)


def shot_value_table(shots: pd.DataFrame) -> pd.DataFrame:
    """Puntos esperados por intento en cada ``(banda, familia)``.

    Las celdas con pocos intentos heredan el valor de su banda de distancia:
    una familia rara a 30 pies no tiene muestra propia, pero la banda sí.
    """
    if shots.empty:
        return pd.DataFrame(columns=list(VALUE_TABLE_COLUMNS))
    cells = shots.groupby(["band", "family"], observed=True).agg(
        xpps=("points", "mean"), attempts=("points", "size")
    )
    bands = shots.groupby("band", observed=True)["points"].mean()
    table = cells.reset_index()
    scarce = table["attempts"] < MIN_CELL_ATTEMPTS
    table.loc[scarce, "xpps"] = table.loc[scarce, "band"].map(bands).to_numpy()
    return table[list(VALUE_TABLE_COLUMNS)]


def _previous_season_tables(shots: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Tabla de referencia de cada temporada, hecha con la anterior."""
    tables: dict[int, pd.DataFrame] = {}
    for season in sorted(shots["season"].unique()):
        previous = shots[shots["season"] == season - 1]
        if previous.empty:
            LOGGER.warning(
                "pbp: sin temporada anterior a %d; esa temporada queda sin "
                "calidad de tiro para no usar su propia referencia",
                season,
            )
            continue
        tables[int(season)] = shot_value_table(previous)
    return tables


def team_shot_quality(shots: pd.DataFrame) -> pd.DataFrame:
    """Calidad de tiro y acierto sobre lo esperado, por equipo y partido.

    ``shot_quality`` son los puntos por intento que valen los tiros elegidos a
    conversión de liga; ``shot_making`` es lo que el equipo saca por encima (o
    por debajo) de eso. ``rim_rate`` y ``mid_rate`` son la selección cruda:
    proporción de tiros al aro y de media distancia.

    Returns:
        Una fila por ``(game_id, team_id)``; las temporadas sin referencia
        anterior se omiten.
    """
    if shots.empty:
        return pd.DataFrame(columns=list(QUALITY_COLUMNS))

    pieces: list[pd.DataFrame] = []
    for season, table in _previous_season_tables(shots).items():
        season_shots = shots[shots["season"] == season].merge(
            table[["band", "family", "xpps"]], on=["band", "family"], how="left"
        )
        band_means = table.groupby("band", observed=True)["xpps"].mean()
        season_shots["xpps"] = season_shots["xpps"].fillna(
            season_shots["band"].map(band_means)
        )
        season_shots["is_rim"] = season_shots["distance"] <= RIM_BAND_LIMIT
        season_shots["is_mid"] = (season_shots["distance"] > MID_BAND_LOWER) & (
            season_shots["distance"] <= MID_BAND_UPPER
        )
        grouped = season_shots.groupby(["game_id", "team_id"], observed=True).agg(
            attempts=("points", "size"),
            shot_quality=("xpps", "mean"),
            actual=("points", "mean"),
            rim_rate=("is_rim", "mean"),
            mid_rate=("is_mid", "mean"),
        )
        grouped["shot_making"] = grouped["actual"] - grouped["shot_quality"]
        pieces.append(grouped.drop(columns=["actual"]).reset_index())

    if not pieces:
        return pd.DataFrame(columns=list(QUALITY_COLUMNS))
    return pd.concat(pieces, ignore_index=True)[list(QUALITY_COLUMNS)]


def attach_team_labels(quality: pd.DataFrame, paths: Sequence[Path]) -> pd.DataFrame:
    """Traduce ``team_id`` a la abreviatura canónica del proyecto."""
    from nba_sh.teams import is_valid_team, normalize_team

    if quality.empty:
        return pd.DataFrame(columns=["game_id", "team", *QUALITY_COLUMNS[2:]])
    frames = [
        pd.read_parquet(path, columns=list(ROSTER_READ_COLUMNS)) for path in paths
    ]
    rosters = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    pairs = pd.concat(
        [
            rosters[["game_id", "home_team_id", "home_team_abbrev"]].rename(
                columns={"home_team_id": "team_id", "home_team_abbrev": "team"}
            ),
            rosters[["game_id", "away_team_id", "away_team_abbrev"]].rename(
                columns={"away_team_id": "team_id", "away_team_abbrev": "team"}
            ),
        ],
        ignore_index=True,
    )
    pairs["game_id"] = pairs["game_id"].astype(str)
    pairs["team_id"] = pairs["team_id"].astype(int)
    merged = quality.merge(pairs, on=["game_id", "team_id"], how="left")
    # Los equipos del All-Star (LEB, SHQ...) no son de la liga: se descartan.
    merged["team"] = [
        normalize_team(label)
        if isinstance(label, str) and is_valid_team(label)
        else None
        for label in merged["team"]
    ]
    missing = int(merged["team"].isna().sum())
    if missing:
        LOGGER.info("pbp: %d filas descartadas por equipo no liguero", missing)
    return merged.dropna(subset=["team"]).drop(columns=["team_id"])


def load_shot_quality(
    seasons: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Calidad de tiro lista para el feature store, de caché o de la red.

    Returns:
        ``game_id``, ``team``, ``attempts``, ``shot_quality``, ``shot_making``,
        ``rim_rate``, ``mid_rate``.
    """
    paths = list(download_play_by_play(seasons, cache_dir=cache_dir).values())
    return attach_team_labels(team_shot_quality(load_shots(paths)), paths)


__all__ = [
    "CORNER_THREE_DISTANCE",
    "DISTANCE_EDGES",
    "HOOP_X",
    "MIN_CELL_ATTEMPTS",
    "QUALITY_COLUMNS",
    "SHOT_FAMILIES",
    "attach_team_labels",
    "classify_family",
    "distance_band",
    "is_three_point_shot",
    "download_play_by_play",
    "load_shot_quality",
    "load_shots",
    "THREE_POINT_ARC",
    "shot_distance",
    "shot_value_table",
    "team_shot_quality",
]
