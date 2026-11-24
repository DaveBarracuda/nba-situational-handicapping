"""Box scores completos de equipo y jugador (hoopR-nba-data).

Cubre los dos huecos que deja el scoreboard de ESPN
(:mod:`nba_sh.ingest.espn_source`):

1. **Posesiones exactas.** El scoreboard no publica pérdidas ni rebotes
   ofensivos, así que había que estimarlas; el box score de equipo sí los trae
   y permite aplicar la fórmula completa de la guía (§3.4):
   ``Poss = FGA + 0.44*FTA - OREB + TOV``.
2. **Disponibilidad y lesiones.** El box score de jugador trae minutos,
   ``plus_minus`` y ``did_not_play`` por partido, de donde se reconstruye quién
   faltó y cuánto pesaba, que es lo que pide la guía §5.3 ("minutos ponderados
   por net rating on/off", nunca "estrella fuera = -X puntos").

Fuente: volcados parquet de `sportsdataverse/hoopR-nba-data`, derivados de la
misma API de ESPN, así que los ``game_id`` coinciden con los del scoreboard.
Son 6 ficheros de ~0,7 MB para tres temporadas, frente a 3.692 peticiones al
endpoint ``summary``.

**Aviso de causalidad.** Los minutos y el on/off de cada ausente se calculan
solo con partidos **anteriores** (ventana móvil + encogimiento), pero *saber
quién faltó* sale del box score final. Equivale a suponer que conocías el
lineup al apostar: cierto para el informe oficial de lesiones (17:30 ET del día
previo), optimista para un scratch de última hora. Con un feed histórico de
informes de lesiones, sustituye :func:`build_availability` y nada más cambia.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from nba_sh.config import CONFIG, POSSESSIONS_SCALE, Config
from nba_sh.schemas import (
    INJURY_COLUMNS,
    STATUS_OUT,
    TEAM_BOX_COLUMNS,
    empty_frame,
)
from nba_sh.teams import is_valid_team, normalize_team

LOGGER = logging.getLogger(__name__)

HOOPR_BASE = "https://github.com/sportsdataverse/hoopR-nba-data/raw/main/nba"
TEAM_BOX = "team_box"
PLAYER_BOX = "player_box"
BOX_KINDS = (TEAM_BOX, PLAYER_BOX)
DEFAULT_CACHE_DIR = Path("data/raw/hoopr")
DEFAULT_TIMEOUT_SECONDS = 120
USER_AGENT = "nba-sh/1.0"
REGULAR_SEASON_TYPE = 2

FREE_THROW_POSSESSION_FACTOR = 0.44
REGULATION_MINUTES = 48.0
OVERTIME_MINUTES = 5.0
REGULATION_PERIODS = 4
MIN_OFF_COURT_MINUTES = 2.0

# Reconstrucción de la rotación y del on/off.
ROTATION_WINDOW_GAMES = 10
ROTATION_MIN_MINUTES = 15.0
ROTATION_MIN_APPEARANCES = 3
ON_OFF_SHRINKAGE_GAMES = 10.0
ON_OFF_CLIP_PER_100 = 30.0
RECONSTRUCTED_HOURS_BEFORE_TIP = 24.0

NORMALIZED_TEAM_COLUMNS = (
    "game_id",
    "team",
    "home_away",
    "score",
    "opponent_score",
    "field_goals_attempted",
    "free_throws_attempted",
    "offensive_rebounds",
    "turnovers",
    "possessions",
)
PLAYER_BOX_COLUMNS = (
    "game_id",
    "team",
    "player_id",
    "player_name",
    "minutes",
    "plus_minus",
    "did_not_play",
)


@dataclass(frozen=True)
class BoxScores:
    """Box scores normalizados de equipo y jugador.

    ``team`` trae lo justo para calcular posesiones; ``canonical_team`` es la
    tabla del :class:`~nba_sh.schemas.Dataset`, con el detalle de tiro que usa
    el perfil de estilo. **La del Dataset es la canónica**: pasar ``team`` deja
    las features de tiro a 0 sin que nada falle.
    """

    team: pd.DataFrame
    player: pd.DataFrame
    canonical_team: pd.DataFrame = field(
        default_factory=lambda: empty_frame(TEAM_BOX_COLUMNS)
    )

    @property
    def n_games(self) -> int:
        """Juegos cubiertos por el box score de equipo."""
        return int(self.team["game_id"].nunique())


def season_label_to_hoopr(season: str) -> int:
    """``"2023-24"`` -> ``2024`` (hoopR nombra por año de final)."""
    return int(season.split("-")[0]) + 1


def download_box_scores(
    seasons: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> dict[tuple[str, int], Path]:
    """Descarga (y cachea) los parquet de box scores de cada temporada.

    Args:
        seasons: temporadas en formato NBA (``"2023-24"``).
        cache_dir: directorio de caché; es data cruda, no se sobrescribe.
        timeout: timeout por petición.
        force: vuelve a descargar aunque exista el fichero.

    Returns:
        Mapa ``(kind, año_hoopr) -> ruta local``.
    """
    import requests  # import diferido: el paquete funciona sin red

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, int], Path] = {}
    for season in seasons:
        year = season_label_to_hoopr(season)
        for kind in BOX_KINDS:
            path = cache_dir / f"{kind}_{year}.parquet"
            if force or not path.exists():
                url = f"{HOOPR_BASE}/{kind}/parquet/{kind}_{year}.parquet"
                response = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=timeout
                )
                response.raise_for_status()
                path.write_bytes(response.content)
                LOGGER.info(
                    "hoopr: %s -> %s (%d bytes)", url, path, len(response.content)
                )
            paths[(kind, year)] = path
    return paths


def _read_concat(paths: Sequence[Path]) -> pd.DataFrame:
    """Lee y concatena parquet, quedándose con la temporada regular."""
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame[frame["season_type"] == REGULAR_SEASON_TYPE].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    return frame


def load_team_box(paths: Sequence[Path]) -> pd.DataFrame:
    """Normaliza el box score de equipo y calcula posesiones exactas."""
    frame = _read_concat(paths)
    frame["team"] = frame["team_abbreviation"].map(
        lambda value: normalize_team(value) if is_valid_team(value) else None
    )
    frame = frame.dropna(subset=["team"])
    normalized = pd.DataFrame(
        {
            "game_id": frame["game_id"].to_numpy(),
            "team": frame["team"].to_numpy(),
            "home_away": frame["team_home_away"].to_numpy(),
        }
    )
    for source, target in (
        ("field_goals_attempted", "field_goals_attempted"),
        ("free_throws_attempted", "free_throws_attempted"),
        ("offensive_rebounds", "offensive_rebounds"),
        ("total_turnovers", "turnovers"),
        ("team_score", "score"),
        ("opponent_team_score", "opponent_score"),
    ):
        normalized[target] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
    normalized["possessions"] = (
        normalized["field_goals_attempted"]
        + FREE_THROW_POSSESSION_FACTOR * normalized["free_throws_attempted"]
        - normalized["offensive_rebounds"]
        + normalized["turnovers"]
    )
    normalized = normalized.dropna(subset=["possessions"])
    return normalized[list(NORMALIZED_TEAM_COLUMNS)].reset_index(drop=True)


def load_player_box(paths: Sequence[Path]) -> pd.DataFrame:
    """Normaliza el box score de jugador (minutos y plus/minus numéricos)."""
    frame = _read_concat(paths)
    frame["team"] = frame["team_abbreviation"].map(
        lambda value: normalize_team(value) if is_valid_team(value) else None
    )
    frame = frame.dropna(subset=["team"])
    normalized = pd.DataFrame(
        {
            "game_id": frame["game_id"].to_numpy(),
            "team": frame["team"].to_numpy(),
            "player_id": frame["athlete_id"].astype(str).to_numpy(),
            "player_name": frame["athlete_display_name"].astype(str).to_numpy(),
            "minutes": pd.to_numeric(frame["minutes"], errors="coerce").to_numpy(),
            "plus_minus": pd.to_numeric(frame["plus_minus"], errors="coerce").to_numpy(),
            "did_not_play": frame["did_not_play"].fillna(False).astype(bool).to_numpy(),
        }
    )
    return normalized[list(PLAYER_BOX_COLUMNS)].reset_index(drop=True)


def load_box_scores(
    seasons: Sequence[str],
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> BoxScores:
    """Descarga si hace falta y devuelve los box scores normalizados."""
    paths = download_box_scores(seasons, cache_dir=cache_dir, timeout=timeout)
    team_paths = [path for (kind, _), path in paths.items() if kind == TEAM_BOX]
    player_paths = [path for (kind, _), path in paths.items() if kind == PLAYER_BOX]
    team = load_team_box(team_paths)
    box = BoxScores(
        team=team,
        player=load_player_box(player_paths),
        canonical_team=canonical_team_box(team_paths, team),
    )
    LOGGER.info("hoopr: %d juegos, %d filas de jugador", box.n_games, len(box.player))
    return box


def canonical_team_box(paths: Sequence[Path], team_box: pd.DataFrame) -> pd.DataFrame:
    """Tabla ``team_box`` canónica del :class:`~nba_sh.schemas.Dataset`.

    Añade al box normalizado el detalle de tiro que usa el perfil de estilo
    (guía 5.4): tiros de 3, puntos en la pintura, faltas y contraataque.

    Args:
        paths: parquet de box score de equipo ya descargados.
        team_box: salida de :func:`load_team_box` (aporta las posesiones).
    """
    frame = _read_concat(paths)
    frame["team"] = frame["team_abbreviation"].map(
        lambda value: normalize_team(value) if is_valid_team(value) else None
    )
    frame = frame.dropna(subset=["team"]).drop(columns=["turnovers"], errors="ignore")
    frame = frame.rename(
        columns={
            "team_score": "score",
            "opponent_team_score": "opponent_score",
            "total_turnovers": "turnovers",
        }
    )
    frame["possessions"] = frame["game_id"].map(
        exact_possessions(team_box)
    )
    for column in TEAM_BOX_COLUMNS:
        if column not in ("game_id", "team"):
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame[list(TEAM_BOX_COLUMNS)].reset_index(drop=True)


def exact_possessions(team_box: pd.DataFrame) -> pd.Series:
    """Posesiones del partido: media de las de ambos equipos."""
    return team_box.groupby("game_id")["possessions"].mean()


def apply_exact_possessions(
    games: pd.DataFrame, team_box: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Sustituye las posesiones estimadas por las exactas donde existan.

    Returns:
        La tabla de juegos y cuántas filas se actualizaron.
    """
    exact = exact_possessions(team_box)
    updated = games.copy()
    mapped = updated["game_id"].map(exact)
    replaced = int(mapped.notna().sum())
    updated["possessions"] = mapped.fillna(updated["possessions"]).astype(float)
    LOGGER.info("posesiones exactas aplicadas a %d/%d juegos", replaced, len(updated))
    return updated, replaced


def _game_minutes(periods: float) -> float:
    """Minutos de juego, contando prórrogas."""
    extra = max(float(periods) - REGULATION_PERIODS, 0.0)
    return REGULATION_MINUTES + OVERTIME_MINUTES * extra


def player_game_impact(games: pd.DataFrame, player_box: pd.DataFrame) -> pd.DataFrame:
    """On/off por jugador y partido, en puntos por 100 posesiones.

    ``plus_minus`` es el margen acumulado con el jugador en pista (verificado:
    la suma por equipo-partido es exactamente 5x el margen). Con eso:

        net_on  = pm / minutos_jugados          (por 48 min)
        net_off = (margen - pm) / minutos_fuera (por 48 min)
        on_off  = net_on - net_off              -> se escala a 100 posesiones
    """
    context = games[
        [
            "game_id",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "periods",
            "possessions",
        ]
    ].copy()
    context["margin_home"] = pd.to_numeric(games["home_score"], errors="coerce") - (
        pd.to_numeric(games["away_score"], errors="coerce")
    )
    merged = player_box.merge(context, on="game_id", how="inner")
    merged = merged[merged["minutes"].notna() & (merged["minutes"] > 0.0)].copy()
    if merged.empty:
        return merged.assign(on_off_per_100=[])

    is_home = merged["team"].to_numpy() == merged["home_team"].to_numpy()
    margin = np.where(
        is_home, merged["margin_home"].to_numpy(), -merged["margin_home"].to_numpy()
    )
    game_minutes = np.array(
        [_game_minutes(value) for value in merged["periods"].to_numpy(dtype=float)]
    )
    minutes = merged["minutes"].to_numpy(dtype=float)
    plus_minus = merged["plus_minus"].fillna(0.0).to_numpy(dtype=float)
    off_minutes = np.maximum(game_minutes - minutes, MIN_OFF_COURT_MINUTES)

    net_on = plus_minus / minutes * REGULATION_MINUTES
    net_off = (margin - plus_minus) / off_minutes * REGULATION_MINUTES
    possessions_per_48 = (
        merged["possessions"].to_numpy(dtype=float) * REGULATION_MINUTES / game_minutes
    )
    on_off = (net_on - net_off) * POSSESSIONS_SCALE / possessions_per_48
    merged["on_off_per_100"] = np.clip(on_off, -ON_OFF_CLIP_PER_100, ON_OFF_CLIP_PER_100)

    return (
        merged[
            [
                "game_id",
                "game_date",
                "season",
                "team",
                "player_id",
                "minutes",
                "on_off_per_100",
            ]
        ]
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )


def _long_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Calendario en formato equipo-juego, ordenado por equipo y fecha."""
    schedule = games[["game_id", "game_date", "season", "home_team", "away_team"]]
    sides = [
        schedule.rename(columns={column: "team"})[
            ["game_id", "game_date", "season", "team"]
        ]
        for column in ("home_team", "away_team")
    ]
    return pd.concat(sides, ignore_index=True).sort_values(
        ["team", "game_date", "game_id"]
    )


def build_availability(
    games: pd.DataFrame, player_box: pd.DataFrame, *, config: Config = CONFIG
) -> pd.DataFrame:
    """Reconstruye las ausencias de rotación con pesos causales.

    Para cada equipo y partido, en orden cronológico:

    1. la rotación son los jugadores que aparecieron en los últimos
       ``ROTATION_WINDOW_GAMES`` partidos del equipo, con al menos
       ``ROTATION_MIN_MINUTES`` de media y ``ROTATION_MIN_APPEARANCES`` apariciones;
    2. los de la rotación que no juegan ese partido se registran como baja;
    3. sus minutos y su on/off salen **solo de partidos anteriores**, con
       encogimiento hacia 0 para no sobre-reaccionar a muestras pequeñas.

    El estado se reinicia en cada temporada (las plantillas cambian).

    Returns:
        Tabla con el esquema ``INJURY_COLUMNS``.
    """
    del config  # los umbrales de reconstrucción son propios de la fuente
    impact = player_game_impact(games, player_box)
    if impact.empty:
        return empty_frame(INJURY_COLUMNS)

    appearances: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in impact.itertuples(index=False):
        appearances.setdefault((row.team, row.game_id), []).append(
            {
                "player_id": row.player_id,
                "minutes": float(row.minutes),
                "on_off": float(row.on_off_per_100),
            }
        )

    rows: list[dict[str, object]] = []
    for team, team_games in _long_schedule(games).groupby("team", sort=False):
        history: dict[str, dict[str, float]] = {}
        last_seen: dict[str, int] = {}
        current_season: str | None = None
        for index, game in enumerate(team_games.itertuples(index=False)):
            if current_season is not None and game.season != current_season:
                history, last_seen = {}, {}
            current_season = game.season

            played_rows = appearances.get((team, game.game_id), [])
            played_ids = {entry["player_id"] for entry in played_rows}
            for player_id, stats in history.items():
                if player_id in played_ids:
                    continue
                games_since = index - last_seen.get(player_id, -ROTATION_WINDOW_GAMES)
                if games_since > ROTATION_WINDOW_GAMES:
                    continue
                if stats["appearances"] < ROTATION_MIN_APPEARANCES:
                    continue
                minutes_pg = stats["minutes_sum"] / stats["appearances"]
                if minutes_pg < ROTATION_MIN_MINUTES:
                    continue
                rows.append(
                    {
                        "game_id": game.game_id,
                        "team": team,
                        "player_id": player_id,
                        "minutes_pg": round(minutes_pg, 2),
                        "on_off_net": round(
                            stats["on_off_sum"]
                            / (stats["appearances"] + ON_OFF_SHRINKAGE_GAMES),
                            3,
                        ),
                        "status": STATUS_OUT,
                        "hours_before_tip": RECONSTRUCTED_HOURS_BEFORE_TIP,
                    }
                )

            for entry in played_rows:
                player_id = str(entry["player_id"])
                stats = history.setdefault(
                    player_id,
                    {"appearances": 0.0, "minutes_sum": 0.0, "on_off_sum": 0.0},
                )
                stats["appearances"] += 1.0
                stats["minutes_sum"] += float(entry["minutes"])
                stats["on_off_sum"] += float(entry["on_off"])
                last_seen[player_id] = index

    if not rows:
        return empty_frame(INJURY_COLUMNS)
    availability = pd.DataFrame(rows, columns=list(INJURY_COLUMNS))
    LOGGER.info(
        "disponibilidad: %d ausencias de rotación en %d juegos",
        len(availability),
        availability["game_id"].nunique(),
    )
    return availability


__all__ = [
    "BOX_KINDS",
    "DEFAULT_CACHE_DIR",
    "HOOPR_BASE",
    "ON_OFF_CLIP_PER_100",
    "PLAYER_BOX",
    "RECONSTRUCTED_HOURS_BEFORE_TIP",
    "ROTATION_MIN_APPEARANCES",
    "ROTATION_MIN_MINUTES",
    "ROTATION_WINDOW_GAMES",
    "NORMALIZED_TEAM_COLUMNS",
    "TEAM_BOX",
    "BoxScores",
    "apply_exact_possessions",
    "build_availability",
    "canonical_team_box",
    "download_box_scores",
    "exact_possessions",
    "load_box_scores",
    "load_player_box",
    "load_team_box",
    "player_game_impact",
    "season_label_to_hoopr",
]
