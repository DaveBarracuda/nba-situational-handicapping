"""Ingesta real alternativa: API pública de ESPN (guía 3.1).

La guía lista la API de ESPN como fuente de scores y calendarios ("no
documentada, estable"). Es la vía de respaldo cuando ``stats.nba.com`` no es
alcanzable (bloquea muchos entornos) o basketball-reference responde 403.

Ventaja práctica: el endpoint de scoreboard acepta un **mes completo**
(``dates=YYYYMM``), así que una temporada son 8 peticiones en vez de 170.

Limitación honesta: ESPN no publica pérdidas ni rebotes ofensivos por equipo,
que son dos de los cuatro términos de la fórmula de posesiones. Aquí se
estiman (ver :func:`estimate_possessions_from_shooting`): el ritmo relativo
entre partidos se conserva porque lo dominan FGA y FTA, pero si necesitas
posesiones exactas usa ``nba_api`` o el box score completo de ESPN
(``summary?event=<id>``, una petición por partido).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nba_sh.schemas import (
    GAME_COLUMNS,
    INJURY_COLUMNS,
    ODDS_COLUMNS,
    REFEREE_COLUMNS,
    SEASON_TYPE_REGULAR,
    TIMING_CLOSE,
    Dataset,
    empty_frame,
)
from nba_sh.teams import is_valid_team, normalize_team

LOGGER = logging.getLogger(__name__)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
# Único endpoint que publica el cuadro arbitral: una petición por partido.
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
USER_AGENT = "nba-sh/1.0"
EVENTS_PER_REQUEST = 1_000
DEFAULT_TIMEOUT_SECONDS = 30
REQUEST_PAUSE_SECONDS = 0.4
MARKET_TIMEZONE = "America/New_York"

REGULAR_SEASON_TYPE = 2
STATUS_FINAL = "STATUS_FINAL"
STATUS_SCHEDULED = "STATUS_SCHEDULED"
# Posesiones de una jornada sin jugar: la media de la liga. El feature
# store solo la usa como respaldo si no hay media móvil del equipo.
SLATE_POSSESSIONS = 101.5
REGULATION_PERIODS = 4
MAX_OFFICIALS = 3
# Partidos minimos de un colegiado antes de fiarse de su perfil.
MIN_OFFICIAL_GAMES = 10
SEASON_MONTHS_FIRST_YEAR = (10, 11, 12)
SEASON_MONTHS_SECOND_YEAR = (1, 2, 3, 4, 5, 6)
REQUIRED_STATISTICS = (
    "fieldGoalsAttempted",
    "fieldGoalsMade",
    "freeThrowsAttempted",
)

# Términos de la fórmula de posesiones que ESPN no publica.
FREE_THROW_POSSESSION_FACTOR = 0.44
OFFENSIVE_REBOUND_RATE = 0.23
LEAGUE_AVERAGE_TURNOVERS = 13.3


def season_months(season: str) -> tuple[str, ...]:
    """Meses ``YYYYMM`` que cubren una temporada (``"2023-24"``)."""
    start_year = int(season.split("-")[0])
    first = [f"{start_year}{month:02d}" for month in SEASON_MONTHS_FIRST_YEAR]
    second = [f"{start_year + 1}{month:02d}" for month in SEASON_MONTHS_SECOND_YEAR]
    return tuple(first + second)


def estimate_possessions_from_shooting(
    field_goals_attempted: float,
    field_goals_made: float,
    free_throws_attempted: float,
) -> float:
    """Posesiones estimadas sin pérdidas ni rebotes ofensivos publicados.

    Parte de ``Poss = FGA + 0.44*FTA - OREB + TOV`` y sustituye:

    * ``OREB`` por la tasa de liga sobre los tiros fallados (~23%),
    * ``TOV`` por la media de liga (~13.3 por equipo y partido).

    Es la aproximación estándar cuando solo hay datos de tiro.
    """
    missed_shots = max(float(field_goals_attempted) - float(field_goals_made), 0.0)
    return float(
        float(field_goals_attempted)
        + FREE_THROW_POSSESSION_FACTOR * float(free_throws_attempted)
        - OFFENSIVE_REBOUND_RATE * missed_shots
        + LEAGUE_AVERAGE_TURNOVERS
    )


def fetch_month(
    month: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> list[dict[str, Any]]:
    """Descarga los partidos de un mes (``YYYYMM``) y devuelve sus eventos."""
    import requests  # import diferido: el paquete funciona sin red

    response = requests.get(
        SCOREBOARD_URL,
        params={"dates": month, "limit": EVENTS_PER_REQUEST},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    events = response.json().get("events", [])
    LOGGER.info("espn: %s -> %d eventos", month, len(events))
    return events


def _team_statistics(competitor: Mapping[str, Any]) -> dict[str, float]:
    """Estadísticas de equipo de un competidor, como números."""
    statistics: dict[str, float] = {}
    for entry in competitor.get("statistics") or ():
        try:
            statistics[str(entry.get("name"))] = float(entry.get("displayValue"))
        except (TypeError, ValueError):
            continue
    return statistics


def _periods(competition: Mapping[str, Any], competitor: Mapping[str, Any]) -> int:
    """Periodos jugados (incluye prórrogas)."""
    status_period = (competition.get("status") or {}).get("period")
    linescores = competitor.get("linescores") or ()
    return int(max(status_period or 0, len(linescores), REGULATION_PERIODS))


def events_to_games(events: Iterable[Mapping[str, Any]], *, season: str) -> pd.DataFrame:
    """Convierte eventos de ESPN en la tabla canónica de juegos.

    Solo conserva partidos de temporada regular ya finalizados y con
    estadísticas de tiro de ambos equipos.
    """
    rows: list[dict[str, object]] = []
    for event in events:
        if int((event.get("season") or {}).get("type", 0)) != REGULAR_SEASON_TYPE:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = ((competition.get("status") or {}).get("type") or {}).get("name")
        if status != STATUS_FINAL:
            continue

        sides = {
            str(competitor.get("homeAway")): competitor
            for competitor in competition.get("competitors") or ()
        }
        if set(sides) != {"home", "away"}:
            continue

        home, away = sides["home"], sides["away"]
        home_code = home["team"].get("abbreviation")
        away_code = away["team"].get("abbreviation")
        if not (is_valid_team(home_code) and is_valid_team(away_code)):
            LOGGER.warning("equipos no reconocidos: %s vs %s", home_code, away_code)
            continue

        home_stats = _team_statistics(home)
        away_stats = _team_statistics(away)
        if any(
            key not in home_stats or key not in away_stats
            for key in REQUIRED_STATISTICS
        ):
            continue

        tipoff_utc = pd.Timestamp(competition.get("date") or event.get("date"))
        if tipoff_utc.tzinfo is None:
            tipoff_utc = tipoff_utc.tz_localize("UTC")
        game_date = tipoff_utc.tz_convert(MARKET_TIMEZONE).normalize().tz_localize(None)

        possessions = (
            estimate_possessions_from_shooting(
                home_stats["fieldGoalsAttempted"],
                home_stats["fieldGoalsMade"],
                home_stats["freeThrowsAttempted"],
            )
            + estimate_possessions_from_shooting(
                away_stats["fieldGoalsAttempted"],
                away_stats["fieldGoalsMade"],
                away_stats["freeThrowsAttempted"],
            )
        ) / 2.0

        rows.append(
            {
                "game_id": str(event.get("id")),
                "season": season,
                "season_type": SEASON_TYPE_REGULAR,
                "game_date": game_date,
                "tipoff_utc": tipoff_utc,
                "home_team": normalize_team(home_code),
                "away_team": normalize_team(away_code),
                "home_score": int(float(home.get("score"))),
                "away_score": int(float(away.get("score"))),
                "periods": _periods(competition, home),
                "possessions": possessions,
                "crew_id": None,
            }
        )

    games = pd.DataFrame(rows, columns=list(GAME_COLUMNS))
    if games.empty:
        return games
    return (
        games.drop_duplicates(subset=["game_id"])
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )


def fetch_season(
    season: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> pd.DataFrame:
    """Descarga una temporada regular completa, mes a mes."""
    frames: list[pd.DataFrame] = []
    for month in season_months(season):
        monthly = events_to_games(fetch_month(month, timeout=timeout), season=season)
        if not monthly.empty:
            frames.append(monthly)
        if pause_seconds:
            time.sleep(pause_seconds)
    if not frames:
        return empty_frame(GAME_COLUMNS)
    games = pd.concat(frames, ignore_index=True)
    games = games.drop_duplicates(subset=["game_id"])
    LOGGER.info("espn: temporada %s -> %d juegos", season, len(games))
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def events_to_slate(
    events: Iterable[Mapping[str, Any]], *, season: str
) -> pd.DataFrame:
    """Convierte eventos de ESPN en la jornada **por jugar**.

    Es la contraparte de :func:`events_to_games`: ahí se exigen partidos
    terminados con estadísticas; aquí se quieren justo los que aún no se han
    jugado, sin marcador, para poder predecirlos y apostarlos en papel.

    Las columnas de resultado (``home_score``, ``away_score``) quedan a NaN:
    son el objetivo, no una entrada. ``periods`` se pone en los 4 de
    reglamento y ``possessions`` en la media de liga, ambos como respaldo.
    """
    rows: list[dict[str, object]] = []
    for event in events:
        if int((event.get("season") or {}).get("type", 0)) != REGULAR_SEASON_TYPE:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        status = (competition.get("status") or {}).get("type") or {}
        if status.get("completed") or status.get("name") == STATUS_FINAL:
            continue

        sides = {
            str(competitor.get("homeAway")): competitor
            for competitor in competition.get("competitors") or ()
        }
        if set(sides) != {"home", "away"}:
            continue
        home_code = (sides["home"].get("team") or {}).get("abbreviation")
        away_code = (sides["away"].get("team") or {}).get("abbreviation")
        if not (is_valid_team(home_code) and is_valid_team(away_code)):
            LOGGER.warning("equipos no reconocidos: %s vs %s", home_code, away_code)
            continue

        tipoff_utc = pd.Timestamp(competition.get("date") or event.get("date"))
        if tipoff_utc.tzinfo is None:
            tipoff_utc = tipoff_utc.tz_localize("UTC")
        rows.append(
            {
                "game_id": str(event.get("id")),
                "season": season,
                "season_type": SEASON_TYPE_REGULAR,
                "game_date": tipoff_utc.tz_convert(MARKET_TIMEZONE)
                .normalize()
                .tz_localize(None),
                "tipoff_utc": tipoff_utc.tz_convert("UTC").tz_localize(None),
                "home_team": normalize_team(home_code),
                "away_team": normalize_team(away_code),
                "home_score": np.nan,
                "away_score": np.nan,
                "periods": REGULATION_PERIODS,
                "possessions": SLATE_POSSESSIONS,
                "crew_id": None,
            }
        )

    slate = pd.DataFrame(rows, columns=list(GAME_COLUMNS))
    if slate.empty:
        return slate
    return slate.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def fetch_slate(
    day: pd.Timestamp | str,
    *,
    season: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Jornada por jugar de un día concreto (una petición).

    Args:
        day: fecha en hora del mercado (ET).
        season: etiqueta de temporada (``"2026-27"``).
        timeout: timeout de la petición.

    Returns:
        Tabla de juegos con el esquema canónico y marcadores a NaN.
    """
    import requests  # import diferido: el paquete funciona sin red

    target = pd.Timestamp(day)
    response = requests.get(
        SCOREBOARD_URL,
        params={"dates": target.strftime("%Y%m%d"), "limit": EVENTS_PER_REQUEST},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    events = response.json().get("events") or []
    slate = events_to_slate(events, season=season)
    # ESPN agrupa por fecha UTC: un partido de la noche anterior puede venir
    # en la misma respuesta. Se filtra por la fecha de mercado.
    if not slate.empty:
        slate = slate[slate["game_date"] == target.normalize()].reset_index(drop=True)
    LOGGER.info("espn: jornada del %s -> %d juegos por jugar", target.date(), len(slate))
    return slate


PICKCENTER_PROVIDER = "DraftKings"


def _number(value: object) -> float:
    """Número de la fuente, o NaN si no lo es."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def pickcenter_to_odds_row(
    game_id: str, pickcenter: Sequence[Mapping[str, Any]], *, provider: str
) -> dict[str, object] | None:
    """Fila de cuotas desde el ``pickcenter`` de un partido.

    ESPN publica ahí la línea del libro **ya jugado el partido**, que es la vía
    para reconstruir un cierre que no se capturó en vivo. Su campo ``spread``
    es el del equipo **local** —verificado contra 10 partidos: MIN favorito da
    −10.5 y CHA no favorito da +2.5—, el mismo convenio que ``spread_home``.
    """
    entry = next(
        (
            item
            for item in pickcenter
            if str((item.get("provider") or {}).get("name")) == provider
        ),
        None,
    )
    if entry is None:
        return None
    home_odds = entry.get("homeTeamOdds") or {}
    away_odds = entry.get("awayTeamOdds") or {}
    return {
        "game_id": str(game_id),
        "book": provider,
        "timing": TIMING_CLOSE,
        "ml_home": _number(home_odds.get("moneyLine")),
        "ml_away": _number(away_odds.get("moneyLine")),
        "spread_home": _number(entry.get("spread")),
        "spread_home_price": _number(home_odds.get("spreadOdds")),
        "spread_away_price": _number(away_odds.get("spreadOdds")),
        "total_line": _number(entry.get("overUnder")),
        "over_price": _number(entry.get("overOdds")),
        "under_price": _number(entry.get("underOdds")),
    }


def fetch_closing_lines(
    game_ids: Sequence[str],
    *,
    provider: str = PICKCENTER_PROVIDER,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> pd.DataFrame:
    """Reconstruye la línea de cierre de partidos ya jugados.

    Es el rescate para cuando la captura en vivo no ocurrió: sin ella esas
    apuestas se quedan sin CLV para siempre. **No es lo mismo que una captura
    propia a T−30**: ESPN no documenta el momento en que fijó estos números, y
    por eso quien la guarda debe marcarla con su propia fuente.

    Args:
        game_ids: partidos a reconstruir.
        provider: libro del ``pickcenter`` (por defecto DraftKings).
        timeout: timeout por petición.
        pause_seconds: pausa entre peticiones.

    Returns:
        DataFrame con el esquema ``ODDS_COLUMNS``; los partidos sin ese libro
        no aparecen.
    """
    import requests  # import diferido: el paquete funciona sin red

    rows: list[dict[str, object]] = []
    for game_id in game_ids:
        try:
            response = requests.get(
                SUMMARY_URL,
                params={"event": str(game_id)},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            response.raise_for_status()
            centers = response.json().get("pickcenter") or []
        except (OSError, ValueError) as error:
            LOGGER.warning("sin pickcenter para el juego %s: %s", game_id, error)
            centers = []
        row = pickcenter_to_odds_row(game_id, centers, provider=provider)
        if row is None:
            LOGGER.warning("el juego %s no tiene línea de %s", game_id, provider)
        else:
            rows.append(row)
        if pause_seconds:
            time.sleep(pause_seconds)
    if not rows:
        return empty_frame(ODDS_COLUMNS)
    LOGGER.info("espn: %d cierres reconstruidos de %s", len(rows), provider)
    return pd.DataFrame(rows, columns=list(ODDS_COLUMNS))


def fetch_officials(
    game_ids: Sequence[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> pd.DataFrame:
    """Asignaciones arbitrales, una petición por partido (guía 5.6).

    Es la única vía: ni hoopR ni el scoreboard publican árbitros, y
    stats.nba.com no es alcanzable. Devuelve ``game_id``, ``official_1..3`` y
    ``n_officials``.
    """
    import requests  # import diferido: el paquete funciona sin red

    rows: list[dict[str, object]] = []
    for game_id in game_ids:
        try:
            response = requests.get(
                SUMMARY_URL,
                params={"event": str(game_id)},
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            response.raise_for_status()
            officials = (response.json().get("gameInfo") or {}).get("officials") or []
            names = [
                str(item.get("fullName", "")).strip()
                for item in officials
                if str(item.get("fullName", "")).strip()
            ][:MAX_OFFICIALS]
        except (OSError, ValueError) as error:
            # requests.RequestException hereda de OSError y JSONDecodeError de
            # ValueError: la red y el parseo no cortan la ingesta, un error de
            # programación sí.
            LOGGER.warning("sin árbitros para el juego %s: %s", game_id, error)
            names = []
        row: dict[str, object] = {"game_id": str(game_id), "n_officials": len(names)}
        for index in range(MAX_OFFICIALS):
            row[f"official_{index + 1}"] = names[index] if index < len(names) else None
        rows.append(row)
        if pause_seconds:
            time.sleep(pause_seconds)
    return pd.DataFrame(rows)


def build_referee_tables(
    games: pd.DataFrame, officials: pd.DataFrame, team_box: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfil arbitral causal por partido (guía 5.6).

    El perfil de un cuadro es la media de los perfiles de sus tres colegiados,
    y el de cada colegiado son sus **partidos anteriores**: tiros libres y
    faltas por encuentro. Cada partido recibe su propio ``crew_id`` porque la
    composición cambia cada noche.

    Returns:
        ``(games con crew_id, tabla referees)``.
    """
    if officials is None or officials.empty or team_box is None or team_box.empty:
        return games.copy(), empty_frame(REFEREE_COLUMNS)

    per_game = team_box.groupby("game_id").agg(
        fta=("free_throws_attempted", "sum"), fouls=("fouls", "sum")
    )
    context = (
        games[["game_id", "game_date"]]
        .merge(officials, on="game_id", how="left")
        .join(per_game, on="game_id")
        .sort_values(["game_date", "game_id"])
    )

    history: dict[str, dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for record in context.itertuples(index=False):
        crew = [
            getattr(record, f"official_{index + 1}") for index in range(MAX_OFFICIALS)
        ]
        crew = [name for name in crew if isinstance(name, str) and name]
        seen = [
            history[name]
            for name in crew
            if name in history and history[name]["games"] >= MIN_OFFICIAL_GAMES
        ]
        crew_id = f"{'|'.join(sorted(crew))}@{record.game_id}" if crew else None
        row: dict[str, object] = {"game_id": record.game_id, "crew_id": crew_id}
        if crew_id is not None and seen:
            row["crew_fta_rate"] = float(
                np.mean([item["fta"] / item["games"] for item in seen])
            )
            row["crew_foul_rate"] = float(
                np.mean([item["fouls"] / item["games"] for item in seen])
            )
        rows.append(row)
        for name in crew:
            state = history.setdefault(name, {"games": 0.0, "fta": 0.0, "fouls": 0.0})
            state["games"] += 1.0
            state["fta"] += float(record.fta)
            state["fouls"] += float(record.fouls)

    crews = pd.DataFrame(rows)
    for column in ("crew_fta_rate", "crew_foul_rate"):
        if column not in crews.columns:
            crews[column] = np.nan
    updated = games.drop(columns=["crew_id"], errors="ignore").merge(
        crews[["game_id", "crew_id"]], on="game_id", how="left"
    )
    referees = (
        crews.dropna(subset=["crew_id", "crew_fta_rate"])[
            ["crew_id", "crew_foul_rate", "crew_fta_rate"]
        ]
        .drop_duplicates(subset=["crew_id"])
        .reset_index(drop=True)
    )
    LOGGER.info(
        "árbitros: %d cuadros con historial sobre %d partidos", len(referees), len(games)
    )
    return updated, referees


def fetch_dataset(
    seasons: Sequence[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    pause_seconds: float = REQUEST_PAUSE_SECONDS,
) -> Dataset:
    """Descarga varias temporadas y devuelve un :class:`Dataset`.

    Cuotas, lesiones y árbitros quedan vacíos: cada uno tiene su fuente
    (:mod:`nba_sh.ingest.odds_source`, :mod:`nba_sh.ingest.injury_source`).
    """
    frames = [
        season_games
        for season_games in (
            fetch_season(season, timeout=timeout, pause_seconds=pause_seconds)
            for season in seasons
        )
        if not season_games.empty
    ]
    games = pd.concat(frames, ignore_index=True) if frames else empty_frame(GAME_COLUMNS)
    return Dataset(
        games=games,
        odds=empty_frame(ODDS_COLUMNS),
        injuries=empty_frame(INJURY_COLUMNS),
        referees=empty_frame(REFEREE_COLUMNS),
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "FREE_THROW_POSSESSION_FACTOR",
    "LEAGUE_AVERAGE_TURNOVERS",
    "OFFENSIVE_REBOUND_RATE",
    "SCOREBOARD_URL",
    "PICKCENTER_PROVIDER",
    "SLATE_POSSESSIONS",
    "STATUS_SCHEDULED",
    "SUMMARY_URL",
    "estimate_possessions_from_shooting",
    "events_to_games",
    "events_to_slate",
    "build_referee_tables",
    "fetch_officials",
    "fetch_dataset",
    "fetch_closing_lines",
    "fetch_month",
    "fetch_slate",
    "pickcenter_to_odds_row",
    "fetch_season",
    "season_months",
]
