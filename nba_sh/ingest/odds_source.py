"""Ingesta de líneas reales: The Odds API (guía 3.2).

Guarda **apertura y cierre** de cada línea: la CLV es la métrica de calidad
más honesta del sistema, y sin los dos extremos no se puede calcular.

La clave de API se lee de la variable de entorno ``ODDS_API_KEY``; nunca se
escribe en el código ni se registra en logs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from nba_sh.odds_utils import round_to_half
from nba_sh.schemas import ODDS_COLUMNS, TIMING_PREGAME, empty_frame
from nba_sh.teams import is_valid_team, normalize_team

LOGGER = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY_ENV = "ODDS_API_KEY"
SPORT_KEY = "basketball_nba"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_TIMEOUT_SECONDS = 30

MARKET_H2H = "h2h"
MARKET_SPREADS = "spreads"
MARKET_TOTALS = "totals"
OVER_LABEL = "Over"
UNDER_LABEL = "Under"


def api_key(explicit: str | None = None) -> str:
    """Devuelve la clave de API, del argumento o del entorno.

    Raises:
        RuntimeError: si no hay clave disponible.
    """
    key = explicit or os.environ.get(ODDS_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"falta la clave de The Odds API: exporta {ODDS_API_KEY_ENV}")
    return key


def fetch_odds_payload(
    *,
    key: str | None = None,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Descarga las cuotas vigentes de la NBA.

    Returns:
        La respuesta JSON cruda (una entrada por partido).
    """
    import requests  # import diferido: el paquete funciona sin red

    response = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": api_key(key),
            "regions": regions,
            "markets": markets,
            "oddsFormat": DEFAULT_ODDS_FORMAT,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    LOGGER.info("the-odds-api: %d partidos con cuotas", len(payload))
    return payload


def _outcome_map(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Indexa las salidas de un mercado por nombre."""
    return {str(outcome.get("name")): outcome for outcome in outcomes}


def payload_to_odds(
    payload: Sequence[Mapping[str, Any]],
    *,
    timing: str = TIMING_PREGAME,
    bookmaker_key: str | None = None,
    game_id_by_teams: Mapping[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Convierte la respuesta de The Odds API al esquema canónico.

    Args:
        payload: respuesta JSON de :func:`fetch_odds_payload`.
        timing: momento de captura (``open``, ``t_minus_60`` o ``close``).
        bookmaker_key: libro concreto; por defecto, el primero disponible.
        game_id_by_teams: mapa ``(home, away) -> game_id`` para casar con la
            tabla de juegos; si falta, se genera un id sintético.

    Returns:
        DataFrame con el esquema ``ODDS_COLUMNS``.
    """
    rows: list[dict[str, object]] = []
    for event in payload:
        home_raw, away_raw = event.get("home_team"), event.get("away_team")
        if not (is_valid_team(home_raw) and is_valid_team(away_raw)):
            LOGGER.warning(
                "evento con equipos no reconocidos: %s vs %s", home_raw, away_raw
            )
            continue
        home, away = normalize_team(home_raw), normalize_team(away_raw)
        bookmakers = event.get("bookmakers") or []
        if bookmaker_key is not None:
            bookmakers = [book for book in bookmakers if book.get("key") == bookmaker_key]
        if not bookmakers:
            continue
        book = bookmakers[0]
        markets = {market.get("key"): market for market in book.get("markets", [])}

        head_to_head = _outcome_map(markets.get(MARKET_H2H, {}).get("outcomes", []))
        spreads = _outcome_map(markets.get(MARKET_SPREADS, {}).get("outcomes", []))
        totals = _outcome_map(markets.get(MARKET_TOTALS, {}).get("outcomes", []))
        if not (head_to_head and spreads and totals):
            continue
        if OVER_LABEL not in totals or UNDER_LABEL not in totals:
            continue
        if str(home_raw) not in head_to_head or str(home_raw) not in spreads:
            continue

        commence = pd.Timestamp(event.get("commence_time"))
        mapped_id = (
            game_id_by_teams.get((home, away)) if game_id_by_teams is not None else None
        )
        game_id = mapped_id or f"{commence.strftime('%Y%m%d')}-{away}-{home}"

        rows.append(
            {
                "game_id": str(game_id),
                "book": str(book.get("key")),
                "timing": timing,
                "ml_home": float(head_to_head[str(home_raw)]["price"]),
                "ml_away": float(head_to_head[str(away_raw)]["price"]),
                "spread_home": round_to_half(float(spreads[str(home_raw)]["point"])),
                "spread_home_price": float(spreads[str(home_raw)]["price"]),
                "spread_away_price": float(spreads[str(away_raw)]["price"]),
                "total_line": round_to_half(float(totals[OVER_LABEL]["point"])),
                "over_price": float(totals[OVER_LABEL]["price"]),
                "under_price": float(totals[UNDER_LABEL]["price"]),
            }
        )
    if not rows:
        return empty_frame(ODDS_COLUMNS)
    return pd.DataFrame(rows, columns=list(ODDS_COLUMNS))


def merge_snapshots(*snapshots: pd.DataFrame) -> pd.DataFrame:
    """Une capturas de distintos momentos (apertura, T-60, cierre)."""
    frames = [frame for frame in snapshots if frame is not None and not frame.empty]
    if not frames:
        return empty_frame(ODDS_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(
        subset=["game_id", "book", "timing"], keep="last"
    ).reset_index(drop=True)


__all__ = [
    "DEFAULT_MARKETS",
    "DEFAULT_REGIONS",
    "ODDS_API_BASE",
    "ODDS_API_KEY_ENV",
    "SPORT_KEY",
    "api_key",
    "fetch_odds_payload",
    "merge_snapshots",
    "payload_to_odds",
]
