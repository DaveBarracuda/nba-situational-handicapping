"""Las cuatro operaciones del paper-trading.

**Ninguna envía una apuesta a ningún sitio.** Todo queda en `data/paper/`.

El ciclo de un día, con los horarios de §14.1 de la guía:

    1. ``capture --timing open``   por la mañana, la línea de apertura
    2. ``bet``                     tras el informe de bajas (17:30 ET)
    3. ``capture --timing close``  justo antes del primer tipoff
    4. ``settle``                  al día siguiente, con los resultados

La CLV sale de comparar (3) con la línea que se tomó en (2). Por eso el orden
importa y por eso se guarda qué captura se usó para apostar: sin las dos
puntas no hay CLV, y sin CLV esto no mide nada en menos de una década.

La clave de The Odds API se lee de la variable de entorno ``ODDS_API_KEY`` y
no se escribe nunca, ni en disco ni en los logs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from nba_sh.betting import BankrollState, BettingEngine, settle_bets
from nba_sh.config import CONFIG, Config
from nba_sh.ingest.espn_source import (
    fetch_closing_lines,
    fetch_season,
    fetch_slate,
)
from nba_sh.ingest.odds_source import fetch_odds_payload, payload_to_odds
from nba_sh.paper.ledger import (
    SOURCE_API,
    SOURCE_FILE,
    SOURCE_PICKCENTER,
    PaperLedger,
)
from nba_sh.paper.slate import predict_slate
from nba_sh.pipeline import prepare
from nba_sh.schemas import TIMING_CLOSE, TIMING_PREGAME, Dataset
from nba_sh.storage import Storage

LOGGER = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_GLOB = "real_*"
MARKET_TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class CaptureResult:
    """Resultado de una captura de cuotas."""

    path: Path
    n_lines: int
    n_games: int
    timing: str
    source: str


@dataclass(frozen=True)
class BetResult:
    """Resultado de una jornada de apuestas de papel."""

    path: Path | None
    n_bets: int
    n_evaluated: int
    game_date: pd.Timestamp
    bankroll: float
    rejections: dict[str, int]


@dataclass(frozen=True)
class SettleResult:
    """Resultado de una liquidación."""

    path: Path | None
    n_settled: int
    n_with_clv: int
    n_still_pending: int


def market_today() -> pd.Timestamp:
    """La fecha de hoy en hora del mercado (ET), sin zona."""
    return pd.Timestamp.now(tz=MARKET_TIMEZONE).normalize().tz_localize(None)


def load_history(*, storage: Storage | None = None, config: Config = CONFIG) -> Dataset:
    """Dataset histórico validado, del snapshot más reciente.

    Raises:
        FileNotFoundError: si no hay ningún snapshot que cargar.
    """
    store = storage or Storage()
    directories = [
        path for path in store.raw_dir.glob(DEFAULT_SNAPSHOT_GLOB) if path.is_dir()
    ]
    if not directories:
        raise FileNotFoundError(
            f"no hay snapshots en {store.raw_dir}; ingiere data real primero"
        )
    # Por fecha de escritura, no por nombre: "real_shots" ordena después de
    # "real_5seasons" alfabéticamente y es más viejo.
    latest = max(directories, key=lambda path: path.stat().st_mtime)
    LOGGER.info("paper: histórico de %s", latest.name)
    return prepare(store.read_snapshot(latest), config=config).dataset


def _game_id_by_teams(games: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Mapa ``(local, visitante) -> game_id`` para casar cuotas con juegos."""
    return {
        (str(row.home_team), str(row.away_team)): str(row.game_id)
        for row in games.itertuples(index=False)
    }


def capture_odds(
    *,
    timing: str,
    season: str,
    day: pd.Timestamp | None = None,
    payload_file: Path | None = None,
    ledger: PaperLedger | None = None,
    bookmaker: str | None = None,
) -> CaptureResult:
    """Captura las cuotas de la jornada y las guarda en el libro.

    Args:
        timing: ``open``, ``t_minus_60`` o ``close``.
        season: etiqueta de temporada para pedir el calendario.
        day: jornada a casar; por defecto hoy en hora del mercado.
        payload_file: JSON guardado en vez de la API (para probar la tubería).
        ledger: libro destino.
        bookmaker: clave del libro concreto; por defecto el primero.

    Returns:
        :class:`CaptureResult` con la ruta escrita.
    """
    book = ledger or PaperLedger()
    target = pd.Timestamp(day) if day is not None else market_today()
    slate = fetch_slate(target, season=season)

    # Una línea capturada después del tipoff no es el cierre, es una línea en
    # vivo: etiquetarla como cierre envenenaría la CLV sin que nada fallara.
    # Se descartan esos partidos y se dice cuántos.
    if timing == TIMING_CLOSE and not slate.empty:
        now = pd.Timestamp.utcnow().tz_localize(None)
        started = pd.to_datetime(slate["tipoff_utc"]) <= now
        if started.any():
            LOGGER.warning(
                "paper: %d de %d partidos ya habían empezado; su línea no es "
                "cierre y queda fuera de la captura",
                int(started.sum()),
                len(slate),
            )
            slate = slate[~started].reset_index(drop=True)
        if slate.empty:
            LOGGER.error(
                "paper: la jornada del %s ya está en marcha; no hay cierre que "
                "capturar",
                target.date(),
            )

    if payload_file is not None:
        payload = json.loads(Path(payload_file).read_text(encoding="utf-8"))
        source = SOURCE_FILE
        LOGGER.warning(
            "paper: cuotas desde %s — sirven para probar la tubería, no para "
            "concluir nada",
            payload_file,
        )
    else:
        payload = fetch_odds_payload()
        source = SOURCE_API

    odds = payload_to_odds(
        payload,
        timing=timing,
        bookmaker_key=bookmaker,
        game_id_by_teams=_game_id_by_teams(slate),
    )
    known = set(slate["game_id"].astype(str))
    matched = odds[odds["game_id"].astype(str).isin(known)].reset_index(drop=True)
    if len(matched) < len(odds):
        LOGGER.info(
            "paper: %d líneas descartadas por no casar con la jornada del %s",
            len(odds) - len(matched),
            target.date(),
        )
    path = book.write_odds(matched, timing=timing, source=source)
    return CaptureResult(
        path=path,
        n_lines=len(matched),
        n_games=int(matched["game_id"].nunique()) if not matched.empty else 0,
        timing=timing,
        source=source,
    )


def reconstruct_close(
    *,
    game_ids: Sequence[str] | None = None,
    ledger: PaperLedger | None = None,
) -> CaptureResult:
    """Rescata el cierre de partidos ya jugados desde el pickcenter de ESPN.

    Sin esto, una captura de cierre que no ocurrió deja esas apuestas sin CLV
    para siempre. Con esto se recupera la línea real del libro, pero marcada
    con su propia fuente (``espn-pickcenter``): es mercado de verdad y no una
    captura propia a T−30, y el reporte lo dirá.

    Args:
        game_ids: partidos a reconstruir; por defecto, los de las apuestas del
            libro que no tienen cierre guardado.
        ledger: libro destino.

    Returns:
        :class:`CaptureResult` con la ruta escrita.

    Raises:
        ValueError: si no hay ningún partido que reconstruir.
    """
    book = ledger or PaperLedger()
    if game_ids is None:
        placed = pd.concat([book.bets(), book.settled()], ignore_index=True)
        if placed.empty:
            raise ValueError("no hay apuestas en el libro")
        existing = book.odds(timing=TIMING_CLOSE)
        covered = set(existing["game_id"].astype(str)) if not existing.empty else set()
        game_ids = sorted(set(placed["game_id"].astype(str)) - covered)
    if not game_ids:
        raise ValueError("todas las apuestas del libro ya tienen cierre guardado")

    odds = fetch_closing_lines(game_ids)
    if odds.empty:
        raise ValueError("ESPN no devolvió ninguna línea de cierre")
    path = book.write_odds(odds, timing=TIMING_CLOSE, source=SOURCE_PICKCENTER)
    LOGGER.warning(
        "paper: cierre reconstruido de %d partidos desde el pickcenter de ESPN; "
        "no es una captura propia a T-30 y queda marcado como tal",
        len(odds),
    )
    return CaptureResult(
        path=path,
        n_lines=len(odds),
        n_games=int(odds["game_id"].nunique()),
        timing=TIMING_CLOSE,
        source=SOURCE_PICKCENTER,
    )


def _bankroll_state(ledger: PaperLedger, *, config: Config) -> BankrollState:
    """Bankroll acumulado: el inicial más el P&L de lo ya liquidado."""
    state = BankrollState.start(config=config)
    settled = ledger.settled()
    if settled.empty or "pnl" not in settled.columns:
        return state
    realised = float(pd.to_numeric(settled["pnl"], errors="coerce").fillna(0.0).sum())
    return replace(state, bankroll=state.bankroll + realised)


def place_bets(
    *,
    season: str,
    day: pd.Timestamp | None = None,
    timing: str = TIMING_PREGAME,
    history: Dataset | None = None,
    ledger: PaperLedger | None = None,
    config: Config = CONFIG,
) -> BetResult:
    """Corre el modelo sobre la jornada y registra las apuestas de papel.

    Usa la última captura de cuotas del ``timing`` pedido: esa es la línea que
    se "toma", y contra ella se medirá la CLV cuando llegue el cierre.

    Raises:
        ValueError: si no hay jornada o no hay cuotas capturadas.
    """
    book = ledger or PaperLedger()
    target = pd.Timestamp(day) if day is not None else market_today()
    slate = fetch_slate(target, season=season)
    if slate.empty:
        raise ValueError(f"no hay jornada el {target.date()}")

    lines = book.latest_odds(timing=timing)
    if not lines.empty:
        lines = lines[
            lines["game_id"].astype(str).isin(set(slate["game_id"].astype(str)))
        ]
    if lines.empty:
        raise ValueError(
            f"no hay cuotas capturadas con timing {timing!r} para el "
            f"{target.date()}: corre primero `paper capture`"
        )
    snapshot = str(lines["captured_at"].max())

    indexed = lines.set_index("game_id")
    spread = indexed["spread_home"].reindex(slate["game_id"].astype(str))
    total = indexed["total_line"].reindex(slate["game_id"].astype(str))

    dataset = history if history is not None else load_history(config=config)
    predictions = predict_slate(
        dataset,
        slate,
        spread_home=spread.to_numpy(),
        total_line=total.to_numpy(),
        config=config,
    )

    engine = BettingEngine(config=config, timing=timing)
    # ``compute_candidates`` busca con ``lines.index``: las líneas van
    # indexadas por game_id, no con el índice reseteado.
    result = engine.evaluate(
        predictions,
        indexed,
        state=_bankroll_state(book, config=config),
    )
    # El motivo vacío son las que pasaron el filtro de valor y cayó el límite:
    # no es un descarte del edge y confunde en el log.
    reasons = (
        result.evaluated["rejection_reason"].dropna()
        if not result.evaluated.empty
        else pd.Series(dtype="object")
    )
    rejections = reasons[reasons.astype(str) != ""].value_counts().to_dict()
    path = (
        book.write_bets(result.bets, odds_snapshot=snapshot)
        if not result.bets.empty
        else None
    )
    return BetResult(
        path=path,
        n_bets=len(result.bets),
        n_evaluated=len(result.evaluated),
        game_date=target,
        bankroll=result.state.bankroll,
        rejections={str(key): int(value) for key, value in rejections.items()},
    )


def settle(
    *,
    season: str,
    ledger: PaperLedger | None = None,
    results: pd.DataFrame | None = None,
    resettle: bool = False,
) -> SettleResult:
    """Liquida las apuestas pendientes cuyos partidos ya terminaron.

    La CLV compara la línea tomada con la del cierre guardado. Una apuesta sin
    captura de cierre se liquida con resultado y P&L, pero su CLV queda a NaN:
    es lo honesto, no se inventa.

    Args:
        season: temporada de la que traer resultados.
        ledger: libro de paper-trading.
        results: resultados ya cargados (evita pedir el calendario).
        resettle: vuelve a liquidar también lo ya liquidado. Es lo que hace
            falta cuando el cierre aparece **después** (reconstruido del
            pickcenter, por ejemplo): el libro conserva la liquidación más
            reciente por apuesta, así que repetirla es seguro e idempotente.
    """
    book = ledger or PaperLedger()
    pending = book.bets() if resettle else book.pending_bets()
    if pending.empty:
        return SettleResult(path=None, n_settled=0, n_with_clv=0, n_still_pending=0)

    finished = (results if results is not None else fetch_season(season)).copy()
    finished["actual_margin"] = finished["home_score"] - finished["away_score"]
    finished["actual_total"] = finished["home_score"] + finished["away_score"]
    known = set(finished["game_id"].astype(str))
    ready = pending[pending["game_id"].astype(str).isin(known)]
    if ready.empty:
        return SettleResult(
            path=None, n_settled=0, n_with_clv=0, n_still_pending=len(pending)
        )

    closing = book.latest_odds(timing=TIMING_CLOSE)
    captures = book.odds()
    settled = settle_bets(
        ready,
        finished,
        closing_odds=closing.set_index("game_id") if not closing.empty else None,
        bet_odds=(
            captures.sort_values("captured_at")
            .drop_duplicates(subset=["game_id"], keep="first")
            .set_index("game_id")
            if not captures.empty
            else None
        ),
    )
    path = book.write_settled(settled)
    with_clv = int(pd.to_numeric(settled["clv"], errors="coerce").notna().sum())
    return SettleResult(
        path=path,
        n_settled=len(settled),
        n_with_clv=with_clv,
        n_still_pending=len(pending) - len(ready),
    )


__all__ = [
    "BetResult",
    "CaptureResult",
    "SettleResult",
    "capture_odds",
    "reconstruct_close",
    "load_history",
    "market_today",
    "place_bets",
    "settle",
]
