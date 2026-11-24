"""Libro de paper-trading: append-only en parquet.

**Nada de esto envía una apuesta a ningún sitio.** El libro solo escribe
ficheros en `data/paper/`, y existe para medir una cosa que de otra forma
tardaría temporadas en resolverse: la **CLV**, cuánto mejor es la línea que
cogió el modelo que la que el mercado acabó fijando.

Cada operación escribe un fichero nuevo con su timestamp y nunca modifica uno
anterior. El libro es la concatenación del directorio, así que un proceso que
muere a medias no corrompe nada y dos capturas del mismo día no se pisan.

    data/paper/
      odds/open/20261123T160000Z.parquet       una captura de cuotas
      odds/close/20261124T000000Z.parquet
      bets/20261123T223000Z.parquet            una jornada de apuestas
      settled/20261124T120000Z.parquet         una liquidación

El campo ``source`` de cada captura dice de dónde salió (``the-odds-api`` o
``file``). El reporte no mezcla fuentes sin decirlo: una captura de fichero
sirve para probar la tubería, no para concluir nada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_sh.betting.engine import LOG_COLUMNS
from nba_sh.betting.settle import SETTLEMENT_COLUMNS
from nba_sh.schemas import ODDS_COLUMNS, TIMINGS

LOGGER = logging.getLogger(__name__)

DEFAULT_PAPER_DIR = Path("data/paper")
ODDS_DIR = "odds"
BETS_DIR = "bets"
SETTLED_DIR = "settled"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

SOURCE_API = "the-odds-api"
SOURCE_FILE = "file"
# Cierre reconstruido del pickcenter de ESPN, con el partido ya jugado. Es
# mercado real, pero no es una captura propia a T-30: ESPN no documenta el
# momento en que fijó esos números. Se marca aparte para que el reporte pueda
# decirlo en vez de hacerla pasar por lo otro.
SOURCE_PICKCENTER = "espn-pickcenter"
CAPTURE_COLUMNS: tuple[str, ...] = (*ODDS_COLUMNS, "captured_at", "source")
PAPER_BET_COLUMNS: tuple[str, ...] = (*LOG_COLUMNS, "odds_snapshot")
PAPER_SETTLED_COLUMNS: tuple[str, ...] = (
    *PAPER_BET_COLUMNS,
    *SETTLEMENT_COLUMNS,
    "settled_at",
)


def utc_stamp(moment: datetime | pd.Timestamp | None = None) -> str:
    """Marca de tiempo UTC para nombrar ficheros."""
    now = pd.Timestamp(moment) if moment is not None else pd.Timestamp.now(tz=UTC)
    if now.tzinfo is None:
        now = now.tz_localize(UTC)
    return now.tz_convert(UTC).strftime(TIMESTAMP_FORMAT)


@dataclass(frozen=True)
class PaperLedger:
    """Acceso al libro. Cada escritura es un fichero nuevo."""

    root: Path = DEFAULT_PAPER_DIR

    def _dir(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _read_dir(self, path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
        """Concatena los parquet de un directorio, en orden de nombre."""
        if not path.exists():
            return pd.DataFrame(columns=list(columns))
        files = sorted(path.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=list(columns))
        frames = [pd.read_parquet(file) for file in files]
        present = [frame for frame in frames if not frame.empty]
        if not present:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(present, ignore_index=True)

    # --- cuotas ---------------------------------------------------------

    def write_odds(
        self,
        odds: pd.DataFrame,
        *,
        timing: str,
        source: str = SOURCE_API,
        captured_at: pd.Timestamp | None = None,
    ) -> Path:
        """Guarda una captura de cuotas y devuelve su ruta.

        Raises:
            ValueError: si el timing no es uno de los de la guía.
        """
        if timing not in TIMINGS:
            raise ValueError(f"timing desconocido: {timing!r} (usa {TIMINGS})")
        moment = pd.Timestamp(captured_at or pd.Timestamp.now(tz=UTC))
        stamped = odds.copy()
        stamped["timing"] = timing
        stamped["captured_at"] = moment
        stamped["source"] = source
        path = self._dir(ODDS_DIR, timing) / f"{utc_stamp(moment)}.parquet"
        stamped.reindex(columns=list(CAPTURE_COLUMNS)).to_parquet(path, index=False)
        LOGGER.info("paper: %d líneas de %s guardadas en %s", len(stamped), timing, path)
        return path

    def odds(self, *, timing: str | None = None) -> pd.DataFrame:
        """Todas las capturas, o las de un timing."""
        timings = [timing] if timing is not None else list(TIMINGS)
        frames = [
            self._read_dir(self.root / ODDS_DIR / name, CAPTURE_COLUMNS)
            for name in timings
        ]
        # Concatenar marcos vacíos con columnas hace que pandas avise del
        # dtype; con un solo timing capturado los demás están vacíos.
        present = [frame for frame in frames if not frame.empty]
        if not present:
            return pd.DataFrame(columns=list(CAPTURE_COLUMNS))
        return pd.concat(present, ignore_index=True)

    def latest_odds(self, *, timing: str) -> pd.DataFrame:
        """La captura más reciente de cada juego en ese timing.

        Un mismo partido se captura varias veces: para apostar vale la última
        línea vista, y para la CLV la última antes del salto.
        """
        captures = self.odds(timing=timing)
        if captures.empty:
            return captures
        ordered = captures.sort_values("captured_at")
        return ordered.drop_duplicates(subset=["game_id"], keep="last").reset_index(
            drop=True
        )

    # --- apuestas -------------------------------------------------------

    def write_bets(
        self,
        bets: pd.DataFrame,
        *,
        odds_snapshot: str,
        placed_at: pd.Timestamp | None = None,
    ) -> Path:
        """Registra una jornada de apuestas de papel."""
        moment = pd.Timestamp(placed_at or pd.Timestamp.now(tz=UTC))
        stamped = bets.copy()
        stamped["odds_snapshot"] = odds_snapshot
        path = self._dir(BETS_DIR) / f"{utc_stamp(moment)}.parquet"
        stamped.reindex(columns=list(PAPER_BET_COLUMNS)).to_parquet(path, index=False)
        LOGGER.info("paper: %d apuestas de papel en %s", len(stamped), path)
        return path

    def bets(self) -> pd.DataFrame:
        """Todas las apuestas de papel registradas."""
        return self._read_dir(self.root / BETS_DIR, PAPER_BET_COLUMNS)

    # --- liquidación ----------------------------------------------------

    def write_settled(
        self, settled: pd.DataFrame, *, settled_at: pd.Timestamp | None = None
    ) -> Path:
        """Registra una liquidación (resultado, P&L y CLV)."""
        moment = pd.Timestamp(settled_at or pd.Timestamp.now(tz=UTC))
        stamped = settled.copy()
        stamped["settled_at"] = moment
        path = self._dir(SETTLED_DIR) / f"{utc_stamp(moment)}.parquet"
        stamped.reindex(columns=list(PAPER_SETTLED_COLUMNS)).to_parquet(
            path, index=False
        )
        LOGGER.info("paper: %d apuestas liquidadas en %s", len(stamped), path)
        return path

    def settled(self) -> pd.DataFrame:
        """Todas las liquidaciones, la última por apuesta.

        Una apuesta puede liquidarse dos veces si se repite el comando; vale
        la liquidación más reciente.
        """
        rows = self._read_dir(self.root / SETTLED_DIR, PAPER_SETTLED_COLUMNS)
        if rows.empty:
            return rows
        ordered = rows.sort_values("settled_at")
        return ordered.drop_duplicates(subset=["bet_id"], keep="last").reset_index(
            drop=True
        )

    def pending_bets(self) -> pd.DataFrame:
        """Apuestas registradas que aún no se han liquidado."""
        placed = self.bets()
        if placed.empty:
            return placed
        done = self.settled()
        known = set(done["bet_id"]) if not done.empty else set()
        return placed[~placed["bet_id"].isin(known)].reset_index(drop=True)


__all__ = [
    "CAPTURE_COLUMNS",
    "DEFAULT_PAPER_DIR",
    "PAPER_BET_COLUMNS",
    "PAPER_SETTLED_COLUMNS",
    "SOURCE_API",
    "SOURCE_FILE",
    "SOURCE_PICKCENTER",
    "PaperLedger",
    "utc_stamp",
]
