"""Paper-trading con CLV: mide sin arriesgar dinero.

**Ninguna función de este paquete envía una apuesta a ningún sitio.** Todo lo
que hace es escribir ficheros en `data/paper/`.

Existe por una razón concreta y medida: probar un edge de 1 punto porcentual
por resultados exige ~19.600 apuestas (unas 16 temporadas). La CLV —cuánto
mejor es la línea que cogiste que la que el mercado acabó fijando— quita el
azar del partido y resuelve lo mismo con dos órdenes de magnitud menos
muestra. Es la única forma de saber si este modelo sirve antes de 2040.

- :mod:`nba_sh.paper.ledger`: libro append-only de cuotas, apuestas y liquidaciones.
- :mod:`nba_sh.paper.slate`: predicción de una jornada por jugar.
- :mod:`nba_sh.paper.session`: capturar, apostar, liquidar.
- :mod:`nba_sh.paper.report`: CLV con potencia estadística.
- :mod:`nba_sh.paper.record`: historial y acumulado de resultados.
"""

from nba_sh.paper.ledger import PaperLedger
from nba_sh.paper.record import TrackRecord, bet_log, record_from_ledger
from nba_sh.paper.report import ClvReport, clv_report, report_from_ledger
from nba_sh.paper.session import (
    capture_odds,
    load_history,
    place_bets,
    reconstruct_close,
    settle,
)
from nba_sh.paper.slate import predict_slate

__all__ = [
    "ClvReport",
    "PaperLedger",
    "TrackRecord",
    "bet_log",
    "capture_odds",
    "clv_report",
    "load_history",
    "place_bets",
    "predict_slate",
    "reconstruct_close",
    "record_from_ledger",
    "report_from_ledger",
    "settle",
]
