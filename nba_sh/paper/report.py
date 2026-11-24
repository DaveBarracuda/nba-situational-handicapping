"""Reporte de CLV con potencia estadística.

Este es el entregable del paper-trading. La diferencia con un reporte de ROI
es toda la razón de montar esto, y conviene tenerla a la vista:

- Probar un edge de **1 punto porcentual** por resultados exige ~19.600
  apuestas, unas 16 temporadas apostando todos los partidos. Con 2.356
  apuestas (lo que midió este proyecto contra el cierre de ESPN) el edge
  mínimo detectable era 2,89 pp: solo descarta edges grandes.
- La CLV no pregunta si acertaste, pregunta si cogiste un número mejor que el
  que el mercado acabó fijando. Quita el azar del partido —sd del margen ≈ 16
  puntos— y deja solo el del movimiento de línea, un orden de magnitud menor.
  Eso son ~100 veces menos muestra para la misma precisión.

Por eso el reporte no imprime solo la media: imprime el error estándar, el
estadístico t y **cuántas apuestas faltan** para resolver el umbral que te
importe. Y si la CLV media es negativa con muestra suficiente, lo dice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from nba_sh.paper.ledger import SOURCE_FILE, SOURCE_PICKCENTER, PaperLedger

LOGGER = logging.getLogger(__name__)

# CLV media que se considera un edge perseguible, en puntos de probabilidad.
DEFAULT_CLV_TARGET = 0.01
CONFIDENCE = 0.95
POWER = 0.80
MIN_BETS_FOR_VERDICT = 100
NAN = float("nan")


@dataclass(frozen=True)
class ClvReport:
    """CLV medida y lo que hace falta para decidir."""

    n_bets: int
    n_with_clv: int
    mean_clv: float
    sd_clv: float
    standard_error: float
    t_statistic: float
    p_value: float
    beat_close_rate: float
    mean_clv_points: float
    bets_needed: int
    detectable_clv: float
    sources: dict[str, int] = field(default_factory=dict)
    by_market: dict[str, float] = field(default_factory=dict)
    roi: float = NAN
    # Apuestas registradas que aún esperan resultado. No entran en la CLV,
    # pero sin esto el reporte dice "0 apuestas" justo después de registrar
    # cinco, que es desconcertante.
    n_pending: int = 0
    verdict: str = ""

    @staticmethod
    def empty() -> ClvReport:
        """Reporte sin datos."""
        return ClvReport(
            n_bets=0,
            n_with_clv=0,
            mean_clv=NAN,
            sd_clv=NAN,
            standard_error=NAN,
            t_statistic=NAN,
            p_value=NAN,
            beat_close_rate=NAN,
            mean_clv_points=NAN,
            bets_needed=0,
            detectable_clv=NAN,
        )

    def as_dict(self) -> dict[str, object]:
        """Vista serializable para artefactos."""
        return {
            "n_bets": self.n_bets,
            "n_with_clv": self.n_with_clv,
            "mean_clv": self.mean_clv,
            "sd_clv": self.sd_clv,
            "standard_error": self.standard_error,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "beat_close_rate": self.beat_close_rate,
            "mean_clv_points": self.mean_clv_points,
            "bets_needed": self.bets_needed,
            "detectable_clv": self.detectable_clv,
            "sources": self.sources,
            "by_market": self.by_market,
            "roi": self.roi,
            "n_pending": self.n_pending,
            "verdict": self.verdict,
        }


def bets_needed_for(target: float, sd: float) -> int:
    """Apuestas para detectar una CLV media de ``target`` al 80% de potencia."""
    if not np.isfinite(sd) or sd <= 0.0 or target <= 0.0:
        return 0
    z_alpha = float(norm.ppf(1.0 - (1.0 - CONFIDENCE) / 2.0))
    z_beta = float(norm.ppf(POWER))
    return int(np.ceil((z_alpha + z_beta) ** 2 * sd**2 / target**2))


def detectable_clv_with(n_bets: int, sd: float) -> float:
    """CLV media mínima detectable con ``n_bets`` apuestas."""
    if n_bets < 2 or not np.isfinite(sd) or sd <= 0.0:
        return NAN
    z_alpha = float(norm.ppf(1.0 - (1.0 - CONFIDENCE) / 2.0))
    z_beta = float(norm.ppf(POWER))
    return float((z_alpha + z_beta) * sd / np.sqrt(n_bets))


def _verdict(report: ClvReport, *, target: float) -> str:
    """Una frase: qué se puede concluir con la muestra que hay."""
    if report.n_with_clv == 0:
        pending = (
            f"{report.n_pending} apuestas registradas esperando resultado. "
            if report.n_pending
            else ""
        )
        return (
            f"{pending}Sin CLV medida todavía: hace falta una captura de cierre "
            "para cada apuesta (`paper capture --timing close`, antes del tipoff) "
            "y luego `paper settle`."
        )
    # Las cuotas sintéticas invalidan la medición. Un cierre reconstruido del
    # pickcenter de ESPN no, porque es mercado real — pero hay que decirlo,
    # porque no es una captura propia a T−30.
    if SOURCE_FILE in report.sources:
        return (
            "Hay cuotas sintéticas en el libro (fuente `file`): esto prueba la "
            "tubería, no mide nada."
        )
    caveat = (
        " Ojo: el cierre está reconstruido del pickcenter de ESPN, no capturado "
        "a T−30."
        if SOURCE_PICKCENTER in report.sources
        else ""
    )
    if report.n_with_clv < MIN_BETS_FOR_VERDICT:
        return (
            f"Muestra insuficiente: {report.n_with_clv} apuestas con CLV. Para "
            f"resolver {100 * target:+.2f} pp hacen falta ~{report.bets_needed}."
            f"{caveat}"
        )
    if report.p_value < 1.0 - CONFIDENCE and report.mean_clv > 0.0:
        return (
            f"CLV media {100 * report.mean_clv:+.3f} pp, significativa "
            f"(p={report.p_value:.3f}). Es la primera señal real: el modelo coge "
            "mejores números que el cierre."
        )
    if report.p_value < 1.0 - CONFIDENCE and report.mean_clv < 0.0:
        return (
            f"CLV media {100 * report.mean_clv:+.3f} pp, significativamente "
            f"NEGATIVA (p={report.p_value:.3f}). El modelo coge peores números que "
            "el cierre: no sigas."
        )
    return (
        f"CLV media {100 * report.mean_clv:+.3f} pp, no distinguible de 0 "
        f"(p={report.p_value:.3f}). Con esta muestra el mínimo detectable es "
        f"{100 * report.detectable_clv:+.3f} pp."
    )


def clv_report(
    settled: pd.DataFrame, *, target: float = DEFAULT_CLV_TARGET
) -> ClvReport:
    """Construye el reporte de CLV a partir de las apuestas liquidadas.

    Args:
        settled: apuestas liquidadas (del libro de paper-trading).
        target: CLV media que se considera un edge perseguible.

    Returns:
        :class:`ClvReport` con la medición y su potencia.
    """
    if settled.empty:
        base = ClvReport.empty()
        return replace(base, verdict=_verdict(base, target=target))

    clv = pd.to_numeric(settled.get("clv"), errors="coerce").dropna()
    points = pd.to_numeric(settled.get("clv_points"), errors="coerce").dropna()
    stake = pd.to_numeric(settled.get("stake"), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(settled.get("pnl"), errors="coerce").fillna(0.0)

    n = int(clv.size)
    sd = float(clv.std(ddof=1)) if n > 1 else NAN
    standard_error = sd / np.sqrt(n) if n > 1 and np.isfinite(sd) else NAN
    mean = float(clv.mean()) if n else NAN
    if n > 1 and np.isfinite(standard_error) and standard_error > 0.0:
        t_statistic = mean / standard_error
        p_value = float(2.0 * student_t.sf(abs(t_statistic), df=n - 1))
    else:
        t_statistic, p_value = NAN, NAN

    with_clv = settled.assign(
        _clv=pd.to_numeric(settled.get("clv"), errors="coerce")
    ).dropna(subset=["_clv"])
    report = ClvReport(
        n_bets=int(len(settled)),
        n_with_clv=n,
        mean_clv=mean,
        sd_clv=sd,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
        beat_close_rate=float((clv > 0.0).mean()) if n else NAN,
        mean_clv_points=float(points.mean()) if points.size else NAN,
        bets_needed=bets_needed_for(target, sd),
        detectable_clv=detectable_clv_with(n, sd),
        sources=(
            {
                str(key): int(value)
                for key, value in settled["source"].value_counts().items()
            }
            if "source" in settled.columns
            else {}
        ),
        by_market={
            str(market): float(group.mean())
            for market, group in with_clv.groupby("market")["_clv"]
        },
        roi=float(pnl.sum() / stake.sum()) if stake.sum() > 0 else NAN,
    )
    return replace(report, verdict=_verdict(report, target=target))


def report_from_ledger(
    *, ledger: PaperLedger | None = None, target: float = DEFAULT_CLV_TARGET
) -> ClvReport:
    """Reporte de CLV leyendo el libro del disco.

    Añade a cada apuesta la fuente de sus cuotas, para que el veredicto pueda
    negarse a concluir si hay capturas que no vienen de la API.
    """
    book = ledger or PaperLedger()
    settled = book.settled()
    captures = book.odds()
    if not settled.empty and not captures.empty and "source" not in settled.columns:
        by_game = (
            captures.drop_duplicates(subset=["game_id"], keep="last")
            .set_index("game_id")["source"]
            .to_dict()
        )
        settled = settled.assign(source=settled["game_id"].astype(str).map(by_game))
    report = replace(
        clv_report(settled, target=target), n_pending=len(book.pending_bets())
    )
    return replace(report, verdict=_verdict(report, target=target))


__all__ = [
    "CONFIDENCE",
    "DEFAULT_CLV_TARGET",
    "MIN_BETS_FOR_VERDICT",
    "POWER",
    "ClvReport",
    "bets_needed_for",
    "clv_report",
    "detectable_clv_with",
    "report_from_ledger",
]
