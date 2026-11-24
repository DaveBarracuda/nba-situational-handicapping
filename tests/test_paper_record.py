"""Tests del record: acumulado, probabilidad bajo azar y lectura honesta."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_sh.paper.ledger import PaperLedger
from nba_sh.paper.record import (
    BREAK_EVEN_RATE,
    MIN_DECIDED_FOR_SIGNAL,
    bet_log,
    coin_flip_p_value,
    record_from_ledger,
    track_record,
)

STAKE = 100.0
WIN_PNL = 100.0 * 100 / 110
LOSS_PNL = -100.0
FIVE_OF_FIVE_PROBABILITY = 0.5**5


def _settled(
    results: list[str], *, market: str = "spread", clv: float | None = None
) -> pd.DataFrame:
    """Apuestas liquidadas con los resultados dados."""
    size = len(results)
    return pd.DataFrame(
        {
            "bet_id": [f"b{index}" for index in range(size)],
            "game_id": [f"g{index}" for index in range(size)],
            "game_date": pd.date_range("2026-11-23", periods=size, freq="D"),
            "market": [market] * size,
            "side": ["home"] * size,
            "line": [-3.5] * size,
            "american": [-110.0] * size,
            "stake": [STAKE] * size,
            "result": results,
            "pnl": [
                WIN_PNL if value == "win" else (0.0 if value == "push" else LOSS_PNL)
                for value in results
            ],
            "clv": [clv] * size if clv is not None else [np.nan] * size,
            "clv_points": [1.0] * size if clv is not None else [np.nan] * size,
            "closing_line": [-4.5] * size,
        }
    )


@pytest.mark.unit
def test_an_empty_record_says_so() -> None:
    """Sin nada liquidado, el record lo dice y no divide por cero."""
    record = track_record(pd.DataFrame(), pending=3)
    assert record.n_bets == 0
    assert record.pending == 3
    assert np.isnan(record.roi)
    assert "Sin apuestas liquidadas" in record.verdict


@pytest.mark.unit
def test_the_record_counts_results_and_money() -> None:
    """Victorias, derrotas, stake, P&L y ROI salen de las filas."""
    record = track_record(_settled(["win", "win", "loss"]))
    assert (record.wins, record.losses, record.pushes) == (2, 1, 0)
    assert record.staked == pytest.approx(3 * STAKE)
    assert record.pnl == pytest.approx(2 * WIN_PNL + LOSS_PNL)
    assert record.roi == pytest.approx(record.pnl / record.staked)
    assert record.hit_rate == pytest.approx(2 / 3)


@pytest.mark.unit
def test_pushes_do_not_count_as_decided() -> None:
    """Un push no es victoria ni derrota: no entra en el porcentaje."""
    record = track_record(_settled(["win", "push", "loss"]))
    assert record.pushes == 1
    assert record.hit_rate == pytest.approx(0.5)


@pytest.mark.unit
def test_a_perfect_small_record_refuses_to_claim_anything() -> None:
    """El caso real: 5-0 sale por azar el 3,1% de las veces.

    Es la vacuna del módulo: sin esto, un 5-0 con ROI +100% se lee como
    habilidad cuando es una moneda cayendo cinco veces del mismo lado.
    """
    record = track_record(_settled(["win"] * 5))
    assert record.wins == 5
    assert record.roi > 0.9
    assert record.p_value_coin == pytest.approx(FIVE_OF_FIVE_PROBABILITY)
    assert "no es evidencia de nada" in record.verdict
    assert "3.1%" in record.verdict
    assert "Mira la CLV" in record.verdict


@pytest.mark.unit
def test_a_large_winning_record_is_still_hedged() -> None:
    """Con muestra suficiente se reconoce la señal, pero la CLV manda."""
    wins, losses = 60, 40
    record = track_record(_settled(["win"] * wins + ["loss"] * losses))
    assert record.wins + record.losses >= MIN_DECIDED_FOR_SIGNAL
    assert record.hit_rate > BREAK_EVEN_RATE
    assert record.p_value_coin < 0.05
    assert "la CLV decide" in record.verdict


@pytest.mark.unit
def test_a_losing_record_is_not_dressed_up() -> None:
    """Un récord perdedor se reporta tal cual."""
    record = track_record(_settled(["win"] * 12 + ["loss"] * 28))
    assert record.roi < 0
    assert "El azar explicaría" in record.verdict


@pytest.mark.unit
@pytest.mark.parametrize(
    ("wins", "decided", "expected"),
    [(5, 5, 0.5**5), (0, 5, 1.0), (3, 5, 0.5)],
)
def test_coin_flip_p_value_matches_the_binomial(
    wins: int, decided: int, expected: float
) -> None:
    """La probabilidad es la cola binomial, no una aproximación."""
    assert coin_flip_p_value(wins, decided) == pytest.approx(expected)
    assert np.isnan(coin_flip_p_value(0, 0))


@pytest.mark.unit
def test_the_breakdown_separates_markets() -> None:
    """Cada mercado lleva su propio récord: no todos son igual de blandos."""
    spreads = _settled(["win", "loss"], market="spread")
    totals = _settled(["win", "win"], market="total")
    totals["bet_id"] = ["t0", "t1"]
    record = track_record(pd.concat([spreads, totals], ignore_index=True))
    assert set(record.by_market) == {"spread", "total"}
    assert record.by_market["total"]["wins"] == 2
    assert record.by_market["spread"]["losses"] == 1


@pytest.mark.unit
def test_the_bet_log_runs_the_balance_in_order() -> None:
    """El historial va en orden cronológico con el balance acumulado."""
    log = bet_log(_settled(["win", "loss", "win"]))
    assert log["game_date"].is_monotonic_increasing
    assert log["running_pnl"].iloc[-1] == pytest.approx(2 * WIN_PNL + LOSS_PNL)
    assert bet_log(pd.DataFrame()).empty


@pytest.mark.unit
def test_the_record_reads_the_ledger(tmp_path: Path) -> None:
    """De disco: liquidadas más las que esperan resultado."""
    ledger = PaperLedger(root=tmp_path)
    ledger.write_settled(_settled(["win", "loss"]))
    ledger.write_bets(
        pd.DataFrame([{"bet_id": "later", "game_id": "gX"}]), odds_snapshot="snap"
    )
    record = record_from_ledger(ledger=ledger)
    assert record.n_bets == 2
    assert record.pending == 1
