"""Tests del catálogo de equipos y de las reglas de peso situacionales."""

from __future__ import annotations

import pytest

from nba_sh.geo import venue_distance_miles
from nba_sh.situational_rules import (
    apply_fatigue_multiplier,
    cap_total,
    combine_same_sign,
    momentum_points,
    sum_capped,
)
from nba_sh.teams import (
    ARENAS,
    VALID_TEAMS,
    arena,
    conference,
    is_rivalry,
    is_valid_team,
    normalize_team,
    timezone_crossings,
)

EXPECTED_TEAMS = 30
DENVER_ALTITUDE_FEET = 5280.0
MOMENTUM_KWARGS = {"sigma_gate": 1.5, "points_per_sigma": 0.35, "cap": 1.0}


@pytest.mark.unit
def test_league_has_thirty_teams_with_arenas() -> None:
    """Las 30 franquicias tienen sede catalogada."""
    assert len(VALID_TEAMS) == EXPECTED_TEAMS
    assert set(ARENAS) == set(VALID_TEAMS)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SEA", "OKC"),
        ("NJN", "BKN"),
        ("bkn", "BKN"),
        ("Boston Celtics", "BOS"),
        ("  lal ", "LAL"),
    ],
)
def test_normalize_team_handles_aliases(raw: str, expected: str) -> None:
    """Los alias históricos y los nombres completos se normalizan."""
    assert normalize_team(raw) == expected


@pytest.mark.unit
def test_normalize_team_rejects_unknown() -> None:
    """Un identificador desconocido es un error explícito, no un silencio."""
    assert not is_valid_team("XXX")
    with pytest.raises(ValueError):
        normalize_team("XXX")


@pytest.mark.unit
def test_altitude_and_conference() -> None:
    """Denver y Utah son sedes de altitud; las conferencias son correctas."""
    assert arena("DEN").altitude_feet == DENVER_ALTITUDE_FEET
    assert arena("DEN").is_altitude_venue
    assert arena("UTA").is_altitude_venue
    assert not arena("MIA").is_altitude_venue
    assert conference("BOS") == "East"
    assert conference("LAL") == "West"


@pytest.mark.unit
def test_timezone_crossings_and_distance() -> None:
    """Los cruces de huso y las distancias son simétricos y coherentes."""
    assert timezone_crossings("BOS", "LAL") == 3
    assert timezone_crossings("BOS", "NYK") == 0
    assert venue_distance_miles("BOS", "BOS") == 0.0
    assert venue_distance_miles("BOS", "LAL") == pytest.approx(2600.0, rel=0.05)
    assert venue_distance_miles("BOS", "LAL") == pytest.approx(
        venue_distance_miles("LAL", "BOS")
    )


@pytest.mark.unit
def test_rivalries_are_symmetric() -> None:
    """La rivalidad no depende del orden."""
    assert is_rivalry("BOS", "LAL")
    assert is_rivalry("LAL", "BOS")
    assert not is_rivalry("BOS", "SAC")


@pytest.mark.unit
def test_combine_same_sign_matches_guide_example() -> None:
    """Regla (a): lookahead -1.5 y sandwich -1.0 dan -1.9, no -2.5."""
    assert combine_same_sign([-1.5, -1.0]) == pytest.approx(-1.9)


@pytest.mark.unit
def test_combine_same_sign_keeps_signs_separate() -> None:
    """Los spots positivos y negativos se combinan por separado."""
    assert combine_same_sign([-1.5, 0.75]) == pytest.approx(-0.75)
    assert combine_same_sign([]) == 0.0


@pytest.mark.unit
def test_fatigue_multiplier_halves_only_when_fatigued() -> None:
    """Regla (c): el spot vale la mitad si el equipo ya está fatigado."""
    assert apply_fatigue_multiplier(-2.0, is_fatigued=True, multiplier=0.5) == -1.0
    assert apply_fatigue_multiplier(-2.0, is_fatigued=False, multiplier=0.5) == -2.0


@pytest.mark.unit
def test_momentum_gate_and_cap() -> None:
    """Regla (b): por debajo de 1.5 sigma el momentum no aporta nada."""
    assert momentum_points(1.0, **MOMENTUM_KWARGS) == 0.0
    assert momentum_points(2.0, **MOMENTUM_KWARGS) == pytest.approx(0.175)
    assert momentum_points(-2.0, **MOMENTUM_KWARGS) == pytest.approx(-0.175)
    assert momentum_points(20.0, **MOMENTUM_KWARGS) == pytest.approx(1.0)


@pytest.mark.unit
def test_total_cap_is_enforced() -> None:
    """Regla (e): |Σ ajustes| nunca pasa de 4 puntos."""
    assert cap_total(9.0, cap=4.0) == 4.0
    assert cap_total(-9.0, cap=4.0) == -4.0
    assert sum_capped([-3.0, -3.0], cap=4.0) == -4.0
    assert sum_capped([1.0, 0.5], cap=4.0) == pytest.approx(1.5)
