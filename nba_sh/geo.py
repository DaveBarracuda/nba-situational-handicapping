"""Geografía: distancias entre arenas (guía 5.2, viajes y fatiga)."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from nba_sh.teams import arena

EARTH_RADIUS_MILES = 3958.7613


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia del círculo máximo entre dos puntos, en millas."""
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = radians(lon2 - lon1)
    inner = (
        sin(delta_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_MILES * asin(sqrt(max(inner, 0.0))))


def venue_distance_miles(from_team: str, to_team: str) -> float:
    """Millas entre las arenas de dos equipos (0 si es la misma sede)."""
    origin = arena(from_team)
    destination = arena(to_team)
    if origin.team == destination.team:
        return 0.0
    return haversine_miles(
        origin.latitude, origin.longitude, destination.latitude, destination.longitude
    )


__all__ = ["EARTH_RADIUS_MILES", "haversine_miles", "venue_distance_miles"]
