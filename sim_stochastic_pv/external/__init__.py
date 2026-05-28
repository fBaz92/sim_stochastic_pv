"""
External API clients for geolocation, solar resource, and climate data.

This subpackage provides thin, synchronous, side-effect-free wrappers around
three free public APIs that the wizard "Luogo" step relies on to remove the
manual data-entry burden from the user:

- :class:`NominatimClient` — OpenStreetMap geocoding (name → lat/lon).
- :class:`PVGISClient` — European Commission JRC PVGIS v5.2 PVcalc
  (lat/lon/tilt/azimuth → monthly PV energy yield).
- :class:`OpenMeteoClient` — Open-Meteo Archive API (lat/lon → 30-year
  monthly normals of temperature and cloud cover).

All three clients:

- speak HTTP over ``httpx`` (sync ``httpx.Client``), already a runtime
  dependency of the project; no new dependency introduced;
- expect to be instantiated with a short-lived ``httpx.Client`` instance
  (or default to one with sensible timeouts), so that they can be easily
  mocked in tests;
- raise :class:`ExternalAPIError` on any non-2xx HTTP response or schema
  surprise, with a message including the upstream URL and the offending
  payload — explicit failure instead of silent fallback.

The clients are intentionally split per provider so each can be tested in
isolation and so users of the library can adopt only the pieces they need
(e.g. a script that just wants geocoding without touching PVGIS).
"""

from __future__ import annotations

from .errors import ExternalAPIError
from .nominatim_client import GeocodeResult, NominatimClient
from .openmeteo_client import ClimateNormals, DailyArchive, OpenMeteoClient
from .pvgis_client import PVGISClient, PVGISMonthlyYield

__all__ = [
    "ExternalAPIError",
    "GeocodeResult",
    "NominatimClient",
    "ClimateNormals",
    "DailyArchive",
    "OpenMeteoClient",
    "PVGISClient",
    "PVGISMonthlyYield",
]
