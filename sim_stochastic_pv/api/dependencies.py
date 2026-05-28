from __future__ import annotations

from functools import lru_cache

from ..application import SimulationApplication
from ..external import NominatimClient, OpenMeteoClient, PVGISClient
from ..persistence import PersistenceService
from ..output import ResultBuilder
from ..db.session import init_db


@lru_cache()
def get_persistence_service() -> PersistenceService:
    """
    Provide a cached PersistenceService instance for API routes.
    """
    init_db()
    return PersistenceService()


def get_result_builder() -> ResultBuilder:
    """
    Provide a ResultBuilder for optional CLI-style exports.
    """
    return ResultBuilder()


def get_application_service() -> SimulationApplication:
    """
    Provide a SimulationApplication configured for API usage.
    """
    persistence = get_persistence_service()
    # API does not save graphical outputs by default
    return SimulationApplication(
        save_outputs=False,
        persistence=persistence,
        result_builder=None,
    )


# ---------------------------------------------------------------------------
# Phase 14 — external API clients (Nominatim, PVGIS, Open-Meteo).
#
# Each provider gets its own factory so that:
#
# - the dependency-injection seam stays granular (a test that mocks PVGIS
#   doesn't have to also rebuild Nominatim);
# - the lifecycle is per-request (the factory is called once per endpoint
#   invocation); the underlying httpx client is short-lived but its
#   connection pool is reused across the same process via httpx defaults.
#
# Tests override these via ``app.dependency_overrides[<factory>] = <stub>``.
# ---------------------------------------------------------------------------


def get_nominatim_client() -> NominatimClient:
    """
    Provide a Nominatim geocoding client for API routes.

    Returns a fresh :class:`NominatimClient` per request. The default
    User-Agent (set on the class) identifies this application to the
    public Nominatim instance, in line with the OSMF usage policy.
    """
    return NominatimClient()


def get_pvgis_client() -> PVGISClient:
    """
    Provide a PVGIS PVcalc client for API routes.
    """
    return PVGISClient()


def get_openmeteo_client() -> OpenMeteoClient:
    """
    Provide an Open-Meteo Archive client for API routes.
    """
    return OpenMeteoClient()
