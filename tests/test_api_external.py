"""
End-to-end tests for the Phase-14 external API endpoints.

All three external clients are stubbed via ``app.dependency_overrides`` so
the tests never hit the public Internet. The persistence layer uses the
shared in-memory SQLite fixture from ``conftest.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.external import (
    ClimateNormals,
    GeocodeResult,
    NominatimClient,
    OpenMeteoClient,
    PVGISClient,
    PVGISMonthlyYield,
)
from sim_stochastic_pv.persistence import PersistenceService


# ---------------------------------------------------------------------------
# Stub clients (no network)
# ---------------------------------------------------------------------------


class _StubNominatim:
    """Returns a single canned :class:`GeocodeResult` regardless of input."""

    def search(self, query: str, limit: int = 5, accept_language: str = "it,en"):
        return [
            GeocodeResult(
                display_name="Pavullo nel Frignano, Modena, Italia",
                latitude=44.336,
                longitude=10.831,
                place_class="place",
                place_type="town",
                importance=0.71,
            )
        ]


class _StubPVGIS:
    """Returns a deterministic :class:`PVGISMonthlyYield`."""

    def fetch_monthly_yield(
        self,
        latitude: float,
        longitude: float,
        tilt_degrees: float,
        azimuth_degrees: float,
        peakpower_kwp: float = 1.0,
        loss_pct: float = 14.0,
        pv_tech: str = "crystSi",
        mounting_place: str = "free",
    ) -> PVGISMonthlyYield:
        e_m = (45, 65, 100, 125, 155, 165, 170, 145, 110, 85, 55, 40)
        h_i = tuple(int(e * 1.05) for e in e_m)
        return PVGISMonthlyYield(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            tilt_degrees=tilt_degrees,
            azimuth_degrees=azimuth_degrees,
            loss_pct=loss_pct,
            peakpower_kwp=peakpower_kwp,
            monthly_e_kwh=tuple(float(v) for v in e_m),
            monthly_h_i_kwh_per_m2=tuple(float(v) for v in h_i),
        )


class _StubOpenMeteo:
    """
    Returns 12 constant monthly normals (40% cloud → p_sunny=0.6) for
    :meth:`fetch_climate_normals`, and a synthetic 10-year daily archive
    with a clean seasonal sinusoid for :meth:`fetch_daily_archive` (used
    by the Phase-15 ``from_location`` endpoint).
    """

    def fetch_climate_normals(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = 10,
        end_year: int | None = None,
    ) -> ClimateNormals:
        return ClimateNormals(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            years_window=(2014, 2023),
            avg_tmax_c=tuple([20.0] * 12),
            avg_tmin_c=tuple([8.0] * 12),
            avg_tmean_c=tuple([14.0] * 12),
            p_sunny=tuple([0.6] * 12),
        )

    def fetch_daily_archive(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = 10,
        end_year: int | None = None,
    ):
        from sim_stochastic_pv.external import DailyArchive  # noqa: PLC0415
        import datetime as dt  # noqa: PLC0415
        import math  # noqa: PLC0415

        dates: list[str] = []
        tmean: list[float] = []
        tmax: list[float] = []
        tmin: list[float] = []
        start = dt.date(2014, 1, 1)
        end = dt.date(2023, 12, 31)
        d = start
        while d <= end:
            doy = d.timetuple().tm_yday - 1
            seasonal = 12.0 - 10.0 * math.cos(2 * math.pi * doy / 365.25)
            dates.append(d.isoformat())
            tmean.append(seasonal)
            tmax.append(seasonal + 5.0)
            tmin.append(seasonal - 5.0)
            d += dt.timedelta(days=1)
        return DailyArchive(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            years_window=(2014, 2023),
            dates=tuple(dates),
            t_mean_c=tuple(tmean),
            t_max_c=tuple(tmax),
            t_min_c=tuple(tmin),
        )


# ---------------------------------------------------------------------------
# Test-client factory with all four overrides
# ---------------------------------------------------------------------------


def _create_test_client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    app.dependency_overrides[dependencies.get_nominatim_client] = _StubNominatim
    app.dependency_overrides[dependencies.get_pvgis_client] = _StubPVGIS
    app.dependency_overrides[dependencies.get_openmeteo_client] = _StubOpenMeteo
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/external/geocode
# ---------------------------------------------------------------------------


def test_geocode_endpoint_returns_results(persistence: PersistenceService) -> None:
    """POST /api/external/geocode returns the candidate list."""
    client = _create_test_client(persistence)
    resp = client.post(
        "/api/external/geocode",
        json={"query": "Pavullo nel Frignano", "limit": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["display_name"].startswith("Pavullo nel Frignano")
    assert body[0]["latitude"] == pytest.approx(44.336)
    assert body[0]["longitude"] == pytest.approx(10.831)


def test_geocode_rejects_empty_query(persistence: PersistenceService) -> None:
    """Empty / missing ``query`` field → 422 from Pydantic validation."""
    client = _create_test_client(persistence)
    resp = client.post("/api/external/geocode", json={"query": "", "limit": 3})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/external/climate-normals
# ---------------------------------------------------------------------------


def test_climate_normals_endpoint(persistence: PersistenceService) -> None:
    """GET /api/external/climate-normals returns the 12-month preview."""
    client = _create_test_client(persistence)
    resp = client.get(
        "/api/external/climate-normals",
        params={"lat": 44.336, "lon": 10.831, "lookback_years": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["start_year"] == 2014
    assert body["end_year"] == 2023
    assert len(body["avg_tmax_c"]) == 12
    assert len(body["p_sunny"]) == 12
    # 40% cloud cover → p_sunny ≈ 0.6 every month
    assert all(abs(p - 0.6) < 1e-9 for p in body["p_sunny"])


def test_climate_normals_validates_query_params(persistence: PersistenceService) -> None:
    """Latitude > 90 → 422."""
    client = _create_test_client(persistence)
    resp = client.get(
        "/api/external/climate-normals",
        params={"lat": 999.0, "lon": 10.0, "lookback_years": 5},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/profiles/solar/from_location
# ---------------------------------------------------------------------------


def _make_from_location_payload(**overrides: Any) -> dict:
    base = {
        "name": "Pavullo_API_Test",
        "location_name": "Pavullo nel Frignano, Modena, Italia",
        "latitude": 44.336,
        "longitude": 10.831,
        "tilt_degrees": 35.0,
        "azimuth_degrees": 180.0,
    }
    base.update(overrides)
    return base


def test_from_location_creates_solar_profile(persistence: PersistenceService) -> None:
    """End-to-end: PVGIS + OpenMeteo orchestration writes a SolarProfileModel."""
    client = _create_test_client(persistence)
    resp = client.post(
        "/api/profiles/solar/from_location",
        json=_make_from_location_payload(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # 12 monthly values, derived from the stubbed PVGIS payload via avg_daily.
    assert len(body["avg_daily_kwh_per_kwp"]) == 12
    # Numbers should be positive and roughly in the [0.5, 6] range for a 1kWp
    # nominal request — verifies the per-day conversion math at least roughly.
    assert all(0.0 <= v < 20.0 for v in body["avg_daily_kwh_per_kwp"])

    # p_sunny seeded from Open-Meteo (40% cloud → 0.6)
    assert all(abs(p - 0.6) < 1e-9 for p in body["p_sunny"])

    # Source attribution + notes are populated for auditability.
    assert body["source"] == "PVGIS+OpenMeteo"
    assert "PVGIS" in body["notes"] and "Open-Meteo" in body["notes"]

    # The record is queryable via the DB.
    saved = persistence.solar.get_solar_profile_by_name("Pavullo_API_Test")
    assert saved is not None
    assert saved.optimal_tilt_degrees == pytest.approx(35.0)


def test_from_location_conflicts_on_duplicate_name(persistence: PersistenceService) -> None:
    """Second call with the same name and overwrite=False → 409."""
    client = _create_test_client(persistence)
    payload = _make_from_location_payload(name="Pavullo_Dup")

    first = client.post("/api/profiles/solar/from_location", json=payload)
    assert first.status_code == 200, first.text

    second = client.post("/api/profiles/solar/from_location", json=payload)
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]


def test_from_location_overwrite_upserts(persistence: PersistenceService) -> None:
    """Same name + overwrite=True updates the existing record."""
    client = _create_test_client(persistence)
    payload = _make_from_location_payload(name="Pavullo_Upsert")

    first = client.post("/api/profiles/solar/from_location", json=payload)
    assert first.status_code == 200
    first_id = first.json()["id"]

    # Same name, different tilt + overwrite=True → upsert
    payload2 = _make_from_location_payload(
        name="Pavullo_Upsert", tilt_degrees=15.0, overwrite=True
    )
    second = client.post("/api/profiles/solar/from_location", json=payload2)
    assert second.status_code == 200
    assert second.json()["id"] == first_id
    assert second.json()["optimal_tilt_degrees"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Phase 15 — /api/profiles/climate/from_location + /preview
# ---------------------------------------------------------------------------


def _make_climate_payload(**overrides):
    base = {
        "name": "Pavullo_Climate",
        "location_name": "Pavullo nel Frignano, Modena, Italia",
        "latitude": 44.336,
        "longitude": 10.831,
        "lookback_years": 10,
        "climate_trend_c_per_year": 0.0,
    }
    base.update(overrides)
    return base


def test_climate_from_location_creates_profile(persistence: PersistenceService) -> None:
    """End-to-end: Open-Meteo daily archive → calibration → persisted record."""
    client = _create_test_client(persistence)
    resp = client.post(
        "/api/profiles/climate/from_location",
        json=_make_climate_payload(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Pavullo_Climate"
    assert body["source"] == "OpenMeteo Archive"
    assert "harmonic" in body and set(body["harmonic"]) >= {"a0", "a1", "a2"}
    # 12 monthly param dicts
    assert len(body["monthly_params"]) == 12
    # Lookback window populated
    assert body["lookback_window"] == {"start_year": 2014, "end_year": 2023}
    # Persisted in DB
    saved = persistence.climate.get_climate_profile_by_name("Pavullo_Climate")
    assert saved is not None


def test_climate_from_location_conflicts_on_duplicate(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    payload = _make_climate_payload(name="Dup_C")
    assert client.post(
        "/api/profiles/climate/from_location", json=payload
    ).status_code == 200
    resp = client.post("/api/profiles/climate/from_location", json=payload)
    assert resp.status_code == 409


def test_climate_preview_returns_fan_chart_shape(
    persistence: PersistenceService,
) -> None:
    """After creating a profile, GET /preview returns the band + sample paths."""
    client = _create_test_client(persistence)
    create = client.post(
        "/api/profiles/climate/from_location",
        json=_make_climate_payload(name="Preview_C"),
    )
    assert create.status_code == 200
    profile_id = create.json()["id"]

    resp = client.get(
        f"/api/profiles/climate/{profile_id}/preview",
        params={"n_paths": 25, "n_years": 1, "seed": 42},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["days"]) == 365
    assert len(body["mean_c"]) == 365
    assert len(body["p05_c"]) == 365
    assert len(body["p95_c"]) == 365
    # 25 sample paths
    assert len(body["sample_paths_c"]) == 25
    assert len(body["sample_paths_c"][0]) == 365
    # Band ordering invariant for the entire horizon
    for i in range(365):
        assert body["p05_c"][i] <= body["mean_c"][i] <= body["p95_c"][i]

    # 12 monthly hourly distributions present and well-ordered
    for key in (
        "monthly_p05_c", "monthly_p25_c", "monthly_p50_c",
        "monthly_p75_c", "monthly_p95_c",
        "monthly_min_c", "monthly_max_c",
    ):
        assert len(body[key]) == 12
    for m in range(12):
        assert body["monthly_min_c"][m] <= body["monthly_p05_c"][m]
        assert body["monthly_p05_c"][m] <= body["monthly_p25_c"][m]
        assert body["monthly_p25_c"][m] <= body["monthly_p50_c"][m]
        assert body["monthly_p50_c"][m] <= body["monthly_p75_c"][m]
        assert body["monthly_p75_c"][m] <= body["monthly_p95_c"][m]
        assert body["monthly_p95_c"][m] <= body["monthly_max_c"][m]


def test_climate_preview_404_on_missing(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    resp = client.get("/api/profiles/climate/999999/preview")
    assert resp.status_code == 404


def test_climate_list_and_delete(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    create = client.post(
        "/api/profiles/climate/from_location",
        json=_make_climate_payload(name="ListDel_C"),
    )
    assert create.status_code == 200
    profile_id = create.json()["id"]

    list_resp = client.get("/api/profiles/climate")
    assert list_resp.status_code == 200
    names = [r["name"] for r in list_resp.json()]
    assert "ListDel_C" in names

    del_resp = client.delete(f"/api/profiles/climate/{profile_id}")
    assert del_resp.status_code == 204
    assert client.delete(f"/api/profiles/climate/{profile_id}").status_code == 404
