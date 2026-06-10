"""
End-to-end tests for the external API endpoints (geocode, climate normals,
climate-profile management).

All external clients are stubbed via ``app.dependency_overrides`` so the
tests never hit the public Internet. The persistence layer uses the shared
in-memory SQLite fixture from ``conftest.py``. Profile *creation* is
exercised in ``test_api_locations.py`` (unified import flow); here we only
need a created profile as a fixture for the management endpoints, so we go
through the same import endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _external_stubs import StubNominatim, StubOpenMeteo, StubPVGIS
from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService


def _create_test_client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    app.dependency_overrides[dependencies.get_nominatim_client] = StubNominatim
    app.dependency_overrides[dependencies.get_pvgis_client] = StubPVGIS
    app.dependency_overrides[dependencies.get_openmeteo_client] = StubOpenMeteo
    return TestClient(app)


def _create_climate_profile(client: TestClient, name: str) -> int:
    """Create a climate profile through the unified import flow; return its id."""
    resp = client.post(
        "/api/locations/import",
        json={
            "name": name,
            "latitude": 44.336,
            "longitude": 10.831,
            "include_solar": False,
            "include_climate": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["climate_profile"] is not None, body
    return body["climate_profile"]["id"]


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
# Climate-profile management (list / preview / delete)
# ---------------------------------------------------------------------------


def test_climate_preview_returns_fan_chart_shape(
    persistence: PersistenceService,
) -> None:
    """After creating a profile, GET /preview returns the band + sample paths."""
    client = _create_test_client(persistence)
    profile_id = _create_climate_profile(client, "Preview_C")

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


def test_climate_extremes_check(persistence: PersistenceService) -> None:
    """Backtest endpoint: a profile calibrated on the stub archive must
    pass its own extremes check (high coverage, small bias, OK verdict)."""
    client = _create_test_client(persistence)
    profile_id = _create_climate_profile(client, "Extremes_C")

    resp = client.get(
        f"/api/profiles/climate/{profile_id}/extremes-check",
        params={"n_paths": 20, "lookback_years": 10, "seed": 7},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Stub archive covers 2014–2023 → 10 complete observed years.
    assert body["observed_years"] == list(range(2014, 2024))
    assert len(body["observed_annual_tmax"]) == 10
    assert body["n_years"] == 10
    # Band ordering on both tails.
    assert body["sim_tmax_p05"] < body["sim_tmax_p50"] < body["sim_tmax_p95"]
    assert body["sim_tmin_p05"] < body["sim_tmin_p50"] < body["sim_tmin_p95"]
    # Self-consistency: the model was calibrated on this very archive.
    assert body["tmax_coverage"] >= 0.6
    assert abs(body["tmax_median_bias_c"]) <= 1.5
    assert body["verdict"].startswith("OK")


def test_climate_extremes_check_404_on_missing(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    resp = client.get("/api/profiles/climate/999999/extremes-check")
    assert resp.status_code == 404


def test_climate_list_and_delete(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    profile_id = _create_climate_profile(client, "ListDel_C")

    list_resp = client.get("/api/profiles/climate")
    assert list_resp.status_code == 200
    names = [r["name"] for r in list_resp.json()]
    assert "ListDel_C" in names

    del_resp = client.delete(f"/api/profiles/climate/{profile_id}")
    assert del_resp.status_code == 204
    assert client.delete(f"/api/profiles/climate/{profile_id}").status_code == 404
