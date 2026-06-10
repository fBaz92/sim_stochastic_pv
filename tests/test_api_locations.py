"""
End-to-end tests for the installation-site endpoints (``/api/locations``).

Covers the unified import flow that fixed the historical save bugs:
- one call saves location + solar + climate, all linked via ``location_id``;
- re-import upserts (no 409 surprises, same ids, refreshed data);
- an upstream outage never leaves half-written state — the failed component
  is reported explicitly while the rest is saved;
- delete detaches profiles by default, deletes them on request.

External clients are stubbed via ``app.dependency_overrides`` (see
``_external_stubs.py``); persistence uses the in-memory SQLite fixture.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _external_stubs import (
    ArchiveFailingOpenMeteo,
    FailingPVGIS,
    StubNominatim,
    StubOpenMeteo,
    StubPVGIS,
)
from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService


def _create_test_client(
    persistence: PersistenceService,
    *,
    pvgis=StubPVGIS,
    meteo=StubOpenMeteo,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    app.dependency_overrides[dependencies.get_nominatim_client] = StubNominatim
    app.dependency_overrides[dependencies.get_pvgis_client] = pvgis
    app.dependency_overrides[dependencies.get_openmeteo_client] = meteo
    return TestClient(app)


def _import_payload(**overrides):
    base = {
        "name": "Pavullo",
        "address": "Pavullo nel Frignano",
        "display_name": "Pavullo nel Frignano, Modena, Italia",
        "latitude": 44.336,
        "longitude": 10.831,
        "tilt_degrees": 35.0,
        "azimuth_degrees": 180.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path: one call saves everything, linked
# ---------------------------------------------------------------------------


def test_import_creates_location_and_linked_profiles(
    persistence: PersistenceService,
) -> None:
    """POST /api/locations/import persists location + solar + climate, linked."""
    client = _create_test_client(persistence)
    resp = client.post("/api/locations/import", json=_import_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # No component errors on the happy path.
    assert body["solar_error"] is None
    assert body["climate_error"] is None

    location = body["location"]
    assert location["name"] == "Pavullo"
    assert location["display_name"].startswith("Pavullo nel Frignano")

    # Full profile records returned and linked to the location.
    assert body["solar_profile"] is not None
    assert body["solar_profile"]["location_id"] == location["id"]
    assert len(body["solar_profile"]["avg_daily_kwh_per_kwp"]) == 12
    assert all(abs(p - 0.6) < 1e-9 for p in body["solar_profile"]["p_sunny"])

    assert body["climate_profile"] is not None
    assert body["climate_profile"]["location_id"] == location["id"]
    assert len(body["climate_profile"]["monthly_params"]) == 12

    # Summaries embedded in the location reflect the same links.
    assert len(location["solar_profiles"]) == 1
    assert len(location["climate_profiles"]) == 1
    assert location["solar_profiles"][0]["source"] == "PVGIS+OpenMeteo"

    # And the rows are queryable from the DB with the FK set.
    saved_solar = persistence.solar.get_solar_profile_by_name("Pavullo")
    assert saved_solar is not None
    assert saved_solar.location_id == location["id"]
    saved_climate = persistence.climate.get_climate_profile_by_name("Pavullo")
    assert saved_climate is not None
    assert saved_climate.location_id == location["id"]


def test_reimport_upserts_same_ids(persistence: PersistenceService) -> None:
    """Re-importing the same name refreshes data without duplicating rows."""
    client = _create_test_client(persistence)
    first = client.post("/api/locations/import", json=_import_payload())
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        "/api/locations/import", json=_import_payload(tilt_degrees=20.0)
    )
    assert second.status_code == 200
    second_body = second.json()

    assert second_body["location"]["id"] == first_body["location"]["id"]
    assert second_body["solar_profile"]["id"] == first_body["solar_profile"]["id"]
    assert second_body["solar_profile"]["optimal_tilt_degrees"] == pytest.approx(20.0)
    # Still exactly one location and one solar profile in the DB.
    assert len(persistence.locations.list_locations()) == 1
    assert len(persistence.solar.list_solar_profiles()) == 1


def test_import_adopts_legacy_same_named_profile(
    persistence: PersistenceService,
) -> None:
    """A pre-existing unlinked profile with the same name gets adopted."""
    legacy = persistence.upsert_solar_profile({
        "name": "Pavullo",
        "location_name": "Pavullo (legacy)",
        "latitude": 44.3,
        "longitude": 10.8,
        "optimal_tilt_degrees": 30.0,
        "optimal_azimuth_degrees": 180.0,
        "avg_daily_kwh_per_kwp": [3.0] * 12,
        "p_sunny": [0.5] * 12,
    })
    assert legacy.location_id is None

    client = _create_test_client(persistence)
    resp = client.post("/api/locations/import", json=_import_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["solar_profile"]["id"] == legacy.id
    assert body["solar_profile"]["location_id"] == body["location"]["id"]


# ---------------------------------------------------------------------------
# Partial failures: explicit errors, no half-written profiles
# ---------------------------------------------------------------------------


def test_pvgis_outage_saves_location_with_explicit_solar_error(
    persistence: PersistenceService,
) -> None:
    """PVGIS down → no solar profile row, but location saved + error message."""
    client = _create_test_client(persistence, pvgis=FailingPVGIS)
    resp = client.post("/api/locations/import", json=_import_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["solar_profile"] is None
    assert "PVGIS" in body["solar_error"]
    # Climate is independent and still succeeds.
    assert body["climate_profile"] is not None
    assert body["climate_error"] is None
    # The address is never lost.
    assert persistence.locations.get_location_by_name("Pavullo") is not None
    # No half-written solar row.
    assert persistence.solar.get_solar_profile_by_name("Pavullo") is None


def test_archive_outage_saves_solar_with_explicit_climate_error(
    persistence: PersistenceService,
) -> None:
    """Open-Meteo archive down → solar saved, climate error explicit."""
    client = _create_test_client(persistence, meteo=ArchiveFailingOpenMeteo)
    resp = client.post("/api/locations/import", json=_import_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["solar_profile"] is not None
    assert body["solar_error"] is None
    assert body["climate_profile"] is None
    assert "Open-Meteo" in body["climate_error"]
    assert persistence.climate.get_climate_profile_by_name("Pavullo") is None


def test_import_location_only_saves_address(
    persistence: PersistenceService,
) -> None:
    """include_solar=False, include_climate=False → just save the site."""
    client = _create_test_client(persistence)
    resp = client.post(
        "/api/locations/import",
        json=_import_payload(include_solar=False, include_climate=False),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["solar_profile"] is None and body["solar_error"] is None
    assert body["climate_profile"] is None and body["climate_error"] is None
    assert persistence.locations.get_location_by_name("Pavullo") is not None


# ---------------------------------------------------------------------------
# List / update / delete
# ---------------------------------------------------------------------------


def test_list_locations_includes_profile_summaries(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    client.post("/api/locations/import", json=_import_payload())
    client.post(
        "/api/locations/import",
        json=_import_payload(name="Milano", latitude=45.46, longitude=9.19,
                             include_climate=False),
    )

    resp = client.get("/api/locations")
    assert resp.status_code == 200
    body = resp.json()
    assert [loc["name"] for loc in body] == ["Milano", "Pavullo"]
    milano = body[0]
    assert len(milano["solar_profiles"]) == 1
    assert milano["climate_profiles"] == []
    pavullo = body[1]
    assert len(pavullo["solar_profiles"]) == 1
    assert len(pavullo["climate_profiles"]) == 1
    assert pavullo["solar_profiles"][0]["updated_at"] is not None


def test_update_location_rename_and_conflict(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    a = client.post("/api/locations/import", json=_import_payload()).json()
    client.post(
        "/api/locations/import",
        json=_import_payload(name="Milano", latitude=45.46, longitude=9.19),
    )

    ok = client.put(
        f"/api/locations/{a['location']['id']}", json={"name": "Pavullo_2"}
    )
    assert ok.status_code == 200
    assert ok.json()["name"] == "Pavullo_2"

    clash = client.put(
        f"/api/locations/{a['location']['id']}", json={"name": "Milano"}
    )
    assert clash.status_code == 409

    missing = client.put("/api/locations/999999", json={"name": "X"})
    assert missing.status_code == 404


def test_delete_location_detaches_profiles_by_default(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    body = client.post("/api/locations/import", json=_import_payload()).json()
    location_id = body["location"]["id"]

    resp = client.delete(f"/api/locations/{location_id}")
    assert resp.status_code == 204

    # Profiles survive, detached.
    solar = persistence.solar.get_solar_profile_by_name("Pavullo")
    assert solar is not None and solar.location_id is None
    climate = persistence.climate.get_climate_profile_by_name("Pavullo")
    assert climate is not None and climate.location_id is None
    assert persistence.locations.get_location_by_name("Pavullo") is None


def test_delete_location_with_profiles(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    body = client.post("/api/locations/import", json=_import_payload()).json()
    location_id = body["location"]["id"]

    resp = client.delete(
        f"/api/locations/{location_id}", params={"delete_profiles": "true"}
    )
    assert resp.status_code == 204
    assert persistence.solar.get_solar_profile_by_name("Pavullo") is None
    assert persistence.climate.get_climate_profile_by_name("Pavullo") is None


def test_delete_missing_location_404(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    assert client.delete("/api/locations/999999").status_code == 404
