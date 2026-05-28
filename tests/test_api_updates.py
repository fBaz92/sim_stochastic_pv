"""
API tests for the PUT-update endpoints introduced for full DB editing.

Covers:

- PUT ``/api/inverters/{id}`` / panels / batteries (rename + value changes).
- PUT ``/api/profiles/load/{id}`` and ``/api/profiles/price/{id}``.
- PUT ``/api/configurations/{id}`` and the new GET ``/api/configurations/{id}``.
- PUT ``/api/profiles/solar/{id}`` (partial update via ``exclude_unset``)
  and the matching DELETE.
- PUT ``/api/profiles/climate/{id}`` (partial metadata update).

Each test exercises the happy path, the 404-on-missing-id branch, and the
409-on-name-collision branch where applicable. The fixtures reuse the
in-memory SQLite session factory from :mod:`tests.conftest`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(persistence: PersistenceService) -> TestClient:
    """Construct a TestClient with the persistence dependency overridden."""
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def _make_inverter(client: TestClient, name: str = "Inv A", power: float = 3.0) -> dict:
    resp = client.post(
        "/api/inverters",
        json={"name": name, "p_ac_max_kw": power, "price_eur": 1000.0},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _make_panel(client: TestClient, name: str = "Pan A", power_w: float = 400.0) -> dict:
    resp = client.post(
        "/api/panels",
        json={"name": name, "power_w": power_w, "price_eur": 120.0},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _make_battery(client: TestClient, name: str = "Bat A", capacity: float = 5.0) -> dict:
    resp = client.post(
        "/api/batteries",
        json={
            "name": name,
            "capacity_kwh": capacity,
            "cycles_life": 5000,
            "price_eur": 4000.0,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Hardware: inverter
# ---------------------------------------------------------------------------


def test_update_inverter_rename_and_values(persistence: PersistenceService) -> None:
    client = _client(persistence)
    inv = _make_inverter(client, name="Inv A", power=3.0)

    resp = client.put(
        f"/api/inverters/{inv['id']}",
        json={"name": "Inv A renamed", "p_ac_max_kw": 4.2, "price_eur": 1500.0},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == inv["id"]
    assert body["name"] == "Inv A renamed"
    # The power lives in two places on the wire: the dedicated DB column
    # ``nominal_power_kw`` and the JSON specs blob under ``p_ac_max_kw``.
    assert body["nominal_power_kw"] == 4.2
    assert body["specs"]["p_ac_max_kw"] == 4.2
    assert body["specs"]["price_eur"] == 1500.0

    # And the catalog reflects the change without producing a duplicate.
    all_invs = client.get("/api/inverters").json()
    assert len(all_invs) == 1
    assert all_invs[0]["name"] == "Inv A renamed"


def test_update_inverter_missing_id_returns_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    resp = client.put(
        "/api/inverters/9999",
        json={"name": "Ghost", "p_ac_max_kw": 1.0},
    )
    assert resp.status_code == 404


def test_update_inverter_name_collision_returns_409(persistence: PersistenceService) -> None:
    client = _client(persistence)
    a = _make_inverter(client, name="Inv A")
    _make_inverter(client, name="Inv B")
    resp = client.put(
        f"/api/inverters/{a['id']}",
        json={"name": "Inv B", "p_ac_max_kw": 3.0},
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Hardware: panel
# ---------------------------------------------------------------------------


def test_update_panel_rename(persistence: PersistenceService) -> None:
    client = _client(persistence)
    pan = _make_panel(client, name="Pan A", power_w=400)
    resp = client.put(
        f"/api/panels/{pan['id']}",
        json={"name": "Pan A renamed", "power_w": 420.0, "price_eur": 130.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Pan A renamed"
    assert body["power_w"] == 420.0
    assert body["specs"]["price_eur"] == 130.0


def test_update_panel_missing_returns_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    resp = client.put("/api/panels/123", json={"name": "x", "power_w": 1.0})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Hardware: battery
# ---------------------------------------------------------------------------


def test_update_battery_rename_and_capacity(persistence: PersistenceService) -> None:
    client = _client(persistence)
    bat = _make_battery(client, name="Bat A", capacity=5.0)
    resp = client.put(
        f"/api/batteries/{bat['id']}",
        json={
            "name": "Bat A renamed",
            "capacity_kwh": 8.0,
            "cycles_life": 6000,
            "price_eur": 5500.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Bat A renamed"
    assert body["capacity_kwh"] == 8.0
    assert body["specs"]["cycles_life"] == 6000


# ---------------------------------------------------------------------------
# Load and price profile
# ---------------------------------------------------------------------------


def test_update_load_profile_rename(persistence: PersistenceService) -> None:
    client = _client(persistence)
    create = client.post(
        "/api/profiles/load",
        json={"name": "lp", "profile_type": "arera", "data": {}},
    )
    assert create.status_code == 200
    pid = create.json()["id"]

    upd = client.put(
        f"/api/profiles/load/{pid}",
        json={"name": "lp renamed", "profile_type": "arera", "data": {}},
    )
    assert upd.status_code == 200
    assert upd.json()["name"] == "lp renamed"


def test_update_price_profile_rename(persistence: PersistenceService) -> None:
    client = _client(persistence)
    create = client.post(
        "/api/profiles/price",
        json={"name": "pp", "data": {"base_price_eur_per_kwh": 0.25}},
    )
    assert create.status_code == 200
    pid = create.json()["id"]

    upd = client.put(
        f"/api/profiles/price/{pid}",
        json={"name": "pp renamed", "data": {"base_price_eur_per_kwh": 0.30}},
    )
    assert upd.status_code == 200
    assert upd.json()["name"] == "pp renamed"
    assert upd.json()["data"]["base_price_eur_per_kwh"] == 0.30


def test_update_load_profile_missing_returns_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    resp = client.put(
        "/api/profiles/load/999",
        json={"name": "x", "profile_type": "arera", "data": {}},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Saved configurations: GET, PUT
# ---------------------------------------------------------------------------


def test_get_configuration_by_id(persistence: PersistenceService) -> None:
    client = _client(persistence)
    create = client.post(
        "/api/configurations",
        json={"name": "cfg1", "config_type": "scenario", "data": {"a": 1}},
    )
    assert create.status_code == 200
    cid = create.json()["id"]

    resp = client.get(f"/api/configurations/{cid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "cfg1"
    assert resp.json()["data"]["a"] == 1


def test_get_configuration_missing_returns_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    resp = client.get("/api/configurations/4242")
    assert resp.status_code == 404


def test_update_configuration_rename_and_data(persistence: PersistenceService) -> None:
    client = _client(persistence)
    create = client.post(
        "/api/configurations",
        json={"name": "cfg1", "config_type": "scenario", "data": {"a": 1}},
    )
    cid = create.json()["id"]

    resp = client.put(
        f"/api/configurations/{cid}",
        json={"name": "cfg1 renamed", "config_type": "scenario", "data": {"a": 2}},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "cfg1 renamed"
    assert resp.json()["data"]["a"] == 2


def test_update_configuration_name_collision_returns_409(persistence: PersistenceService) -> None:
    client = _client(persistence)
    a = client.post(
        "/api/configurations",
        json={"name": "cfg A", "config_type": "scenario", "data": {}},
    ).json()
    client.post(
        "/api/configurations",
        json={"name": "cfg B", "config_type": "scenario", "data": {}},
    )
    resp = client.put(
        f"/api/configurations/{a['id']}",
        json={"name": "cfg B", "config_type": "scenario", "data": {}},
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Solar profile: PUT and DELETE
# ---------------------------------------------------------------------------


def _seed_solar(persistence: PersistenceService, name: str = "Loc1") -> int:
    record = persistence.upsert_solar_profile(
        {
            "name": name,
            "location_name": "Test Location",
            "latitude": 44.0,
            "longitude": 10.0,
            "optimal_tilt_degrees": 35.0,
            "optimal_azimuth_degrees": 180.0,
            "avg_daily_kwh_per_kwp": [3.0] * 12,
            "p_sunny": [0.5] * 12,
            "weather_persistence": [0.3] * 12,
        }
    )
    return record.id


def test_update_solar_profile_partial_rename(persistence: PersistenceService) -> None:
    client = _client(persistence)
    pid = _seed_solar(persistence, name="Loc1")
    resp = client.put(
        f"/api/profiles/solar/{pid}",
        json={"name": "Loc1 renamed", "notes": "moved to v2"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Loc1 renamed"
    assert body["notes"] == "moved to v2"
    # Unspecified fields preserved (e.g. tilt).
    assert body["optimal_tilt_degrees"] == 35.0


def test_update_solar_profile_collision_returns_409(persistence: PersistenceService) -> None:
    client = _client(persistence)
    a_id = _seed_solar(persistence, name="A")
    _seed_solar(persistence, name="B")
    resp = client.put(f"/api/profiles/solar/{a_id}", json={"name": "B"})
    assert resp.status_code == 409


def test_delete_solar_profile(persistence: PersistenceService) -> None:
    client = _client(persistence)
    pid = _seed_solar(persistence)
    resp = client.delete(f"/api/profiles/solar/{pid}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # Second delete must 404.
    resp2 = client.delete(f"/api/profiles/solar/{pid}")
    assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# Climate profile: PUT
# ---------------------------------------------------------------------------


def _seed_climate(persistence: PersistenceService, name: str = "Clima1") -> int:
    blob = {
        "name": name,
        "location_name": "Test Climate",
        "latitude": 44.0,
        "longitude": 10.0,
        "source": "OpenMeteo Archive",
        "harmonic": {"a0": 12.0, "a1": 1.0, "a2": 0.5},
        "monthly_params": [
            {
                "t_std_residual_c": 1.5,
                "persistence_phi": 0.6,
                "t_amplitude_c": 4.0,
                "gpd_upper": None,
                "gpd_lower": None,
            }
            for _ in range(12)
        ],
        "climate_trend_c_per_year": 0.0,
        "lookback_window": {"start_year": 2014, "end_year": 2023},
        "notes": "seed",
    }
    record = persistence.climate.upsert_climate_profile(blob)
    return record.id


def test_update_climate_profile_partial_rename(persistence: PersistenceService) -> None:
    client = _client(persistence)
    pid = _seed_climate(persistence, name="Clima1")
    resp = client.put(
        f"/api/profiles/climate/{pid}",
        json={"name": "Clima1 renamed", "notes": "edited"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "Clima1 renamed"
    assert body["notes"] == "edited"
    # Calibrated payload preserved.
    assert body["harmonic"]["a0"] == 12.0


def test_update_climate_profile_missing_returns_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    resp = client.put("/api/profiles/climate/9999", json={"name": "x"})
    assert resp.status_code == 404
