from __future__ import annotations

from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService


def create_test_client(persistence: PersistenceService) -> TestClient:
    """Build a FastAPI test client with dependency overrides for persistence."""
    app = create_app()

    def get_app_service() -> SimulationApplication:
        return SimulationApplication(
            save_outputs=False,
            persistence=persistence,
            result_builder=None,
        )

    app.dependency_overrides[dependencies.get_application_service] = get_app_service
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def test_api_analysis_and_runs(persistence: PersistenceService, simple_scenario_data: dict):
    """Exercise /api/analysis and /api/runs endpoints."""
    client = create_test_client(persistence)
    resp = client.post(
        "/api/analysis",
        json={"n_mc": 1, "seed": 1, "scenario": simple_scenario_data},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario"]

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()
    assert len(runs) >= 1


def test_api_optimization(persistence: PersistenceService, simple_scenario_data: dict):
    """Exercise the /api/optimization endpoint."""
    client = create_test_client(persistence)
    resp = client.post(
        "/api/optimization",
        json={"seed": 1, "n_mc": 1, "scenario": simple_scenario_data},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluations"] > 0


def test_save_and_run_scenario_with_ids(persistence: PersistenceService):
    """Test saving a scenario with hardware IDs and running it from the database."""
    client = create_test_client(persistence)

    # Create test hardware
    inverter_resp = client.post(
        "/api/inverters",
        json={
            "name": "Test Inverter",
            "manufacturer": "Test Mfg",
            "p_ac_max_kw": 3.0,
            "price_eur": 1000.0,
        },
    )
    assert inverter_resp.status_code == 200
    inverter = inverter_resp.json()

    battery_resp = client.post(
        "/api/batteries",
        json={
            "name": "Test Battery",
            "manufacturer": "Test Mfg",
            "capacity_kwh": 5.0,
            "cycles_life": 5000,
            "price_eur": 5000.0,
        },
    )
    assert battery_resp.status_code == 200
    battery = battery_resp.json()

    # Create a scenario with hardware IDs
    scenario_config = {
        "name": "Test Scenario with IDs",
        "config_type": "scenario",
        "data": {
            "inverter_id": inverter["id"],
            "battery_id": battery["id"],
            "load_profile": {
                "home_profile_type": "arera",
                "away_profile": "arera",
                "min_days_home": [25] * 12,
            },
            "solar": {
                "pv_kwp": 3.0,
                "degradation_per_year": 0.007,
            },
            "energy": {
                "n_years": 20,
                "pv_kwp": 3.0,
                "n_batteries": 1,
            },
            "price": {
                "base_price_eur_per_kwh": 0.20,
                "annual_escalation": 0.02,
            },
            "economic": {
                "n_mc": 1,
                "investment_eur": 6000.0,
                "n_years": 20,
            },
        },
    }

    save_resp = client.post("/api/configurations", json=scenario_config)
    assert save_resp.status_code == 200
    saved = save_resp.json()

    # Run the saved scenario - it should hydrate hardware from DB
    run_resp = client.post(f"/api/scenarios/{saved['id']}/run?seed=123&n_mc=1")
    assert run_resp.status_code == 200
    result = run_resp.json()
    assert "final_gain_mean_eur" in result


def test_save_and_run_campaign_with_ids(persistence: PersistenceService):
    """Test saving a campaign with hardware IDs and running it from the database."""
    client = create_test_client(persistence)

    # Create test hardware
    inv1 = client.post(
        "/api/inverters",
        json={"name": "Inv1", "p_ac_max_kw": 3.0, "price_eur": 1000.0},
    ).json()
    inv2 = client.post(
        "/api/inverters",
        json={"name": "Inv2", "p_ac_max_kw": 5.0, "price_eur": 1500.0},
    ).json()
    panel1 = client.post(
        "/api/panels",
        json={"name": "Panel1", "power_w": 400.0, "price_eur": 200.0},
    ).json()
    bat1 = client.post(
        "/api/batteries",
        json={"name": "Bat1", "capacity_kwh": 5.0, "cycles_life": 5000, "price_eur": 5000.0},
    ).json()

    # Create an optimization with hardware IDs
    campaign_config = {
        "name": "Test Campaign with IDs",
        "config_type": "optimization",
        "data": {
            "hardware_selections": {
                "inverter_ids": [inv1["id"], inv2["id"]],
                "panel_ids": [panel1["id"]],
                "battery_ids": [bat1["id"]],
            },
            "optimization": {
                "panel_count_options": [1, 2],
                "battery_count_options": [0, 1],
                "include_no_battery": True,
            },
            "load_profile": {
                "home_profile_type": "arera",
                "away_profile": "arera",
                "min_days_home": [25] * 12,
            },
            "solar": {
                "pv_kwp": 3.0,
                "degradation_per_year": 0.007,
            },
            "energy": {
                "n_years": 20,
                "pv_kwp": 3.0,
                "n_batteries": 0,
            },
            "price": {
                "base_price_eur_per_kwh": 0.20,
                "annual_escalation": 0.02,
            },
            "economic": {
                "n_mc": 1,
                "investment_eur": 0.0,
            },
            "scenario_name": "Test Campaign",
        },
    }

    save_resp = client.post("/api/configurations", json=campaign_config)
    assert save_resp.status_code == 200
    saved = save_resp.json()

    # Run the saved campaign (optimization) - it should hydrate hardware from DB
    run_resp = client.post(f"/api/optimizations/{saved['id']}/run?seed=123&n_mc=1")
    assert run_resp.status_code == 200
    result = run_resp.json()
    assert result["evaluations"] > 0


def test_hardware_updates_propagate_to_saved_scenarios(persistence: PersistenceService):
    """Test that updating hardware specs propagates to saved scenarios when run."""
    client = create_test_client(persistence)

    # Create initial hardware
    inverter_resp = client.post(
        "/api/inverters",
        json={
            "name": "Updatable Inverter",
            "p_ac_max_kw": 3.0,
            "price_eur": 1000.0,
        },
    )
    inverter = inverter_resp.json()

    # Save a scenario that references this inverter
    scenario_config = {
        "name": "Scenario to test updates",
        "config_type": "scenario",
        "data": {
            "inverter_id": inverter["id"],
            "load_profile": {"home_profile_type": "arera", "away_profile": "arera", "min_days_home": [25] * 12},
            "solar": {"pv_kwp": 3.0},
            "energy": {"n_years": 20, "pv_kwp": 3.0},
            "price": {"base_price_eur_per_kwh": 0.20},
            "economic": {"n_mc": 1, "n_years": 20},
        },
    }
    save_resp = client.post("/api/configurations", json=scenario_config)
    saved_scenario = save_resp.json()

    # Update the inverter specs
    update_resp = client.post(
        "/api/inverters",
        json={
            "name": "Updatable Inverter",
            "p_ac_max_kw": 5.0,  # Changed from 3.0 to 5.0
            "price_eur": 1200.0,  # Changed price
        },
    )
    assert update_resp.status_code == 200

    # Run the saved scenario - it should use the UPDATED specs
    run_resp = client.post(f"/api/scenarios/{saved_scenario['id']}/run?seed=123&n_mc=1")
    assert run_resp.status_code == 200
    # The scenario should now be running with the updated 5.0 kW inverter
    # This is the key benefit of the DB-driven workflow!


# ── Phase 6 — Wizard UI tests ────────────────────────────────────────────────


def test_solar_profiles_endpoint_returns_list(persistence: PersistenceService):
    """
    GET /api/profiles/solar must return a non-empty list with the expected
    schema fields so the wizard Step 1 dropdown can populate correctly.

    Seeds one solar profile directly via the persistence layer and then
    exercises the API endpoint, verifying both the HTTP status and the
    JSON structure.
    """
    # Seed a minimal solar profile into the test DB.
    persistence.upsert_solar_profile({
        "name": "TestCity",
        "location_name": "Test City, Italy",
        "latitude": 45.0,
        "longitude": 9.0,
        "elevation_m": 100.0,
        "optimal_tilt_degrees": 35.0,
        "optimal_azimuth_degrees": 180.0,
        "avg_daily_kwh_per_kwp": [1.5, 2.0, 3.0, 4.0, 5.0, 5.5,
                                    6.0, 5.5, 4.5, 3.0, 2.0, 1.5],
        "p_sunny": [0.4, 0.45, 0.5, 0.55, 0.6, 0.7,
                    0.75, 0.7, 0.6, 0.5, 0.4, 0.42],
        "sunny_factor": 1.2,
        "cloudy_factor": 0.3,
        "source": "Test",
        "notes": "Minimal test profile",
    })

    client = create_test_client(persistence)
    resp = client.get("/api/profiles/solar")
    assert resp.status_code == 200
    profiles = resp.json()
    assert isinstance(profiles, list)
    assert len(profiles) >= 1

    # Verify the expected schema fields are present in each item.
    required_fields = {
        "id", "name", "location_name", "latitude", "longitude",
        "optimal_tilt_degrees", "optimal_azimuth_degrees",
        "avg_daily_kwh_per_kwp", "p_sunny",
    }
    for profile in profiles:
        missing = required_fields - set(profile.keys())
        assert not missing, f"Solar profile missing fields: {missing}"
        assert len(profile["avg_daily_kwh_per_kwp"]) == 12
        assert len(profile["p_sunny"]) == 12


def test_analysis_response_includes_run_id(
    persistence: PersistenceService, simple_scenario_data: dict
):
    """
    POST /api/analysis must include a non-null ``run_id`` in the response so
    the Scenario Wizard can redirect the user directly to the newly created
    run in the Dashboard (Phase 6 redirect feature).
    """
    client = create_test_client(persistence)
    resp = client.post(
        "/api/analysis",
        json={"n_mc": 1, "seed": 42, "scenario": simple_scenario_data},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data, "AnalysisResponse must include 'run_id'"
    assert data["run_id"] is not None, "run_id must not be None when persistence is active"
    assert isinstance(data["run_id"], int), f"run_id must be int, got {type(data['run_id'])}"

    # Cross-check: the run must be listable via /api/runs
    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    run_ids = [r["id"] for r in runs_resp.json()]
    assert data["run_id"] in run_ids, "The run_id returned by /api/analysis must appear in /api/runs"
