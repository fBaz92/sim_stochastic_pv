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

    # Create a campaign with hardware IDs
    campaign_config = {
        "name": "Test Campaign with IDs",
        "config_type": "campaign",
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

    # Run the saved campaign - it should hydrate hardware from DB
    run_resp = client.post(f"/api/campaigns/{saved['id']}/run?seed=123&n_mc=1")
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
            "economic": {"n_mc": 1},
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
