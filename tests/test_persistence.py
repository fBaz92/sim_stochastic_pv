from __future__ import annotations

from sim_stochastic_pv.persistence import PersistenceService


def test_persistence_records_components_and_runs(persistence: PersistenceService):
    """Verify components, scenarios, and runs can be stored and retrieved."""
    inverter = persistence.upsert_inverter({"name": "Inverter X", "p_ac_max_kw": 1.2})
    assert inverter is not None

    panel = persistence.upsert_panel({"name": "Panel Y", "power_w": 400.0})
    assert panel is not None

    battery = persistence.upsert_battery(
        {"name": "Battery Z", "specs": {"capacity_kwh": 2.0, "cycles_life": 3000}}
    )
    assert battery is not None

    scenario = persistence.record_scenario(
        "Scenario Test",
        config={"pv_kwp": 1.0},
        metadata={"note": "unit test"},
        inverter=inverter,
        panel=panel,
        battery=battery,
    )
    assert scenario.id is not None

    run = persistence.record_run_result(
        "analysis",
        {"final_gain": 100.0},
        scenario=scenario,
        output_dir="results/test",
    )
    assert run.id is not None

    runs = persistence.list_run_results(limit=5)
    assert len(runs) == 1
    assert runs[0].summary["final_gain"] == 100.0
