from __future__ import annotations

import pytest

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


class TestHardwareRepository:
    """Tests for hardware repository access via PersistenceService."""

    def test_upsert_inverter_creates_new(self, persistence: PersistenceService):
        """Upsert creates new inverter if it doesn't exist."""
        inverter = persistence.upsert_inverter({
            "name": "NewInverter",
            "p_ac_max_kw": 5.0,
            "p_dc_max_kw": 5.5,
            "price_eur": 1500.0,
        })

        assert inverter.id is not None
        assert inverter.name == "NewInverter"
        assert inverter.specs["p_ac_max_kw"] == 5.0

    def test_upsert_inverter_updates_existing(self, persistence: PersistenceService):
        """Upsert updates existing inverter with same name."""
        # Create first
        inv1 = persistence.upsert_inverter({
            "name": "UpdatedInverter",
            "p_ac_max_kw": 3.0,
            "price_eur": 1000.0,
        })

        # Update (same name, different specs)
        inv2 = persistence.upsert_inverter({
            "name": "UpdatedInverter",
            "p_ac_max_kw": 5.0,
            "price_eur": 1500.0,
        })

        # Should have same ID but updated specs
        assert inv1.id == inv2.id
        assert inv2.specs["p_ac_max_kw"] == 5.0
        assert inv2.specs["price_eur"] == 1500.0

    def test_list_inverters(self, persistence: PersistenceService):
        """Can list all inverters."""
        persistence.upsert_inverter({"name": "Inv1", "p_ac_max_kw": 3.0})
        persistence.upsert_inverter({"name": "Inv2", "p_ac_max_kw": 5.0})

        inverters = persistence.list_inverters()
        assert len(inverters) >= 2
        names = [inv.name for inv in inverters]
        assert "Inv1" in names
        assert "Inv2" in names

    def test_upsert_panel(self, persistence: PersistenceService):
        """Can upsert panel."""
        panel = persistence.upsert_panel({
            "name": "Panel400W",
            "power_w": 400.0,
            "price_eur": 120.0,
        })

        assert panel.id is not None
        assert panel.name == "Panel400W"
        assert panel.specs["power_w"] == 400.0

    def test_upsert_battery(self, persistence: PersistenceService):
        """Can upsert battery."""
        battery = persistence.upsert_battery({
            "name": "Battery5kWh",
            "specs": {
                "capacity_kwh": 5.0,
                "cycles_life": 6000,
            },
            "price_eur": 2000.0,
        })

        assert battery.id is not None
        assert battery.name == "Battery5kWh"
        assert battery.specs["specs"]["capacity_kwh"] == 5.0


class TestConfigurationRepository:
    """Tests for configuration repository access."""

    def test_save_scenario_configuration(self, persistence: PersistenceService):
        """Can save scenario configuration."""
        config_data = {
            "load_profile": {"type": "monthly"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        config = persistence.save_configuration(
            config_type="scenario",
            name="TestScenario",
            data=config_data,
        )

        assert config.id is not None
        assert config.name == "TestScenario"
        assert config.config_type == "scenario"
        assert config.data == config_data

    def test_save_optimization_configuration(self, persistence: PersistenceService):
        """Can save optimization configuration."""
        config_data = {
            "load_profile": {"type": "monthly"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            "optimization": {
                "inverter_options": [{"name": "Inv1"}],
                "panel_options": [{"name": "Panel1"}],
            },
        }

        config = persistence.save_configuration(
            config_type="optimization",
            name="TestOptimization",
            data=config_data,
        )

        assert config.id is not None
        assert config.name == "TestOptimization"
        assert config.config_type == "optimization"

    def test_get_configuration_by_id(self, persistence: PersistenceService):
        """Can retrieve configuration by ID."""
        config_data = {"test": "data"}
        saved = persistence.save_configuration(
            config_type="scenario",
            name="GetByID",
            data=config_data,
        )

        retrieved = persistence.get_configuration_by_id(saved.id)
        assert retrieved is not None
        assert retrieved.id == saved.id
        assert retrieved.data == config_data

    def test_get_configuration_by_name(self, persistence: PersistenceService):
        """Can retrieve configuration by name."""
        config_data = {"test": "data"}
        persistence.save_configuration(
            config_type="scenario",
            name="GetByName",
            data=config_data,
        )

        retrieved = persistence.get_configuration_by_name("GetByName")
        assert retrieved is not None
        assert retrieved.name == "GetByName"

    def test_list_configurations(self, persistence: PersistenceService):
        """Can list configurations by type."""
        persistence.save_configuration(
            config_type="scenario",
            name="Scenario1",
            data={},
        )
        persistence.save_configuration(
            config_type="scenario",
            name="Scenario2",
            data={},
        )
        persistence.save_configuration(
            config_type="optimization",
            name="Opt1",
            data={},
        )

        scenarios = persistence.list_configurations("scenario")
        assert len(scenarios) >= 2

        optimizations = persistence.list_configurations("optimization")
        assert len(optimizations) >= 1


class TestHydration:
    """Tests for hydration functions."""

    def test_hydrate_scenario(self, persistence: PersistenceService):
        """hydrate_scenario resolves top-level inverter_id to energy.inverter_p_ac_max_kw."""
        # Create hardware
        inverter = persistence.upsert_inverter({
            "name": "TestInv",
            "p_ac_max_kw": 5.0,
            "price_eur": 1500.0,
        })

        # Scenario with inverter_id at top level (as used by the API)
        scenario_data = {
            "inverter_id": inverter.id,
            "load_profile": {"type": "monthly"},
            "solar": {"type": "default"},
            "energy": {
                "pv_kwp": 3.5,
            },
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        # Hydrate
        hydrated = persistence.hydrate_scenario(scenario_data)

        # inverter_id at top level is preserved in the hydrated dict
        # (hydrate_scenario injects specs into the energy section)
        assert "energy" in hydrated
        # The inverter's p_ac_max_kw should be injected into the energy section
        assert hydrated["energy"].get("inverter_p_ac_max_kw") == 5.0

    def test_hydrate_optimization(self, persistence: PersistenceService):
        """hydrate_optimization expands hardware selection IDs."""
        # Create hardware
        inv1 = persistence.upsert_inverter({"name": "Inv1", "p_ac_max_kw": 3.0})
        inv2 = persistence.upsert_inverter({"name": "Inv2", "p_ac_max_kw": 5.0})

        # Optimization with hardware selection IDs
        opt_data = {
            "load_profile": {"type": "monthly"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            "hardware_selections": {
                "inverter_ids": [inv1.id, inv2.id],
            },
            "optimization": {},
        }

        # Hydrate
        hydrated = persistence.hydrate_optimization(opt_data)

        # Should expand to inverter_options
        assert "inverter_options" in hydrated["optimization"]
        assert len(hydrated["optimization"]["inverter_options"]) == 2


class TestExecutionRepository:
    """Tests for execution repository access."""

    def test_record_optimization(self, persistence: PersistenceService):
        """Can record optimization execution."""
        opt = persistence.record_optimization(
            label="TestOpt",
            request_payload={"test": "data"},
            metadata={"evaluations": 10},
        )

        assert opt.id is not None
        assert opt.label == "TestOpt"

    def test_record_scenario(self, persistence: PersistenceService):
        """Can record scenario."""
        scenario = persistence.record_scenario(
            name="TestScenario",
            config={"pv_kwp": 3.5},
            metadata={},
        )

        assert scenario.id is not None
        assert scenario.config["pv_kwp"] == 3.5

    def test_record_run_result(self, persistence: PersistenceService):
        """Can record run result."""
        scenario = persistence.record_scenario(
            name="TestScenario",
            config={},
            metadata={},
        )

        run = persistence.record_run_result(
            result_type="analysis",
            summary={"final_gain": 500.0},
            scenario=scenario,
        )

        assert run.id is not None
        assert run.summary["final_gain"] == 500.0
