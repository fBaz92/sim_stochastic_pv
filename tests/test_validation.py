"""Tests for configuration validation module."""

from __future__ import annotations

import pytest

from sim_stochastic_pv.validation import validate_scenario, validate_optimization


class TestValidateScenario:
    """Tests for validate_scenario function."""

    def test_valid_scenario_returns_no_errors(self):
        """Valid scenario configuration should return empty error list."""
        valid_scenario = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100, "n_years": 20},
            "price": {"type": "escalating"},
        }
        errors = validate_scenario(valid_scenario)
        assert errors == []

    def test_missing_required_sections(self):
        """Missing required sections should be reported."""
        incomplete_scenario = {
            "load_profile": {"type": "home_away"},
            # Missing: solar, energy, economic, price
        }
        errors = validate_scenario(incomplete_scenario)

        assert len(errors) == 4
        assert "Missing required section: 'solar'" in errors
        assert "Missing required section: 'energy'" in errors
        assert "Missing required section: 'economic'" in errors
        assert "Missing required section: 'price'" in errors

    def test_load_profile_validation(self):
        """Load profile section validation."""
        # Missing type
        scenario = {
            "load_profile": {},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }
        errors = validate_scenario(scenario)
        assert any("load_profile.type is required" in e for e in errors)

        # Wrong type (not a dict)
        scenario["load_profile"] = "invalid"
        errors = validate_scenario(scenario)
        assert any("load_profile must be a dict/object" in e for e in errors)

    def test_solar_validation(self):
        """Solar section validation."""
        scenario = {
            "load_profile": {"type": "home_away"},
            "solar": {},  # Missing type
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }
        errors = validate_scenario(scenario)
        assert any("solar.type is required" in e for e in errors)

    def test_energy_validation(self):
        """Energy section validation."""
        base_scenario = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        # Missing pv_kwp
        scenario = {**base_scenario, "energy": {}}
        errors = validate_scenario(scenario)
        assert any("energy.pv_kwp is required" in e for e in errors)

        # pv_kwp not a number
        scenario = {**base_scenario, "energy": {"pv_kwp": "3.5"}}
        errors = validate_scenario(scenario)
        assert any("energy.pv_kwp must be a number" in e for e in errors)

        # pv_kwp not positive
        scenario = {**base_scenario, "energy": {"pv_kwp": 0}}
        errors = validate_scenario(scenario)
        assert any("energy.pv_kwp must be positive" in e for e in errors)

        scenario = {**base_scenario, "energy": {"pv_kwp": -1.5}}
        errors = validate_scenario(scenario)
        assert any("energy.pv_kwp must be positive" in e for e in errors)

    def test_economic_validation(self):
        """Economic section validation."""
        base_scenario = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "price": {"type": "escalating"},
        }

        # n_mc not an integer
        scenario = {**base_scenario, "economic": {"n_mc": 100.5}}
        errors = validate_scenario(scenario)
        assert any("economic.n_mc must be an integer" in e for e in errors)

        # n_mc not positive
        scenario = {**base_scenario, "economic": {"n_mc": 0}}
        errors = validate_scenario(scenario)
        assert any("economic.n_mc must be positive" in e for e in errors)

        # n_years not an integer
        scenario = {**base_scenario, "economic": {"n_years": "20"}}
        errors = validate_scenario(scenario)
        assert any("economic.n_years must be an integer" in e for e in errors)

        # n_years not positive
        scenario = {**base_scenario, "economic": {"n_years": -5}}
        errors = validate_scenario(scenario)
        assert any("economic.n_years must be positive" in e for e in errors)

    def test_price_validation(self):
        """Price section validation."""
        scenario = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {},  # Missing type
        }
        errors = validate_scenario(scenario)
        assert any("price.type is required" in e for e in errors)


class TestValidateOptimization:
    """Tests for validate_optimization function."""

    def test_valid_optimization_returns_no_errors(self):
        """Valid optimization configuration should return empty error list."""
        valid_optimization = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100, "n_years": 20},
            "price": {"type": "escalating"},
            "optimization": {
                "inverter_options": [{"name": "Inv1", "p_ac_max_kw": 5.0}],
            },
        }
        errors = validate_optimization(valid_optimization)
        assert errors == []

    def test_includes_scenario_validation(self):
        """Optimization validation should include scenario validation."""
        incomplete_optimization = {
            # Missing scenario sections
            "optimization": {
                "inverter_options": [{"name": "Inv1"}],
            },
        }
        errors = validate_optimization(incomplete_optimization)

        # Should have errors from scenario validation
        assert any("Missing required section: 'load_profile'" in e for e in errors)
        assert any("Missing required section: 'solar'" in e for e in errors)

    def test_missing_optimization_section(self):
        """Missing optimization section should be reported."""
        scenario_only = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            # Missing: optimization
        }
        errors = validate_optimization(scenario_only)
        assert any("Missing required section: 'optimization'" in e for e in errors)

    def test_optimization_not_dict(self):
        """Optimization section must be a dict."""
        invalid = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            "optimization": "invalid",
        }
        errors = validate_optimization(invalid)
        assert any("optimization must be a dict/object" in e for e in errors)

    def test_inverter_options_validation(self):
        """Inverter options validation."""
        base_config = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        # Not a list
        config = {**base_config, "optimization": {"inverter_options": "invalid"}}
        errors = validate_optimization(config)
        assert any("optimization.inverter_options must be a list" in e for e in errors)

        # Empty list
        config = {**base_config, "optimization": {"inverter_options": []}}
        errors = validate_optimization(config)
        assert any("optimization.inverter_options cannot be empty" in e for e in errors)

    def test_panel_options_validation(self):
        """Panel options validation."""
        base_config = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        # Not a list
        config = {**base_config, "optimization": {"panel_options": 123}}
        errors = validate_optimization(config)
        assert any("optimization.panel_options must be a list" in e for e in errors)

        # Empty list
        config = {**base_config, "optimization": {"panel_options": []}}
        errors = validate_optimization(config)
        assert any("optimization.panel_options cannot be empty" in e for e in errors)

    def test_battery_options_validation(self):
        """Battery options validation."""
        base_config = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
        }

        # Not a list
        config = {**base_config, "optimization": {"battery_options": {}}}
        errors = validate_optimization(config)
        assert any("optimization.battery_options must be a list" in e for e in errors)

        # Empty list
        config = {**base_config, "optimization": {"battery_options": []}}
        errors = validate_optimization(config)
        assert any("optimization.battery_options cannot be empty" in e for e in errors)

    def test_requires_at_least_one_option_list(self):
        """Optimization must have at least one non-empty hardware option list."""
        config = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            "optimization": {},  # No hardware options
        }
        errors = validate_optimization(config)
        assert any(
            "optimization must have at least one of: inverter_options, panel_options, or battery_options"
            in e
            for e in errors
        )

    def test_valid_with_single_option_type(self):
        """Valid optimization with just one type of hardware option."""
        # Only inverters
        config = {
            "load_profile": {"type": "home_away"},
            "solar": {"type": "default"},
            "energy": {"pv_kwp": 3.5},
            "economic": {"n_mc": 100},
            "price": {"type": "escalating"},
            "optimization": {
                "inverter_options": [{"name": "Inv1"}],
            },
        }
        errors = validate_optimization(config)
        assert errors == []

        # Only panels
        config["optimization"] = {"panel_options": [{"name": "Panel1"}]}
        errors = validate_optimization(config)
        assert errors == []

        # Only batteries
        config["optimization"] = {"battery_options": [{"name": "Battery1"}]}
        errors = validate_optimization(config)
        assert errors == []

    def test_multiple_validation_errors_accumulated(self):
        """Multiple validation errors should all be reported."""
        bad_config = {
            # Missing all required sections
            "optimization": {
                "inverter_options": [],  # Empty
                "panel_options": "invalid",  # Not a list
            }
        }
        errors = validate_optimization(bad_config)

        # Should have multiple errors
        assert len(errors) > 5
        assert any("Missing required section" in e for e in errors)
        assert any("inverter_options cannot be empty" in e for e in errors)
        assert any("panel_options must be a list" in e for e in errors)
