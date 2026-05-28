"""
Phase 11 — JSON round-trip and validation tests for tax_bonus + inflation.

Covers:
- ``build_default_economic_config`` correctly hydrates the new sub-blocks.
- ``validate_scenario`` accepts well-formed blocks and rejects malformed ones.
- Legacy scenarios without the new blocks still hydrate identically.
"""

from __future__ import annotations

from sim_stochastic_pv.scenario_builder import build_default_economic_config
from sim_stochastic_pv.simulation import InflationConfig, TaxBonusConfig
from sim_stochastic_pv.validation import validate_scenario


def _base_scenario_dict() -> dict:
    """Minimum well-formed scenario, used as a starting point in tests."""
    return {
        "load_profile": {"type": "home_away"},
        "solar": {"type": "default"},
        "energy": {"pv_kwp": 3.0, "n_years": 20},
        "economic": {"n_mc": 50, "investment_eur": 6000.0},
        "price": {"type": "escalating", "base_price_eur_per_kwh": 0.22},
    }


class TestScenarioBuilderHydration:
    def test_legacy_scenario_yields_no_inflation_object_no_bonus(self):
        """Pre-Phase-11 JSON ⇒ ``inflation`` and ``tax_bonus`` stay None."""
        data = _base_scenario_dict()
        cfg = build_default_economic_config(scenario_data=data)
        assert cfg.inflation is None
        assert cfg.tax_bonus is None
        # The legacy scalar is still wired (default 0.025) so the simulator
        # behaviour is unchanged.
        assert cfg.inflation_rate == 0.025

    def test_legacy_scalar_inflation_rate_is_preserved(self):
        data = _base_scenario_dict()
        data["economic"]["inflation_rate"] = 0.035
        cfg = build_default_economic_config(scenario_data=data)
        assert cfg.inflation_rate == 0.035
        assert cfg.inflation is None  # no rich object built from scalar

    def test_inflation_block_builds_config(self):
        data = _base_scenario_dict()
        data["economic"]["inflation"] = {
            "mode": "stochastic",
            "mean": 0.03,
            "std": 0.012,
            "min_clip": -0.01,
            "max_clip": 0.08,
        }
        cfg = build_default_economic_config(scenario_data=data)
        assert isinstance(cfg.inflation, InflationConfig)
        assert cfg.inflation.mode == "stochastic"
        assert cfg.inflation.mean == 0.03
        assert cfg.inflation.std == 0.012

    def test_inflation_mode_defaults_to_deterministic(self):
        data = _base_scenario_dict()
        data["economic"]["inflation"] = {"mean": 0.025}  # mode missing
        cfg = build_default_economic_config(scenario_data=data)
        assert cfg.inflation is not None
        assert cfg.inflation.mode == "deterministic"

    def test_tax_bonus_block_builds_config(self):
        data = _base_scenario_dict()
        data["economic"]["tax_bonus"] = {
            "enabled": True,
            "fraction_of_investment": 0.5,
            "duration_years": 10,
        }
        cfg = build_default_economic_config(scenario_data=data)
        assert isinstance(cfg.tax_bonus, TaxBonusConfig)
        assert cfg.tax_bonus.enabled is True
        assert cfg.tax_bonus.fraction_of_investment == 0.5
        assert cfg.tax_bonus.duration_years == 10

    def test_n_mc_override_wins_over_json(self):
        data = _base_scenario_dict()
        cfg = build_default_economic_config(n_mc=200, scenario_data=data)
        assert cfg.n_mc == 200


class TestScenarioValidationNewBlocks:
    def test_empty_blocks_are_ignored(self):
        data = _base_scenario_dict()
        # Neither block is present — must validate clean.
        assert validate_scenario(data) == []

    def test_well_formed_blocks_pass(self):
        data = _base_scenario_dict()
        data["economic"]["tax_bonus"] = {
            "enabled": True,
            "fraction_of_investment": 0.5,
            "duration_years": 10,
        }
        data["economic"]["inflation"] = {
            "mode": "stochastic",
            "mean": 0.025,
            "std": 0.015,
            "min_clip": -0.01,
            "max_clip": 0.08,
        }
        assert validate_scenario(data) == []

    def test_tax_bonus_invalid_fraction_rejected(self):
        data = _base_scenario_dict()
        data["economic"]["tax_bonus"] = {"fraction_of_investment": 1.5}
        errors = validate_scenario(data)
        assert any("fraction_of_investment" in e for e in errors)

    def test_tax_bonus_invalid_duration_rejected(self):
        data = _base_scenario_dict()
        data["economic"]["tax_bonus"] = {"duration_years": 0}
        errors = validate_scenario(data)
        assert any("duration_years" in e for e in errors)

    def test_inflation_invalid_mode_rejected(self):
        data = _base_scenario_dict()
        data["economic"]["inflation"] = {"mode": "exponential"}
        errors = validate_scenario(data)
        assert any("inflation.mode" in e for e in errors)

    def test_inflation_negative_std_rejected(self):
        data = _base_scenario_dict()
        data["economic"]["inflation"] = {"mode": "stochastic", "std": -0.01}
        errors = validate_scenario(data)
        assert any("std must be non-negative" in e for e in errors)

    def test_inflation_clip_order_rejected(self):
        data = _base_scenario_dict()
        data["economic"]["inflation"] = {
            "mode": "stochastic",
            "min_clip": 0.05,
            "max_clip": 0.0,
        }
        errors = validate_scenario(data)
        assert any("min_clip must be <=" in e for e in errors)
