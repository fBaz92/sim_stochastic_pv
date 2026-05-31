"""
Phase 11 — Step 5 payload tests.

Verify that the analysis summary now exposes:
- ``tax_bonus_total_eur`` (always present, 0.0 by default).
- ``plots_data["inflation"]`` (None in deterministic mode, dict with
  ``years``, ``mean_factor``, ``sample_paths`` in stochastic mode).
- ``plots_data["cashflow_table"]`` (dict with the monthly mean columns).
"""

from __future__ import annotations

import json

from sim_stochastic_pv.application import SimulationApplication


def _legacy_scenario(n_years: int = 3, n_mc: int = 8) -> dict:
    return {
        "load_profile": {"home_profile_type": "arera", "away_profile": "arera"},
        "solar": {"pv_kwp": 1.5},
        "energy": {
            "n_years": n_years,
            "pv_kwp": 1.5,
            "battery_specs": {"capacity_kwh": 2.5, "cycles_life": 5000},
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 1.5,
        },
        "price": {
            "base_price_eur_per_kwh": 0.22,
            "annual_escalation": 0.02,
        },
        "economic": {
            "n_mc": n_mc,
            "investment_eur": 6000.0,
        },
        "scenario_name": "phase11_payload_test",
    }


class TestSummaryPayload:
    def test_legacy_run_includes_tax_bonus_total_zero(self):
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=_legacy_scenario(), seed=1)
        assert summary["tax_bonus_total_eur"] == 0.0

    def test_legacy_run_has_no_inflation_block(self):
        """Deterministic-inflation runs return None for plots_data['inflation']."""
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=_legacy_scenario(), seed=1)
        assert "inflation" in summary["plots_data"]
        assert summary["plots_data"]["inflation"] is None

    def test_legacy_run_has_cashflow_table(self):
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=_legacy_scenario(), seed=1)
        cf = summary["plots_data"]["cashflow_table"]
        assert set(cf.keys()) == {
            "months",
            "mean_savings_eur",
            "mean_savings_real_eur",
            "bonus_per_month_eur",
            "export_eur",
            "mean_profit_cum_eur",
            "mean_profit_cum_real_eur",
            "mean_price_eur_per_kwh",
            "mean_inflation_factor",
        }
        n_months = len(cf["months"])
        for k, v in cf.items():
            if k == "months":
                continue
            assert len(v) == n_months, f"column {k} length mismatch"

    def test_bonus_total_matches_investment_times_fraction(self):
        scen = _legacy_scenario(n_years=4)
        scen["economic"]["tax_bonus"] = {
            "enabled": True,
            "fraction_of_investment": 0.5,
            "duration_years": 4,
        }
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=scen, seed=1)
        assert summary["tax_bonus_total_eur"] > 0
        assert abs(summary["tax_bonus_total_eur"] - 6000.0 * 0.5) < 1e-6
        # bonus_per_month_eur column reflects the same totals (sparse at
        # months 11, 23, 35, 47).
        cf = summary["plots_data"]["cashflow_table"]
        nonzero = [v for v in cf["bonus_per_month_eur"] if v > 0]
        assert len(nonzero) == 4

    def test_stochastic_inflation_run_has_fan_chart_payload(self):
        scen = _legacy_scenario(n_years=4, n_mc=20)
        scen["economic"]["inflation"] = {
            "mode": "stochastic",
            "mean": 0.03,
            "std": 0.01,
        }
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=scen, seed=42)
        inf = summary["plots_data"]["inflation"]
        assert inf is not None
        assert set(inf.keys()) == {
            "years",
            "mean_factor",
            "p05_factor",
            "p95_factor",
            "mean_rate",
            "sample_paths",
        }
        assert len(inf["years"]) == 4
        assert len(inf["mean_factor"]) == 4
        # Year 0 factor is always 1.0 by convention.
        assert inf["mean_factor"][0] == 1.0
        # At least one sample path included (capped at 20).
        assert 1 <= len(inf["sample_paths"]) <= 20
        for path in inf["sample_paths"]:
            assert len(path) == 4

    def test_summary_is_json_serialisable(self):
        """The full summary must be safe to store as JSON in RunResultRecord."""
        scen = _legacy_scenario()
        scen["economic"]["tax_bonus"] = {
            "enabled": True,
            "fraction_of_investment": 0.5,
            "duration_years": 5,
        }
        scen["economic"]["inflation"] = {
            "mode": "stochastic",
            "mean": 0.025,
            "std": 0.01,
        }
        app = SimulationApplication()
        summary = app.run_analysis(scenario_data=scen, seed=42)
        # Should not raise — important because the persistence layer
        # serialises summary via json.dumps before storing.
        json.dumps(summary, default=float)
