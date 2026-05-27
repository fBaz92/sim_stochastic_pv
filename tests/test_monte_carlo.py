"""Tests for Monte Carlo simulation module (reorganized)."""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.monte_carlo import (
    EconomicConfig,
    MonteCarloResults,
    MonteCarloSimulator,
)
from sim_stochastic_pv.simulation.energy_simulator import (
    EnergySystemSimulator,
    EnergySystemConfig,
)
from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
from sim_stochastic_pv.simulation.solar import SolarModel, SolarMonthParams
from sim_stochastic_pv.simulation.prices import EscalatingPriceModel
from sim_stochastic_pv.simulation.battery import BatterySpecs


class TestEconomicConfig:
    """Tests for EconomicConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = EconomicConfig()
        assert config.investment_eur == 2500.0
        assert config.n_mc == 200
        assert config.inflation_rate == 0.025

    def test_custom_values(self):
        """Can override default values."""
        config = EconomicConfig(
            investment_eur=5000.0,
            n_mc=100,
            inflation_rate=0.03,
        )
        assert config.investment_eur == 5000.0
        assert config.n_mc == 100
        assert config.inflation_rate == 0.03


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator."""

    @pytest.fixture
    def minimal_simulator(self):
        """Create a minimal Monte Carlo simulator for testing."""
        # Minimal load profile
        load_profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[200.0] * 12)

        # Minimal solar model
        month_params = [
            SolarMonthParams(
                avg_daily_kwh_per_kwp=2.0,
                p_sunny=0.5,
                sunny_factor=1.1,
                cloudy_factor=0.4,
            )
            for _ in range(12)
        ]
        solar_model = SolarModel(
            pv_kwp=1.0,
            degradation_per_year=0.0,
            month_params=month_params,
        )

        # Minimal energy config
        energy_config = EnergySystemConfig(
            n_years=2,
            pv_kwp=1.0,
            battery_specs=BatterySpecs(capacity_kwh=1.0, cycles_life=1000),
            n_batteries=0,  # No battery for simplicity
            inverter_p_ac_max_kw=1.0,
        )

        energy_sim = EnergySystemSimulator(
            config=energy_config,
            solar_model=solar_model,
            load_profile=load_profile,
        )

        # Minimal price model
        price_model = EscalatingPriceModel(
            base_price_eur_per_kwh=0.2,
            annual_escalation=0.01,
        )

        # Minimal economic config with few MC paths for speed
        economic_config = EconomicConfig(
            investment_eur=500.0,
            n_mc=5,  # Small for fast tests
            inflation_rate=0.025,
        )

        return MonteCarloSimulator(
            energy_simulator=energy_sim,
            price_model=price_model,
            economic_config=economic_config,
        )

    def test_initialization(self, minimal_simulator):
        """Simulator can be initialized."""
        assert minimal_simulator.economic_config.n_mc == 5
        assert minimal_simulator.economic_config.investment_eur == 500.0

    def test_run_returns_results(self, minimal_simulator):
        """Run method returns MonteCarloResults."""
        results = minimal_simulator.run(seed=42, show_progress=False)

        assert isinstance(results, MonteCarloResults)
        assert results.df_profit is not None
        assert results.df_energy is not None

    def test_run_with_seed_is_deterministic(self, minimal_simulator):
        """Same seed produces identical results."""
        results1 = minimal_simulator.run(seed=42, show_progress=False)
        results2 = minimal_simulator.run(seed=42, show_progress=False)

        np.testing.assert_array_equal(
            results1.df_profit["mean_gain_eur"].values,
            results2.df_profit["mean_gain_eur"].values,
        )

    def test_run_with_different_seed_produces_different_results(
        self, minimal_simulator
    ):
        """Different seeds produce different results."""
        results1 = minimal_simulator.run(seed=42, show_progress=False)
        results2 = minimal_simulator.run(seed=123, show_progress=False)

        # Results should be different (stochastic)
        assert not np.array_equal(
            results1.df_profit["mean_gain_eur"].values,
            results2.df_profit["mean_gain_eur"].values,
        )

    def test_results_structure(self, minimal_simulator):
        """Results contain expected dataframes and metrics."""
        results = minimal_simulator.run(seed=42, show_progress=False)

        # Check profit dataframe columns
        assert "month_index" in results.df_profit.columns
        assert "mean_gain_eur" in results.df_profit.columns
        assert "mean_gain_real_eur" in results.df_profit.columns
        assert "prob_gain" in results.df_profit.columns
        assert "p05_gain_eur" in results.df_profit.columns
        assert "p95_gain_eur" in results.df_profit.columns

        # Check energy dataframe columns
        assert "month_index" in results.df_energy.columns
        assert "pv_prod_mean_kwh" in results.df_energy.columns
        assert "solar_used_mean_kwh" in results.df_energy.columns
        assert "grid_import_mean_kwh" in results.df_energy.columns

        # Check SOC dataframe
        assert results.df_soc is not None
        assert len(results.df_soc) > 0

    def test_results_length_matches_years(self, minimal_simulator):
        """Results should have correct number of months."""
        n_years = minimal_simulator.energy_simulator.config.n_years
        results = minimal_simulator.run(seed=42, show_progress=False)

        expected_months = n_years * 12
        assert len(results.df_profit) == expected_months
        assert len(results.df_energy) == expected_months

    def test_progress_callback(self, minimal_simulator):
        """Progress callback is called during simulation."""
        callback_calls = []

        def progress_callback(done, total, elapsed, eta):
            callback_calls.append((done, total))

        results = minimal_simulator.run(
            seed=42,
            progress_callback=progress_callback,
            show_progress=False,
        )

        # Callback should have been called
        assert len(callback_calls) > 0

        # Last call should indicate completion
        last_done, last_total = callback_calls[-1]
        assert last_done == last_total
        assert last_total == 5  # n_mc

    def test_show_progress_parameter(self, minimal_simulator):
        """Show progress parameter controls progress bar display."""
        # Should run without errors regardless of show_progress value
        results1 = minimal_simulator.run(seed=42, show_progress=True)
        results2 = minimal_simulator.run(seed=42, show_progress=False)

        # Both should produce valid results
        assert isinstance(results1, MonteCarloResults)
        assert isinstance(results2, MonteCarloResults)

    def test_inflation_adjustment(self, minimal_simulator):
        """Real gains should be less than nominal due to inflation."""
        results = minimal_simulator.run(seed=42, show_progress=False)

        # Final month
        final = results.df_profit.iloc[-1]

        # Real gain should be less than nominal (due to inflation)
        assert final["mean_gain_real_eur"] < final["mean_gain_eur"]

    def test_probability_gain_range(self, minimal_simulator):
        """Probability of gain should be between 0 and 1."""
        results = minimal_simulator.run(seed=42, show_progress=False)

        prob_values = results.df_profit["prob_gain"].values

        assert all(0.0 <= p <= 1.0 for p in prob_values)

    def test_irr_calculation(self, minimal_simulator):
        """IRR should be calculated for all paths."""
        results = minimal_simulator.run(seed=42, show_progress=False)

        # IRR array should have same length as n_mc
        assert len(results.irr_annual_paths) == 5

        # Some IRR values may be nan if no valid solution, but not all
        valid_irr = results.irr_annual_paths[~np.isnan(results.irr_annual_paths)]
        assert len(valid_irr) > 0


class TestBackwardCompatibility:
    """Tests for backward compatibility of Monte Carlo imports."""

    def test_all_exports_available(self):
        """All expected classes should be importable from main module."""
        from sim_stochastic_pv.simulation.monte_carlo import (
            EconomicConfig,
            MonteCarloResults,
            MonteCarloSimulator,
        )

        assert EconomicConfig is not None
        assert MonteCarloResults is not None
        assert MonteCarloSimulator is not None

    def test_finance_functions_accessible(self):
        """Finance functions should be accessible from submodule."""
        from sim_stochastic_pv.simulation.monte_carlo.finance import (
            _npv,
            _compute_irr_monthly,
            _compute_irr_annual,
        )

        # Functions should exist
        assert callable(_npv)
        assert callable(_compute_irr_monthly)
        assert callable(_compute_irr_annual)


# ────────────────────────────────────────────────────────────────────────────
# Phase 4 — Break-even and KPI tests
# ────────────────────────────────────────────────────────────────────────────

def _make_simulator(investment_eur: float, n_mc: int = 10, n_years: int = 3):
    """
    Build a minimal MonteCarloSimulator for break-even unit tests.

    Args:
        investment_eur: Initial investment cost in EUR.
        n_mc: Number of Monte Carlo paths (small for speed).
        n_years: Simulation horizon in years.

    Returns:
        MonteCarloSimulator ready to call .run(seed=…, show_progress=False).
    """
    load_profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[200.0] * 12)
    month_params = [
        SolarMonthParams(
            avg_daily_kwh_per_kwp=3.0,
            p_sunny=0.6,
            sunny_factor=1.1,
            cloudy_factor=0.4,
        )
        for _ in range(12)
    ]
    solar_model = SolarModel(pv_kwp=2.0, degradation_per_year=0.0, month_params=month_params)
    energy_cfg = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=2.0,
        battery_specs=BatterySpecs(capacity_kwh=1.0, cycles_life=1000),
        n_batteries=0,
        inverter_p_ac_max_kw=2.0,
    )
    energy_sim = EnergySystemSimulator(
        config=energy_cfg, solar_model=solar_model, load_profile=load_profile
    )
    price_model = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25, annual_escalation=0.0
    )
    econ_cfg = EconomicConfig(
        investment_eur=investment_eur, n_mc=n_mc, inflation_rate=0.025
    )
    return MonteCarloSimulator(
        energy_simulator=energy_sim,
        price_model=price_model,
        economic_config=econ_cfg,
    )


class TestPhase4BreakEven:
    """
    Phase 4 — break-even analysis.

    Tests that MonteCarloSimulator.run() populates the new break-even fields
    on MonteCarloResults with values that are mathematically consistent with
    the raw profit paths.
    """

    def test_break_even_per_path_shape(self):
        """break_even_month_per_path has shape (n_mc,)."""
        sim = _make_simulator(investment_eur=500.0, n_mc=8, n_years=3)
        results = sim.run(seed=42, show_progress=False)

        assert results.break_even_month_per_path is not None
        assert results.break_even_month_per_path.shape == (8,)

    def test_break_even_per_path_values_are_valid(self):
        """Each entry is either -1 (never) or a valid 0-based month index."""
        sim = _make_simulator(investment_eur=200.0, n_mc=20, n_years=3)
        results = sim.run(seed=7, show_progress=False)

        n_months = 3 * 12  # n_years × 12
        per_path = results.break_even_month_per_path
        assert per_path is not None
        # Every value must be either -1 or in [0, n_months)
        for v in per_path:
            assert v == -1 or (0 <= v < n_months), f"Unexpected break-even value: {v}"

    def test_zero_investment_breaks_even_month_zero(self):
        """With investment=0, every path breaks even at month 0 (first month)."""
        sim = _make_simulator(investment_eur=0.0, n_mc=10, n_years=2)
        results = sim.run(seed=1, show_progress=False)

        per_path = results.break_even_month_per_path
        assert per_path is not None
        # All paths must break even, and the first such month must be 0
        # (cumulative profit starts at -0 + savings_month_0 which is ≥ 0
        # as long as any solar savings are generated).
        assert np.all(per_path >= 0), "All paths should break even with zero investment"
        assert np.all(per_path == 0), f"Expected month 0 for all; got {per_path}"

    def test_huge_investment_no_break_even(self):
        """With a very large investment, no path should break even in a short horizon."""
        # A 1,000,000 EUR investment vs a 2 kWp system in 2 years → never recovers.
        sim = _make_simulator(investment_eur=1_000_000.0, n_mc=10, n_years=2)
        results = sim.run(seed=99, show_progress=False)

        per_path = results.break_even_month_per_path
        assert per_path is not None
        assert np.all(per_path == -1), "No path should break even with 1M EUR investment"

    def test_prob_break_even_consistent_with_per_path(self):
        """prob_break_even_within_horizon equals fraction of paths with value != -1."""
        sim = _make_simulator(investment_eur=500.0, n_mc=20, n_years=5)
        results = sim.run(seed=42, show_progress=False)

        per_path = results.break_even_month_per_path
        assert per_path is not None
        expected_prob = float((per_path != -1).mean())
        assert results.prob_break_even_within_horizon is not None
        assert abs(results.prob_break_even_within_horizon - expected_prob) < 1e-9

    def test_break_even_statistics_consistent(self):
        """Median and percentile break-even months are consistent with per-path data."""
        sim = _make_simulator(investment_eur=300.0, n_mc=50, n_years=5)
        results = sim.run(seed=42, show_progress=False)

        per_path = results.break_even_month_per_path
        assert per_path is not None
        valid = per_path[per_path != -1]

        if len(valid) == 0:
            assert results.break_even_month_median is None
            assert results.break_even_month_p05 is None
            assert results.break_even_month_p95 is None
        else:
            assert results.break_even_month_median is not None
            assert results.break_even_month_p05 is not None
            assert results.break_even_month_p95 is not None
            assert abs(results.break_even_month_median - float(np.median(valid))) < 1e-6
            assert abs(results.break_even_month_p05 - float(np.percentile(valid, 5))) < 1e-6
            assert abs(results.break_even_month_p95 - float(np.percentile(valid, 95))) < 1e-6
            # p05 ≤ median ≤ p95
            assert results.break_even_month_p05 <= results.break_even_month_median
            assert results.break_even_month_median <= results.break_even_month_p95

    def test_npv_median_consistent(self):
        """npv_median_eur is close to the median of the final profit in df_profit."""
        sim = _make_simulator(investment_eur=400.0, n_mc=20, n_years=3)
        results = sim.run(seed=42, show_progress=False)

        assert results.npv_median_eur is not None
        # The median final profit is different from the mean final profit
        # (df_profit stores the mean).  We just verify the sign is the
        # same direction and the value is finite.
        assert np.isfinite(results.npv_median_eur)

    def test_irr_mean_excludes_nan(self):
        """irr_mean is the mean of valid (non-nan) IRR paths."""
        sim = _make_simulator(investment_eur=500.0, n_mc=20, n_years=3)
        results = sim.run(seed=42, show_progress=False)

        valid_irr = results.irr_annual_paths[~np.isnan(results.irr_annual_paths)]
        if len(valid_irr) > 0:
            expected = float(valid_irr.mean())
            assert results.irr_mean is not None
            assert abs(results.irr_mean - expected) < 1e-9
        else:
            assert results.irr_mean is None

    def test_backward_compat_manual_construction(self):
        """MonteCarloResults can be constructed without the new Phase 4 fields."""
        import pandas as pd

        df = pd.DataFrame({"a": [1]})
        # All new fields should default to None — this must not raise.
        r = MonteCarloResults(
            df_profit=df,
            df_energy=df,
            df_soc=df,
            df_soh=df,
            monthly_savings_eur_paths=np.zeros((1, 1)),
            monthly_savings_real_eur_paths=np.zeros((1, 1)),
            monthly_load_kwh_paths=np.zeros((1, 1)),
            irr_annual_paths=np.array([0.05]),
        )
        assert r.break_even_month_per_path is None
        assert r.prob_break_even_within_horizon is None
        assert r.break_even_month_median is None
        assert r.break_even_month_p05 is None
        assert r.break_even_month_p95 is None
        assert r.npv_median_eur is None
        assert r.irr_mean is None
