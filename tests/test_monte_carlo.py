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
