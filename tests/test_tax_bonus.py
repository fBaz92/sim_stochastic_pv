"""
Tests for TaxBonusConfig integration in MonteCarloSimulator.

Phase 11 — Step 3. The tax bonus is modelled as a sparse cash inflow
placed at the end of each of the first ``duration_years`` years (month
indices 11, 23, 35, ...). Disabled bonuses must leave the simulator
output untouched; enabled bonuses must improve IRR and break-even.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.monte_carlo import (
    EconomicConfig,
    MonteCarloSimulator,
    TaxBonusConfig,
)
from sim_stochastic_pv.simulation.energy_simulator import (
    EnergySystemSimulator,
    EnergySystemConfig,
)
from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
from sim_stochastic_pv.simulation.solar import SolarModel, SolarMonthParams
from sim_stochastic_pv.simulation.prices import EscalatingPriceModel
from sim_stochastic_pv.simulation.battery import BatterySpecs


def _build_minimal_simulator(
    n_years: int = 4,
    n_mc: int = 5,
    investment_eur: float = 6000.0,
    tax_bonus: TaxBonusConfig | None = None,
) -> MonteCarloSimulator:
    """Tiny simulator wired with deterministic-ish components for fast tests."""
    load_profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[300.0] * 12)
    month_params = [
        SolarMonthParams(
            avg_daily_kwh_per_kwp=3.5,
            p_sunny=0.6,
            sunny_factor=1.1,
            cloudy_factor=0.5,
        )
        for _ in range(12)
    ]
    solar_model = SolarModel(
        pv_kwp=3.0,
        degradation_per_year=0.0,
        month_params=month_params,
    )
    energy_config = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=3.0,
        battery_specs=BatterySpecs(capacity_kwh=5.0, cycles_life=5000),
        n_batteries=0,
        inverter_p_ac_max_kw=3.0,
    )
    energy_sim = EnergySystemSimulator(
        config=energy_config,
        solar_model=solar_model,
        load_profile=load_profile,
    )
    price_model = EscalatingPriceModel(
        base_price_eur_per_kwh=0.22,
        annual_escalation=0.02,
    )
    econ = EconomicConfig(
        investment_eur=investment_eur,
        n_mc=n_mc,
        tax_bonus=tax_bonus,
    )
    return MonteCarloSimulator(
        energy_simulator=energy_sim,
        price_model=price_model,
        economic_config=econ,
    )


class TestTaxBonusDataclass:
    def test_defaults(self):
        cfg = TaxBonusConfig()
        assert cfg.enabled is False
        assert cfg.fraction_of_investment == 0.5
        assert cfg.duration_years == 10


class TestBonusDisabled:
    """When disabled, the bonus must be a true no-op."""

    def test_disabled_bonus_matches_no_bonus(self):
        sim_none = _build_minimal_simulator(tax_bonus=None)
        sim_disabled = _build_minimal_simulator(
            tax_bonus=TaxBonusConfig(enabled=False)
        )
        r_none = sim_none.run(seed=42, show_progress=False)
        r_disabled = sim_disabled.run(seed=42, show_progress=False)
        np.testing.assert_array_equal(
            r_none.df_profit["mean_gain_eur"].values,
            r_disabled.df_profit["mean_gain_eur"].values,
        )
        assert r_disabled.tax_bonus_total_eur == 0.0
        assert np.all(r_disabled.bonus_per_month_eur == 0.0)


class TestBonusEnabled:
    """When enabled, the bonus must be sparse, total-correct, and improve IRR."""

    def test_sparsity_pattern_year_end_only(self):
        sim = _build_minimal_simulator(
            n_years=4,
            tax_bonus=TaxBonusConfig(
                enabled=True, fraction_of_investment=0.5, duration_years=4
            ),
        )
        r = sim.run(seed=1, show_progress=False)
        bonus = r.bonus_per_month_eur
        # Non-zero only at month indices 11, 23, 35, 47.
        nonzero_idx = np.flatnonzero(bonus)
        np.testing.assert_array_equal(nonzero_idx, [11, 23, 35, 47])

    def test_total_equals_investment_times_fraction(self):
        """When duration_years <= n_years, total bonus = investment * fraction."""
        sim = _build_minimal_simulator(
            n_years=10,
            investment_eur=8000.0,
            tax_bonus=TaxBonusConfig(
                enabled=True, fraction_of_investment=0.5, duration_years=10
            ),
        )
        r = sim.run(seed=1, show_progress=False)
        assert r.tax_bonus_total_eur == pytest.approx(8000.0 * 0.5)

    def test_truncation_when_duration_exceeds_horizon(self):
        """duration_years > n_years ⇒ user loses the late instalments."""
        sim = _build_minimal_simulator(
            n_years=5,  # horizon too short
            investment_eur=6000.0,
            tax_bonus=TaxBonusConfig(
                enabled=True,
                fraction_of_investment=0.5,
                duration_years=10,  # only 5 of 10 are paid
            ),
        )
        r = sim.run(seed=1, show_progress=False)
        expected_yearly = 6000.0 * 0.5 / 10
        # 5 years × yearly instalment = half of what the user would
        # have received with a long enough horizon.
        assert r.tax_bonus_total_eur == pytest.approx(expected_yearly * 5)
        # Bonus is paid in months 11, 23, 35, 47, 59 — total 5 entries.
        assert np.count_nonzero(r.bonus_per_month_eur) == 5

    def test_bonus_increases_final_gain_and_irr(self):
        """Enabling the bonus must strictly increase mean gain (nominal)."""
        no_bonus = _build_minimal_simulator(
            n_years=10,
            investment_eur=6000.0,
            tax_bonus=None,
        )
        with_bonus = _build_minimal_simulator(
            n_years=10,
            investment_eur=6000.0,
            tax_bonus=TaxBonusConfig(
                enabled=True, fraction_of_investment=0.5, duration_years=10
            ),
        )
        r_no = no_bonus.run(seed=42, show_progress=False)
        r_yes = with_bonus.run(seed=42, show_progress=False)

        final_no = r_no.df_profit["mean_gain_eur"].iloc[-1]
        final_yes = r_yes.df_profit["mean_gain_eur"].iloc[-1]
        assert final_yes - final_no == pytest.approx(6000.0 * 0.5, rel=1e-9)

        # Mean IRR should also be strictly higher (more cash inflows
        # earlier, same investment).
        irr_no = np.nanmean(r_no.irr_annual_paths)
        irr_yes = np.nanmean(r_yes.irr_annual_paths)
        assert irr_yes > irr_no

    def test_bonus_present_in_real_curve_too(self):
        """The real (inflation-adjusted) curve must also reflect the bonus."""
        no_bonus = _build_minimal_simulator(
            n_years=10, investment_eur=6000.0, tax_bonus=None
        )
        with_bonus = _build_minimal_simulator(
            n_years=10,
            investment_eur=6000.0,
            tax_bonus=TaxBonusConfig(
                enabled=True, fraction_of_investment=0.5, duration_years=10
            ),
        )
        r_no = no_bonus.run(seed=42, show_progress=False)
        r_yes = with_bonus.run(seed=42, show_progress=False)
        final_real_no = r_no.df_profit["mean_gain_real_eur"].iloc[-1]
        final_real_yes = r_yes.df_profit["mean_gain_real_eur"].iloc[-1]
        # Bonus erodes with inflation: increment must be POSITIVE but
        # strictly LESS than the nominal 3000 EUR.
        delta_real = final_real_yes - final_real_no
        assert 0 < delta_real < 3000.0
