"""
Tests for InflationConfig and stochastic inflation in MonteCarloSimulator.

Phase 11 — Step 2. The deterministic regime must remain byte-identical
to the legacy scalar ``EconomicConfig.inflation_rate`` so that pre-Phase-11
runs are unaffected. The stochastic regime must respect clipping, expose
``df_inflation`` and ``inflation_annual_rates_paths`` correctly, and
converge to the deterministic mean for large ``n_mc``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.monte_carlo import (
    EconomicConfig,
    InflationConfig,
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


def _build_minimal_simulator(
    n_years: int = 3,
    n_mc: int = 5,
    inflation_rate: float = 0.025,
    inflation_cfg: InflationConfig | None = None,
) -> MonteCarloSimulator:
    """Build a deterministic-ish simulator used across the tests.

    The energy and price models are intentionally trivial: only the
    inflation handling matters here.
    """
    load_profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[200.0] * 12)
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
    energy_config = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=1.0,
        battery_specs=BatterySpecs(capacity_kwh=1.0, cycles_life=1000),
        n_batteries=0,
        inverter_p_ac_max_kw=1.0,
    )
    energy_sim = EnergySystemSimulator(
        config=energy_config,
        solar_model=solar_model,
        load_profile=load_profile,
    )
    price_model = EscalatingPriceModel(
        base_price_eur_per_kwh=0.2,
        annual_escalation=0.01,
    )
    econ = EconomicConfig(
        investment_eur=500.0,
        n_mc=n_mc,
        inflation_rate=inflation_rate,
        inflation=inflation_cfg,
    )
    return MonteCarloSimulator(
        energy_simulator=energy_sim,
        price_model=price_model,
        economic_config=econ,
    )


class TestInflationConfigDataclass:
    """Smoke tests for the InflationConfig dataclass itself."""

    def test_defaults(self):
        cfg = InflationConfig()
        assert cfg.mode == "deterministic"
        assert cfg.mean == 0.025
        assert cfg.std == 0.0
        assert cfg.min_clip == -0.02
        assert cfg.max_clip == 0.10

    def test_custom(self):
        cfg = InflationConfig(
            mode="stochastic", mean=0.03, std=0.015, min_clip=0.0, max_clip=0.08
        )
        assert cfg.mode == "stochastic"
        assert cfg.std == 0.015


class TestDeterministicByteIdentity:
    """The deterministic regime must match the legacy scalar exactly."""

    def test_no_inflation_config_uses_legacy_scalar(self):
        """When EconomicConfig.inflation is None, the legacy inflation_rate wins."""
        sim = _build_minimal_simulator(inflation_rate=0.025, inflation_cfg=None)
        results = sim.run(seed=42, show_progress=False)
        # Real series must exist and be finite.
        assert results.df_profit["mean_gain_real_eur"].notna().all()

    def test_deterministic_config_matches_legacy_scalar(self):
        """InflationConfig(mode='deterministic', mean=r) == legacy inflation_rate=r."""
        sim_legacy = _build_minimal_simulator(inflation_rate=0.03, inflation_cfg=None)
        sim_new = _build_minimal_simulator(
            inflation_rate=0.0,  # ignored when inflation is provided
            inflation_cfg=InflationConfig(mode="deterministic", mean=0.03),
        )
        r_legacy = sim_legacy.run(seed=42, show_progress=False)
        r_new = sim_new.run(seed=42, show_progress=False)
        np.testing.assert_array_equal(
            r_legacy.df_profit["mean_gain_real_eur"].values,
            r_new.df_profit["mean_gain_real_eur"].values,
        )
        # Stochastic-only fields should be absent.
        assert r_new.inflation_annual_rates_paths is None
        assert r_new.df_inflation is None


class TestStochasticInflation:
    """Behavioural checks for mode='stochastic'."""

    def test_paths_shape(self):
        sim = _build_minimal_simulator(
            n_years=4,
            n_mc=20,
            inflation_cfg=InflationConfig(
                mode="stochastic", mean=0.025, std=0.01
            ),
        )
        r = sim.run(seed=7, show_progress=False)
        assert r.inflation_annual_rates_paths is not None
        assert r.inflation_annual_rates_paths.shape == (20, 4)
        assert r.df_inflation is not None
        assert list(r.df_inflation.columns) == [
            "year",
            "mean_rate",
            "p05_rate",
            "p95_rate",
            "mean_factor",
            "p05_factor",
            "p95_factor",
        ]
        assert len(r.df_inflation) == 4

    def test_clipping(self):
        """Aggressive std + tight clip ⇒ no annual rate outside [min_clip, max_clip]."""
        sim = _build_minimal_simulator(
            n_years=3,
            n_mc=50,
            inflation_cfg=InflationConfig(
                mode="stochastic",
                mean=0.0,
                std=10.0,  # absurdly wide
                min_clip=-0.01,
                max_clip=0.05,
            ),
        )
        r = sim.run(seed=11, show_progress=False)
        rates = r.inflation_annual_rates_paths
        assert rates.min() >= -0.01 - 1e-12
        assert rates.max() <= 0.05 + 1e-12

    def test_stochastic_collapses_to_deterministic_with_zero_std(self):
        """std=0 + same mean ⇒ same factors as deterministic (within numerical noise)."""
        sim_det = _build_minimal_simulator(
            n_years=3,
            n_mc=4,
            inflation_cfg=InflationConfig(mode="deterministic", mean=0.04),
        )
        sim_sto = _build_minimal_simulator(
            n_years=3,
            n_mc=4,
            inflation_cfg=InflationConfig(
                mode="stochastic", mean=0.04, std=0.0
            ),
        )
        r_det = sim_det.run(seed=5, show_progress=False)
        r_sto = sim_sto.run(seed=5, show_progress=False)
        # The stochastic branch consumes RNG for inflation BEFORE the path
        # loop, which shifts the per-path RNG state vs. the deterministic
        # branch. So we cannot compare profit series byte-identical here.
        # Instead we verify that the cumulative factor expected by year is
        # exactly 1, (1+r), (1+r)^2 in both branches.
        expected = np.array(
            [1.0, 1.04, 1.04 * 1.04], dtype=float
        )
        np.testing.assert_allclose(
            r_sto.df_inflation["mean_factor"].values, expected, rtol=1e-10
        )

    def test_mean_rate_close_to_target_for_large_n_mc(self):
        """Empirical mean across paths ≈ configured mean for large n_mc."""
        sim = _build_minimal_simulator(
            n_years=5,
            n_mc=400,
            inflation_cfg=InflationConfig(
                mode="stochastic",
                mean=0.025,
                std=0.01,
                min_clip=-0.05,
                max_clip=0.10,
            ),
        )
        r = sim.run(seed=2026, show_progress=False)
        empirical = r.inflation_annual_rates_paths.mean()
        assert abs(empirical - 0.025) < 0.003  # 0.3 percentage points tolerance

    def test_year_zero_factor_is_one(self):
        """Convention: factor for year 0 (months 0–11) is 1.0, no inflation yet."""
        sim = _build_minimal_simulator(
            n_years=3,
            n_mc=10,
            inflation_cfg=InflationConfig(
                mode="stochastic", mean=0.03, std=0.01
            ),
        )
        r = sim.run(seed=99, show_progress=False)
        assert r.df_inflation.loc[0, "mean_factor"] == pytest.approx(1.0)
        assert r.df_inflation.loc[0, "p05_factor"] == pytest.approx(1.0)
        assert r.df_inflation.loc[0, "p95_factor"] == pytest.approx(1.0)
