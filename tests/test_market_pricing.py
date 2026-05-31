"""
Tests for the market price provider and its economic integration.

Covers three concerns:

* :class:`MarketPriceProvider` arithmetic — the *ritiro dedicato* export price
  ``max(wholesale, PMG)``, the inflation-indexed PMG floor, monthly aggregation,
  trajectory wrapping, horizon clamping, and the optional retail tariff.
* The :meth:`PriceModel.get_price_hourly` hook — its default must be flat across
  the day so any monthly-only consumer stays byte-identical.
* The Monte Carlo integration — a run without a provider is byte-identical to a
  run whose provider values everything at zero, and a run with a real surface
  folds the export revenue into the profit/IRR/break-even curves.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.market import PriceSurface
from sim_stochastic_pv.market.config import GAS_SCENARIOS, ITALIAN_MIX
from sim_stochastic_pv.market.horizon import MixTrend, build_price_surface
from sim_stochastic_pv.simulation import (
    BatterySpecs,
    EconomicConfig,
    EnergySystemConfig,
    EnergySystemSimulator,
    EscalatingPriceModel,
    GBMPriceModel,
    MarketPriceProvider,
    MonteCarloSimulator,
    MonthlyAverageLoadProfile,
    SolarModel,
)
from sim_stochastic_pv.simulation.solar import make_default_solar_params_for_pavullo


# ── Fixtures / builders ───────────────────────────────────────────────────


def _flat_surface(
    value: float,
    *,
    n_trajectories: int = 1,
    n_years: int = 2,
) -> PriceSurface:
    """A surface with a single constant wholesale price everywhere."""
    grid = np.full((n_trajectories, n_years, 12, 24), float(value))
    return PriceSurface(
        price_eur_per_kwh=grid,
        n_trajectories=n_trajectories,
        n_years=n_years,
    )


def _per_year_surface(year_values, *, n_trajectories: int = 1) -> PriceSurface:
    """A surface whose price is constant within a year but varies by year."""
    year_values = np.asarray(year_values, dtype=float)
    n_years = year_values.size
    grid = np.empty((n_trajectories, n_years, 12, 24))
    for y in range(n_years):
        grid[:, y, :, :] = year_values[y]
    return PriceSurface(
        price_eur_per_kwh=grid,
        n_trajectories=n_trajectories,
        n_years=n_years,
    )


def _make_simulator(n_years: int = 2) -> EnergySystemSimulator:
    """A small PV-heavy system that exports surplus most sunny days."""
    solar = SolarModel(
        pv_kwp=6.0,
        month_params=make_default_solar_params_for_pavullo(),
        degradation_per_year=0.0,
    )
    load = MonthlyAverageLoadProfile(monthly_avg_kwh=[150.0] * 12)
    config = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=6.0,
        n_batteries=1,
        battery_specs=BatterySpecs(capacity_kwh=5.0),
        inverter_p_ac_max_kw=4.0,
    )
    return EnergySystemSimulator(config, solar, load)


# ── MarketPriceProvider: constructor validation ───────────────────────────


def test_provider_rejects_non_price_surface():
    with pytest.raises(TypeError):
        MarketPriceProvider(object())  # type: ignore[arg-type]


def test_provider_rejects_wrong_surface_shape():
    bad = PriceSurface(
        price_eur_per_kwh=np.zeros((2, 12, 24)),  # 3-D, missing year axis
        n_trajectories=2,
        n_years=1,
    )
    with pytest.raises(ValueError):
        MarketPriceProvider(bad)


def test_provider_rejects_negative_pmg():
    with pytest.raises(ValueError):
        MarketPriceProvider(_flat_surface(0.05), pmg_base_eur_per_kwh=-0.01)


def test_provider_rejects_markup_below_minus_one():
    with pytest.raises(ValueError):
        MarketPriceProvider(_flat_surface(0.05), retail_markup_fraction=-1.5)


def test_provider_rejects_negative_fixed_components():
    with pytest.raises(ValueError):
        MarketPriceProvider(
            _flat_surface(0.05),
            retail_markup_fraction=0.5,
            retail_fixed_components_eur_per_kwh=-0.1,
        )


# ── export_price_grid: max(wholesale, PMG) ─────────────────────────────────


def test_export_price_floor_binds_when_wholesale_low():
    surface = _per_year_surface([0.03, 0.10])
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.05)
    grid = provider.export_price_grid(
        trajectory_index=0, inflation_factor_by_year=np.array([1.0, 1.0])
    )
    assert grid.shape == (2, 12, 24)
    # Year 0: PMG 0.05 > wholesale 0.03 → floor binds.
    assert np.allclose(grid[0], 0.05)
    # Year 1: wholesale 0.10 > PMG 0.05 → market price.
    assert np.allclose(grid[1], 0.10)


def test_export_price_pmg_escalates_with_inflation():
    surface = _flat_surface(0.0, n_years=2)  # wholesale never binds
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.10)
    grid = provider.export_price_grid(
        trajectory_index=0, inflation_factor_by_year=np.array([1.0, 1.05])
    )
    assert np.allclose(grid[0], 0.10)
    assert np.allclose(grid[1], 0.105)


def test_export_price_pmg_zero_reduces_to_wholesale():
    surface = _per_year_surface([0.07, 0.09])
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.0)
    grid = provider.export_price_grid(
        trajectory_index=0, inflation_factor_by_year=np.array([1.0, 1.0])
    )
    assert np.allclose(grid[0], 0.07)
    assert np.allclose(grid[1], 0.09)


# ── value_export_grid: monthly aggregation ─────────────────────────────────


def test_value_export_grid_exact_revenue_and_energy():
    surface = _per_year_surface([0.03, 0.10])
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.05)

    export = np.zeros((2, 12, 24))
    export[0, 0, 10] = 2.0  # year 0, January, 10:00 → 2 kWh
    export[1, 5, 15] = 4.0  # year 1, June, 15:00 → 4 kWh

    eur, kwh = provider.value_export_grid(
        export,
        trajectory_index=0,
        inflation_factor_by_year=np.array([1.0, 1.0]),
    )
    assert eur.shape == (24,) and kwh.shape == (24,)
    # Month 0 (y0 Jan): 2 kWh × max(0.03, 0.05) = 0.10 EUR.
    assert eur[0] == pytest.approx(0.10)
    assert kwh[0] == pytest.approx(2.0)
    # Month 17 (y1 m5=June): 4 kWh × max(0.10, 0.05) = 0.40 EUR.
    assert eur[17] == pytest.approx(0.40)
    assert kwh[17] == pytest.approx(4.0)
    # Everything else is zero.
    mask = np.ones(24, dtype=bool)
    mask[[0, 17]] = False
    assert np.allclose(eur[mask], 0.0)
    assert np.allclose(kwh[mask], 0.0)


def test_value_export_grid_rejects_bad_export_shape():
    provider = MarketPriceProvider(_flat_surface(0.05))
    with pytest.raises(ValueError):
        provider.value_export_grid(
            np.zeros((2, 12, 23)),  # 23 hours
            trajectory_index=0,
            inflation_factor_by_year=np.array([1.0, 1.0]),
        )


def test_value_export_grid_rejects_year_count_mismatch():
    provider = MarketPriceProvider(_flat_surface(0.05))
    with pytest.raises(ValueError):
        provider.value_export_grid(
            np.zeros((2, 12, 24)),
            trajectory_index=0,
            inflation_factor_by_year=np.array([1.0, 1.0, 1.0]),  # 3 ≠ 2
        )


# ── trajectory wrapping & horizon clamping ─────────────────────────────────


def test_trajectory_index_wraps_modulo():
    grid = np.empty((2, 1, 12, 24))
    grid[0] = 0.01
    grid[1] = 0.09
    surface = PriceSurface(price_eur_per_kwh=grid, n_trajectories=2, n_years=1)
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.0)
    # index 5 % 2 == 1 → trajectory 1 (0.09).
    out = provider.export_price_grid(
        trajectory_index=5, inflation_factor_by_year=np.array([1.0])
    )
    assert np.allclose(out, 0.09)


def test_horizon_clamping_repeats_last_year_when_surface_shorter():
    surface = _per_year_surface([0.02, 0.08])  # 2 years
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.0)
    grid = provider.export_price_grid(
        trajectory_index=0,
        inflation_factor_by_year=np.array([1.0, 1.0, 1.0, 1.0]),  # 4 years
    )
    assert grid.shape == (4, 12, 24)
    assert np.allclose(grid[1], 0.08)
    assert np.allclose(grid[2], 0.08)  # last year repeated
    assert np.allclose(grid[3], 0.08)


def test_horizon_clamping_truncates_when_surface_longer():
    surface = _per_year_surface([0.02, 0.08, 0.20])  # 3 years
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.0)
    grid = provider.export_price_grid(
        trajectory_index=0, inflation_factor_by_year=np.array([1.0])  # 1 year
    )
    assert grid.shape == (1, 12, 24)
    assert np.allclose(grid[0], 0.02)


# ── retail tariff (optional feature) ───────────────────────────────────────


def test_retail_price_grid_raises_when_not_configured():
    provider = MarketPriceProvider(_flat_surface(0.05))
    with pytest.raises(ValueError):
        provider.retail_price_grid(
            trajectory_index=0, inflation_factor_by_year=np.array([1.0, 1.0])
        )


def test_retail_price_grid_formula():
    surface = _flat_surface(0.05, n_years=2)
    provider = MarketPriceProvider(
        surface,
        retail_markup_fraction=0.8,
        retail_fixed_components_eur_per_kwh=0.10,
    )
    grid = provider.retail_price_grid(
        trajectory_index=0, inflation_factor_by_year=np.array([1.0, 1.0])
    )
    # 0.05 * 1.8 + 0.10 = 0.19.
    assert np.allclose(grid, 0.19)


# ── PriceModel.get_price_hourly default is flat across the day ──────────────


def test_get_price_hourly_flat_for_escalating_model():
    model = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25,
        annual_escalation=0.02,
        use_stochastic_escalation=False,
    )
    model.reset_for_run(rng=np.random.default_rng(0), n_years=3)
    for year in range(3):
        for month in range(12):
            monthly = model.get_price(year, month)
            hourly = [model.get_price_hourly(year, month, h) for h in range(24)]
            assert all(abs(h - monthly) < 1e-12 for h in hourly)


def test_get_price_hourly_flat_for_gbm_model():
    model = GBMPriceModel(
        base_price_eur_per_kwh=0.25,
        drift_annual=0.02,
        volatility_annual=0.12,
    )
    model.reset_for_run(rng=np.random.default_rng(1), n_years=2)
    for year in range(2):
        for month in range(12):
            monthly = model.get_price(year, month)
            for h in range(24):
                assert model.get_price_hourly(year, month, h) == pytest.approx(monthly)


# ── Monte Carlo integration ────────────────────────────────────────────────


def test_mc_without_provider_leaves_export_fields_empty():
    esim = _make_simulator(n_years=2)
    price = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25, use_stochastic_escalation=False
    )
    econ = EconomicConfig(investment_eur=9000.0, n_mc=4, inflation_rate=0.02)
    res = MonteCarloSimulator(esim, price, econ).run(seed=42, show_progress=False)
    assert res.df_export is None
    assert res.monthly_export_eur_paths is None
    assert res.monthly_export_kwh_paths is None
    assert res.export_revenue_total_mean_eur == 0.0
    assert res.export_kwh_total_mean == 0.0


def test_mc_zero_price_provider_is_byte_identical_to_market_off():
    """A provider that values everything at zero must not perturb the run."""
    esim = _make_simulator(n_years=2)
    price = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25, use_stochastic_escalation=False
    )
    econ = EconomicConfig(investment_eur=9000.0, n_mc=5, inflation_rate=0.02)

    res_off = MonteCarloSimulator(esim, price, econ).run(seed=7, show_progress=False)

    zero_provider = MarketPriceProvider(
        _flat_surface(0.0, n_trajectories=3, n_years=2), pmg_base_eur_per_kwh=0.0
    )
    res_zero = MonteCarloSimulator(
        esim, price, econ, market_price_provider=zero_provider
    ).run(seed=7, show_progress=False)

    # Savings, profit and IRR are byte-identical: export revenue added zero.
    assert np.array_equal(
        res_off.monthly_savings_eur_paths, res_zero.monthly_savings_eur_paths
    )
    assert np.array_equal(
        res_off.df_profit["mean_gain_eur"].to_numpy(),
        res_zero.df_profit["mean_gain_eur"].to_numpy(),
    )
    # The provider run still exposes the (all-zero) export diagnostics.
    assert res_zero.df_export is not None
    assert np.allclose(res_zero.monthly_export_eur_paths, 0.0)
    assert res_zero.export_revenue_total_mean_eur == pytest.approx(0.0)


def test_mc_export_revenue_folds_into_profit_and_savings():
    esim = _make_simulator(n_years=2)
    price = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25, use_stochastic_escalation=False
    )
    econ = EconomicConfig(investment_eur=9000.0, n_mc=5, inflation_rate=0.02)

    res_off = MonteCarloSimulator(esim, price, econ).run(seed=11, show_progress=False)

    provider = MarketPriceProvider(
        _flat_surface(0.05, n_trajectories=3, n_years=2),
        pmg_base_eur_per_kwh=0.07,
    )
    res_on = MonteCarloSimulator(
        esim, price, econ, market_price_provider=provider
    ).run(seed=11, show_progress=False)

    # Export revenue is strictly positive (PV-heavy system exports surplus).
    assert res_on.export_revenue_total_mean_eur > 0.0
    assert res_on.export_kwh_total_mean > 0.0
    assert res_on.df_export.shape[0] == 2 * 12

    # The savings difference equals exactly the export revenue, element-wise:
    # same seed → identical energy/RNG; the only delta is the folded export.
    delta = (
        res_on.monthly_savings_eur_paths - res_off.monthly_savings_eur_paths
    )
    assert np.allclose(delta, res_on.monthly_export_eur_paths)

    # The final mean profit improves by exactly the mean total export revenue
    # (no tax bonus configured here).
    final_off = float(res_off.df_profit["mean_gain_eur"].iloc[-1])
    final_on = float(res_on.df_profit["mean_gain_eur"].iloc[-1])
    assert final_on - final_off == pytest.approx(
        res_on.export_revenue_total_mean_eur, rel=1e-9
    )

    # A strictly positive cash inflow cannot worsen the median NPV.
    assert res_on.npv_median_eur >= res_off.npv_median_eur


def test_energy_simulator_surfaces_hourly_self_consumption():
    esim = _make_simulator(n_years=2)
    rng = np.random.default_rng(0)
    out = esim.run_one_path(rng)
    self_cons = esim.last_self_consumption_kwh_by_year_month_hour
    assert self_cons is not None
    assert self_cons.shape == (2, 12, 24)
    # Monthly self-consumption == load served by PV+battery == savings kWh
    # (= monthly_load - monthly_grid_import). out tuple: [3]=grid_import,[4]=load.
    monthly_grid_import = out[3]
    monthly_load = out[4]
    monthly_savings_kwh = monthly_load - monthly_grid_import
    monthly_self_cons = self_cons.sum(axis=2).reshape(-1)
    assert np.allclose(monthly_self_cons, monthly_savings_kwh, atol=1e-9)


def test_mc_market_drives_purchase_matches_flat_retail():
    """Flat market retail R must reproduce a flat PriceModel at R."""
    esim = _make_simulator(n_years=2)
    econ = EconomicConfig(investment_eur=9000.0, n_mc=4, inflation_rate=0.0)

    # Flat wholesale 0.10, markup 0.5, fixed 0.05 → retail 0.20 EUR/kWh flat.
    surface = _flat_surface(0.10, n_trajectories=3, n_years=2)
    provider = MarketPriceProvider(
        surface,
        pmg_base_eur_per_kwh=0.0,
        retail_markup_fraction=0.5,
        retail_fixed_components_eur_per_kwh=0.05,
    )
    flat_price = EscalatingPriceModel(
        base_price_eur_per_kwh=0.20,
        annual_escalation=0.0,
        use_stochastic_escalation=False,
        seasonal_factors=[0.0] * 12,
    )

    baseline = MonteCarloSimulator(esim, flat_price, econ).run(seed=5, show_progress=False)
    market = MonteCarloSimulator(
        esim,
        flat_price,
        econ,
        market_price_provider=provider,
        value_export=False,  # isolate the purchase side
        market_drives_purchase=True,
    ).run(seed=5, show_progress=False)

    # Same RNG, same kWh, flat retail == flat price → identical savings.
    assert np.allclose(
        baseline.monthly_savings_eur_paths, market.monthly_savings_eur_paths
    )
    # value_export=False → no export diagnostics surfaced.
    assert market.df_export is None
    assert market.monthly_export_eur_paths is None


def test_mc_market_drives_purchase_requires_retail():
    esim = _make_simulator(n_years=2)
    econ = EconomicConfig(investment_eur=9000.0, n_mc=2, inflation_rate=0.0)
    # Provider WITHOUT a retail configuration.
    provider = MarketPriceProvider(_flat_surface(0.05, n_years=2))
    sim = MonteCarloSimulator(
        esim,
        EscalatingPriceModel(use_stochastic_escalation=False),
        econ,
        market_price_provider=provider,
        market_drives_purchase=True,
    )
    with pytest.raises(ValueError):
        sim.run(seed=1, show_progress=False)


def test_cashflow_table_itemizes_export_revenue():
    """The economic cash-flow table breaks out export revenue as its own
    column, consistent with the total KPI, and zero when no market is wired."""
    from sim_stochastic_pv.application import _build_cashflow_table_payload

    esim = _make_simulator(n_years=2)
    econ = EconomicConfig(investment_eur=9000.0, n_mc=4, inflation_rate=0.02)
    price = EscalatingPriceModel(use_stochastic_escalation=False)
    provider = MarketPriceProvider(
        _flat_surface(0.05, n_trajectories=3, n_years=2), pmg_base_eur_per_kwh=0.08
    )

    res = MonteCarloSimulator(
        esim, price, econ, market_price_provider=provider
    ).run(seed=11, show_progress=False)
    cf = _build_cashflow_table_payload(res)
    assert "export_eur" in cf
    assert sum(cf["export_eur"]) == pytest.approx(
        res.export_revenue_total_mean_eur, rel=1e-9
    )

    res0 = MonteCarloSimulator(esim, price, econ).run(seed=11, show_progress=False)
    cf0 = _build_cashflow_table_payload(res0)
    assert all(x == 0.0 for x in cf0["export_eur"])


def test_mc_with_real_built_surface_end_to_end():
    """Wiring works against an actual (small) market-built price surface."""
    esim = _make_simulator(n_years=2)
    price = EscalatingPriceModel(
        base_price_eur_per_kwh=0.25, use_stochastic_escalation=False
    )
    econ = EconomicConfig(investment_eur=9000.0, n_mc=4, inflation_rate=0.02)

    trend = MixTrend(base_mix=ITALIAN_MIX)
    surface = build_price_surface(
        trend, GAS_SCENARIOS["base"], n_years=2, n_trajectories=2, seed=0
    )
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.04)
    res = MonteCarloSimulator(
        esim, price, econ, market_price_provider=provider
    ).run(seed=3, show_progress=False)

    assert res.df_export is not None
    assert res.export_revenue_total_mean_eur > 0.0
    assert res.monthly_export_eur_paths.shape == (4, 2 * 12)
