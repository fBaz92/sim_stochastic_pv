"""Tests for mix evolution over the horizon and the wholesale price surface.

The capacity/mix trend logic is pure arithmetic and tested exactly. The
price-surface builder is exercised with small trajectory counts and short
horizons so each dispatch-backed case stays well under the 5 s budget
(~0.7 s per dispatch); physical expectations (summer midday cheaper than the
evening peak, gas drift lifting later-year prices) are asserted on
across-trajectory means with comfortable margins.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.market.config import ITALIAN_MIX, GAS_SCENARIOS
from sim_stochastic_pv.market.horizon import (
    TechCapacityTrend,
    MixTrend,
    PriceSurface,
    build_price_surface,
)


# ── Capacity / mix trend (pure, exact) ────────────────────────────────────

def test_capacity_growth_compounds() -> None:
    """Compound annual growth applies year on year from the base capacity."""
    t = TechCapacityTrend(base_capacity_gw=30.0, annual_growth_pct=10.0)
    assert t.capacity_for_year(0) == pytest.approx(30.0)
    assert t.capacity_for_year(1) == pytest.approx(33.0)
    assert t.capacity_for_year(2) == pytest.approx(36.3)


def test_capacity_step_introduces_technology_midhorizon() -> None:
    """A step keeps capacity at base until the step year, then re-bases."""
    nuke = TechCapacityTrend(base_capacity_gw=0.0, step_year=5,
                             step_capacity_gw=4.0, annual_growth_pct=10.0)
    assert nuke.capacity_for_year(4) == pytest.approx(0.0)
    assert nuke.capacity_for_year(5) == pytest.approx(4.0)
    assert nuke.capacity_for_year(6) == pytest.approx(4.4)  # 4 * 1.1


def test_capacity_trend_validation() -> None:
    """The trend rejects inconsistent step/capacity and negative values."""
    with pytest.raises(ValueError):
        TechCapacityTrend(base_capacity_gw=-1.0)
    with pytest.raises(ValueError):
        TechCapacityTrend(base_capacity_gw=1.0, step_year=3)  # no step capacity
    with pytest.raises(ValueError):
        TechCapacityTrend(base_capacity_gw=1.0, step_capacity_gw=2.0)  # no year


def test_build_mix_for_year_applies_trends_only_to_capacity() -> None:
    """build_mix_for_year overrides capacity_gw, leaving other params intact."""
    trend = MixTrend(
        base_mix=ITALIAN_MIX,
        capacity_trends={
            "solar": TechCapacityTrend(30.0, annual_growth_pct=10.0),
            "nuclear": TechCapacityTrend(0.0, step_year=5, step_capacity_gw=8.0),
        },
    )
    y0 = trend.build_mix_for_year(0)
    y6 = trend.build_mix_for_year(6)
    assert y0["solar"]["capacity_gw"] == pytest.approx(30.0)
    assert y6["solar"]["capacity_gw"] == pytest.approx(30.0 * 1.1 ** 6)
    # nuclear absent (0 GW) at year 0, present (flat, no growth) from year 5.
    assert y0["nuclear"]["capacity_gw"] == pytest.approx(0.0)
    assert y6["nuclear"]["capacity_gw"] == pytest.approx(8.0)
    # Untrended technology keeps its base capacity; non-capacity params kept.
    assert y6["gas"]["capacity_gw"] == ITALIAN_MIX["gas"]["capacity_gw"]
    assert y6["solar"]["capex_per_kw"] == ITALIAN_MIX["solar"]["capex_per_kw"]


def test_mixtrend_rejects_unknown_trend_technology() -> None:
    """A trend for a technology absent from the base mix is rejected."""
    with pytest.raises(ValueError):
        MixTrend(base_mix={"gas": ITALIAN_MIX["gas"]},
                 capacity_trends={"solar": TechCapacityTrend(10.0)})


def test_fuel_drift_scales_mu() -> None:
    """Gas/CO2 mean prices drift compoundly; sigma/theta are unchanged."""
    trend = MixTrend(base_mix=ITALIAN_MIX, gas_mu_drift_annual=0.02,
                     co2_mu_drift_annual=0.05)
    base = {"mu": 35.0, "sigma": 8.0, "theta": 0.1}
    g2 = trend.gas_scenario_for_year(base, 2)
    assert g2["mu"] == pytest.approx(35.0 * 1.02 ** 2)
    assert g2["sigma"] == 8.0 and g2["theta"] == 0.1
    c3 = trend.co2_scenario_for_year({"mu": 65.0, "sigma": 10.0, "theta": 0.05}, 3)
    assert c3["mu"] == pytest.approx(65.0 * 1.05 ** 3)


# ── Price surface (dispatch-backed, kept small) ───────────────────────────

def _flat_trend() -> MixTrend:
    """A flat default mix (no trends), the cheapest surface to build."""
    return MixTrend(base_mix=ITALIAN_MIX)


def test_surface_shape_units_and_helpers() -> None:
    """The surface has shape (K, N, 12, 24), sane EUR/kWh values, and helpers."""
    s = build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=2, n_trajectories=2, seed=0)
    assert isinstance(s, PriceSurface)
    assert s.price_eur_per_kwh.shape == (2, 2, 12, 24)
    assert (s.price_eur_per_kwh > 0).all()
    assert (s.price_eur_per_kwh < 1.0).all()  # EUR/kWh, not EUR/MWh
    assert s.mean_grid().shape == (2, 12, 24)
    assert s.percentile_grid(95).shape == (2, 12, 24)
    assert s.trajectory_grid(0).shape == (2, 12, 24)
    # price_at uses 1-based month; matches the underlying array.
    assert s.price_at(1, 0, 7, 13) == pytest.approx(
        s.price_eur_per_kwh[1, 0, 6, 13])


def test_surface_reproducible_with_seed() -> None:
    """Same seed yields a byte-identical surface."""
    a = build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=1, n_trajectories=2, seed=7)
    b = build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=1, n_trajectories=2, seed=7)
    assert np.allclose(a.price_eur_per_kwh, b.price_eur_per_kwh)


def test_surface_summer_midday_cheaper_than_evening() -> None:
    """Solar pushes the summer midday price below the evening peak."""
    s = build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=1, n_trajectories=3, seed=0)
    g = s.mean_grid()[0]  # year 0, shape (12, 24)
    july = g[6]           # month index 6 = July
    midday = july[11:15].mean()   # 11:00-14:00
    evening = july[19:22].mean()  # 19:00-21:00
    assert midday < evening


def test_surface_gas_drift_raises_later_year_price() -> None:
    """A positive gas drift lifts the mean price in later horizon years."""
    trend = MixTrend(base_mix=ITALIAN_MIX, gas_mu_drift_annual=0.5)
    s = build_price_surface(trend, GAS_SCENARIOS["base"],
                            n_years=2, n_trajectories=2, seed=0)
    year_means = s.price_eur_per_kwh.mean(axis=(0, 2, 3))  # (n_years,)
    assert year_means[1] > year_means[0]


def test_surface_interpolates_intermediate_years() -> None:
    """Unsimulated years are the linear midpoint of the representative years."""
    s = build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=3, n_trajectories=2,
                            representative_years=[0, 2], seed=0)
    surf = s.price_eur_per_kwh  # (2, 3, 12, 24)
    midpoint = 0.5 * (surf[:, 0] + surf[:, 2])
    assert np.allclose(surf[:, 1], midpoint)


def test_build_surface_validation() -> None:
    """Invalid horizon / trajectory / representative-year inputs are rejected."""
    with pytest.raises(ValueError):
        build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=0, n_trajectories=2)
    with pytest.raises(ValueError):
        build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=2, n_trajectories=0)
    with pytest.raises(ValueError):
        build_price_surface(_flat_trend(), GAS_SCENARIOS["base"],
                            n_years=2, n_trajectories=2,
                            representative_years=[0, 5])  # out of range
