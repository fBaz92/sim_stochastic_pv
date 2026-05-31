"""Tests for fuel/CO2 price models, availability models, and the factory.

The stochastic price processes are validated on statistical properties
(stationary mean, positive lag-1 autocorrelation) rather than individual
values. Availability models are checked on physical invariants (solar zero
at night, capacity factors in [0, 1]).
"""

from __future__ import annotations

import numpy as np

from sim_stochastic_pv.market.config import ITALIAN_MIX, GAS_SCENARIOS
from sim_stochastic_pv.market.grid import TimeGrid
from sim_stochastic_pv.market.generators import (
    FuelPriceModel,
    ConstantFuelPrice,
    CarbonPriceModel,
    SolarAvailability,
    WindAvailability,
    build_generators,
)

_N = 35040


def test_fuel_price_path_shape_floor_and_reproducible() -> None:
    """O-U fuel path is floored at 1.0, starts at mu, and is seed-stable."""
    fm = FuelPriceModel(mu=35.0, sigma=8.0, theta=0.1)
    p1 = fm.generate_path(_N, np.random.default_rng(0))
    p2 = fm.generate_path(_N, np.random.default_rng(0))
    assert p1.shape == (_N,)
    assert np.allclose(p1, p2)
    assert p1[0] == 35.0
    assert (p1 >= 1.0).all()


def test_fuel_price_is_mean_reverting() -> None:
    """The O-U path stays near mu and has strong lag-1 autocorrelation."""
    fm = FuelPriceModel(mu=50.0, sigma=10.0, theta=0.3)
    p = fm.generate_path(_N, np.random.default_rng(3))
    assert abs(p.mean() - 50.0) < 15.0
    lag1 = np.corrcoef(p[:-1], p[1:])[0, 1]
    assert lag1 > 0.9


def test_constant_fuel_price() -> None:
    """ConstantFuelPrice returns a flat path at the configured price."""
    p = ConstantFuelPrice(3.0).generate_path(_N)
    assert p.shape == (_N,) and np.all(p == 3.0)


def test_carbon_price_path() -> None:
    """CO2 O-U path starts at mu, is floored at 1.0, reproducible."""
    cm = CarbonPriceModel(mu=65.0, sigma=10.0, theta=0.05)
    a = cm.generate_path(_N, np.random.default_rng(1))
    b = cm.generate_path(_N, np.random.default_rng(1))
    assert a[0] == 65.0 and (a >= 1.0).all() and np.allclose(a, b)


def test_solar_zero_at_night_and_bounded() -> None:
    """Solar availability is zero at night and stays within [0, 1]."""
    tg = TimeGrid()
    prof = SolarAvailability().generate_profile(tg, np.random.default_rng(2))
    assert prof.shape == (tg.n,)
    assert prof.min() >= 0.0 and prof.max() <= 1.0
    night = np.isin(tg.hour, [0, 1, 2, 3, 4, 5, 21, 22, 23])
    assert np.all(prof[night] == 0.0)
    assert prof[~night].max() > 0.0


def test_wind_profile_bounded() -> None:
    """Wind capacity factor lies within [0, 1]."""
    tg = TimeGrid()
    prof = WindAvailability().generate_profile(tg, np.random.default_rng(4))
    assert prof.shape == (tg.n,)
    assert prof.min() >= 0.0 and prof.max() <= 1.0


def test_build_generators_skips_zero_capacity_and_routes_fuel() -> None:
    """Factory drops zero-capacity techs and assigns the right fuel model."""
    gens = build_generators(ITALIAN_MIX, GAS_SCENARIOS["base"])
    by_type = {g.gen_type: g for g in gens}
    assert "nuclear" not in by_type and "coal" not in by_type  # capacity 0
    assert "gas" in by_type and "solar" in by_type
    assert isinstance(by_type["gas"].fuel_model, FuelPriceModel)
    assert by_type["solar"].fuel_model is None


def test_generator_srmc_increases_with_fuel() -> None:
    """A gas unit's SRMC rises when the gas mean price rises."""
    tg = TimeGrid()
    co2 = CarbonPriceModel()
    cheap = build_generators({"gas": ITALIAN_MIX["gas"]},
                             {"mu": 20.0, "sigma": 0.0, "theta": 0.1})[0]
    dear = build_generators({"gas": ITALIAN_MIX["gas"]},
                            {"mu": 80.0, "sigma": 0.0, "theta": 0.1})[0]
    cheap.prepare_run(tg, np.random.default_rng(0), co2)
    dear.prepare_run(tg, np.random.default_rng(0), co2)
    assert dear.srmc().mean() > cheap.srmc().mean()
