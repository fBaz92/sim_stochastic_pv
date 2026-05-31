"""Tests for the market Monte Carlo runner.

Keeps every case fast by using few runs and avoiding the grid-storage
dispatch loop (the slow path). One case enables interconnections to confirm
imports lower the clearing price; that single run is ~2 s, within budget.
"""

from __future__ import annotations

import numpy as np

from sim_stochastic_pv.market import run_monte_carlo
from sim_stochastic_pv.market.config import (
    ITALIAN_MIX,
    GAS_SCENARIOS,
    PRICE_AREAS,
    INTERCONNECTIONS,
    PRICE_AREA_CORRELATIONS,
)


def test_run_shapes_and_reproducibility() -> None:
    """Outputs have the documented shapes and are reproducible by seed."""
    a = run_monte_carlo(mix_config=ITALIAN_MIX,
                        gas_scenario=GAS_SCENARIOS["base"], n_runs=2, seed=11)
    b = run_monte_carlo(mix_config=ITALIAN_MIX,
                        gas_scenario=GAS_SCENARIOS["base"], n_runs=2, seed=11)
    assert a.avg_price.shape == (2,)
    assert a.monthly_prices.shape == (2, 12)
    assert np.allclose(a.avg_price, b.avg_price)
    assert np.allclose(a.monthly_prices, b.monthly_prices)
    assert (a.avg_price > 0).all()


def test_crisis_gas_raises_average_price() -> None:
    """A crisis gas scenario yields a higher mean price than the base one."""
    base = run_monte_carlo(mix_config=ITALIAN_MIX,
                           gas_scenario=GAS_SCENARIOS["base"], n_runs=2, seed=5)
    crisis = run_monte_carlo(mix_config=ITALIAN_MIX,
                             gas_scenario=GAS_SCENARIOS["crisis"], n_runs=2,
                             seed=5)
    assert crisis.avg_price.mean() > base.avg_price.mean()


def test_interconnections_lower_price() -> None:
    """Enabling imports lowers the clearing price (imports cap merit order)."""
    closed = run_monte_carlo(mix_config=ITALIAN_MIX,
                             gas_scenario=GAS_SCENARIOS["base"], n_runs=1, seed=1)
    coupled = run_monte_carlo(
        mix_config=ITALIAN_MIX, gas_scenario=GAS_SCENARIOS["base"], n_runs=1,
        seed=1, interconnections_cfg=INTERCONNECTIONS, price_areas_cfg=PRICE_AREAS,
        price_area_correlations=PRICE_AREA_CORRELATIONS,
    )
    assert coupled.interconnection_names  # links were built
    assert coupled.avg_price[0] < closed.avg_price[0]
