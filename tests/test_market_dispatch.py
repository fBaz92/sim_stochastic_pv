"""Tests for the merit-order dispatch engine.

Uses a small all-synchronous mix (gas + must-run hydro) so the inertia-fix
phase never triggers and curtailment stays zero: under those conditions the
clearing price equals exactly the highest SRMC among dispatched units, which
makes the marginal-pricing invariant cheaply checkable.
"""

from __future__ import annotations

import numpy as np

from sim_stochastic_pv.market.config import ITALIAN_MIX, GAS_SCENARIOS
from sim_stochastic_pv.market.grid import TimeGrid, LoadProfile
from sim_stochastic_pv.market.generators import CarbonPriceModel, build_generators
from sim_stochastic_pv.market.dispatch import dispatch_year


def _sync_mix() -> dict:
    """A two-technology synchronous mix: gas + must-run hydro."""
    return {"gas": ITALIAN_MIX["gas"],
            "hydro_mustrun": ITALIAN_MIX["hydro_mustrun"]}


def _run(gas_scenario: dict, seed: int = 0):
    tg = TimeGrid()
    co2 = CarbonPriceModel()
    gens = build_generators(_sync_mix(), gas_scenario)
    rng = np.random.default_rng(seed)
    for g in gens:
        g.prepare_run(tg, rng, co2)
    load = LoadProfile(tg).generate(rng, noise_sigma=0.0)
    return gens, load, dispatch_year(gens, load)


def test_dispatch_shapes_and_nonnegativity() -> None:
    """Dispatch returns full-year arrays with non-negative power and price."""
    gens, load, res = _run(GAS_SCENARIOS["base"])
    assert res.marginal_price.shape == (load.size,)
    assert res.power.shape == (len(gens), load.size)
    assert (res.power >= 0.0).all()
    assert (res.marginal_price >= 0.0).all()
    assert (res.power.sum(axis=0) <= load + 1e-9).all()  # never over-serve


def test_marginal_price_equals_max_srmc_of_dispatched() -> None:
    """Clearing price = max SRMC among dispatched units (no inertia fix)."""
    gens, load, res = _run(GAS_SCENARIOS["base"])
    assert res.curtailment.sum() == 0.0  # all-synchronous → no curtailment
    srmc = np.array([g.srmc() for g in gens])  # (n_units, T)
    masked = np.where(res.power > 0, srmc, -np.inf)
    expected = np.maximum(masked.max(axis=0), 0.0)
    assert np.allclose(res.marginal_price, expected)


def test_price_monotonic_in_gas_price() -> None:
    """A higher gas mean price yields a higher average clearing price."""
    _, _, base = _run(GAS_SCENARIOS["base"])
    _, _, crisis = _run(GAS_SCENARIOS["crisis"])
    assert crisis.marginal_price.mean() > base.marginal_price.mean()
