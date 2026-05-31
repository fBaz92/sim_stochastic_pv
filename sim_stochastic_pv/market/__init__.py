"""
Underlying electricity-market model: endogenous wholesale price from a mix.

A Monte Carlo engine that derives a sensible hourly wholesale electricity
price from a generation mix and market conditions, so that PV surplus can be
valued realistically (dedicated-withdrawal / ritiro dedicato). Given an
installed mix (gas, hydro, solar, wind, nuclear, coal, …), stochastic fuel,
coal and CO2 prices, weather-driven renewable availability, cross-border
interconnections and grid-scale storage, it runs a merit-order dispatch and
returns the system marginal price. Calibrated on the Italian power system as
reference case (60 GW peak) but technology-agnostic.

This is the system-level counterpart to the household PV simulator in
:mod:`sim_stochastic_pv.simulation`: it produces the price *surface* the PV
economics consume, it does not itself model a single dwelling.

Modules:
    config: Global parameters, default mix/fuel scenarios, and coefficients.
    grid: Temporal backbone (TimeGrid) and system load profile.
    generators: Fuel/carbon price models, availability models, Generator.
    dispatch: Vectorized merit-order dispatch with inertia and storage.
    storage: Grid-scale battery (BESS) physical model.
    interconnections: Cross-border links and virtual import generators.
    price_areas: Correlated foreign price-area stochastic processes.
    reliability: Outage/availability models for interconnection links.
    simulation: Monte Carlo runner and scenario sweep utilities.
"""

from __future__ import annotations

from sim_stochastic_pv.market.config import ITALIAN_MIX, GAS_SCENARIOS, P_PEAK_GW, N_MC_RUNS
from sim_stochastic_pv.market.grid import TimeGrid, LoadProfile
from sim_stochastic_pv.market.generators import Generator, build_generators
from sim_stochastic_pv.market.dispatch import dispatch_year, DispatchResult
from sim_stochastic_pv.market.simulation import (
    SimulationConfig, MonteCarloResult,
    run_monte_carlo, sweep_technology, build_sensitivity_heatmap,
)
