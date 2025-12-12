"""
Core photovoltaic simulation models.

This package collects all components involved in the stochastic PV + battery
simulation engine:

* Deterministic building blocks such as the solar irradiance model,
  load profiles, batteries, and inverter dispatch logic.
* Hourly energy system simulation (`energy_simulator`) that couples PV
  production, consumption, and storage.
* Economic post-processing via Monte Carlo analysis and optimization helpers.

Modules are kept together to make the simulation domain self-contained and to
provide a single namespace that higher layers (`application`, FastAPI routes,
CLI) can import from.
"""

from __future__ import annotations

from .battery import BatteryBank, BatterySpecs
from .energy_simulator import EnergySystemConfig, EnergySystemSimulator
from .inverter import InverterAC
from .load_profiles import (
    AreraLoadProfile,
    HomeAwayLoadProfile,
    LoadProfile,
    LoadScenarioBlueprint,
    MonthlyAverageLoadProfile,
    VariableLoadProfile,
    make_flat_monthly_load_profiles,
)
from .monte_carlo import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from .optimizer import (
    BatteryOption,
    InverterOption,
    OptimizationRequest,
    PanelOption,
    ScenarioEvaluation,
    ScenarioOptimizer,
)
from .prices import EscalatingPriceModel, PriceModel
from .pv_model import PVModelSingleDiode
from .solar import SolarModel, SolarMonthParams, make_default_solar_params_for_pavullo

__all__ = [
    # Hardware + physical models
    "BatteryBank",
    "BatterySpecs",
    "InverterAC",
    "PVModelSingleDiode",
    "SolarModel",
    "SolarMonthParams",
    "make_default_solar_params_for_pavullo",
    # Load profiles
    "LoadProfile",
    "AreraLoadProfile",
    "HomeAwayLoadProfile",
    "MonthlyAverageLoadProfile",
    "VariableLoadProfile",
    "LoadScenarioBlueprint",
    "make_flat_monthly_load_profiles",
    # Energy system simulator
    "EnergySystemConfig",
    "EnergySystemSimulator",
    # Economics + Monte Carlo
    "EconomicConfig",
    "MonteCarloResults",
    "MonteCarloSimulator",
    "PriceModel",
    "EscalatingPriceModel",
    # Optimization helpers
    "ScenarioOptimizer",
    "ScenarioEvaluation",
    "OptimizationRequest",
    "InverterOption",
    "PanelOption",
    "BatteryOption",
]
