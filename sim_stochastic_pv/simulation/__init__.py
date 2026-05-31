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
from .electrical import (
    DEFAULT_DERATING_EXPONENT_K,
    ElectricalKPIs,
    ElectricalModel,
    InverterElectricalSpecs,
    PanelElectricalSpecs,
    PvString,
    aggregate_kpis,
    cell_temperature_c,
    missing_inverter_fields,
    missing_panel_fields,
    v_string_at_cell_temperature,
)
from .energy_simulator import EnergySystemConfig, EnergySystemSimulator
from .inverter import InverterAC
from .load_profiles import (
    APPLIANCE_PRESETS,
    ApplianceEvent,
    ApplianceProfileConfig,
    AppliancesKPIs,
    AreraLoadProfile,
    DEFAULT_PHI_INTRA_DAY,
    DEFAULT_SIGMA_LOG,
    EventBasedApplianceProfile,
    HomeAwayLoadProfile,
    LoadProfile,
    LoadScenarioBlueprint,
    MonthlyAverageLoadProfile,
    StochasticLoadConfig,
    StochasticLoadProfile,
    VariableLoadProfile,
    WeeklyPatternLoadProfile,
    WEEKLY_PRESETS,
    aggregate_appliances_kpis,
    get_preset as get_appliance_preset,
    make_flat_monthly_load_profiles,
)
from .thermal_load import (
    DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2,
    HeatPumpConfig,
    HouseThermalConfig,
    HvacController,
    INSULATION_PRESETS,
    PRESET_GOOD_W_PER_C_PER_M2,
    PRESET_POOR_W_PER_C_PER_M2,
    PRESET_STANDARD_W_PER_C_PER_M2,
    SetpointConfig,
    ThermalLoadConfig,
    ThermalLoadKPIs,
    aggregate_thermal_kpis,
)
from .monte_carlo import (
    EconomicConfig,
    InflationConfig,
    MonteCarloResults,
    MonteCarloSimulator,
    TaxBonusConfig,
)
from .optimizer import (
    BatteryOption,
    InverterOption,
    OptimizationRequest,
    PanelOption,
    ScenarioEvaluation,
    ScenarioOptimizer,
)
from .market_pricing import MarketPriceProvider
from .prices import (
    EscalatingPriceModel,
    GBMPriceModel,
    MeanRevertingPriceModel,
    PriceModel,
)
from .pv_model import PVModelSingleDiode
from .solar import SolarModel, SolarMonthParams, make_default_solar_params_for_pavullo

__all__ = [
    # Hardware + physical models
    "BatteryBank",
    "BatterySpecs",
    "InverterAC",
    # Electrical model (Phase 16 — opt-in MPPT-window derating)
    "DEFAULT_DERATING_EXPONENT_K",
    "ElectricalKPIs",
    "ElectricalModel",
    "InverterElectricalSpecs",
    "PanelElectricalSpecs",
    "PvString",
    "aggregate_kpis",
    "cell_temperature_c",
    "missing_inverter_fields",
    "missing_panel_fields",
    "v_string_at_cell_temperature",
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
    "WeeklyPatternLoadProfile",
    "WEEKLY_PRESETS",
    "LoadScenarioBlueprint",
    "make_flat_monthly_load_profiles",
    # Phase 17 — stochastic load decorator
    "DEFAULT_PHI_INTRA_DAY",
    "DEFAULT_SIGMA_LOG",
    "StochasticLoadConfig",
    "StochasticLoadProfile",
    # Phase 17-bis — appliance event-based decorator
    "APPLIANCE_PRESETS",
    "ApplianceEvent",
    "ApplianceProfileConfig",
    "AppliancesKPIs",
    "EventBasedApplianceProfile",
    "aggregate_appliances_kpis",
    "get_appliance_preset",
    # Phase 17 — thermal load (HVAC)
    "DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2",
    "HeatPumpConfig",
    "HouseThermalConfig",
    "HvacController",
    "INSULATION_PRESETS",
    "PRESET_GOOD_W_PER_C_PER_M2",
    "PRESET_POOR_W_PER_C_PER_M2",
    "PRESET_STANDARD_W_PER_C_PER_M2",
    "SetpointConfig",
    "ThermalLoadConfig",
    "ThermalLoadKPIs",
    "aggregate_thermal_kpis",
    # Energy system simulator
    "EnergySystemConfig",
    "EnergySystemSimulator",
    # Economics + Monte Carlo
    "EconomicConfig",
    "InflationConfig",
    "TaxBonusConfig",
    "MonteCarloResults",
    "MonteCarloSimulator",
    "PriceModel",
    "EscalatingPriceModel",
    "GBMPriceModel",
    "MeanRevertingPriceModel",
    "MarketPriceProvider",
    # Optimization helpers
    "ScenarioOptimizer",
    "ScenarioEvaluation",
    "OptimizationRequest",
    "InverterOption",
    "PanelOption",
    "BatteryOption",
]
