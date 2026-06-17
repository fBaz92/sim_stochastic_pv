"""
Synthetic electricity consumption profile generators.

Implements deterministic and stochastic load models used by the energy
simulation (ARERA baseline, home/away, variable occupancy, weekly pattern, etc.).
"""

from .base import BL_TABLE, LoadProfile
from .monthly import MonthlyAverageLoadProfile
from .arera import AreraLoadProfile
from .home_away import HomeAwayLoadProfile
from .variable import VariableLoadProfile
from .weekly import WeeklyPatternLoadProfile, WEEKLY_PRESETS
from .helpers import make_flat_monthly_load_profiles, get_load_w
from .blueprint import LoadScenarioBlueprint
from .stochastic import (
    DEFAULT_PHI_INTRA_DAY,
    DEFAULT_SIGMA_LOG,
    StochasticLoadConfig,
    StochasticLoadProfile,
)
from .appliances import (
    APPLIANCE_PRESETS,
    ApplianceEvent,
    ApplianceProfileConfig,
    AppliancesKPIs,
    EventBasedApplianceProfile,
    aggregate_appliances_kpis,
    get_preset,
)
from .presence import (
    DEFAULT_PRESENCE_CALENDAR,
    HOUSE_TYPE_PRESETS,
    HouseTypePreset,
    MonthPresencePattern,
    PRESENCE_CALENDAR_PRESETS,
    PresenceCalendar,
)
from .bolletta import (
    ARERA_BASELINE_ANNUAL_KWH,
    annual_kwh_from_bimonthly,
    build_scaled_arera_factory,
    compute_arera_baseline_annual_kwh,
    fit_bolletta_profile,
)

__all__ = [
    # Base
    "LoadProfile",
    "BL_TABLE",
    # Profile implementations
    "MonthlyAverageLoadProfile",
    "AreraLoadProfile",
    "HomeAwayLoadProfile",
    "VariableLoadProfile",
    "WeeklyPatternLoadProfile",
    # Weekly presets
    "WEEKLY_PRESETS",
    # Helpers
    "make_flat_monthly_load_profiles",
    "get_load_w",
    # Blueprint
    "LoadScenarioBlueprint",
    # Phase 17 — stochastic decorator
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
    "get_preset",
    # Presence calendar (occupancy backbone of home/away profiles)
    "MonthPresencePattern",
    "PresenceCalendar",
    "HouseTypePreset",
    "HOUSE_TYPE_PRESETS",
    "DEFAULT_PRESENCE_CALENDAR",
    "PRESENCE_CALENDAR_PRESETS",
    # Bill-based auto-fit
    "ARERA_BASELINE_ANNUAL_KWH",
    "compute_arera_baseline_annual_kwh",
    "fit_bolletta_profile",
    "build_scaled_arera_factory",
    "annual_kwh_from_bimonthly",
]
