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
]
