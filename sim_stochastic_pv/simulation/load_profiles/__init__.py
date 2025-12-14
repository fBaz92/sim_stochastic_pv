"""
Synthetic electricity consumption profile generators.

Implements deterministic and stochastic load models used by the energy
simulation (ARERA baseline, home/away, variable occupancy, etc.).
"""

from .base import BL_TABLE, LoadProfile
from .monthly import MonthlyAverageLoadProfile
from .arera import AreraLoadProfile
from .home_away import HomeAwayLoadProfile
from .variable import VariableLoadProfile
from .helpers import make_flat_monthly_load_profiles, get_load_w
from .blueprint import LoadScenarioBlueprint

__all__ = [
    # Base
    "LoadProfile",
    "BL_TABLE",
    # Profile implementations
    "MonthlyAverageLoadProfile",
    "AreraLoadProfile",
    "HomeAwayLoadProfile",
    "VariableLoadProfile",
    # Helpers
    "make_flat_monthly_load_profiles",
    "get_load_w",
    # Blueprint
    "LoadScenarioBlueprint",
]
