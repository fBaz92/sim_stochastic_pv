"""
Monte Carlo economic analysis utilities for PV systems.
"""

from .core import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from .finance import _npv, _compute_irr_monthly, _compute_irr_annual

__all__ = [
    "EconomicConfig",
    "MonteCarloResults",
    "MonteCarloSimulator",
    # Financial functions (internal use, but exported for compatibility)
    "_npv",
    "_compute_irr_monthly",
    "_compute_irr_annual",
]
