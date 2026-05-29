"""
API route modules for simulation application.

This package organizes FastAPI route handlers by business domain:
- hardware: Hardware catalog CRUD operations (inverters, panels, batteries)
- simulation: Direct simulation execution (analysis, optimization)
- profiles: Load and price profile management
- configurations: Saved configuration management (scenarios, campaigns)
- execution: Database-driven execution endpoints
- thermal_lab: Insulation comparison + indoor-temperature preview (Phase 19)

All routers are prefixed with /api when included in the main application.
"""

from __future__ import annotations

from .configurations import router as configurations_router
from .execution import router as execution_router
from .external import router as external_router
from .hardware import router as hardware_router
from .jobs import router as jobs_router
from .profiles import router as profiles_router
from .simulation import router as simulation_router
from .thermal_lab import router as thermal_lab_router

__all__ = [
    "hardware_router",
    "simulation_router",
    "profiles_router",
    "configurations_router",
    "execution_router",
    "external_router",
    "jobs_router",
    "thermal_lab_router",
]
