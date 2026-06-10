"""
Pydantic schemas for API request/response validation.

This package contains all Pydantic models used for API validation,
organized by domain:
- hardware: Inverter, Panel, Battery schemas
- simulation: Analysis and Optimization request/response schemas
- profiles: Load and Price profile schemas
- configurations: Saved configuration schemas
- common: Shared base schemas and utilities

All schemas are re-exported from this module for backwards compatibility.

Example:
    ```python
    # Both import styles work:
    from sim_stochastic_pv.api.schemas import InverterResponse
    from sim_stochastic_pv.api.schemas.hardware import InverterResponse
    ```
"""

from __future__ import annotations

# Re-export all schemas for backward compatibility
from .common import _coerce_to_dict, _merge_specs_defaults
from .configurations import (
    SavedConfigurationCreate,
    SavedConfigurationResponse,
    ScenarioResponse,
)
from .external import (
    ClimateNormalsResponse,
    ClimateProfilePreviewResponse,
    ClimateProfileResponse,
    GeocodeRequest,
    GeocodeResultResponse,
)
from .locations import (
    ClimateProfileSummary,
    LocationImportRequest,
    LocationImportResponse,
    LocationResponse,
    LocationUpdate,
    SolarProfileSummary,
)
from .hardware import (
    BatteryCreate,
    BatteryResponse,
    InverterCreate,
    InverterResponse,
    PanelCreate,
    PanelResponse,
)
from .profiles import (
    LoadProfileCreate,
    LoadProfileResponse,
    PriceProfileCreate,
    PriceProfileResponse,
)
from .simulation import (
    AnalysisRequest,
    AnalysisResponse,
    OptimizationRequest,
    OptimizationResponse,
    RunResult,
)
from .thermal_lab import (
    ThermalLabCompareRequest,
    ThermalLabCompareResponse,
    ThermalTimeseriesRequest,
    ThermalTimeseriesResponse,
    ThermalVariantResultSchema,
)

__all__ = [
    # Utility functions
    "_coerce_to_dict",
    "_merge_specs_defaults",
    # Hardware schemas
    "InverterResponse",
    "InverterCreate",
    "PanelResponse",
    "PanelCreate",
    "BatteryResponse",
    "BatteryCreate",
    # Simulation schemas
    "AnalysisRequest",
    "AnalysisResponse",
    "OptimizationRequest",
    "OptimizationResponse",
    "RunResult",
    # Profile schemas
    "LoadProfileResponse",
    "LoadProfileCreate",
    "PriceProfileResponse",
    "PriceProfileCreate",
    # Configuration schemas
    "SavedConfigurationResponse",
    "SavedConfigurationCreate",
    "ScenarioResponse",
    # External / geolocation schemas (Phase 14)
    "ClimateNormalsResponse",
    "GeocodeRequest",
    "GeocodeResultResponse",
    # Climate profile schemas (Phase 15)
    "ClimateProfilePreviewResponse",
    "ClimateProfileResponse",
    # Installation sites (locations + unified import flow)
    "ClimateProfileSummary",
    "LocationImportRequest",
    "LocationImportResponse",
    "LocationResponse",
    "LocationUpdate",
    "SolarProfileSummary",
    # Thermal-lab schemas (Phase 19)
    "ThermalLabCompareRequest",
    "ThermalLabCompareResponse",
    "ThermalTimeseriesRequest",
    "ThermalTimeseriesResponse",
    "ThermalVariantResultSchema",
]
