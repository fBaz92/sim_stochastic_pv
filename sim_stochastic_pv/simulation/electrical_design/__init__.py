"""
Electrical designer engine — string sizing, current checks, DC cables
and string protections for a residential PV plant.

This package ports the reference sizing spreadsheet ("Dimensionamento
FV") into the codebase as pure functions over datasheet dataclasses, so
the web UI can recompute every derived cell on each input change and the
Monte Carlo engine can verify the same design dynamically (clipping,
hourly cable losses, thermal derating) — the part a static spreadsheet
cannot do.

Modules:

- :mod:`.inputs` — site + design-requirement dataclasses.
- :mod:`.sizing` — temperature-corrected voltages/currents, admissible
  modules-per-string range, string-voltage checks, temperature margins,
  plant sizing (strings / total power / DC-AC ratio).
- :mod:`.currents` — per-MPPT current and physical-input checks.
- :mod:`.cables` — DC cable loss table per cross-section + recommendation.
- :mod:`.protections` — string-fuse sizing per CEI EN 62548.
- :mod:`.evaluate` — one-call orchestrator returning the full evaluation.
"""

from .inputs import DesignRequirements, DesignSite
from .sizing import (
    PlantSizing,
    StringSizingBounds,
    StringVoltageChecks,
    TemperatureCorrectedValues,
    TemperatureMargins,
    compute_plant_sizing,
    compute_string_bounds,
    compute_temperature_corrected,
    compute_temperature_margins,
    check_string_voltages,
)
from .currents import CurrentChecks, check_mppt_currents
from .cables import (
    CableParams,
    CableSectionRow,
    CableTable,
    DEFAULT_SECTIONS_MM2,
    compute_cable_table,
)
from .protections import (
    STANDARD_GPV_FUSE_SIZES_A,
    ProtectionSizing,
    size_string_protection,
)
from .evaluate import DesignEvaluation, evaluate_design
from .production import (
    CableLossSpec,
    ProductionPreviewResult,
    simulate_production_preview,
)

__all__ = [
    "DesignRequirements",
    "DesignSite",
    "TemperatureCorrectedValues",
    "StringSizingBounds",
    "StringVoltageChecks",
    "TemperatureMargins",
    "PlantSizing",
    "compute_temperature_corrected",
    "compute_string_bounds",
    "check_string_voltages",
    "compute_temperature_margins",
    "compute_plant_sizing",
    "CurrentChecks",
    "check_mppt_currents",
    "CableParams",
    "CableSectionRow",
    "CableTable",
    "DEFAULT_SECTIONS_MM2",
    "compute_cable_table",
    "STANDARD_GPV_FUSE_SIZES_A",
    "ProtectionSizing",
    "size_string_protection",
    "DesignEvaluation",
    "evaluate_design",
    "CableLossSpec",
    "ProductionPreviewResult",
    "simulate_production_preview",
]
