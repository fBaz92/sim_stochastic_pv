"""
Pydantic schemas for the "Electricity Market" lab endpoints.

Map the JSON payloads of ``/api/market/*`` to the
:mod:`sim_stochastic_pv.simulation.market_lab` dataclasses. Light validation
(types, ranges, known scenario/technology keys) lives here at the boundary;
the market engine enforces the deeper invariants.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, field_validator

from ...market.config import (
    CO2_SCENARIOS,
    COAL_SCENARIOS,
    GAS_SCENARIOS,
    ITALIAN_MIX,
)

# Allowed keys, derived from the packaged presets so the schema and the engine
# never drift apart.
_VALID_TECHS = set(ITALIAN_MIX.keys())
_VALID_GAS = set(GAS_SCENARIOS.keys())
_VALID_CO2 = set(CO2_SCENARIOS.keys())
_VALID_COAL = set(COAL_SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Request building blocks
# ---------------------------------------------------------------------------


class TechTrendSchema(BaseModel):
    """Capacity-evolution spec for one technology (growth + optional step)."""

    annual_growth_pct: float = 0.0
    step_year: Optional[Annotated[int, Field(ge=0)]] = None
    step_capacity_gw: Optional[Annotated[float, Field(ge=0.0)]] = None


class MarketLabRunRequest(BaseModel):
    """
    Configuration for a market-lab run.

    All fields default to a plausible Italian base case, so an empty body runs
    a sensible default scenario. ``capacities_gw`` overrides only the listed
    technologies; the rest keep their packaged values.
    """

    capacities_gw: dict[str, Annotated[float, Field(ge=0.0)]] = Field(
        default_factory=dict
    )
    capacity_trends: dict[str, TechTrendSchema] = Field(default_factory=dict)
    gas_scenario: str = "base"
    co2_scenario: Optional[str] = None
    coal_scenario: Optional[str] = None
    gas_mu_drift_annual: float = 0.0
    co2_mu_drift_annual: float = 0.0
    n_years: Annotated[int, Field(ge=1, le=30)] = 20
    n_trajectories: Annotated[int, Field(ge=1, le=100)] = 8
    n_runs: Annotated[int, Field(ge=1, le=100)] = 6
    seed: int = 42
    display_year: Annotated[int, Field(ge=0)] = 0

    @field_validator("capacities_gw", "capacity_trends")
    @classmethod
    def _known_techs(cls, value: dict) -> dict:
        unknown = set(value) - _VALID_TECHS
        if unknown:
            raise ValueError(
                f"unknown technologies {sorted(unknown)}; "
                f"valid: {sorted(_VALID_TECHS)}"
            )
        return value

    @field_validator("gas_scenario")
    @classmethod
    def _known_gas(cls, value: str) -> str:
        if value not in _VALID_GAS:
            raise ValueError(f"gas_scenario must be one of {sorted(_VALID_GAS)}")
        return value

    @field_validator("co2_scenario")
    @classmethod
    def _known_co2(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in _VALID_CO2:
            raise ValueError(f"co2_scenario must be one of {sorted(_VALID_CO2)} or null")
        return value

    @field_validator("coal_scenario")
    @classmethod
    def _known_coal(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in _VALID_COAL:
            raise ValueError(f"coal_scenario must be one of {sorted(_VALID_COAL)} or null")
        return value


class MarketProfileSaveRequest(BaseModel):
    """Request to materialise a lab configuration into a saved market profile."""

    name: Annotated[str, Field(min_length=1, max_length=255)]
    description: Optional[str] = None
    config: MarketLabRunRequest
    pmg_base_eur_per_kwh: Annotated[float, Field(ge=0.0)] = 0.04
    retail_markup_fraction: Optional[Annotated[float, Field(ge=-1.0)]] = None
    retail_fixed_components_eur_per_kwh: Annotated[float, Field(ge=0.0)] = 0.0


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class MarketLabResponse(BaseModel):
    """Aggregates for the four lab visualisations + the capacity trends."""

    techs: list[str]
    years: list[int]
    capacity_by_year_gw: dict[str, list[float]]
    display_year: int
    # Month × hour mean wholesale price (12 rows × 24 cols), EUR/kWh.
    price_heatmap_eur_per_kwh: list[list[float]]
    annual_price_mean_eur_per_kwh: list[float]
    annual_price_p05_eur_per_kwh: list[float]
    annual_price_p95_eur_per_kwh: list[float]
    duration_curve_x: list[float]
    duration_curve_price_eur_per_kwh: list[float]
    price_setter_techs: list[str]
    # Month × hour index into ``price_setter_techs`` (-1 = none), 12 × 24.
    price_setter_dominant: list[list[int]]
    price_setter_share_year: dict[str, float]
    mean_price_eur_per_kwh: float
    # Fuel/CO2 mean-price trajectories over the horizon.
    gas_price_by_year_eur_per_mwh: list[float]
    co2_price_by_year_eur_per_ton: list[float]
    n_trajectories: int
    n_runs: int


class MarketProfileRef(BaseModel):
    """Lightweight reference to a saved market profile (for pickers/listing)."""

    id: int
    name: str
    description: Optional[str] = None


class MarketProfileSaveResponse(BaseModel):
    """Result of saving a market profile."""

    id: int
    name: str


class MarketProfileDetail(BaseModel):
    """
    Full editable detail of a saved market profile.

    Returns the PMG / retail parameters plus the ``build_config`` blob that
    produced the profile, so the lab UI can reload a saved market into its
    editor. ``config`` is the (free-form) build configuration as it was stored;
    the frontend reads the keys it recognises and falls back to defaults for
    the rest (the seeded default profile stores a lighter config than a
    lab-designed one).
    """

    id: int
    name: str
    description: Optional[str] = None
    pmg_base_eur_per_kwh: float = 0.0
    retail_markup_fraction: Optional[float] = None
    retail_fixed_components_eur_per_kwh: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
