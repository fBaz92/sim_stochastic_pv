"""
Pydantic schemas for the Phase-19 thermal-lab endpoints.

These map the JSON request/response payloads of
``POST /api/thermal-lab/compare`` and ``POST /api/thermal-lab/timeseries`` to
the :mod:`sim_stochastic_pv.simulation.thermal_lab` dataclasses. Light
validation (types, ranges, list lengths) lives here at the boundary; the deep
domain invariants (heating setpoint < cooling setpoint hour-by-hour, valid
insulation preset, positive COP) are enforced by the frozen dataclasses'
``__post_init__`` and surfaced by the route as HTTP 400.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared input building blocks
# ---------------------------------------------------------------------------


class HeatPumpSchema(BaseModel):
    """Heat-pump characterisation (COP + max electric power)."""

    cop_heating: Annotated[float, Field(gt=0.0, description="Heating COP")] = 3.5
    cop_cooling: Annotated[float, Field(gt=0.0, description="Cooling COP")] = 3.0
    p_elec_max_kw: Annotated[
        float, Field(gt=0.0, description="Max electric draw (kW)")
    ] = 3.0


class SetpointSchema(BaseModel):
    """
    Comfort setpoints — single values and/or optional time-of-day schedules.

    When a 24-entry schedule is provided it overrides the scalar for the
    home hours; the scalar stays the representative single-number summary.
    """

    t_setpoint_heating_c: float = 20.0
    t_setpoint_cooling_c: float = 26.0
    t_setpoint_away_c: Optional[float] = None
    heating_schedule_c: Optional[
        Annotated[list[float], Field(min_length=24, max_length=24)]
    ] = None
    cooling_schedule_c: Optional[
        Annotated[list[float], Field(min_length=24, max_length=24)]
    ] = None


class HouseVariantSchema(BaseModel):
    """
    One house configuration. ``label`` defaults so the schema can double as a
    single-house input for the timeseries preview.
    """

    label: Annotated[str, Field(min_length=1, max_length=80)] = "Casa"
    floor_area_m2: Annotated[float, Field(gt=0.0)] = 100.0
    insulation_preset: Annotated[str, Field(max_length=40)] = "standard"
    ua_w_per_c_per_m2: Optional[Annotated[float, Field(ge=0.0)]] = None
    capacitance_kwh_per_c_per_m2: Annotated[float, Field(ge=0.0)] = 0.05
    internal_gains_kw: Annotated[float, Field(ge=0.0)] = 0.0


class ThermalLabPriceSchema(BaseModel):
    """
    Optional electricity price model (Phase 19-bis).

    Maps to the same ``price`` block consumed by
    :func:`scenario_builder.build_default_price_model`. When present on a
    compare request the annual HVAC cost is computed against this model
    (per-path price trajectory); when absent the flat
    ``electricity_price_eur_per_kwh`` scalar is used.

    Attributes:
        model_type: ``"escalating"`` (deterministic + jitter), ``"gbm"``
            (geometric Brownian motion) or ``"mean_reverting"`` (OU).
        base_price_eur_per_kwh: Starting price (€/kWh).
        annual_escalation: Yearly escalation for the ``escalating`` model.
        use_stochastic_escalation: Whether the escalating model adds jitter.
        drift_annual: Log-drift for GBM (or OU).
        volatility_annual: Annual volatility for GBM / OU.
        long_term_price_eur_per_kwh: OU equilibrium (defaults to base).
        mean_reversion_speed_annual: OU reversion speed.
    """

    model_type: Annotated[str, Field(max_length=40)] = "gbm"
    base_price_eur_per_kwh: Annotated[float, Field(gt=0.0)] = 0.25
    annual_escalation: float = 0.02
    use_stochastic_escalation: bool = True
    drift_annual: float = 0.025
    volatility_annual: Annotated[float, Field(ge=0.0)] = 0.10
    long_term_price_eur_per_kwh: Optional[Annotated[float, Field(gt=0.0)]] = None
    mean_reversion_speed_annual: Annotated[float, Field(ge=0.0)] = 0.30


# ---------------------------------------------------------------------------
# Compare endpoint
# ---------------------------------------------------------------------------


class ThermalLabCompareRequest(BaseModel):
    """Request body for ``POST /api/thermal-lab/compare``."""

    climate_profile_id: int
    n_paths: Annotated[int, Field(ge=1, le=200, description="MC paths")] = 30
    n_years: Annotated[int, Field(ge=1, le=20, description="Years per path")] = 1
    seed: int = 42
    dynamic: bool = False
    home_hours_of_day: Optional[list[Annotated[int, Field(ge=0, le=23)]]] = None
    electricity_price_eur_per_kwh: Annotated[float, Field(ge=0.0)] = 0.25
    price: Optional[ThermalLabPriceSchema] = None
    heat_pump: HeatPumpSchema = Field(default_factory=HeatPumpSchema)
    setpoint: SetpointSchema = Field(default_factory=SetpointSchema)
    house_variants: Annotated[list[HouseVariantSchema], Field(min_length=1)]


class ThermalVariantResultSchema(BaseModel):
    """Per-variant aggregated comparison result."""

    label: str
    ua_kw_per_c: float
    hvac_kwh_annual_mean: float
    hvac_kwh_annual_p05: float
    hvac_kwh_annual_p95: float
    heating_kwh_annual_mean: float
    cooling_kwh_annual_mean: float
    annual_cost_eur_mean: float
    annual_cost_eur_p05: float
    annual_cost_eur_p95: float
    comfort_breach_hours_per_year_mean: float
    p_elec_hvac_peak_kw_mean: float
    t_in_min_c: float
    t_in_max_c: float
    daily_hvac_kwh: list[float] = Field(..., min_length=365, max_length=365)
    daily_indoor_min_c: Optional[list[float]] = None
    daily_indoor_max_c: Optional[list[float]] = None
    worst_heating_day_index: Optional[int] = None
    worst_cooling_day_index: Optional[int] = None


class ThermalLabCompareResponse(BaseModel):
    """Response body for ``POST /api/thermal-lab/compare``."""

    days: list[int] = Field(..., min_length=365, max_length=365)
    daily_outdoor_mean_c: list[float] = Field(..., min_length=365, max_length=365)
    variants: list[ThermalVariantResultSchema]
    n_paths: int
    n_years: int


# ---------------------------------------------------------------------------
# Timeseries preview endpoint
# ---------------------------------------------------------------------------


class ThermalTimeseriesRequest(BaseModel):
    """Request body for ``POST /api/thermal-lab/timeseries``."""

    climate_profile_id: int
    n_days: Annotated[int, Field(ge=1, le=60, description="Horizon (days)")] = 14
    seed: int = 42
    dynamic: bool = True
    home_hours_of_day: Optional[list[Annotated[int, Field(ge=0, le=23)]]] = None
    start_day: Annotated[int, Field(ge=0, le=364, description="Day-of-year offset")] = 0
    heat_pump: HeatPumpSchema = Field(default_factory=HeatPumpSchema)
    setpoint: SetpointSchema = Field(default_factory=SetpointSchema)
    house: HouseVariantSchema = Field(default_factory=HouseVariantSchema)


class ThermalTimeseriesResponse(BaseModel):
    """
    Response body for ``POST /api/thermal-lab/timeseries``.

    Setpoint arrays use ``None`` where the physics value is ``±inf`` (away
    hours with HVAC off) so the payload stays valid JSON.
    """

    hours: list[int]
    t_outdoor_c: list[float]
    t_indoor_c: Optional[list[float]] = None
    p_elec_hvac_kw: list[float]
    t_set_heating_c: list[Optional[float]]
    t_set_cooling_c: list[Optional[float]]
