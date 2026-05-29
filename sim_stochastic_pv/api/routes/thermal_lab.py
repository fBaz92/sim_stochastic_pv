"""
Thermal-lab API endpoints (Phase 19).

Expose the :mod:`sim_stochastic_pv.simulation.thermal_lab` comparison engine
over HTTP so the (future) "Laboratorio termico" webapp section can compare
insulation levels and preview the indoor-temperature trajectory of a single
house configuration.

- ``POST /api/thermal-lab/compare`` — run a Monte Carlo comparison of several
  house variants against a saved climate profile (Phase 15).
- ``POST /api/thermal-lab/timeseries`` — hourly preview (outdoor/indoor
  temperature, electric draw, setpoints) for one house configuration.

The routes stay thin: parse → build the simulation dataclasses (whose
``__post_init__`` enforces the domain invariants) → delegate → shape the
response. Domain ``ValueError``s become HTTP 400; a missing climate profile
becomes HTTP 404.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from ...persistence import PersistenceService
from ...simulation.thermal_lab import (
    HouseVariant,
    ThermalLabConfig,
    compare_house_variants,
    simulate_thermal_timeseries,
)
from ...simulation.thermal_load import (
    HeatPumpConfig,
    HouseThermalConfig,
    SetpointConfig,
)
from .. import dependencies
from ..schemas.thermal_lab import (
    HeatPumpSchema,
    HouseVariantSchema,
    SetpointSchema,
    ThermalLabCompareRequest,
    ThermalLabCompareResponse,
    ThermalTimeseriesRequest,
    ThermalTimeseriesResponse,
    ThermalVariantResultSchema,
)


router = APIRouter(prefix="/api/thermal-lab", tags=["thermal-lab"])


# ---------------------------------------------------------------------------
# Schema → dataclass adapters
# ---------------------------------------------------------------------------


def _to_house_config(schema: HouseVariantSchema) -> HouseThermalConfig:
    """Build a :class:`HouseThermalConfig` from its request schema."""
    return HouseThermalConfig(
        floor_area_m2=schema.floor_area_m2,
        insulation_preset=schema.insulation_preset,
        ua_w_per_c_per_m2=schema.ua_w_per_c_per_m2,
        capacitance_kwh_per_c_per_m2=schema.capacitance_kwh_per_c_per_m2,
        internal_gains_kw=schema.internal_gains_kw,
    )


def _to_heat_pump(schema: HeatPumpSchema) -> HeatPumpConfig:
    """Build a :class:`HeatPumpConfig` from its request schema."""
    return HeatPumpConfig(
        cop_heating=schema.cop_heating,
        cop_cooling=schema.cop_cooling,
        p_elec_max_kw=schema.p_elec_max_kw,
    )


def _to_setpoint(schema: SetpointSchema) -> SetpointConfig:
    """Build a :class:`SetpointConfig`, coercing schedules to tuples."""
    return SetpointConfig(
        t_setpoint_heating_c=schema.t_setpoint_heating_c,
        t_setpoint_cooling_c=schema.t_setpoint_cooling_c,
        t_setpoint_away_c=schema.t_setpoint_away_c,
        heating_schedule_c=(
            tuple(schema.heating_schedule_c)
            if schema.heating_schedule_c is not None
            else None
        ),
        cooling_schedule_c=(
            tuple(schema.cooling_schedule_c)
            if schema.cooling_schedule_c is not None
            else None
        ),
    )


def _finite_or_none(values: np.ndarray) -> list[float | None]:
    """Map a float array to JSON-safe values, replacing ``±inf`` with ``None``.

    Away hours without a setback setpoint carry ``±inf`` setpoints; JSON has
    no representation for infinity, so the preview surfaces them as ``null``.
    """
    return [float(x) if np.isfinite(x) else None for x in values]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/compare", response_model=ThermalLabCompareResponse)
def compare_thermal_variants(
    request: ThermalLabCompareRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> ThermalLabCompareResponse:
    """
    Compare several house variants against a saved climate profile.

    Runs ``n_paths`` Monte Carlo ambient-temperature paths from the climate
    profile and evaluates every house variant on the *same* paths, so the KPI
    differences are purely the envelope. Returns per-variant annual energy /
    cost / comfort-breach KPIs plus a calendar-aligned typical-year daily
    series for charting.

    Errors:
        404 if ``climate_profile_id`` does not exist.
        400 if the configuration violates a domain invariant (e.g. an
        unknown insulation preset or heating ≥ cooling setpoint).
    """
    model = persistence.climate.load_thermal_model(request.climate_profile_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {request.climate_profile_id} not found",
        )

    try:
        config = ThermalLabConfig(
            house_variants=tuple(
                HouseVariant(label=v.label, house=_to_house_config(v))
                for v in request.house_variants
            ),
            heat_pump=_to_heat_pump(request.heat_pump),
            setpoint=_to_setpoint(request.setpoint),
            dynamic=request.dynamic,
            home_hours_of_day=(
                tuple(request.home_hours_of_day)
                if request.home_hours_of_day is not None
                else None
            ),
            electricity_price_eur_per_kwh=request.electricity_price_eur_per_kwh,
        )
        result = compare_house_variants(
            model,
            config,
            n_paths=request.n_paths,
            n_years=request.n_years,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ThermalLabCompareResponse(
        days=result.days.tolist(),
        daily_outdoor_mean_c=result.daily_outdoor_mean_c.tolist(),
        variants=[
            ThermalVariantResultSchema(
                label=v.label,
                ua_kw_per_c=v.ua_kw_per_c,
                hvac_kwh_annual_mean=v.hvac_kwh_annual_mean,
                hvac_kwh_annual_p05=v.hvac_kwh_annual_p05,
                hvac_kwh_annual_p95=v.hvac_kwh_annual_p95,
                annual_cost_eur_mean=v.annual_cost_eur_mean,
                annual_cost_eur_p05=v.annual_cost_eur_p05,
                annual_cost_eur_p95=v.annual_cost_eur_p95,
                comfort_breach_hours_per_year_mean=v.comfort_breach_hours_per_year_mean,
                p_elec_hvac_peak_kw_mean=v.p_elec_hvac_peak_kw_mean,
                t_in_min_c=v.t_in_min_c,
                t_in_max_c=v.t_in_max_c,
                daily_hvac_kwh=v.daily_hvac_kwh.tolist(),
                daily_indoor_min_c=(
                    v.daily_indoor_min_c.tolist()
                    if v.daily_indoor_min_c is not None
                    else None
                ),
                daily_indoor_max_c=(
                    v.daily_indoor_max_c.tolist()
                    if v.daily_indoor_max_c is not None
                    else None
                ),
                worst_heating_day_index=v.worst_heating_day_index,
                worst_cooling_day_index=v.worst_cooling_day_index,
            )
            for v in result.variants
        ],
        n_paths=result.n_paths,
        n_years=result.n_years,
    )


@router.post("/timeseries", response_model=ThermalTimeseriesResponse)
def thermal_timeseries(
    request: ThermalTimeseriesRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> ThermalTimeseriesResponse:
    """
    Hourly preview of one house configuration over a short horizon.

    Returns the outdoor temperature, the (dynamic-mode) indoor-temperature
    trajectory, the electric HVAC draw and the effective setpoints — the data
    behind the "setpoint vs indoor temperature" preview chart.

    Errors:
        404 if ``climate_profile_id`` does not exist.
        400 on a domain-invariant violation in the house/setpoint config.
    """
    model = persistence.climate.load_thermal_model(request.climate_profile_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {request.climate_profile_id} not found",
        )

    try:
        result = simulate_thermal_timeseries(
            model,
            house=_to_house_config(request.house),
            heat_pump=_to_heat_pump(request.heat_pump),
            setpoint=_to_setpoint(request.setpoint),
            dynamic=request.dynamic,
            home_hours_of_day=request.home_hours_of_day,
            n_days=request.n_days,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ThermalTimeseriesResponse(
        hours=result.hours.tolist(),
        t_outdoor_c=result.t_outdoor_c.tolist(),
        t_indoor_c=(
            result.t_indoor_c.tolist() if result.t_indoor_c is not None else None
        ),
        p_elec_hvac_kw=result.p_elec_hvac_kw.tolist(),
        t_set_heating_c=_finite_or_none(result.t_set_heating_c),
        t_set_cooling_c=_finite_or_none(result.t_set_cooling_c),
    )
