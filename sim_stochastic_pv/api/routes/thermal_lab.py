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

import io

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...output.exporters import build_thermal_lab_pdf, build_thermal_lab_xlsx
from ...persistence import PersistenceService
from ...scenario_builder import build_default_price_model
from ...simulation.prices import PriceModel
from ...simulation.thermal import ThermalModel
from ...simulation.thermal_lab import (
    HouseVariant,
    ThermalLabConfig,
    ThermalLabResult,
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

_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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


def _build_lab_config(request: ThermalLabCompareRequest) -> ThermalLabConfig:
    """Assemble the :class:`ThermalLabConfig` from the request (may raise
    ``ValueError`` on a domain-invariant violation)."""
    return ThermalLabConfig(
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


def _build_price_model(request: ThermalLabCompareRequest) -> PriceModel | None:
    """Build the optional electricity :class:`PriceModel` from the request's
    ``price`` block, or ``None`` to use the flat scalar price."""
    if request.price is None:
        return None
    price_cfg = request.price.model_dump(exclude_none=True)
    return build_default_price_model(scenario_data={"price": price_cfg})


def _price_label(request: ThermalLabCompareRequest) -> str:
    """Short human label of the pricing assumption for export headers."""
    if request.price is None:
        return f"prezzo fisso {request.electricity_price_eur_per_kwh} €/kWh"
    p = request.price
    base = f"{p.model_type} (base {p.base_price_eur_per_kwh} €/kWh"
    if p.model_type.lower() in ("gbm", "random_walk", "mean_reverting", "ou"):
        return base + f", drift {p.drift_annual:.1%}, vol {p.volatility_annual:.1%})"
    return base + f", escalation {p.annual_escalation:.1%})"


def _run_comparison(
    request: ThermalLabCompareRequest, persistence: PersistenceService
) -> tuple[ThermalLabResult, str]:
    """
    Shared path for the compare + export endpoints.

    Loads the climate model, builds the config (+ optional price model) and
    runs the comparison. Returns ``(result, climate_display_name)``.

    Raises:
        HTTPException 404: Climate profile not found.
        HTTPException 400: Domain-invariant violation in the config.
    """
    record = persistence.climate.get_climate_profile_by_id(request.climate_profile_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {request.climate_profile_id} not found",
        )
    model: ThermalModel | None = persistence.climate.load_thermal_model(
        request.climate_profile_id
    )
    try:
        config = _build_lab_config(request)
        price_model = _build_price_model(request)
        result = compare_house_variants(
            model,
            config,
            n_paths=request.n_paths,
            n_years=request.n_years,
            seed=request.seed,
            price_model=price_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result, (record.location_name or record.name)


def _variant_to_schema(v) -> ThermalVariantResultSchema:
    """Serialise a :class:`ThermalVariantResult` to its response schema."""
    return ThermalVariantResultSchema(
        label=v.label,
        ua_kw_per_c=v.ua_kw_per_c,
        hvac_kwh_annual_mean=v.hvac_kwh_annual_mean,
        hvac_kwh_annual_p05=v.hvac_kwh_annual_p05,
        hvac_kwh_annual_p95=v.hvac_kwh_annual_p95,
        heating_kwh_annual_mean=v.heating_kwh_annual_mean,
        cooling_kwh_annual_mean=v.cooling_kwh_annual_mean,
        annual_cost_eur_mean=v.annual_cost_eur_mean,
        annual_cost_eur_p05=v.annual_cost_eur_p05,
        annual_cost_eur_p95=v.annual_cost_eur_p95,
        comfort_breach_hours_per_year_mean=v.comfort_breach_hours_per_year_mean,
        p_elec_hvac_peak_kw_mean=v.p_elec_hvac_peak_kw_mean,
        t_in_min_c=v.t_in_min_c,
        t_in_max_c=v.t_in_max_c,
        daily_hvac_kwh=v.daily_hvac_kwh.tolist(),
        daily_indoor_min_c=(
            v.daily_indoor_min_c.tolist() if v.daily_indoor_min_c is not None else None
        ),
        daily_indoor_max_c=(
            v.daily_indoor_max_c.tolist() if v.daily_indoor_max_c is not None else None
        ),
        worst_heating_day_index=v.worst_heating_day_index,
        worst_cooling_day_index=v.worst_cooling_day_index,
    )


def _result_to_response(result: ThermalLabResult) -> ThermalLabCompareResponse:
    """Serialise a full :class:`ThermalLabResult` to the compare response."""
    return ThermalLabCompareResponse(
        days=result.days.tolist(),
        daily_outdoor_mean_c=result.daily_outdoor_mean_c.tolist(),
        variants=[_variant_to_schema(v) for v in result.variants],
        n_paths=result.n_paths,
        n_years=result.n_years,
    )


def _report_dict(
    result: ThermalLabResult,
    request: ThermalLabCompareRequest,
    climate_name: str,
) -> dict:
    """Build the plain mapping the file exporters consume (response payload
    enriched with the run-meta keys they need for the header)."""
    report = _result_to_response(result).model_dump()
    report["climate_name"] = climate_name
    report["dynamic"] = request.dynamic
    report["electricity_price_eur_per_kwh"] = request.electricity_price_eur_per_kwh
    report["price_label"] = _price_label(request)
    return report


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
    result, _climate_name = _run_comparison(request, persistence)
    return _result_to_response(result)


@router.post("/compare/export.xlsx")
def export_compare_xlsx(
    request: ThermalLabCompareRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> StreamingResponse:
    """
    Run the comparison and stream it as an Excel workbook.

    Same body as ``/compare``. The workbook has a KPI sheet, a daily-series
    sheet and (in dynamic mode) an indoor-temperature sheet.

    Errors: 404 (missing profile) / 400 (invalid config).
    """
    result, climate_name = _run_comparison(request, persistence)
    buffer = io.BytesIO()
    build_thermal_lab_xlsx(_report_dict(result, request, climate_name), buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type=_XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": 'attachment; filename="laboratorio_termico.xlsx"'
        },
    )


@router.post("/compare/export.pdf")
def export_compare_pdf(
    request: ThermalLabCompareRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> StreamingResponse:
    """
    Run the comparison and stream it as a PDF report.

    Same body as ``/compare``. The report bundles the KPI table and the
    comparison charts (daily energy + outdoor temperature, cost per variant,
    and the indoor-temperature band in dynamic mode).

    Errors: 404 (missing profile) / 400 (invalid config).
    """
    result, climate_name = _run_comparison(request, persistence)
    buffer = io.BytesIO()
    build_thermal_lab_pdf(_report_dict(result, request, climate_name), buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="laboratorio_termico.pdf"'
        },
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
            start_day=request.start_day,
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
