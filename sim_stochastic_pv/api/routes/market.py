"""
Electricity-market lab API endpoints.

Expose the :mod:`sim_stochastic_pv.simulation.market_lab` orchestrator over
HTTP so the "Mercato Elettrico" webapp section can design a generation mix +
capacity trends + fuel/CO2 scenarios and read back the wholesale price views
(month×hour heatmap, annual fan chart, duration curve, "who sets the price"
heatmap). It also persists a designed market as a reusable market profile.

Routes stay thin: parse → build the orchestrator config → delegate → shape the
response. Engine/domain ``ValueError``/``KeyError`` become HTTP 400; a missing
profile becomes HTTP 404.

- ``POST /api/market/run`` — run the lab and return all visualisation data.
- ``POST /api/market/run/export.{xlsx,pdf}`` — same run, streamed as a report.
- ``GET  /api/market/profiles`` — list saved market profiles.
- ``POST /api/market/profiles`` — materialise a lab config into a saved profile.
- ``DELETE /api/market/profiles/{id}`` — delete a saved profile.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ...output.exporters import build_market_pdf, build_market_xlsx
from ...persistence import PersistenceService
from ...simulation.market_lab import (
    MarketLabConfig,
    MarketLabResult,
    TechTrendSpec,
    build_market_provider,
    run_market_lab,
)
from .. import dependencies
from ..schemas.market import (
    MarketLabResponse,
    MarketLabRunRequest,
    MarketProfileRef,
    MarketProfileSaveRequest,
    MarketProfileSaveResponse,
)

_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

router = APIRouter(prefix="/api/market", tags=["market"])


# ---------------------------------------------------------------------------
# Schema → dataclass adapters
# ---------------------------------------------------------------------------


def _to_config(request: MarketLabRunRequest) -> MarketLabConfig:
    """Build a :class:`MarketLabConfig` from its request schema."""
    return MarketLabConfig(
        capacities_gw=dict(request.capacities_gw),
        capacity_trends={
            tech: TechTrendSpec(
                annual_growth_pct=spec.annual_growth_pct,
                step_year=spec.step_year,
                step_capacity_gw=spec.step_capacity_gw,
            )
            for tech, spec in request.capacity_trends.items()
        },
        gas_scenario=request.gas_scenario,
        co2_scenario=request.co2_scenario,
        coal_scenario=request.coal_scenario,
        gas_mu_drift_annual=request.gas_mu_drift_annual,
        co2_mu_drift_annual=request.co2_mu_drift_annual,
        n_years=request.n_years,
        n_trajectories=request.n_trajectories,
        n_runs=request.n_runs,
        seed=request.seed,
        display_year=request.display_year,
    )


def _to_response(result: MarketLabResult) -> MarketLabResponse:
    """Serialise a :class:`MarketLabResult` to its response schema."""
    return MarketLabResponse(
        techs=result.techs,
        years=result.years,
        capacity_by_year_gw={
            tech: arr.tolist() for tech, arr in result.capacity_by_year_gw.items()
        },
        display_year=result.display_year,
        price_heatmap_eur_per_kwh=result.price_heatmap_eur_per_kwh.tolist(),
        annual_price_mean_eur_per_kwh=result.annual_price_mean_eur_per_kwh.tolist(),
        annual_price_p05_eur_per_kwh=result.annual_price_p05_eur_per_kwh.tolist(),
        annual_price_p95_eur_per_kwh=result.annual_price_p95_eur_per_kwh.tolist(),
        duration_curve_x=result.duration_curve_x.tolist(),
        duration_curve_price_eur_per_kwh=result.duration_curve_price_eur_per_kwh.tolist(),
        price_setter_techs=result.price_setter_techs,
        price_setter_dominant=result.price_setter_dominant.tolist(),
        price_setter_share_year=result.price_setter_share_year,
        mean_price_eur_per_kwh=result.mean_price_eur_per_kwh,
        n_trajectories=result.n_trajectories,
        n_runs=result.n_runs,
    )


def _run(request: MarketLabRunRequest) -> MarketLabResult:
    """Run the lab, mapping engine errors to HTTP 400."""
    try:
        return run_market_lab(_to_config(request))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _report_dict(result: MarketLabResult, request: MarketLabRunRequest) -> dict:
    """Plain mapping consumed by the file exporters (response + run-meta)."""
    report = _to_response(result).model_dump()
    report["gas_scenario"] = request.gas_scenario
    report["co2_scenario"] = request.co2_scenario
    report["coal_scenario"] = request.coal_scenario
    return report


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/run", response_model=MarketLabResponse)
def run_market(request: MarketLabRunRequest) -> MarketLabResponse:
    """
    Run the electricity-market lab and return all visualisation data.

    Builds a price surface for the configured mix/trends/scenarios (heatmap,
    annual fan chart, duration curve) and a single-year Monte Carlo for the
    "who sets the price" breakdown, plus the per-technology capacity trajectory.

    Errors:
        400 if the configuration is invalid (e.g. an unknown scenario key).
    """
    return _to_response(_run(request))


@router.post("/run/export.xlsx")
def export_market_xlsx(request: MarketLabRunRequest) -> StreamingResponse:
    """Run the lab and stream the result as an Excel workbook."""
    result = _run(request)
    buffer = io.BytesIO()
    build_market_xlsx(_report_dict(result, request), buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type=_XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": 'attachment; filename="mercato_elettrico.xlsx"'
        },
    )


@router.post("/run/export.pdf")
def export_market_pdf(request: MarketLabRunRequest) -> StreamingResponse:
    """Run the lab and stream the result as a PDF report."""
    result = _run(request)
    buffer = io.BytesIO()
    build_market_pdf(_report_dict(result, request), buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="mercato_elettrico.pdf"'
        },
    )


@router.get("/profiles", response_model=list[MarketProfileRef])
def list_market_profiles(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[MarketProfileRef]:
    """List saved market profiles (id, name, description) ordered by name."""
    return [
        MarketProfileRef(id=p.id, name=p.name, description=p.description)
        for p in persistence.list_market_profiles()
    ]


@router.post("/profiles", response_model=MarketProfileSaveResponse)
def save_market_profile(
    request: MarketProfileSaveRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> MarketProfileSaveResponse:
    """
    Materialise a lab configuration into a reusable saved market profile.

    Rebuilds the wholesale price surface for the configuration, wraps it with
    the supplied PMG / retail parameters, serialises the whole provider config
    and upserts it by name.

    Errors:
        400 if the configuration is invalid.
    """
    try:
        provider = build_market_provider(
            _to_config(request.config),
            pmg_base_eur_per_kwh=request.pmg_base_eur_per_kwh,
            retail_markup_fraction=request.retail_markup_fraction,
            retail_fixed_components_eur_per_kwh=request.retail_fixed_components_eur_per_kwh,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = provider.to_config_dict(
        build_config=request.config.model_dump(mode="json")
    )
    record = persistence.upsert_market_profile(
        {"name": request.name, "description": request.description, "data": data}
    )
    return MarketProfileSaveResponse(id=record.id, name=record.name)


@router.delete("/profiles/{profile_id}")
def delete_market_profile(
    profile_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """Delete a saved market profile by id (404 if it does not exist)."""
    if not persistence.delete_market_profile(profile_id):
        raise HTTPException(
            status_code=404, detail=f"Market profile {profile_id} not found"
        )
    return {"deleted": True}
