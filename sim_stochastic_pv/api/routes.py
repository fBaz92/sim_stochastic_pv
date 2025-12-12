from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..application import SimulationApplication
from ..persistence import PersistenceService
from . import dependencies, schemas

router = APIRouter(prefix="/api", tags=["simulation"])


@router.post("/analysis", response_model=schemas.AnalysisResponse)
def trigger_analysis(
    payload: schemas.AnalysisRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
):
    """
    Run the single-scenario analysis synchronously.
    """
    payload = payload or schemas.AnalysisRequest()
    summary = app_service.run_analysis(
        n_mc=payload.n_mc,
        seed=payload.seed or 123,
    )
    return schemas.AnalysisResponse(**summary)


@router.post("/optimization", response_model=schemas.OptimizationResponse)
def trigger_optimization(
    payload: schemas.OptimizationRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
):
    """
    Run the full optimization workflow.
    """
    payload = payload or schemas.OptimizationRequest()
    summary = app_service.run_optimization(seed=payload.seed or 321)
    return schemas.OptimizationResponse(**summary)


@router.get("/runs", response_model=list[schemas.RunResult])
def list_runs(
    limit: int = Query(50, ge=1, le=500),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
):
    """
    Return the latest stored run summaries.
    """
    records = persistence.list_run_results(limit=limit)
    return records
