"""
Background-job API endpoints (Phase 12).

Wraps the synchronous ``SimulationApplication.run_analysis`` /
``run_optimization`` calls in a thread-pool job so the browser can
display a progress bar and redirect when the work is done.

Endpoints:

- ``POST /api/jobs/analysis``     — accept the same payload as
  ``/api/analysis``, return a ``{job_id}``.
- ``POST /api/jobs/optimization`` — accept the same payload as
  ``/api/optimization``, return a ``{job_id}``.
- ``GET  /api/jobs/{job_id}``     — poll job status. Returns the dict
  produced by :meth:`JobRecord.to_dict`.
- ``GET  /api/jobs``              — list recent jobs (debug helper).
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from ...application import SimulationApplication
from ...jobs import JobHandle, get_default_store
from .. import dependencies
from ..schemas import simulation as sim_schemas
from ..schemas.designs import DesignCompareRequest

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/analysis")
def submit_analysis_job(
    payload: sim_schemas.AnalysisRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> Dict[str, Any]:
    """
    Schedule a Monte Carlo analysis to run in the background.

    The payload matches :class:`AnalysisRequest` (same fields as
    ``POST /api/analysis``). The endpoint returns immediately with a
    ``job_id`` the client should poll via ``GET /api/jobs/{job_id}``.

    Returns:
        ``{"job_id": "<uuid>", "kind": "analysis"}``
    """
    payload = payload or sim_schemas.AnalysisRequest()
    store = get_default_store()

    def worker(handle: JobHandle) -> None:
        def cb(done: int, total: int, elapsed: float, eta: float) -> None:
            handle.set_progress(
                done, total, message=f"Path {done}/{total} (ETA {eta:.0f}s)"
            )

        summary = app_service.run_analysis(
            n_mc=payload.n_mc,
            seed=payload.seed or 123,
            scenario_data=payload.scenario,
            progress_callback=cb,
        )
        run_id = summary.get("run_id")
        if run_id is not None:
            handle.set_run_id(run_id)

    record = store.submit("analysis", worker)
    return {"job_id": record.id, "kind": record.kind}


@router.post("/optimization")
def submit_optimization_job(
    payload: sim_schemas.OptimizationRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> Dict[str, Any]:
    """
    Schedule an optimization (design) sweep to run in the background.

    Returns:
        ``{"job_id": "<uuid>", "kind": "optimization"}``
    """
    payload = payload or sim_schemas.OptimizationRequest()
    store = get_default_store()

    def worker(handle: JobHandle) -> None:
        def cb(done: int, total: int, current_desc: str) -> None:
            handle.set_progress(
                done, total, message=f"Scenario {done}/{total} — {current_desc}"
            )

        summary = app_service.run_optimization(
            seed=payload.seed or 123,
            n_mc=payload.n_mc,
            scenario_data=payload.scenario,
            progress_callback=cb,
        )
        run_id = summary.get("run_id")
        if run_id is not None:
            handle.set_run_id(run_id)

    record = store.submit("optimization", worker)
    return {"job_id": record.id, "kind": record.kind}


@router.post("/compare")
def submit_compare_job(
    payload: DesignCompareRequest,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> Dict[str, Any]:
    """
    Schedule a design comparison to run in the background.

    Builds the shared scenario context (load, price, horizon) once and
    runs it over every design with a common seed (paired Monte Carlo).
    The result payload lands inline on the job record (``result`` field
    of ``GET /api/jobs/{job_id}``) — no run record is persisted.

    Returns:
        ``{"job_id": "<uuid>", "kind": "comparison"}``
    """
    store = get_default_store()

    base_scenario: Dict[str, Any] = {
        "scenario_name": "confronto_design",
        "energy": {"n_years": payload.n_years},
        "economic": {"n_mc": payload.n_mc},
    }
    if payload.load_profile_id is not None:
        base_scenario["load_profile_id"] = payload.load_profile_id
    else:
        # Standard Italian residential baseline, occupied all year — the
        # same default the offer page uses.
        base_scenario["load_profile"] = {
            "kind": "home_away",
            "home": {"type": "arera"},
            "away": {"type": "arera"},
        }
    if payload.price_profile_id is not None:
        base_scenario["price_profile_id"] = payload.price_profile_id
    else:
        base_scenario["price"] = {
            "base_price_eur_per_kwh": 0.25,
            "annual_escalation": 0.03,
            "use_stochastic_escalation": True,
        }

    def worker(handle: JobHandle) -> None:
        def cb(done: int, total: int, name: str) -> None:
            handle.set_progress(done, total, message=f"Impianto {done}/{total} — {name}")

        result = app_service.run_design_comparison(
            design_ids=payload.design_ids,
            base_scenario=base_scenario,
            n_mc=payload.n_mc,
            seed=payload.seed,
            progress_callback=cb,
        )
        handle.set_result(result)

    record = store.submit("comparison", worker)
    return {"job_id": record.id, "kind": record.kind}


@router.get("/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """Return the latest snapshot for a job."""
    record = get_default_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return record.to_dict()


@router.get("")
def list_jobs(limit: int = 20) -> Dict[str, Any]:
    """List the ``limit`` most recent jobs (newest first)."""
    return {
        "jobs": [j.to_dict() for j in get_default_store().list_recent(limit)],
    }
