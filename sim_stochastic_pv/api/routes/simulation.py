"""
Direct simulation execution API endpoints.

This module provides endpoints for executing simulations with inline
scenario configurations (not from database). For database-driven execution
of saved scenarios/campaigns, see the execution module.

Endpoints:
- POST /analysis: Single-scenario Monte Carlo simulation
- POST /optimization: Multi-scenario parameter sweep optimization
- GET /runs: Historical execution results

These endpoints accept complete scenario configurations in the request body
and execute them immediately. Results are stored in the database if
persistence is enabled.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...application import SimulationApplication
from ...output.exporters import build_cashflow_xlsx, build_pdf_report
from ...persistence import PersistenceService
from .. import dependencies
from ..schemas import simulation as sim_schemas

router = APIRouter(prefix="/api", tags=["simulation"])


@router.post("/analysis", response_model=sim_schemas.AnalysisResponse)
def trigger_analysis(
    payload: sim_schemas.AnalysisRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> sim_schemas.AnalysisResponse:
    """
    Execute a single-scenario Monte Carlo analysis.

    Runs a Monte Carlo simulation for a photovoltaic energy system configuration,
    evaluating economic outcomes across multiple stochastic paths. The simulation
    considers solar production variability, load consumption patterns, electricity
    price escalation, and battery degradation.

    This endpoint accepts an inline scenario configuration. For executing saved
    scenarios from the database, use POST /scenarios/{id}/run instead.

    Args:
        payload: Analysis request with optional scenario configuration, Monte Carlo
            count, and random seed. If None or scenario is None, uses application's
            default scenario configuration.
        app_service: Simulation application service (dependency injected).

    Returns:
        AnalysisResponse containing:
        - scenario: Scenario name/identifier
        - final_gain_mean_eur: Mean net gain (nominal)
        - final_gain_real_mean_eur: Mean net gain (inflation-adjusted)
        - prob_gain: Probability of positive return (0-1)
        - output_dir: Path to saved results (if enabled)
        - plots_data: Embedded visualization data (if enabled)

    Example:
        ```python
        # POST /api/analysis
        {
            "n_mc": 500,
            "seed": 123,
            "scenario": {
                "load_profile": {
                    "home_profile_type": "arera",
                    "away_profile": "arera",
                    "min_days_home": [25, 25, ..., 25]
                },
                "solar": {
                    "pv_kwp": 3.0,
                    "degradation_per_year": 0.007
                },
                "energy": {
                    "n_years": 20,
                    "pv_kwp": 3.0,
                    "battery_specs": {"capacity_kwh": 5.0, "cycles_life": 5000},
                    "n_batteries": 1,
                    "inverter_p_ac_max_kw": 3.0
                },
                "price": {
                    "base_price_eur_per_kwh": 0.20,
                    "annual_escalation": 0.02,
                    "use_stochastic_escalation": true
                },
                "economic": {
                    "n_mc": 500,
                    "investment_eur": 8000.0
                }
            }
        }

        # Response
        {
            "scenario": "default",
            "final_gain_mean_eur": 2450.50,
            "final_gain_real_mean_eur": 1890.25,
            "prob_gain": 0.87,
            "output_dir": "/path/to/results/analysis_20250115_103045"
        }
        ```

    Notes:
        - If payload is None, uses default AnalysisRequest()
        - n_mc in request overrides n_mc in scenario.economic if both provided
        - seed defaults to 123 if not provided
        - Results are automatically saved to database if persistence enabled
        - Computation time scales linearly with n_mc (500 paths ≈ 10-30 seconds)
    """
    payload = payload or sim_schemas.AnalysisRequest()
    summary = app_service.run_analysis(
        n_mc=payload.n_mc,
        seed=payload.seed or 123,
        scenario_data=payload.scenario,
    )
    return sim_schemas.AnalysisResponse(**summary)


@router.post("/optimization", response_model=sim_schemas.OptimizationResponse)
def trigger_optimization(
    payload: sim_schemas.OptimizationRequest | None = None,
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> sim_schemas.OptimizationResponse:
    """
    Execute a multi-scenario parameter sweep optimization.

    Evaluates multiple hardware configurations to find the optimal PV + battery
    system design. Each configuration combination is assessed via Monte Carlo
    simulation, and results are ranked by expected economic return.

    The optimization explores all combinations of:
    - Inverter models (from optimization.inverter_options)
    - Panel models (from optimization.panel_options)
    - Battery models (from optimization.battery_options)
    - Panel counts (from optimization.panel_count_options)
    - Battery counts (from optimization.battery_count_options)

    This endpoint accepts an inline scenario configuration with optimization
    parameters. For executing saved campaigns from the database, use
    POST /campaigns/{id}/run instead.

    Args:
        payload: Optimization request with optional scenario configuration
            (including optimization section), Monte Carlo count, and random seed.
            If None or scenario is None, uses application's default configuration.
        app_service: Simulation application service (dependency injected).

    Returns:
        OptimizationResponse containing:
        - evaluations: Number of scenarios evaluated
        - output_dir: Path to detailed results (if enabled)

    Example:
        ```python
        # POST /api/optimization
        {
            "seed": 321,
            "n_mc": 200,
            "scenario": {
                "optimization": {
                    "inverter_options": [
                        {"name": "Huawei 5kW", "p_ac_max_kw": 5.0, ...},
                        {"name": "SMA 6kW", "p_ac_max_kw": 6.0, ...}
                    ],
                    "panel_options": [
                        {"name": "CS 410W", "power_w": 410, ...}
                    ],
                    "battery_options": [
                        {"name": "Powerwall 13.5kWh", "capacity_kwh": 13.5, ...}
                    ],
                    "panel_count_options": [6, 8, 10],
                    "battery_count_options": [0, 1, 2],
                    "include_no_battery": true
                },
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            }
        }

        # Response
        {
            "evaluations": 36,
            "output_dir": "/path/to/results/optimization_20250115_104530"
        }
        ```

    Notes:
        - If payload is None, uses default OptimizationRequest()
        - Total scenarios = len(inverters) × len(panels) × len(batteries) ×
          len(panel_counts) × len(battery_counts)
        - Large parameter spaces can take significant time (minutes to hours)
        - Detailed results (best config, rankings, plots) saved to output_dir
        - Best configuration selected by highest mean real economic gain
        - seed defaults to 123 if not provided
    """
    payload = payload or sim_schemas.OptimizationRequest()
    summary = app_service.run_optimization(
        seed=payload.seed or 123,
        n_mc=payload.n_mc,
        scenario_data=payload.scenario,
    )
    return sim_schemas.OptimizationResponse(**summary)


@router.get("/runs", response_model=list[sim_schemas.RunResult])
def list_runs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    scenario_name: str | None = Query(None),
    location: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    include_archived: bool = Query(False),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[sim_schemas.RunResult]:
    """
    Retrieve historical simulation execution results.

    Returns a list of past simulation runs (both analyses and optimizations)
    with their summary results. Results are sorted by creation timestamp
    (newest first).

    This endpoint provides access to the execution history without needing
    to re-run simulations. Useful for:
    - Reviewing past analyses
    - Comparing different configuration results
    - Tracking simulation history over time
    - Audit trail of executed scenarios

    Args:
        limit: Maximum number of results to return. Must be between 1 and 500.
            Defaults to 50.
        persistence: Database persistence service (dependency injected).

    Returns:
        List of RunResult objects, each containing:
        - id: Unique database identifier
        - result_type: "analysis" or "optimization"
        - summary: Execution results as JSON (structure varies by type)
        - scenario_id: Link to scenario if from saved scenario (optional)
        - optimization_id: Link to optimization if from campaign (optional)
        - created_at: Timestamp when result was created

    Example:
        ```python
        # GET /api/runs?limit=10
        [
            {
                "id": 42,
                "result_type": "analysis",
                "summary": {
                    "scenario": "3kW PV + 5kWh Battery",
                    "final_gain_mean_eur": 2450.50,
                    "final_gain_real_mean_eur": 1890.25,
                    "prob_gain": 0.87
                },
                "scenario_id": 7,
                "optimization_id": null,
                "created_at": "2025-01-15T10:30:45.123456Z"
            },
            {
                "id": 41,
                "result_type": "optimization",
                "summary": {
                    "evaluations": 36,
                    "best_config": {...}
                },
                "scenario_id": null,
                "optimization_id": 5,
                "created_at": "2025-01-15T09:15:22.654321Z"
            },
            ...
        ]
        ```

    Notes:
        - Results ordered by created_at descending (newest first)
        - Only one of scenario_id or optimization_id will be non-null
        - summary structure varies by result_type
        - Empty list if no runs have been executed
        - Runs are only stored if persistence is enabled in application config
    """
    records = persistence.list_run_results(
        limit=limit,
        offset=offset,
        scenario_name=scenario_name,
        location=location,
        date_from=date_from,
        date_to=date_to,
        include_archived=include_archived,
    )
    return records


# Phase 12 — Dashboard helper endpoints: distinct locations (for the
# filter dropdown), soft archive toggle, hard delete.


@router.get("/runs/locations", response_model=list[str])
def list_run_locations(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[str]:
    """Return the distinct location names across all run results."""
    return persistence.list_distinct_run_locations()


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(
    run_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """Hard-delete a run result. 404 when the id is unknown."""
    removed = persistence.delete_run_result(run_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")


@router.patch("/runs/{run_id}/archive", response_model=sim_schemas.RunResult)
def archive_run(
    run_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
):
    """Mark a run as archived (hidden from the default Dashboard list)."""
    record = persistence.set_run_archived(run_id, archived=True)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return record


@router.patch("/runs/{run_id}/unarchive", response_model=sim_schemas.RunResult)
def unarchive_run(
    run_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
):
    """Clear the archived flag on a run."""
    record = persistence.set_run_archived(run_id, archived=False)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return record


# Phase 11 — Excel export endpoint. Reads the persisted ``summary`` JSON
# of a run (which now includes the ``cashflow_table`` payload) and
# streams an .xlsx workbook back to the client.
@router.get("/runs/{run_id}/export/cashflow.xlsx")
def export_run_cashflow_xlsx(
    run_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> StreamingResponse:
    """
    Export the monthly cash flow of a run as an Excel workbook.

    The workbook has two sheets: "Cash flow medio" (one row per month with
    nominal/real savings, bonus, cumulative profit, price, inflation
    factor) and "KPI" (decision metrics).

    Args:
        run_id: Primary key of the run to export. The run must have been
            executed in Phase 11 or later (older runs lack the
            ``cashflow_table`` payload and respond with 422).

    Returns:
        StreamingResponse with the .xlsx file content.

    Raises:
        HTTPException 404: No run with the given id exists.
        HTTPException 422: Run exists but predates Phase 11 (no
            ``cashflow_table`` in the summary).
    """
    record = persistence.get_run_result(run_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Run with id={run_id} not found"
        )
    buffer = io.BytesIO()
    try:
        build_cashflow_xlsx(record.summary, buffer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    buffer.seek(0)
    filename = f"cashflow_run_{run_id}.xlsx"
    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/runs/{run_id}/export/report.pdf")
def export_run_report_pdf(
    run_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> StreamingResponse:
    """
    Export the full Monte Carlo report of a run as a PDF.

    The PDF includes the Decisione KPI cards, the profit/energy/price/
    inflation charts (matplotlib-rendered) and a cash-flow table. The
    renderer reads only the ``RunResultRecord.summary`` JSON, so older
    runs without the Phase-11 payloads still produce a (smaller) report.

    Args:
        run_id: Primary key of the run to export.

    Returns:
        StreamingResponse with the generated PDF.

    Raises:
        HTTPException 404: No run with the given id exists.
    """
    record = persistence.get_run_result(run_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Run with id={run_id} not found"
        )
    buffer = io.BytesIO()
    build_pdf_report(record.summary, run_id=record.id, buffer=buffer)
    buffer.seek(0)
    filename = f"report_run_{run_id}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
