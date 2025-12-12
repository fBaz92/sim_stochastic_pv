"""
Database-driven execution API endpoints.

This module provides endpoints for executing saved configurations from the
database. Unlike the simulation module (which accepts inline configurations),
these endpoints fetch configurations by ID and hydrate hardware/profile
references before execution.

Endpoints:
- POST /scenarios/{scenario_id}/run: Execute a saved scenario
- POST /campaigns/{campaign_id}/run: Execute a saved campaign

The key benefit of database-driven execution is that hardware and profile
specifications are hydrated at runtime from the database, ensuring scenarios
always use the current specs even if hardware has been updated since the
scenario was saved.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...application import SimulationApplication
from ...persistence import PersistenceService
from .. import dependencies
from ..schemas import simulation as sim_schemas

router = APIRouter(prefix="/api", tags=["execution"])


@router.post("/scenarios/{scenario_id}/run", response_model=sim_schemas.AnalysisResponse)
def run_saved_scenario(
    scenario_id: int,
    seed: int | None = None,
    n_mc: int | None = None,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> sim_schemas.AnalysisResponse:
    """
    Execute a saved scenario configuration by ID.

    Implements the database-driven execution workflow:
    1. Fetch saved configuration from database by ID
    2. Verify it's a scenario (not a campaign)
    3. Hydrate hardware and profile IDs to current specs from database
    4. Execute the analysis with hydrated configuration
    5. Return results (same structure as POST /analysis)

    This ensures that scenarios always use up-to-date hardware specifications,
    even if the hardware catalog has been updated since the scenario was saved.

    Args:
        scenario_id: Database ID of the saved scenario configuration.
        seed: Optional random seed for reproducibility. Overrides any seed
            in the saved configuration. Defaults to 123 if not provided.
        n_mc: Optional Monte Carlo path count. Overrides n_mc in saved
            configuration's economic section. If None, uses saved value.
        persistence: Database persistence service (dependency injected).
        app_service: Simulation application service (dependency injected).

    Returns:
        AnalysisResponse containing:
        - scenario: Scenario name/identifier
        - final_gain_mean_eur: Mean net gain (nominal)
        - final_gain_real_mean_eur: Mean net gain (inflation-adjusted)
        - prob_gain: Probability of positive return (0-1)
        - output_dir: Path to saved results (if enabled)
        - plots_data: Embedded visualization data (if enabled)

    Raises:
        HTTPException 404: If scenario_id not found in database
        HTTPException 400: If configuration exists but is not type "scenario"

    Example:
        ```python
        # POST /api/scenarios/7/run?seed=123&n_mc=500

        # Response (same as POST /api/analysis)
        {
            "scenario": "Standard Home 3kW System",
            "final_gain_mean_eur": 2450.50,
            "final_gain_real_mean_eur": 1890.25,
            "prob_gain": 0.87,
            "output_dir": "/path/to/results/analysis_20250115_103045"
        }
        ```

    Notes:
        - Hardware/profile IDs in saved scenario are hydrated from database
        - If inverter_id=5 in saved scenario, current specs for inverter #5
          are fetched and used (even if specs have changed since save)
        - This automatic spec update is the key benefit of database-driven workflow
        - Query parameters (seed, n_mc) override saved configuration values
        - Results are automatically saved to database if persistence enabled
    """
    # Fetch the saved configuration
    config = persistence.get_configuration_by_id(scenario_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario {scenario_id} not found",
        )

    # Verify it's a scenario (not a campaign)
    if config.config_type != "scenario":
        raise HTTPException(
            status_code=400,
            detail=f"Configuration {scenario_id} is a {config.config_type}, not a scenario",
        )

    # Hydrate the scenario: replace hardware/profile IDs with current specs
    hydrated_scenario = persistence.hydrate_scenario_from_ids(config.data)

    # Run the analysis with the hydrated scenario
    summary = app_service.run_analysis(
        n_mc=n_mc,
        seed=seed or 123,
        scenario_data=hydrated_scenario,
    )
    return sim_schemas.AnalysisResponse(**summary)


@router.post("/campaigns/{campaign_id}/run", response_model=sim_schemas.OptimizationResponse)
def run_saved_campaign(
    campaign_id: int,
    seed: int | None = None,
    n_mc: int | None = None,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    app_service: SimulationApplication = Depends(dependencies.get_application_service),
) -> sim_schemas.OptimizationResponse:
    """
    Execute a saved campaign (optimization) configuration by ID.

    Implements the database-driven execution workflow for campaigns:
    1. Fetch saved configuration from database by ID
    2. Verify it's a campaign (not a single scenario)
    3. Hydrate hardware selection IDs to current specs from database
    4. Execute the optimization with hydrated configuration
    5. Return results (same structure as POST /optimization)

    For campaigns, the hardware_selections section contains arrays of IDs
    (inverter_ids, panel_ids, battery_ids). These are hydrated into the
    optimization.{inverter,panel,battery}_options arrays with current specs.

    Args:
        campaign_id: Database ID of the saved campaign configuration.
        seed: Optional random seed for reproducibility across all scenarios
            in the optimization. Overrides any seed in saved configuration.
            Defaults to 321 if not provided.
        n_mc: Optional Monte Carlo path count per scenario. Overrides n_mc
            in saved configuration's economic section. If None, uses saved value.
        persistence: Database persistence service (dependency injected).
        app_service: Simulation application service (dependency injected).

    Returns:
        OptimizationResponse containing:
        - evaluations: Number of scenarios evaluated
        - output_dir: Path to detailed results (if enabled)

    Raises:
        HTTPException 404: If campaign_id not found in database
        HTTPException 400: If configuration exists but is not type "campaign"

    Example:
        ```python
        # POST /api/campaigns/5/run?seed=321&n_mc=200

        # Response (same as POST /api/optimization)
        {
            "evaluations": 36,
            "output_dir": "/path/to/results/optimization_20250115_104530"
        }
        ```

    Notes:
        - Campaign data must include full scenario structure with all sections:
          hardware_selections, optimization, load_profile, solar, energy, price, economic
        - hardware_selections.inverter_ids array is hydrated to
          optimization.inverter_options with current specs
        - Similarly for panel_ids and battery_ids
        - This ensures optimization always uses current hardware catalog
        - Query parameters (seed, n_mc) override saved configuration values
        - Detailed results (best config, rankings) saved to output_dir
    """
    # Fetch the saved configuration
    config = persistence.get_configuration_by_id(campaign_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Campaign {campaign_id} not found",
        )

    # Verify it's a campaign (not a scenario)
    if config.config_type != "campaign":
        raise HTTPException(
            status_code=400,
            detail=f"Configuration {campaign_id} is a {config.config_type}, not a campaign",
        )

    # Hydrate the campaign: replace hardware selection IDs with current specs
    hydrated_scenario = persistence.hydrate_scenario_from_ids(config.data)

    # Run the optimization with the hydrated scenario
    summary = app_service.run_optimization(
        seed=seed or 321,
        n_mc=n_mc,
        scenario_data=hydrated_scenario,
    )
    return sim_schemas.OptimizationResponse(**summary)
