"""
Simulation execution schemas for API validation.

This module contains Pydantic models for simulation execution endpoints:
- Analysis: Single scenario Monte Carlo simulation
- Optimization: Multi-scenario parameter sweep optimization
- RunResult: Historical execution results

These schemas handle the core simulation workflow where users submit
scenario configurations and receive economic analysis results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnalysisRequest(BaseModel):
    """
    Request schema for single-scenario Monte Carlo analysis.

    Triggers a Monte Carlo simulation of a single photovoltaic energy system
    configuration, evaluating economic outcomes across multiple stochastic paths.

    The simulation considers:
    - Stochastic solar production (weather variability)
    - Stochastic load consumption (usage patterns)
    - Stochastic electricity prices (escalation uncertainty)
    - Battery degradation over system lifetime
    - Real vs nominal economic returns (inflation-adjusted)

    Attributes:
        n_mc: Number of Monte Carlo simulation paths to run (default from scenario config).
            More paths = more accurate statistics but longer computation time.
            Typical values: 100-1000 paths.
        seed: Random number generator seed for reproducibility (optional).
            Using the same seed with same inputs produces identical results.
            Useful for debugging and comparing scenarios.
        scenario: Complete scenario configuration as JSON dict (optional).
            If None, uses default scenario. Must include all required sections:
            - load_profile: Electricity consumption profile configuration
            - solar: PV system configuration and solar parameters
            - energy: Battery and inverter system configuration
            - price: Electricity price model configuration
            - economic: Investment cost and analysis parameters

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
                    "min_days_home": [25] * 12
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
        ```

    Notes:
        - If scenario is None, uses application's default scenario configuration
        - n_mc in request overrides n_mc in scenario.economic if both are provided
        - Seed is optional; if not provided, results will vary between runs
    """

    n_mc: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of Monte Carlo paths (must be >= 1)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    scenario: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complete scenario configuration (JSON), or None for default"
    )


class AnalysisResponse(BaseModel):
    """
    Response schema for completed Monte Carlo analysis.

    Contains aggregated economic results from the Monte Carlo simulation,
    including both nominal and inflation-adjusted (real) returns.

    Attributes:
        scenario: Name or identifier of the simulated scenario.
        final_gain_mean_eur: Mean final net gain in EUR (nominal, not inflation-adjusted).
            Positive = profitable, negative = loss.
            Calculated as: (energy savings + grid export revenue) - initial investment
        final_gain_real_mean_eur: Mean final net gain in EUR (real, inflation-adjusted).
            Accounts for time value of money and inflation effects.
            This is the economically meaningful metric for decision-making.
        prob_gain: Probability of positive return (0 to 1).
            Fraction of Monte Carlo paths where final_gain > 0.
            Example: 0.85 means 85% chance of profit across scenarios.
        output_dir: Directory path where detailed results are saved (optional).
            Contains plots, detailed statistics, and raw simulation data.
            Only present if save_outputs=True in application configuration.
        plots_data: Embedded plot data for visualization (optional).
            Dictionary containing plot data that can be rendered by frontend.
            Structure depends on result_builder configuration.

    Example:
        ```python
        # Response from POST /api/analysis
        {
            "scenario": "3kW PV + 5kWh Battery",
            "final_gain_mean_eur": 2450.50,
            "final_gain_real_mean_eur": 1890.25,
            "prob_gain": 0.87,
            "output_dir": "/path/to/results/analysis_20250115_103045",
            "plots_data": {
                "gain_distribution": [...],
                "monthly_cashflow": [...]
            }
        }
        ```

    Notes:
        - final_gain_real_mean_eur is always <= final_gain_mean_eur due to discounting
        - prob_gain near 1.0 indicates low risk, near 0.5 indicates high uncertainty
        - output_dir is None if application configured with save_outputs=False
    """

    scenario: str = Field(..., description="Scenario name or identifier")
    final_gain_mean_eur: float = Field(..., description="Mean final gain (nominal EUR)")
    final_gain_real_mean_eur: float = Field(..., description="Mean final gain (real/inflation-adjusted EUR)")
    prob_gain: float = Field(..., ge=0.0, le=1.0, description="Probability of positive return (0-1)")
    output_dir: Optional[str] = Field(None, description="Output directory path (if saved)")
    plots_data: Optional[Dict[str, Any]] = Field(None, description="Embedded plot data for visualization")


class OptimizationRequest(BaseModel):
    """
    Request schema for multi-scenario optimization.

    Triggers a parameter sweep optimization that evaluates multiple hardware
    configurations to find the optimal PV + battery system design.

    The optimization explores combinations of:
    - Inverter models (from selected options)
    - Panel models (from selected options)
    - Battery models (from selected options)
    - Panel counts (number of panels to install)
    - Battery counts (number of battery units)

    Each combination is evaluated via Monte Carlo simulation to determine
    expected economic return, helping identify the best configuration.

    Attributes:
        seed: Random number generator seed for reproducibility (optional).
            Ensures all scenario evaluations use consistent random sequences.
        n_mc: Number of Monte Carlo paths per scenario (optional, overrides scenario config).
            Applied to every scenario in the optimization sweep.
            Note: Total computation = n_mc * number_of_scenarios.
        scenario: Scenario configuration with optimization parameters (optional).
            Must include 'optimization' section with:
            - inverter_options: List of inverter hardware objects to consider
            - panel_options: List of panel hardware objects to consider
            - battery_options: List of battery hardware objects to consider
            - panel_count_options: List of panel counts to try (e.g., [6, 8, 10])
            - battery_count_options: List of battery counts to try (e.g., [0, 1, 2])
            - include_no_battery: Whether to include battery-less configurations

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
                        {"name": "SMA 6kW", "p_ac_max_kw": 6.0", ...}
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
                "scenario_name": "Home PV Optimization",
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            }
        }
        ```

    Notes:
        - Number of evaluated scenarios = len(inverters) * len(panels) * len(batteries) *
          len(panel_counts) * len(battery_counts)
        - Large parameter spaces can take significant time (minutes to hours)
        - Results are saved to output directory with comparison plots
    """

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility across all scenarios"
    )
    n_mc: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of Monte Carlo paths per scenario (must be >= 1)"
    )
    scenario: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Scenario configuration including optimization parameters"
    )


class OptimizationResponse(BaseModel):
    """
    Response schema for completed optimization.

    Provides summary statistics from the parameter sweep optimization,
    indicating how many configurations were evaluated.

    Detailed results (best configuration, comparison plots, full rankings)
    are saved to the output directory if save_outputs=True.

    Attributes:
        evaluations: Total number of scenario configurations evaluated.
            Equals len(inverters) * len(panels) * len(batteries) *
            len(panel_counts) * len(battery_counts).
        output_dir: Directory path where detailed results are saved (optional).
            Contains:
            - Best configuration details
            - Comparison plots showing all evaluated scenarios
            - Ranking table sorted by economic return
            - Individual analysis results for each scenario

    Example:
        ```python
        # Response from POST /api/optimization
        {
            "evaluations": 36,
            "output_dir": "/path/to/results/optimization_20250115_104530"
        }
        ```

    Notes:
        - Check output_dir for detailed results including best configuration
        - Optimization prints progress to logs: "[Opt] Evaluated X/Y scenarios..."
        - Best configuration is determined by highest mean real economic gain
    """

    evaluations: int = Field(..., ge=1, description="Number of scenarios evaluated")
    output_dir: Optional[str] = Field(None, description="Output directory path (if saved)")


class RunResult(BaseModel):
    """
    Historical simulation run result schema.

    Represents a stored execution result from either an analysis or optimization run.
    Used for retrieving historical simulation results from the database.

    Results are automatically persisted after each execution if persistence is enabled,
    allowing users to review past analyses without re-running simulations.

    Attributes:
        id: Unique database identifier for this run result.
        result_type: Type of execution that produced this result.
            - "analysis": Single-scenario Monte Carlo simulation
            - "optimization": Multi-scenario parameter sweep
        summary: Execution results as JSON dictionary.
            Structure depends on result_type:
            - For analysis: Contains final_gain_mean_eur, prob_gain, etc.
            - For optimization: Contains evaluations, best_config, etc.
        scenario_id: Foreign key to scenario record if result from saved scenario (optional).
        optimization_id: Foreign key to optimization record if result from campaign (optional).
        created_at: Timestamp when this result was created.

    Example:
        ```python
        # Response from GET /api/runs
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
        }
        ```

    Notes:
        - Only one of scenario_id or optimization_id will be non-null
        - summary structure varies by result_type
        - Runs are ordered by created_at descending (newest first) in GET /api/runs
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    result_type: str = Field(..., description="Type of execution: 'analysis' or 'optimization'")
    summary: Dict[str, Any] = Field(..., description="Execution results (structure varies by type)")
    scenario_id: Optional[int] = Field(None, description="Link to scenario if from saved scenario")
    optimization_id: Optional[int] = Field(None, description="Link to optimization if from campaign")
    created_at: datetime = Field(..., description="Timestamp of result creation")
