"""
Saved configuration management API endpoints.

This module provides CRUD operations for saved scenario and campaign
configurations. These endpoints enable the database-driven workflow where
users can:
1. Save scenario configurations with hardware/profile IDs
2. Save campaign (optimization) configurations with multiple hardware selections
3. List and retrieve saved configurations
4. Execute saved configurations (see execution module)

Configurations support upsert behavior and can reference hardware and profiles
by ID, ensuring that specification updates propagate automatically.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...persistence import PersistenceService
from .. import dependencies
from ..schemas import configurations as config_schemas

router = APIRouter(prefix="/api", tags=["configurations"])


@router.get("/configurations", response_model=list[config_schemas.SavedConfigurationResponse])
def list_configurations(
    type: str | None = None,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[config_schemas.SavedConfigurationResponse]:
    """
    List saved configurations (scenarios and campaigns).

    Returns all saved configuration records, optionally filtered by type.
    Configurations are sorted by creation date (newest first).

    This endpoint provides access to the saved configuration library,
    enabling users to:
    - Browse saved scenarios and campaigns
    - Select configurations for execution
    - Review configuration details before running

    Args:
        type: Optional filter by configuration type. Valid values:
            - "scenario": Only return single-scenario configurations
            - "campaign": Only return optimization campaign configurations
            - None: Return all configurations (default)
        persistence: Database persistence service (dependency injected).

    Returns:
        List of SavedConfigurationResponse objects, each containing:
        - id: Unique database identifier
        - name: Human-readable configuration name
        - config_type: "scenario" or "campaign"
        - data: Complete configuration as JSON (structure varies by type)

    Example:
        ```python
        # GET /api/configurations
        [
            {
                "id": 1,
                "name": "Standard Home 3kW System",
                "config_type": "scenario",
                "data": {
                    "inverter_id": 5,
                    "battery_id": 3,
                    "load_profile": {...},
                    "solar": {...},
                    "energy": {...},
                    "price": {...},
                    "economic": {...}
                }
            },
            {
                "id": 2,
                "name": "Home Optimization Campaign",
                "config_type": "campaign",
                "data": {
                    "hardware_selections": {
                        "inverter_ids": [1, 2, 3],
                        "panel_ids": [1],
                        "battery_ids": [1, 2]
                    },
                    "optimization": {...},
                    "load_profile": {...},
                    ...
                }
            }
        ]

        # GET /api/configurations?type=scenario
        [
            {
                "id": 1,
                "name": "Standard Home 3kW System",
                "config_type": "scenario",
                ...
            }
        ]
        ```

    Notes:
        - Configurations with hardware IDs will be hydrated at execution time
        - Hydration ensures scenarios use current hardware specs from database
        - Empty list if no configurations have been saved
        - Use POST /scenarios/{id}/run or /campaigns/{id}/run to execute
    """
    return persistence.list_configurations(config_type=type)


@router.post("/configurations", response_model=config_schemas.SavedConfigurationResponse)
def create_configuration(
    payload: config_schemas.SavedConfigurationCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> config_schemas.SavedConfigurationResponse:
    """
    Create or update a saved configuration.

    Implements upsert behavior: creates new configuration if name doesn't exist,
    otherwise updates the existing record. This allows easy configuration updates
    without checking for existence first.

    Configurations can store either:
    - Scenarios: Single-scenario configurations with optional hardware/profile IDs
    - Campaigns: Multi-scenario optimization configs with hardware selections

    Args:
        payload: Configuration data including name, type, and complete config.
            Required fields:
            - name: Unique configuration identifier (upsert key)
            - config_type: Must be "scenario" or "campaign"
            - data: Complete configuration structure (varies by type)

            For scenarios, data should include:
            - Optional hardware IDs: inverter_id, battery_id
            - Optional profile IDs: load_profile_id, price_profile_id
            - All scenario sections: load_profile, solar, energy, price, economic

            For campaigns, data should include:
            - hardware_selections with inverter_ids, panel_ids, battery_ids arrays
            - optimization parameters: panel_count_options, battery_count_options
            - All scenario sections: load_profile, solar, energy, price, economic

        persistence: Database persistence service (dependency injected).

    Returns:
        SavedConfigurationResponse with the created or updated configuration
        including database ID and timestamp.

    Example:
        ```python
        # POST /api/configurations - Scenario with hardware IDs
        {
            "name": "My Home System",
            "config_type": "scenario",
            "data": {
                "inverter_id": 5,
                "battery_id": 3,
                "load_profile_id": 1,
                "price_profile_id": 2,
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
                    "n_batteries": 1
                },
                "price": {
                    "base_price_eur_per_kwh": 0.20,
                    "annual_escalation": 0.02
                },
                "economic": {
                    "n_mc": 500,
                    "investment_eur": 8000.0
                }
            }
        }

        # POST /api/configurations - Campaign with hardware selections
        {
            "name": "System Optimization",
            "config_type": "campaign",
            "data": {
                "hardware_selections": {
                    "inverter_ids": [1, 2, 3],
                    "panel_ids": [1],
                    "battery_ids": [1, 2]
                },
                "optimization": {
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
            "id": 15,
            "name": "My Home System",
            "config_type": "scenario",
            "data": {...}
        }
        ```

    Notes:
        - Upsert is based on name matching (case-sensitive)
        - Hardware IDs enable automatic spec updates when hardware is modified
        - When executed, IDs are hydrated to current hardware specs from database
        - Use POST /scenarios/{id}/run or /campaigns/{id}/run to execute saved configs
        - Frontend should save hardware IDs to benefit from database-driven workflow
    """
    return persistence.save_configuration(
        payload.name,
        payload.config_type,
        payload.data,
    )


@router.get("/scenarios", response_model=list[config_schemas.ScenarioResponse])
def list_scenarios(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[config_schemas.ScenarioResponse]:
    """
    List historical scenario execution records.

    Returns scenario records from the execution history table. This is distinct
    from saved configurations - these are scenarios that were actually executed
    and stored in the historical scenarios table.

    Note: This endpoint is maintained for backward compatibility. For new code,
    consider using GET /configurations?type=scenario for saved scenarios or
    GET /runs for execution history.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of ScenarioResponse objects containing historical scenario records
        with their configurations and creation timestamps.

    Example:
        ```python
        # GET /api/scenarios
        [
            {
                "id": 15,
                "name": "3kW PV + 5kWh Battery",
                "config": {
                    "load_profile": {...},
                    "solar": {...},
                    "energy": {...},
                    "price": {...},
                    "economic": {...}
                },
                "created_at": "2025-01-15T10:30:45.123456Z"
            },
            ...
        ]
        ```

    Notes:
        - Returns executed scenarios from scenarios table (not configurations table)
        - Sorted by creation date (newest first)
        - This table is populated when scenarios are executed
        - Consider migrating to /configurations for saved scenarios
    """
    return persistence.list_scenarios()
