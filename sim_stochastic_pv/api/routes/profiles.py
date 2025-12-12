"""
Load and price profile management API endpoints.

This module provides CRUD operations for reusable electricity profiles:
- Load Profiles: Electricity consumption patterns for homes
- Price Profiles: Electricity pricing models and escalation parameters

Profiles can be saved and reused across multiple scenarios, enabling
consistent modeling and easy comparison of different configurations.
Like hardware, profiles support upsert behavior and can be referenced
by ID in scenarios.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...persistence import PersistenceService
from .. import dependencies
from ..schemas import profiles as profile_schemas

router = APIRouter(prefix="/api", tags=["profiles"])


@router.get("/profiles/load", response_model=list[profile_schemas.LoadProfileResponse])
def list_load_profiles(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[profile_schemas.LoadProfileResponse]:
    """
    List all load profiles in the database.

    Returns all saved electricity consumption profiles. Load profiles define
    how electricity usage varies over time, including daily patterns, seasonal
    variations, and occupancy effects.

    Profiles are sorted by creation date (newest first). No pagination is
    currently implemented - returns all records.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of LoadProfileResponse objects containing profile configurations.
        Each includes id, name, profile_type, and type-specific data.

    Example:
        ```python
        # GET /api/profiles/load
        [
            {
                "id": 1,
                "name": "Standard Italian Home",
                "profile_type": "arera",
                "data": {}
            },
            {
                "id": 2,
                "name": "Low Consumption Home",
                "profile_type": "custom",
                "data": {
                    "home_profiles_w": [200, 180, 190, 210, ...]
                }
            },
            ...
        ]
        ```

    Notes:
        - profile_type can be: "arera" (Italian standard), "custom" (monthly),
          or "custom_24h" (hourly)
        - ARERA profiles have empty data (uses built-in profile)
        - Custom profiles include home_profiles_w array (12 or 12×24 values)
    """
    return persistence.list_load_profiles()


@router.post("/profiles/load", response_model=profile_schemas.LoadProfileResponse)
def create_load_profile(
    payload: profile_schemas.LoadProfileCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.LoadProfileResponse:
    """
    Create or update a load profile.

    Implements upsert behavior: creates new profile if name doesn't exist,
    otherwise updates the existing record. Profile updates automatically
    propagate to all scenarios referencing this profile by ID.

    Args:
        payload: Load profile data including name, type, and configuration.
            Required fields: name, profile_type, data
            profile_type must be: "arera", "custom", or "custom_24h"
            data structure depends on profile_type (see schema documentation)
        persistence: Database persistence service (dependency injected).

    Returns:
        LoadProfileResponse with the created or updated profile data including
        database ID and timestamp.

    Example:
        ```python
        # POST /api/profiles/load - ARERA profile
        {
            "name": "Standard Italian Home",
            "profile_type": "arera",
            "data": {}
        }

        # POST /api/profiles/load - Custom monthly profile
        {
            "name": "Night Worker Profile",
            "profile_type": "custom",
            "data": {
                "home_profiles_w": [220, 200, 210, 230, 250, 300,
                                   280, 270, 260, 240, 220, 210],
                "away_profiles_w": [100, 100, 100, 100, 120, 150,
                                   140, 130, 120, 110, 100, 100],
                "min_days_home": [26, 24, 26, 25, 26, 24,
                                 28, 28, 26, 26, 24, 26],
                "home_variation_percentiles": [-0.1, 0.1],
                "away_variation_percentiles": [-0.05, 0.05]
            }
        }

        # POST /api/profiles/load - Custom hourly profile
        {
            "name": "Detailed Hourly Profile",
            "profile_type": "custom_24h",
            "data": {
                "home_profiles_w": [
                    [100, 90, 80, ..., 120],  # January (24 hours)
                    [110, 95, 85, ..., 130],  # February (24 hours)
                    ...  # 12 months total
                ],
                "min_days_home": [25, 25, 25, ..., 25]
            }
        }
        ```

    Notes:
        - Upsert is based on name matching (case-sensitive)
        - ARERA profiles use built-in Italian residential standard
        - Custom profiles: 12 monthly average values (W)
        - Custom_24h profiles: 12 months × 24 hourly values (W)
        - Updating propagates to all scenarios using this profile ID
    """
    return persistence.upsert_load_profile(
        payload.name,
        payload.profile_type,
        payload.data,
    )


@router.get("/profiles/price", response_model=list[profile_schemas.PriceProfileResponse])
def list_price_profiles(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[profile_schemas.PriceProfileResponse]:
    """
    List all price profiles in the database.

    Returns all saved electricity pricing models. Price profiles define
    electricity costs over the system lifetime, including base price and
    escalation parameters (with optional stochastic modeling).

    Profiles are sorted by creation date (newest first). No pagination is
    currently implemented - returns all records.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of PriceProfileResponse objects containing pricing configurations.
        Each includes id, name, and price model parameters in data field.

    Example:
        ```python
        # GET /api/profiles/price
        [
            {
                "id": 1,
                "name": "Italian Residential 2025",
                "data": {
                    "base_price_eur_per_kwh": 0.22,
                    "annual_escalation": 0.025,
                    "use_stochastic_escalation": true,
                    "escalation_variation_percentiles": [-0.05, 0.05]
                }
            },
            ...
        ]
        ```

    Notes:
        - base_price_eur_per_kwh should match current market rates
        - annual_escalation typically 0-5% (0.00 to 0.05)
        - Stochastic escalation adds Monte Carlo uncertainty to price evolution
    """
    return persistence.list_price_profiles()


@router.post("/profiles/price", response_model=profile_schemas.PriceProfileResponse)
def create_price_profile(
    payload: profile_schemas.PriceProfileCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.PriceProfileResponse:
    """
    Create or update a price profile.

    Implements upsert behavior: creates new profile if name doesn't exist,
    otherwise updates the existing record. Profile updates automatically
    propagate to all scenarios referencing this profile by ID.

    Args:
        payload: Price profile data including name and pricing model configuration.
            Required fields: name, data
            data should include:
            - base_price_eur_per_kwh: Initial electricity price (EUR/kWh)
            - annual_escalation: Average annual increase rate (e.g., 0.02 = 2%/year)
            - use_stochastic_escalation: Enable stochastic modeling (boolean)
            - escalation_variation_percentiles: Uncertainty range (e.g., [-0.05, 0.05])
        persistence: Database persistence service (dependency injected).

    Returns:
        PriceProfileResponse with the created or updated profile data including
        database ID and timestamp.

    Example:
        ```python
        # POST /api/profiles/price - Deterministic escalation
        {
            "name": "Conservative Fixed Escalation",
            "data": {
                "base_price_eur_per_kwh": 0.20,
                "annual_escalation": 0.015,
                "use_stochastic_escalation": false
            }
        }

        # POST /api/profiles/price - Stochastic escalation
        {
            "name": "Realistic Variable Escalation",
            "data": {
                "base_price_eur_per_kwh": 0.22,
                "annual_escalation": 0.025,
                "use_stochastic_escalation": true,
                "escalation_variation_percentiles": [-0.05, 0.05]
            }
        }
        ```

    Notes:
        - Upsert is based on name matching (case-sensitive)
        - base_price_eur_per_kwh should reflect current market conditions
        - Stochastic escalation models price uncertainty via Monte Carlo
        - escalation_variation_percentiles define the range of price paths
        - Updating propagates to all scenarios using this profile ID
    """
    return persistence.upsert_price_profile(payload.name, payload.data)
