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

from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from ...persistence import PersistenceService
from ...scenario_builder import build_default_price_model
from ...simulation.prices import (
    PricePreviewResult,
    simulate_price_preview,
)
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


# ---------------------------------------------------------------------------
# Phase 10 — preview routes for the Database section of the UI
# ---------------------------------------------------------------------------


def _preview_payload(result: PricePreviewResult) -> Dict[str, Any]:
    """Convert the dataclass result to the same JSON shape used in Phase 3."""
    return asdict(result)


@router.get("/profiles/price/{profile_id}/preview")
def preview_saved_price_profile(
    profile_id: int,
    n_paths: int = 200,
    n_years: int = 20,
    seed: int = 42,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> Dict[str, Any]:
    """
    Run a stand-alone Monte Carlo preview of a *saved* price profile and
    return the fan-chart payload (mean / p05 / p95 / sample paths).

    This route lets the Database UI show what a stored price profile
    actually looks like in simulation, without having to wire it into a
    full scenario first.

    Args:
        profile_id: DB primary key of the price profile.
        n_paths: Monte Carlo paths to draw (capped server-side at 1000 to
            avoid pathological JSON sizes).
        n_years: Simulation horizon. Capped at 50 years.
        seed: Master seed for reproducibility.
        persistence: Injected DB service.

    Returns:
        JSON payload with the same shape used by the Dashboard
        `plots_data.price` block (Phase 3 schema):
        ``{months, mean_eur_per_kwh, p05_eur_per_kwh, p95_eur_per_kwh,
        sample_paths}``.

    Raises:
        HTTPException 404: profile not found.
        HTTPException 422: parameter values out of allowed range.
    """
    n_paths = min(max(int(n_paths), 1), 1000)
    n_years = min(max(int(n_years), 1), 50)

    profiles = persistence.list_price_profiles()
    record = next((p for p in profiles if p.id == profile_id), None)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Price profile id={profile_id} not found",
        )

    # Reuse the scenario_builder dispatcher: it already understands
    # model_type ∈ {escalating, gbm, mean_reverting} and applies defaults.
    price_model = build_default_price_model(
        scenario_data={"price": record.data}
    )
    result = simulate_price_preview(
        price_model=price_model,
        n_years=n_years,
        n_paths=n_paths,
        seed=seed,
    )
    return _preview_payload(result)


@router.post("/profiles/price/preview")
def preview_price_parameters(
    payload: profile_schemas.PriceProfileCreate,
    n_paths: int = 200,
    n_years: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Same as :func:`preview_saved_price_profile` but for *unsaved* parameters
    — drives the live preview in the price-profile create/edit form.

    The ``payload`` follows the regular price-profile create schema; only
    the ``data`` field matters for the preview (``name`` is accepted to
    keep the schema reuse straightforward).

    Args:
        payload: Pricing parameters in the same shape as
            ``POST /api/profiles/price``.
        n_paths: Capped at 1000 server-side.
        n_years: Capped at 50 server-side.
        seed: Master seed for reproducibility.

    Returns:
        Same JSON shape as :func:`preview_saved_price_profile`.

    Notes:
        - This endpoint is stateless: nothing is written to the DB.
        - Validation errors from the model constructors (e.g. negative
          volatility) surface as HTTP 422 via FastAPI's automatic
          ValueError→422 mapping.
    """
    n_paths = min(max(int(n_paths), 1), 1000)
    n_years = min(max(int(n_years), 1), 50)

    try:
        price_model = build_default_price_model(
            scenario_data={"price": payload.data}
        )
        result = simulate_price_preview(
            price_model=price_model,
            n_years=n_years,
            n_paths=n_paths,
            seed=seed,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _preview_payload(result)
