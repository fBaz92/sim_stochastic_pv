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

import io
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from ...output.exporters import build_template_xlsx, parse_load_profile_xlsx
from ...persistence import PersistenceService
from ...scenario_builder import (
    build_default_appliance_profile_config,
    build_default_price_model,
    build_default_solar_model,
    build_default_stochastic_load_config,
    build_default_thermal_load_config,
    build_regime_load_profile_factory,
)
from ...simulation.load_preview import (
    LoadPreviewResult,
    simulate_load_profile_preview,
)
from ...simulation.load_profiles import (
    DEFAULT_PRESENCE_CALENDAR,
    HOUSE_TYPE_PRESETS,
    PresenceCalendar,
    annual_kwh_from_bimonthly,
    fit_bolletta_profile,
)
from ...simulation.prices import (
    PricePreviewResult,
    simulate_price_preview,
)
from .. import dependencies
from ..schemas import profiles as profile_schemas
from ..schemas.profiles import SolarProfileResponse, SolarProfileUpdate


_SUPPORTED_LOAD_KINDS = {"monthly_avg", "monthly_24h", "weekly"}

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


@router.put("/profiles/load/{profile_id}", response_model=profile_schemas.LoadProfileResponse)
def update_load_profile(
    profile_id: int,
    payload: profile_schemas.LoadProfileCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.LoadProfileResponse:
    """
    Update a load profile by primary key (allows rename).

    Args:
        profile_id: Primary-key ID of the profile to update.
        payload: New profile data (name + profile_type + data).

    Raises:
        HTTPException 404: profile not found.
        HTTPException 409: new ``name`` already used by another profile.
    """
    try:
        record = persistence.update_load_profile(
            profile_id, payload.name, payload.profile_type, payload.data
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Load profile id={profile_id} not found")
    return record


@router.delete("/profiles/load/{profile_id}")
def delete_load_profile(
    profile_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete a load profile by ID.

    Args:
        profile_id: Database primary key of the load profile to delete.
        persistence: Database persistence service (dependency injected).

    Returns:
        JSON ``{"ok": true, "id": <profile_id>}`` on success.

    Raises:
        HTTPException 404: profile not found.
    """
    if not persistence.delete_load_profile(profile_id):
        raise HTTPException(status_code=404, detail=f"Load profile id={profile_id} not found")
    return {"ok": True, "id": profile_id}


def _run_load_preview(
    profile_type: str,
    data: Dict[str, Any],
    params: profile_schemas.LoadProfilePreviewParams,
    persistence: PersistenceService,
) -> LoadPreviewResult:
    """
    Shared core of the load-profile preview endpoints.

    Resolves the regime sub-profile factory and the optional behaviour layers
    (daily variability, discrete appliances, HVAC) from the profile's ``data``
    blocks, hydrates the climate model when requested, and runs the
    representative-week Monte Carlo. Appliances and HVAC apply only to the
    ``home`` regime (the away regime is intentionally a simple shape).

    Args:
        profile_type: Stored profile type (home_away / custom / custom_24h / …).
        data: Profile ``data`` JSON, optionally with ``stochastic`` /
            ``appliances`` / ``thermal`` blocks.
        params: Preview tuning parameters (month, regime, climate, n_paths, seed).
        persistence: Persistence service for the climate-profile lookup.

    Returns:
        The :class:`LoadPreviewResult` to be serialised into the response.

    Raises:
        HTTPException 400: invalid regime/shape or HVAC requested without a
            climate profile.
        HTTPException 404: ``climate_profile_id`` does not resolve.
    """
    regime = str(params.regime or "home").lower()
    if regime not in ("home", "away"):
        raise HTTPException(status_code=400, detail="regime must be 'home' or 'away'")

    try:
        factory = build_regime_load_profile_factory(profile_type, data, regime)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Appliances + HVAC are home-only; the away regime keeps just the (optional)
    # daily variability on top of its semi-constant pattern.
    wrapper = {
        "load_profile": {
            "stochastic": data.get("stochastic"),
            "appliances": data.get("appliances") if regime == "home" else None,
        },
        "thermal_load": data.get("thermal") if regime == "home" else None,
    }
    try:
        stochastic_config = build_default_stochastic_load_config(wrapper)
        appliance_config = build_default_appliance_profile_config(wrapper)
        thermal_load_config = build_default_thermal_load_config(wrapper)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    thermal_model = None
    if params.climate_profile_id is not None:
        thermal_model = persistence.load_thermal_model(int(params.climate_profile_id))
        if thermal_model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Climate profile {params.climate_profile_id} not found",
            )
    if thermal_load_config is not None and thermal_model is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Il modello HVAC del profilo richiede un profilo climatico: "
                "seleziona un clima (climate_profile_id) per l'anteprima."
            ),
        )

    solar_hourly_shape = None
    if appliance_config is not None and any(
        a.schedule_mode == "smart_pv" for a in appliance_config.appliances
    ):
        # A generic daytime PV shape is sufficient to bias smart_pv starts in a
        # preview (the real scenario uses its own solar profile).
        solar_hourly_shape = build_default_solar_model().hourly_shape

    try:
        return simulate_load_profile_preview(
            base_profile_factory=factory,
            regime=regime,
            month=params.month,
            n_paths=params.n_paths,
            seed=params.seed,
            stochastic_config=stochastic_config,
            appliance_config=appliance_config,
            thermal_load_config=thermal_load_config,
            thermal_model=thermal_model,
            solar_hourly_shape=solar_hourly_shape,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/profiles/load/preview",
    response_model=profile_schemas.LoadProfilePreviewResponse,
)
def preview_load_profile_inline(
    payload: profile_schemas.LoadProfilePreviewRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.LoadProfilePreviewResponse:
    """
    Preview an *unsaved* load profile (shape carried in the request body).

    Powers the live preview in the detail-page editor: the user tweaks
    variability / appliances / HVAC and sees the representative week update
    before saving. See :func:`_run_load_preview` for the semantics.
    """
    result = _run_load_preview(payload.profile_type, payload.data, payload, persistence)
    return profile_schemas.LoadProfilePreviewResponse(**asdict(result))


@router.post(
    "/profiles/load/{profile_id}/preview",
    response_model=profile_schemas.LoadProfilePreviewResponse,
)
def preview_load_profile_saved(
    profile_id: int,
    payload: profile_schemas.LoadProfilePreviewParams,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.LoadProfilePreviewResponse:
    """
    Preview a *saved* load profile by id (shape read from the database).

    Raises:
        HTTPException 404: the load profile id does not exist.
    """
    record = persistence.get_load_profile_by_id(profile_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Load profile id={profile_id} not found"
        )
    result = _run_load_preview(
        record.profile_type, record.data or {}, payload, persistence
    )
    return profile_schemas.LoadProfilePreviewResponse(**asdict(result))


@router.get(
    "/profiles/load/house-types",
    response_model=list[profile_schemas.HouseTypeResponse],
)
def list_house_types() -> list[profile_schemas.HouseTypeResponse]:
    """
    List the house-type archetypes for the quick bill-fit entry level.

    Feeds the "tipo di casa" dropdown on the load-profile detail page: each
    entry carries a typical annual consumption the UI uses to pre-fill the bill
    field. Reference data only — it never enters the fit maths.

    Returns:
        List of :class:`HouseTypeResponse`, in the registry's declaration order
        (small apartment → large house → vacation home).
    """
    return [
        profile_schemas.HouseTypeResponse(
            key=key,
            label_it=preset.label_it,
            floor_area_m2=preset.floor_area_m2,
            baseline_annual_kwh=preset.baseline_annual_kwh,
        )
        for key, preset in HOUSE_TYPE_PRESETS.items()
    ]


@router.post(
    "/profiles/load/fit-bolletta",
    response_model=profile_schemas.BollettaFitResponse,
)
def fit_bolletta(
    payload: profile_schemas.BollettaFitRequest,
) -> profile_schemas.BollettaFitResponse:
    """
    Auto-fit a load profile from an annual bill and a presence calendar.

    Computes the ARERA home-scale factor that makes the profile's expected
    annual energy match the declared consumption, and returns a ready-to-save
    profile ``data`` block (see :func:`fit_bolletta_profile`). Stateless —
    nothing is written to the DB; the frontend saves the profile separately.

    Args:
        payload: Bill (annual or bimonthly), optional house type, optional
            presence calendar.

    Returns:
        :class:`BollettaFitResponse` with the scale factor, the home/away energy
        split and the derived profile data.

    Raises:
        HTTPException 422: neither ``annual_kwh`` nor ``bimonthly_kwh`` supplied,
            a non-positive consumption, or an unknown ``house_type``.
    """
    if payload.bimonthly_kwh is not None:
        try:
            target_annual_kwh = annual_kwh_from_bimonthly(payload.bimonthly_kwh)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    elif payload.annual_kwh is not None:
        target_annual_kwh = float(payload.annual_kwh)
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'annual_kwh' or 'bimonthly_kwh'.",
        )

    if target_annual_kwh <= 0:
        raise HTTPException(
            status_code=422, detail="Annual consumption must be greater than 0."
        )

    if payload.house_type is not None and payload.house_type not in HOUSE_TYPE_PRESETS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown house_type '{payload.house_type}'.",
        )

    if payload.presence_calendar is None:
        calendar = DEFAULT_PRESENCE_CALENDAR
    else:
        calendar = PresenceCalendar.from_dict(payload.presence_calendar.model_dump())

    try:
        result = fit_bolletta_profile(
            target_annual_kwh, calendar, house_type=payload.house_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return profile_schemas.BollettaFitResponse(**result)


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


@router.put("/profiles/price/{profile_id}", response_model=profile_schemas.PriceProfileResponse)
def update_price_profile(
    profile_id: int,
    payload: profile_schemas.PriceProfileCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> profile_schemas.PriceProfileResponse:
    """
    Update a price profile by primary key (allows rename).

    Args:
        profile_id: Primary-key ID of the profile to update.
        payload: New profile data.

    Raises:
        HTTPException 404: profile not found.
        HTTPException 409: new ``name`` already used by another profile.
    """
    try:
        record = persistence.update_price_profile(profile_id, payload.name, payload.data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Price profile id={profile_id} not found")
    return record


@router.delete("/profiles/price/{profile_id}")
def delete_price_profile(
    profile_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete a price profile by ID.

    Note: this must be declared *before* the preview routes so that FastAPI
    does not match ``/profiles/price/preview`` as ``profile_id="preview"``.

    Args:
        profile_id: Database primary key of the price profile to delete.
        persistence: Database persistence service (dependency injected).

    Returns:
        JSON ``{"ok": true, "id": <profile_id>}`` on success.

    Raises:
        HTTPException 404: profile not found.
    """
    if not persistence.delete_price_profile(profile_id):
        raise HTTPException(status_code=404, detail=f"Price profile id={profile_id} not found")
    return {"ok": True, "id": profile_id}


# ---------------------------------------------------------------------------
# Phase 6 — solar profile list for the Wizard step 1 ("Luogo")
# ---------------------------------------------------------------------------


@router.get("/profiles/solar", response_model=list[SolarProfileResponse])
def list_solar_profiles(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[SolarProfileResponse]:
    """
    List all solar irradiance profiles stored in the database.

    This endpoint feeds the **Step 1 — Luogo di installazione** dropdown in the
    Scenario Wizard.  Each profile contains the monthly solar production data
    and optimal panel orientation for a specific geographic location, enabling
    the wizard to:

    - Populate the location selector with human-readable names.
    - Render a read-only weather preview (avg daily production, p_sunny,
      weather persistence) once the user picks a location.
    - Pre-fill the tilt/azimuth fields in Step 2 with the optimal values for
      the chosen site.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of :class:`SolarProfileResponse` objects, sorted by name
        (alphabetical ascending).  Contains all seeded locations (Pavullo,
        Milano, Roma, Napoli, Palermo and any user-created entries).

    Example:
        ```
        GET /api/profiles/solar
        →
        [
          {"id": 1, "name": "Milano", "location_name": "Milano, Lombardia, Italy",
           "latitude": 45.46, "longitude": 9.19, "optimal_tilt_degrees": 35.0,
           "avg_daily_kwh_per_kwp": [1.3, 2.1, …], "p_sunny": [0.38, 0.42, …], …},
          …
        ]
        ```

    Notes:
        - Profiles are read-only via this endpoint; creation/update is handled
          via CLI seeding or direct DB access.
        - ``weather_persistence`` is ``None`` for profiles seeded before Phase 1;
          the frontend should treat ``None`` as 0.0 (iid weather, no persistence).
    """
    records = persistence.list_solar_profiles()
    # Sort alphabetically by name for a deterministic dropdown order.
    records_sorted = sorted(records, key=lambda r: r.name)
    return [SolarProfileResponse.model_validate(r) for r in records_sorted]


@router.put("/profiles/solar/{profile_id}", response_model=SolarProfileResponse)
def update_solar_profile(
    profile_id: int,
    payload: SolarProfileUpdate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> SolarProfileResponse:
    """
    Update a solar profile by primary key (allows rename + metadata edits).

    All fields in :class:`SolarProfileUpdate` are optional — only the keys
    explicitly provided are written back to the record. This lets the UI
    rename a profile or correct ``location_name`` / ``notes`` without
    resubmitting the PVGIS-derived monthly arrays.

    Args:
        profile_id: Primary-key ID of the profile to update.
        payload: Partial dict of new field values.

    Raises:
        HTTPException 404: profile not found.
        HTTPException 409: new ``name`` already used by another profile.
    """
    data = payload.model_dump(exclude_unset=True)
    try:
        record = persistence.update_solar_profile(profile_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Solar profile id={profile_id} not found",
        )
    return SolarProfileResponse.model_validate(record)


@router.delete("/profiles/solar/{profile_id}")
def delete_solar_profile(
    profile_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete a solar profile by primary key.

    Scenarios that reference this profile by ID will fail to hydrate the
    location on next execution.

    Args:
        profile_id: Primary-key ID of the profile to delete.

    Returns:
        JSON ``{"ok": true, "id": <profile_id>}`` on success.

    Raises:
        HTTPException 404: profile not found.
    """
    if not persistence.delete_solar_profile(profile_id):
        raise HTTPException(
            status_code=404,
            detail=f"Solar profile id={profile_id} not found",
        )
    return {"ok": True, "id": profile_id}


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


# ─────────────────────────────────────────────────────────────────────
# Phase 11+ — Excel templates and uploaders for inline load profiles.
# The wizard offers, for each non-ARERA profile kind, a "download
# template" and an "import Excel" action so users can fill the values
# in a spreadsheet rather than typing them cell by cell.
# ─────────────────────────────────────────────────────────────────────


@router.get("/load-profiles/template/{kind}.xlsx")
def download_load_profile_template(kind: str) -> StreamingResponse:
    """
    Stream a blank Excel template for an inline load profile.

    The user downloads the template, fills the cells, then uploads it
    via :func:`upload_load_profile_xlsx`.

    Args:
        kind: Profile shape — one of ``monthly_avg``, ``monthly_24h``,
            ``weekly``. ``arera`` is intentionally excluded because its
            table is fixed by Italian regulation.

    Raises:
        HTTPException 404: ``kind`` is not one of the supported values.
    """
    if kind not in _SUPPORTED_LOAD_KINDS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown load profile kind '{kind}'. "
                f"Supported: {sorted(_SUPPORTED_LOAD_KINDS)}."
            ),
        )
    buffer = io.BytesIO()
    build_template_xlsx(kind, buffer)
    buffer.seek(0)
    filename = f"load_profile_{kind}_template.xlsx"
    return StreamingResponse(
        buffer,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/load-profiles/parse-xlsx/{kind}")
async def upload_load_profile_xlsx(
    kind: str, file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Parse an Excel workbook into a JSON-compatible load profile payload.

    Args:
        kind: Expected profile shape (``monthly_avg``, ``monthly_24h``,
            ``weekly``). Determines how the sheet is read.
        file: Multipart upload of an .xlsx workbook produced from the
            matching template (or any compatible layout).

    Returns:
        A JSON-friendly dict ready to be merged into the scenario
        ``load_profile`` block. Shapes:

        - ``monthly_avg`` → ``{"monthly_avg_w": [12 floats]}``
        - ``monthly_24h`` → ``{"monthly_24h_w": [[24 floats] × 12]}``
        - ``weekly``      → ``{"monthly_w": [...12], "weekly_pattern_w": [[24] × 7]}``

    Raises:
        HTTPException 404: ``kind`` is not supported.
        HTTPException 422: Workbook cannot be parsed (malformed cell,
            wrong dimensions, missing sheet).
    """
    if kind not in _SUPPORTED_LOAD_KINDS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown load profile kind '{kind}'. "
                f"Supported: {sorted(_SUPPORTED_LOAD_KINDS)}."
            ),
        )
    content = await file.read()
    try:
        return parse_load_profile_xlsx(kind, io.BytesIO(content))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
