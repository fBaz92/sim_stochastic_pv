"""
Hardware catalog management API endpoints.

This module provides CRUD operations for hardware components in the catalog:
- Inverters: DC to AC power conversion equipment
- Solar Panels: Photovoltaic modules
- Batteries: Energy storage systems

All hardware entries support upsert behavior (create or update by name).
Hardware specifications are stored in a flexible JSON `specs` field to
accommodate varying manufacturer data without requiring schema migrations.

The hardware catalog enables the database-driven workflow where scenarios
reference hardware by ID rather than embedding values, ensuring that hardware
specification updates propagate to all scenarios using that hardware.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...persistence import PersistenceService
from .. import dependencies
from ..schemas import hardware as hw_schemas

router = APIRouter(prefix="/api", tags=["hardware"])


@router.get("/inverters", response_model=list[hw_schemas.InverterResponse])
def list_inverters(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[hw_schemas.InverterResponse]:
    """
    List all inverters in the hardware catalog.

    Returns all inverter records with their full specifications. Inverters
    are sorted by creation date (newest first). No pagination is currently
    implemented - returns all records.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of InverterResponse objects containing all inverter data including
        specifications, pricing, and metadata.

    Example:
        ```python
        # GET /api/inverters
        [
            {
                "id": 1,
                "name": "Huawei SUN2000-5KTL",
                "p_ac_max_kw": 5.0,
                "specs": {"efficiency": 0.98, "mppt_count": 2},
                "price_eur": 1200.0,
                "created_at": "2025-01-15T10:30:00Z"
            },
            ...
        ]
        ```

    Notes:
        - All inverters in catalog are returned (no filtering yet implemented)
        - Inverter specs include manufacturer-specific fields
        - p_ac_max_kw is the key parameter for sizing calculations
    """
    return persistence.list_inverters()


@router.post("/inverters", response_model=hw_schemas.InverterResponse)
def create_inverter(
    payload: hw_schemas.InverterCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.InverterResponse:
    """
    Create or update an inverter in the hardware catalog.

    Implements upsert behavior: creates new inverter if name doesn't exist,
    otherwise updates the existing record. This allows easy catalog updates
    without checking for existence first.

    When updating an existing inverter, all scenarios referencing this inverter
    by ID will automatically use the new specifications on next execution.

    Args:
        payload: Inverter data including name, specifications, and pricing.
            Required fields: name, p_ac_max_kw
            Optional fields: specs (JSON), price_eur, notes
        persistence: Database persistence service (dependency injected).

    Returns:
        InverterResponse with the created or updated inverter data including
        the database ID and creation timestamp.

    Example:
        ```python
        # POST /api/inverters
        {
            "name": "Huawei SUN2000-6KTL",
            "p_ac_max_kw": 6.0,
            "specs": {
                "efficiency": 0.982,
                "mppt_count": 2,
                "max_dc_voltage": 1000
            },
            "price_eur": 1500.0,
            "notes": "High efficiency model with dual MPPT"
        }

        # Response
        {
            "id": 15,
            "name": "Huawei SUN2000-6KTL",
            "p_ac_max_kw": 6.0,
            ...
        }
        ```

    Notes:
        - Upsert is based on name matching (case-sensitive)
        - Updating an inverter propagates to all saved scenarios using that ID
        - specs field allows flexible storage of manufacturer-specific data
    """
    return persistence.upsert_inverter(payload.model_dump())


@router.get("/panels", response_model=list[hw_schemas.PanelResponse])
def list_panels(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[hw_schemas.PanelResponse]:
    """
    List all solar panels in the hardware catalog.

    Returns all panel records with their full specifications. Panels are
    sorted by creation date (newest first). No pagination is currently
    implemented - returns all records.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of PanelResponse objects containing all panel data including
        power rating, efficiency, dimensions, and pricing.

    Example:
        ```python
        # GET /api/panels
        [
            {
                "id": 1,
                "name": "Canadian Solar CS3W-410P",
                "power_w": 410,
                "specs": {
                    "efficiency": 0.206,
                    "width_m": 1.048,
                    "height_m": 2.094,
                    "weight_kg": 22.4
                },
                "price_eur": 150.0,
                "created_at": "2025-01-15T10:30:00Z"
            },
            ...
        ]
        ```

    Notes:
        - power_w is the nominal power output under standard test conditions
        - Panel efficiency affects how much roof space is needed
        - Dimensions are critical for installation planning
    """
    return persistence.list_panels()


@router.post("/panels", response_model=hw_schemas.PanelResponse)
def create_panel(
    payload: hw_schemas.PanelCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.PanelResponse:
    """
    Create or update a solar panel in the hardware catalog.

    Implements upsert behavior: creates new panel if name doesn't exist,
    otherwise updates the existing record. Panel updates automatically
    propagate to all scenarios using this panel by ID.

    Args:
        payload: Panel data including name, power rating, and specifications.
            Required fields: name, power_w
            Optional fields: specs (JSON), price_eur, notes
        persistence: Database persistence service (dependency injected).

    Returns:
        PanelResponse with the created or updated panel data including
        database ID and timestamp.

    Example:
        ```python
        # POST /api/panels
        {
            "name": "Jinko Tiger Neo 420W",
            "power_w": 420,
            "specs": {
                "efficiency": 0.213,
                "width_m": 1.134,
                "height_m": 1.722,
                "technology": "N-type TOPCon"
            },
            "price_eur": 165.0,
            "notes": "High efficiency N-type panel"
        }
        ```

    Notes:
        - Upsert is based on name matching
        - power_w should match manufacturer's STC rating
        - Updating propagates to all scenarios using this panel ID
    """
    return persistence.upsert_panel(payload.model_dump())


@router.get("/batteries", response_model=list[hw_schemas.BatteryResponse])
def list_batteries(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[hw_schemas.BatteryResponse]:
    """
    List all batteries in the hardware catalog.

    Returns all battery records with their full specifications. Batteries are
    sorted by creation date (newest first). No pagination is currently
    implemented - returns all records.

    Args:
        persistence: Database persistence service (dependency injected).

    Returns:
        List of BatteryResponse objects containing all battery data including
        capacity, cycle life, efficiency, and pricing.

    Example:
        ```python
        # GET /api/batteries
        [
            {
                "id": 1,
                "name": "Tesla Powerwall 2",
                "capacity_kwh": 13.5,
                "specs": {
                    "cycles_life": 5000,
                    "roundtrip_efficiency": 0.90,
                    "max_power_kw": 5.0,
                    "chemistry": "Li-ion NMC"
                },
                "price_eur": 8500.0,
                "created_at": "2025-01-15T10:30:00Z"
            },
            ...
        ]
        ```

    Notes:
        - capacity_kwh is usable energy capacity, not total
        - cycles_life affects battery degradation modeling
        - roundtrip_efficiency impacts energy arbitrage economics
    """
    return persistence.list_batteries()


@router.post("/batteries", response_model=hw_schemas.BatteryResponse)
def create_battery(
    payload: hw_schemas.BatteryCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.BatteryResponse:
    """
    Create or update a battery in the hardware catalog.

    Implements upsert behavior: creates new battery if name doesn't exist,
    otherwise updates the existing record. Battery specification updates
    automatically propagate to all scenarios using this battery by ID.

    Args:
        payload: Battery data including name, capacity, and specifications.
            Required fields: name, capacity_kwh
            Optional fields: specs (JSON with cycles_life, efficiency), price_eur, notes
        persistence: Database persistence service (dependency injected).

    Returns:
        BatteryResponse with the created or updated battery data including
        database ID and timestamp.

    Example:
        ```python
        # POST /api/batteries
        {
            "name": "BYD Battery-Box Premium HVS 10.2",
            "capacity_kwh": 10.24,
            "specs": {
                "cycles_life": 6000,
                "roundtrip_efficiency": 0.92,
                "max_power_kw": 5.12,
                "voltage_range": "102V-614V",
                "chemistry": "LiFePO4"
            },
            "price_eur": 6800.0,
            "notes": "Modular LFP battery with long cycle life"
        }
        ```

    Notes:
        - Upsert is based on name matching
        - capacity_kwh should be usable capacity (not nominal)
        - cycles_life in specs is used for degradation modeling
        - Updating propagates to all scenarios using this battery ID
    """
    return persistence.upsert_battery(payload.model_dump())
