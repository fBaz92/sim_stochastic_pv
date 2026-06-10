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

from fastapi import APIRouter, Depends, HTTPException

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


@router.put("/inverters/{inverter_id}", response_model=hw_schemas.InverterResponse)
def update_inverter(
    inverter_id: int,
    payload: hw_schemas.InverterCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.InverterResponse:
    """
    Update an inverter in the hardware catalog by primary key.

    Differs from POST in that the lookup is by ``inverter_id`` instead
    of by name, so callers can edit *any* field — including ``name`` —
    without producing a second record.

    Args:
        inverter_id: Primary-key ID of the inverter to update.
        payload: New inverter data. All fields are written; specs are
            normalized through the create schema so price/specs blob
            stays consistent.

    Raises:
        HTTPException 404: inverter not found.
        HTTPException 409: new ``name`` already used by another inverter.
    """
    try:
        record = persistence.update_inverter(inverter_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Inverter id={inverter_id} not found")
    return record


@router.delete("/inverters/{inverter_id}")
def delete_inverter(
    inverter_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete an inverter from the hardware catalog by ID.

    Permanently removes the inverter record. Scenarios that reference this
    inverter by ID will fail to hydrate after deletion — remove those
    scenarios first if needed.

    Args:
        inverter_id: Database primary key of the inverter to delete.
        persistence: Database persistence service (dependency injected).

    Returns:
        JSON ``{"ok": true, "id": <inverter_id>}`` on success.

    Raises:
        HTTPException 404: inverter not found.
    """
    if not persistence.delete_inverter(inverter_id):
        raise HTTPException(status_code=404, detail=f"Inverter id={inverter_id} not found")
    return {"ok": True, "id": inverter_id}


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


@router.put("/panels/{panel_id}", response_model=hw_schemas.PanelResponse)
def update_panel(
    panel_id: int,
    payload: hw_schemas.PanelCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.PanelResponse:
    """
    Update a panel in the hardware catalog by primary key (allows rename).

    Raises:
        HTTPException 404: panel not found.
        HTTPException 409: new ``name`` already used by another panel.
    """
    try:
        record = persistence.update_panel(panel_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Panel id={panel_id} not found")
    return record


@router.delete("/panels/{panel_id}")
def delete_panel(
    panel_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete a solar panel from the hardware catalog by ID.

    Args:
        panel_id: Database primary key of the panel to delete.
        persistence: Database persistence service (dependency injected).

    Returns:
        JSON ``{"ok": true, "id": <panel_id>}`` on success.

    Raises:
        HTTPException 404: panel not found.
    """
    if not persistence.delete_panel(panel_id):
        raise HTTPException(status_code=404, detail=f"Panel id={panel_id} not found")
    return {"ok": True, "id": panel_id}


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


@router.put("/batteries/{battery_id}", response_model=hw_schemas.BatteryResponse)
def update_battery(
    battery_id: int,
    payload: hw_schemas.BatteryCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.BatteryResponse:
    """
    Update a battery in the hardware catalog by primary key (allows rename).

    Raises:
        HTTPException 404: battery not found.
        HTTPException 409: new ``name`` already used by another battery.
    """
    try:
        record = persistence.update_battery(battery_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Battery id={battery_id} not found")
    return record


@router.delete("/batteries/{battery_id}")
def delete_battery(
    battery_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> dict:
    """
    Delete a battery from the hardware catalog by ID.

    Args:
        battery_id: Database primary key of the battery to delete.
        persistence: Database persistence service (dependency injected).

    Returns:
        JSON ``{"ok": true, "id": <battery_id>}`` on success.

    Raises:
        HTTPException 404: battery not found.
    """
    if not persistence.delete_battery(battery_id):
        raise HTTPException(status_code=404, detail=f"Battery id={battery_id} not found")
    return {"ok": True, "id": battery_id}


# ---------------------------------------------------------------------------
# DC cables (electrical-designer catalogue)
# ---------------------------------------------------------------------------


@router.get("/cables", response_model=list[hw_schemas.CableResponse])
def list_cables(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[hw_schemas.CableResponse]:
    """List the DC cable catalogue ordered by cross-section."""
    return [
        hw_schemas.CableResponse.model_validate(r)
        for r in persistence.hardware.list_cables()
    ]


@router.post("/cables", response_model=hw_schemas.CableResponse)
def upsert_cable(
    payload: hw_schemas.CableCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.CableResponse:
    """Create a cable, or update it when the name already exists."""
    record = persistence.hardware.upsert_cable(payload.model_dump())
    return hw_schemas.CableResponse.model_validate(record)


@router.put("/cables/{cable_id}", response_model=hw_schemas.CableResponse)
def update_cable(
    cable_id: int,
    payload: hw_schemas.CableCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.CableResponse:
    """
    Update a cable by primary key (allows rename).

    Raises:
        HTTPException 404: cable not found.
        HTTPException 409: new name already used by another cable.
    """
    try:
        record = persistence.hardware.update_cable(cable_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Cable {cable_id} not found")
    return hw_schemas.CableResponse.model_validate(record)


@router.delete("/cables/{cable_id}", status_code=204)
def delete_cable(
    cable_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """Delete a cable. Raises 404 when missing."""
    if not persistence.hardware.delete_cable(cable_id):
        raise HTTPException(status_code=404, detail=f"Cable {cable_id} not found")


# ---------------------------------------------------------------------------
# DC protections (electrical-designer catalogue)
# ---------------------------------------------------------------------------


@router.get("/protections", response_model=list[hw_schemas.ProtectionResponse])
def list_protections(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[hw_schemas.ProtectionResponse]:
    """List the DC protection catalogue ordered by kind and rating."""
    return [
        hw_schemas.ProtectionResponse.model_validate(r)
        for r in persistence.hardware.list_protections()
    ]


@router.post("/protections", response_model=hw_schemas.ProtectionResponse)
def upsert_protection(
    payload: hw_schemas.ProtectionCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.ProtectionResponse:
    """Create a protection, or update it when the name already exists."""
    record = persistence.hardware.upsert_protection(payload.model_dump())
    return hw_schemas.ProtectionResponse.model_validate(record)


@router.put("/protections/{protection_id}", response_model=hw_schemas.ProtectionResponse)
def update_protection(
    protection_id: int,
    payload: hw_schemas.ProtectionCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.ProtectionResponse:
    """
    Update a protection by primary key (allows rename).

    Raises:
        HTTPException 404: protection not found.
        HTTPException 409: new name already used by another protection.
    """
    try:
        record = persistence.hardware.update_protection(
            protection_id, payload.model_dump()
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Protection {protection_id} not found"
        )
    return hw_schemas.ProtectionResponse.model_validate(record)


@router.delete("/protections/{protection_id}", status_code=204)
def delete_protection(
    protection_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """Delete a protection. Raises 404 when missing."""
    if not persistence.hardware.delete_protection(protection_id):
        raise HTTPException(
            status_code=404, detail=f"Protection {protection_id} not found"
        )


@router.get("/panels/{panel_id}/curves", response_model=hw_schemas.PanelCurvesResponse)
def panel_curves(
    panel_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> hw_schemas.PanelCurvesResponse:
    """
    I-V / P-V curve families of a catalogue panel (product sheet).

    Fits the single-diode model on the panel's datasheet specs and
    returns the two standard families: irradiance sweep
    (200..1000 W/m2 at 25 degC cell) and cell-temperature sweep
    (-10..70 degC at full sun), each with the MPP highlighted.

    Raises:
        HTTPException 404: panel not found.
        HTTPException 422: the panel specs lack the electrical datasheet
            fields needed by the model (the message names them).
    """
    from ...simulation.panel_curves import compute_panel_curve_families

    panel = next(
        (p for p in persistence.hardware.list_panels() if p.id == panel_id), None
    )
    if panel is None:
        raise HTTPException(status_code=404, detail=f"Panel {panel_id} not found")

    specs = panel.specs or {}
    required = (
        "i_sc_stc_a", "v_oc_stc_v", "i_mpp_stc_a", "v_mpp_stc_v",
        "n_cells_series", "alpha_isc_pct_per_c", "beta_voc_pct_per_c",
    )
    missing = [k for k in required if specs.get(k) in (None, "")]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Il pannello '{panel.name}' non ha il datasheet elettrico "
                f"completo per le curve: campi mancanti {', '.join(missing)}."
            ),
        )

    try:
        irr, temp = compute_panel_curve_families(
            isc_stc=float(specs["i_sc_stc_a"]),
            voc_stc=float(specs["v_oc_stc_v"]),
            imp_stc=float(specs["i_mpp_stc_a"]),
            vmp_stc=float(specs["v_mpp_stc_v"]),
            n_cells_series=int(specs["n_cells_series"]),
            alpha_isc_pct_per_c=float(specs["alpha_isc_pct_per_c"]),
            beta_voc_pct_per_c=float(specs["beta_voc_pct_per_c"]),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    from dataclasses import asdict

    return hw_schemas.PanelCurvesResponse(
        panel_id=panel.id,
        name=panel.name,
        irradiance_family=[
            hw_schemas.PanelCurvePoint(**asdict(c)) for c in irr
        ],
        temperature_family=[
            hw_schemas.PanelCurvePoint(**asdict(c)) for c in temp
        ],
    )
