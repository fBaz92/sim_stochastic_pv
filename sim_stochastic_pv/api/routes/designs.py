"""
Plant-design API endpoints (``/api/designs``).

A plant design ("Impianto") describes one specific PV system. The
``essential`` level captures a received commercial offer (nameplate data +
cost + incentive) and powers the "Analizza un'offerta" flow: the frontend
saves the design here, then submits a normal analysis whose scenario
references it via ``plant_design_id`` — the hydration step
(:func:`~sim_stochastic_pv.persistence.hydration.apply_plant_design`)
expands the design into the simulator configuration.

Endpoints:

- ``GET    /api/designs`` — list designs.
- ``POST   /api/designs`` — create/upsert by name.
- ``PUT    /api/designs/{id}`` — partial update (rename, edit data).
- ``DELETE /api/designs/{id}`` — delete (past runs keep their snapshot).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...persistence import PersistenceService
from .. import dependencies
from ..schemas.designs import (
    PlantDesignCreate,
    PlantDesignResponse,
    PlantDesignUpdate,
)


router = APIRouter(prefix="/api/designs", tags=["designs"])


@router.get("", response_model=list[PlantDesignResponse])
def list_designs(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[PlantDesignResponse]:
    """
    List all plant designs ordered by name.

    Returns:
        All saved designs (both levels). Empty list when none exist.
    """
    records = persistence.designs.list_designs()
    return [PlantDesignResponse.model_validate(r) for r in records]


@router.post("", response_model=PlantDesignResponse)
def upsert_design(
    payload: PlantDesignCreate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> PlantDesignResponse:
    """
    Create a plant design, or update it when the name already exists.

    Upsert-by-name keeps the offer flow friction-free: re-submitting the
    same offer with corrected numbers refreshes the stored design instead
    of failing — past runs are unaffected (their config is snapshotted).

    Args:
        payload: Validated design payload (``essential`` level).

    Returns:
        The saved record.

    Raises:
        HTTPException 422: payload validation errors (via Pydantic).
    """
    record = persistence.designs.upsert_design({
        "name": payload.name,
        "design_level": payload.design_level,
        "description": payload.description,
        "data": payload.data.model_dump(exclude_none=True),
        "location_id": payload.location_id,
        "inverter_id": payload.inverter_id,
        "panel_id": payload.panel_id,
        "battery_id": payload.battery_id,
    })
    return PlantDesignResponse.model_validate(record)


@router.put("/{design_id}", response_model=PlantDesignResponse)
def update_design(
    design_id: int,
    payload: PlantDesignUpdate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> PlantDesignResponse:
    """
    Update a design by primary key (partial — allows rename).

    Raises:
        HTTPException 404: design not found.
        HTTPException 409: new ``name`` already used by another design.
    """
    data = payload.model_dump(exclude_unset=True)
    if "data" in data and data["data"] is not None:
        data["data"] = payload.data.model_dump(exclude_none=True)
    try:
        record = persistence.designs.update_design(design_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Plant design {design_id} not found"
        )
    return PlantDesignResponse.model_validate(record)


@router.delete("/{design_id}", status_code=204)
def delete_design(
    design_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """
    Delete a design. Past runs keep their frozen config snapshot.

    Raises:
        HTTPException 404: design not found.
    """
    if not persistence.designs.delete_design(design_id):
        raise HTTPException(
            status_code=404, detail=f"Plant design {design_id} not found"
        )
