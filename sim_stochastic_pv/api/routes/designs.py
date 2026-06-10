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

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from ...persistence import PersistenceService
from ...scenario_builder import _solar_model_from_db_record
from ...simulation.electrical import (
    ElectricalModel,
    InverterElectricalSpecs,
    PanelElectricalSpecs,
    PvString,
    missing_panel_fields,
)
from ...simulation.electrical_design import (
    CableLossSpec,
    CableParams,
    DesignRequirements,
    DesignSite,
    evaluate_design,
    simulate_production_preview,
)
from ...simulation.electrical_design.cables import (
    COPPER_RESISTIVITY_20C,
    COPPER_TEMP_COEFFICIENT,
)
from .. import dependencies
from ..schemas.designs import (
    DesignEvaluateRequest,
    DesignEvaluateResponse,
    PlantDesignCreate,
    PlantDesignResponse,
    PlantDesignUpdate,
    ProductionPreviewRequest,
    ProductionPreviewResponse,
)


router = APIRouter(prefix="/api/designs", tags=["designs"])


def _panel_specs(payload) -> PanelElectricalSpecs:
    """Map the request panel block onto the engine dataclass."""
    return PanelElectricalSpecs(
        power_w=payload.power_w,
        v_oc_stc_v=payload.v_oc_stc_v,
        v_mpp_stc_v=payload.v_mpp_stc_v,
        i_sc_stc_a=payload.i_sc_stc_a,
        i_mpp_stc_a=payload.i_mpp_stc_a,
        n_cells_series=payload.n_cells_series,
        beta_voc_pct_per_c=payload.beta_voc_pct_per_c,
        gamma_pmax_pct_per_c=payload.gamma_pmax_pct_per_c,
        noct_c=payload.noct_c,
        alpha_isc_pct_per_c=payload.alpha_isc_pct_per_c,
        v_system_max_v=payload.v_system_max_v,
        max_series_fuse_a=payload.max_series_fuse_a,
    )


def _inverter_specs(payload) -> InverterElectricalSpecs:
    """Map the request inverter block onto the engine dataclass."""
    return InverterElectricalSpecs(
        v_dc_min_v=payload.v_dc_min_v,
        v_dc_max_v=payload.v_dc_max_v,
        v_mppt_min_v=payload.v_mppt_min_v,
        v_mppt_max_v=payload.v_mppt_max_v,
        n_mppt_trackers=payload.n_mppt_trackers,
        i_dc_max_per_mppt_a=payload.i_dc_max_per_mppt_a,
        i_sc_max_per_mppt_a=payload.i_sc_max_per_mppt_a,
        max_strings_per_mppt=payload.max_strings_per_mppt,
        v_mppt_full_load_min_v=payload.v_mppt_full_load_min_v,
        v_mppt_full_load_max_v=payload.v_mppt_full_load_max_v,
        p_ac_nom_kw=payload.p_ac_nom_kw,
        efficiency_max=payload.efficiency_max,
    )


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


@router.post("/evaluate", response_model=DesignEvaluateResponse)
def evaluate(payload: DesignEvaluateRequest) -> DesignEvaluateResponse:
    """
    Stateless evaluation of a full electrical design.

    The Progettazione page posts the complete design on every input
    change and repaints all derived cells (temperature-corrected values,
    admissible string range, voltage/current checks with margins, plant
    sizing, temperature margins, fuse sizing, cable table) from the
    response. Nothing is persisted.

    Raises:
        HTTPException 422: a required datasheet field is missing (the
            message names it) or the inputs violate the schema.
    """
    cable = None
    if payload.cable is not None:
        kwargs: dict = {
            "length_one_way_m": payload.cable.length_one_way_m,
            "operating_temperature_c": payload.cable.operating_temperature_c,
        }
        if payload.cable.sections:
            kwargs["sections_mm2"] = tuple(
                s.section_mm2 for s in payload.cable.sections
            )
            if any(s.price_eur_per_m is not None for s in payload.cable.sections):
                kwargs["price_eur_per_m"] = tuple(
                    s.price_eur_per_m if s.price_eur_per_m is not None else 0.0
                    for s in payload.cable.sections
                )
            if any(s.iz_a is not None for s in payload.cable.sections):
                kwargs["iz_a"] = tuple(
                    s.iz_a if s.iz_a is not None else 0.0
                    for s in payload.cable.sections
                )
        cable = CableParams(**kwargs)

    try:
        evaluation = evaluate_design(
            panel=_panel_specs(payload.panel),
            inverter=_inverter_specs(payload.inverter),
            site=DesignSite(
                t_min_c=payload.site.t_min_c,
                t_max_c=payload.site.t_max_c,
                delta_t_cell_c=payload.site.delta_t_cell_c,
            ),
            requirements=DesignRequirements(
                p_ac_required_kw=payload.requirements.p_ac_required_kw,
                target_dc_ac_ratio=payload.requirements.target_dc_ac_ratio,
                n_panels_per_string=payload.requirements.n_panels_per_string,
                safety_factor_isc=payload.requirements.safety_factor_isc,
                max_cable_loss_fraction=payload.requirements.max_cable_loss_fraction,
                fuse_factor_min=payload.requirements.fuse_factor_min,
                fuse_factor_max=payload.requirements.fuse_factor_max,
            ),
            cable=cable,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DesignEvaluateResponse.model_validate(asdict(evaluation))


@router.post("/production-preview", response_model=ProductionPreviewResponse)
def production_preview(
    payload: ProductionPreviewRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> ProductionPreviewResponse:
    """
    Hourly Monte Carlo production preview for a designed plant.

    Runs one synthetic weather year per path on the chosen site profile,
    applies (in order) the optional MPPT-window/temperature derating,
    the hourly ohmic cable loss and the inverter AC cap, and returns the
    annual energy with bands plus the loss split — the dynamic check a
    static sizing sheet cannot do.

    Raises:
        HTTPException 404: solar (or climate) profile not found.
        HTTPException 422: ``use_electrical_model`` without a climate
            profile, or with an incomplete panel datasheet.
    """
    profile = persistence.solar.get_solar_profile_by_id(payload.solar_profile_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Solar profile {payload.solar_profile_id} not found",
        )

    panel = _panel_specs(payload.panel)
    inverter = _inverter_specs(payload.inverter)
    total_panels = payload.n_panels_per_string * payload.n_strings
    pv_kwp = total_panels * float(payload.panel.power_w) / 1000.0

    solar_model = _solar_model_from_db_record(
        profile,
        pv_kwp=pv_kwp,
        degradation_per_year=0.0,
        panel_tilt_degrees=None,
        panel_azimuth_degrees=None,
    )

    thermal_model = None
    electrical_model = None
    if payload.use_electrical_model:
        if payload.climate_profile_id is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "use_electrical_model richiede un climate_profile_id "
                    "(temperature orarie del sito)."
                ),
            )
        missing = missing_panel_fields(panel)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    "use_electrical_model richiede il datasheet completo del "
                    f"modulo: campi mancanti {', '.join(missing)}."
                ),
            )
        thermal_model = persistence.load_thermal_model(payload.climate_profile_id)
        if thermal_model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Climate profile {payload.climate_profile_id} not found",
            )
        electrical_model = ElectricalModel(
            panel=panel,
            inverter=inverter,
            strings=[
                PvString(
                    n_panels=payload.n_panels_per_string,
                    mppt_id=i % max(1, payload.inverter.n_mppt_trackers),
                )
                for i in range(payload.n_strings)
            ],
            n_years=1,
        )

    cable_spec = None
    if payload.cable is not None:
        rho = COPPER_RESISTIVITY_20C * (
            1.0
            + COPPER_TEMP_COEFFICIENT
            * (payload.cable.operating_temperature_c - 20.0)
        )
        resistance = (
            2.0 * payload.cable.length_one_way_m * rho / payload.cable.section_mm2
        )
        cable_spec = CableLossSpec(
            resistance_per_string_ohm=resistance,
            n_strings=payload.n_strings,
            v_mp_string_stc_v=payload.n_panels_per_string
            * float(payload.panel.v_mpp_stc_v),
        )

    result = simulate_production_preview(
        solar_model=solar_model,
        p_ac_max_kw=float(payload.inverter.p_ac_nom_kw),
        inverter_efficiency=float(payload.inverter.efficiency_max or 1.0),
        n_paths=payload.n_paths,
        seed=payload.seed,
        thermal_model=thermal_model,
        electrical_model=electrical_model,
        cable=cable_spec,
    )
    return ProductionPreviewResponse.model_validate(asdict(result))


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
