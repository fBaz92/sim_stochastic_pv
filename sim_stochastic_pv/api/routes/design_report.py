"""
Technical-report endpoint for detailed plant designs.

``GET /api/designs/{id}/report.pdf`` re-runs the electrical-design
engine from the inputs saved in the design's ``designer`` block (so the
report always reflects the engine's current formulas), optionally adds
the hourly Monte Carlo production preview when the design is anchored to
a site with a solar profile, and streams the WeasyPrint-rendered PDF.
"""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...output.exporters import build_design_report_pdf
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


router = APIRouter(prefix="/api/designs", tags=["designs"])


def _panel_from_dict(data: Mapping[str, Any]) -> PanelElectricalSpecs:
    """Rebuild the panel specs dataclass from the saved designer block."""
    return PanelElectricalSpecs(
        power_w=data.get("power_w"),
        v_oc_stc_v=data.get("v_oc_stc_v"),
        v_mpp_stc_v=data.get("v_mpp_stc_v"),
        i_sc_stc_a=data.get("i_sc_stc_a"),
        i_mpp_stc_a=data.get("i_mpp_stc_a"),
        n_cells_series=data.get("n_cells_series"),
        beta_voc_pct_per_c=data.get("beta_voc_pct_per_c"),
        gamma_pmax_pct_per_c=data.get("gamma_pmax_pct_per_c"),
        noct_c=data.get("noct_c"),
        alpha_isc_pct_per_c=data.get("alpha_isc_pct_per_c"),
        v_system_max_v=data.get("v_system_max_v"),
        max_series_fuse_a=data.get("max_series_fuse_a"),
    )


def _inverter_from_dict(data: Mapping[str, Any]) -> InverterElectricalSpecs:
    """Rebuild the inverter specs dataclass from the saved designer block."""
    return InverterElectricalSpecs(
        v_dc_min_v=data.get("v_dc_min_v"),
        v_dc_max_v=data.get("v_dc_max_v"),
        v_mppt_min_v=data.get("v_mppt_min_v"),
        v_mppt_max_v=data.get("v_mppt_max_v"),
        n_mppt_trackers=int(data.get("n_mppt_trackers") or 1),
        i_dc_max_per_mppt_a=data.get("i_dc_max_per_mppt_a"),
        i_sc_max_per_mppt_a=data.get("i_sc_max_per_mppt_a"),
        max_strings_per_mppt=data.get("max_strings_per_mppt"),
        v_mppt_full_load_min_v=data.get("v_mppt_full_load_min_v"),
        v_mppt_full_load_max_v=data.get("v_mppt_full_load_max_v"),
        p_ac_nom_kw=data.get("p_ac_nom_kw"),
        efficiency_max=data.get("efficiency_max"),
    )


@router.get("/{design_id}/report.pdf")
def design_report_pdf(
    design_id: int,
    with_production: bool = Query(
        True,
        description=(
            "Include the hourly MC production preview (requires the design "
            "to be anchored to a site with a solar profile)."
        ),
    ),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> StreamingResponse:
    """
    Render the technical report ("relazione tecnica") of a detailed design.

    The engine is re-run from the saved designer inputs (components, site
    corners, requirements, cable run), so the report carries fresh,
    internally consistent numbers; the cable table uses the current DC
    cable catalogue (prices + Iz ratings).

    Raises:
        HTTPException 404: design not found.
        HTTPException 422: the design has no ``designer`` block (an
            essential offer has nothing to size: the report is a
            detailed-design artifact).
    """
    design = persistence.designs.get_design_by_id(design_id)
    if design is None:
        raise HTTPException(
            status_code=404, detail=f"Plant design {design_id} not found"
        )
    designer = (design.data or {}).get("designer")
    if not designer or not designer.get("panel") or not designer.get("inverter"):
        raise HTTPException(
            status_code=422,
            detail=(
                "La relazione tecnica richiede un progetto di dettaglio "
                "salvato dalla pagina Progettazione (blocco 'designer' "
                "assente su questo impianto)."
            ),
        )

    panel_data = designer["panel"]
    inverter_data = designer["inverter"]
    site_data = designer.get("site") or {}
    req_data = dict(designer.get("requirements") or {})
    req_data.setdefault("n_panels_per_string", designer.get("n_panels_per_string", 1))
    req_data.setdefault(
        "p_ac_required_kw", inverter_data.get("p_ac_nom_kw") or 1.0
    )

    cable_length = float(designer.get("cable_length_one_way_m") or 30.0)
    cable_temp = float(designer.get("cable_operating_temperature_c") or 70.0)

    # Cable comparison from the live catalogue (prices + Iz), like the UI.
    catalogue = persistence.hardware.list_cables()
    cable_kwargs: dict[str, Any] = {
        "length_one_way_m": cable_length,
        "operating_temperature_c": cable_temp,
    }
    if catalogue:
        cable_kwargs["sections_mm2"] = tuple(c.section_mm2 for c in catalogue)
        cable_kwargs["price_eur_per_m"] = tuple(
            c.price_eur_per_m if c.price_eur_per_m is not None else 0.0
            for c in catalogue
        )
        cable_kwargs["iz_a"] = tuple(
            c.iz_a if c.iz_a is not None else 0.0 for c in catalogue
        )

    try:
        evaluation = evaluate_design(
            panel=_panel_from_dict(panel_data),
            inverter=_inverter_from_dict(inverter_data),
            site=DesignSite(
                t_min_c=float(site_data.get("t_min_c", -10.0)),
                t_max_c=float(site_data.get("t_max_c", 40.0)),
                delta_t_cell_c=float(site_data.get("delta_t_cell_c", 30.0)),
            ),
            requirements=DesignRequirements(
                p_ac_required_kw=float(req_data["p_ac_required_kw"]),
                target_dc_ac_ratio=float(req_data.get("target_dc_ac_ratio", 1.2)),
                n_panels_per_string=int(req_data["n_panels_per_string"]),
                safety_factor_isc=float(req_data.get("safety_factor_isc", 1.25)),
                max_cable_loss_fraction=float(
                    req_data.get("max_cable_loss_fraction", 0.01)
                ),
                fuse_factor_min=float(req_data.get("fuse_factor_min", 1.5)),
                fuse_factor_max=float(req_data.get("fuse_factor_max", 2.4)),
            ),
            cable=CableParams(**cable_kwargs),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Optional production preview: needs a site with a solar profile.
    production = None
    location_name = None
    if design.location_id is not None:
        location = persistence.locations.get_location_by_id(design.location_id)
        location_name = location.name if location else None
        if with_production and location is not None:
            solar_profiles, climate_profiles = persistence.locations.linked_profiles(
                design.location_id
            )
            if solar_profiles:
                panel_specs = _panel_from_dict(panel_data)
                inverter_specs = _inverter_from_dict(inverter_data)
                n_per_string = int(req_data["n_panels_per_string"])
                n_strings = int(designer.get("n_strings") or 1)
                pv_kwp = (
                    n_per_string * n_strings * float(panel_data["power_w"]) / 1000.0
                )
                solar_model = _solar_model_from_db_record(
                    solar_profiles[0],
                    pv_kwp=pv_kwp,
                    degradation_per_year=0.0,
                    panel_tilt_degrees=None,
                    panel_azimuth_degrees=None,
                )
                thermal_model = None
                electrical_model = None
                if climate_profiles and not missing_panel_fields(panel_specs):
                    thermal_model = persistence.load_thermal_model(
                        climate_profiles[0].id
                    )
                    electrical_model = ElectricalModel(
                        panel=panel_specs,
                        inverter=inverter_specs,
                        strings=[
                            PvString(
                                n_panels=n_per_string,
                                mppt_id=i % max(1, inverter_specs.n_mppt_trackers),
                            )
                            for i in range(n_strings)
                        ],
                        n_years=1,
                    )
                cable_spec = None
                chosen = designer.get("cable_section_mm2")
                if chosen:
                    rho = COPPER_RESISTIVITY_20C * (
                        1.0 + COPPER_TEMP_COEFFICIENT * (cable_temp - 20.0)
                    )
                    cable_spec = CableLossSpec(
                        resistance_per_string_ohm=2.0 * cable_length * rho / float(chosen),
                        n_strings=n_strings,
                        v_mp_string_stc_v=n_per_string
                        * float(panel_data["v_mpp_stc_v"]),
                    )
                production = simulate_production_preview(
                    solar_model=solar_model,
                    p_ac_max_kw=float(inverter_data["p_ac_nom_kw"]),
                    inverter_efficiency=float(
                        inverter_data.get("efficiency_max") or 1.0
                    ),
                    n_paths=20,
                    seed=42,
                    thermal_model=thermal_model,
                    electrical_model=electrical_model,
                    cable=cable_spec,
                )

    stream = build_design_report_pdf(
        name=design.name,
        location_name=location_name,
        panel=panel_data,
        inverter=inverter_data,
        site=site_data,
        requirements=req_data,
        evaluation=evaluation,
        chosen_section_mm2=designer.get("cable_section_mm2"),
        cable_length_m=cable_length,
        cable_temp_c=cable_temp,
        production=production,
    )
    safe_name = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in design.name
    )
    return StreamingResponse(
        stream,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="relazione_{safe_name}.pdf"'
        },
    )
