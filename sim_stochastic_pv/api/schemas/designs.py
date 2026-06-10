"""
Pydantic schemas for plant designs (``/api/designs``).

A plant design ("Impianto") is the first-class description of one PV
system. The ``essential`` level captures a received commercial offer in
five numbers (AC power, optional DC power, optional storage, turn-key
cost, optional incentive); the ``detailed`` level is produced by the
electrical designer and carries the full layout in the same ``data``
payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaxBonusBlock(BaseModel):
    """
    Tax-incentive block carried by a design.

    Mirrors the scenario's ``economic.tax_bonus`` shape so the resolver
    can copy it verbatim.

    Attributes:
        enabled: Whether the incentive applies.
        fraction_of_investment: Refunded fraction of the investment
            (0–1, e.g. 0.5 for the Italian 50% detrazione).
        duration_years: Number of years the refund is spread over.
    """

    enabled: bool = True
    fraction_of_investment: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    duration_years: Annotated[int, Field(ge=1, le=30)] = 10


class EssentialDesignData(BaseModel):
    """
    Nameplate payload shared by both design levels.

    For an ``essential`` design (a received offer) these are the only
    fields. A ``detailed`` design produced by the electrical designer
    carries the same nameplate summary (so the economic resolver treats
    both levels identically) plus extra blocks — string layout, cable
    pick, evaluation snapshot — which Pydantic passes through verbatim
    (``extra="allow"``).

    Attributes:
        p_ac_kw: Inverter AC nameplate power (kW), as stated by the offer.
        p_dc_kwp: DC peak power (kWp). ``None`` when the offer does not
            state it — the resolver assumes it equals ``p_ac_kw``.
        storage_kwh: Battery capacity (kWh). ``None``/0 = no storage.
        total_cost_eur: Turn-key cost of the offer (€).
        tax_bonus: Optional incentive block.
        notes: Free-text notes about the offer.
    """

    model_config = ConfigDict(extra="allow", json_schema_extra={
        "examples": [{
            "p_ac_kw": 6.0,
            "p_dc_kwp": 6.6,
            "storage_kwh": 10.0,
            "total_cost_eur": 14500.0,
            "tax_bonus": {
                "enabled": True,
                "fraction_of_investment": 0.5,
                "duration_years": 10,
            },
        }]
    })

    p_ac_kw: Annotated[float, Field(gt=0.0, le=1000.0)]
    p_dc_kwp: Annotated[float | None, Field(gt=0.0, le=2000.0)] = None
    storage_kwh: Annotated[float | None, Field(ge=0.0, le=1000.0)] = None
    total_cost_eur: Annotated[float, Field(gt=0.0)]
    tax_bonus: TaxBonusBlock | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _dc_not_absurdly_low(self) -> "EssentialDesignData":
        """A DC power below half the AC rating is almost surely a typo."""
        if self.p_dc_kwp is not None and self.p_dc_kwp < 0.5 * self.p_ac_kw:
            raise ValueError(
                f"p_dc_kwp={self.p_dc_kwp} is less than half of "
                f"p_ac_kw={self.p_ac_kw}: probable typo in the offer data"
            )
        return self


class PlantDesignCreate(BaseModel):
    """
    Create/upsert payload for a plant design.

    Attributes:
        name: Unique design identifier (upsert key).
        design_level: ``"essential"`` (received offer) or ``"detailed"``
            (saved from the electrical designer; same nameplate fields
            plus pass-through extra blocks).
        description: Optional free text.
        data: Validated :class:`EssentialDesignData` payload.
        location_id: Optional installation-site FK; when set, scenarios
            built from this design inherit the site's solar (and climate)
            profile automatically.
        inverter_id: Optional catalogue references, when the offer names
            the hardware.
        panel_id: See ``inverter_id``.
        battery_id: See ``inverter_id``.
    """

    name: Annotated[str, Field(min_length=1, max_length=255)]
    design_level: Literal["essential", "detailed"] = "essential"
    description: str | None = None
    data: EssentialDesignData
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None


class PlantDesignUpdate(BaseModel):
    """
    Partial-update payload (rename, edit data, move site).

    Only the fields explicitly present are written.
    """

    name: Annotated[str | None, Field(min_length=1, max_length=255)] = None
    description: str | None = None
    data: EssentialDesignData | None = None
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None


class PlantDesignResponse(BaseModel):
    """
    Plant design record as returned by the API.

    ``data`` is returned as a plain dict (not re-validated) so future
    ``detailed`` payloads round-trip unchanged.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    design_level: str
    description: str | None = None
    data: dict
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Electrical-designer evaluation (POST /api/designs/evaluate)
# ---------------------------------------------------------------------------


class DesignPanelSpecsParams(BaseModel):
    """
    Module datasheet inputs for the designer.

    Mirrors the designer-relevant subset of
    :class:`~sim_stochastic_pv.simulation.electrical.PanelElectricalSpecs`;
    the UI prefills it from the panel catalogue and lets the user edit.
    ``noct_c`` / ``n_cells_series`` are needed only by the production
    preview's electrical model, hence optional here.
    """

    power_w: Annotated[float, Field(gt=0)]
    v_oc_stc_v: Annotated[float, Field(gt=0)]
    v_mpp_stc_v: Annotated[float, Field(gt=0)]
    i_sc_stc_a: Annotated[float, Field(gt=0)]
    i_mpp_stc_a: Annotated[float, Field(gt=0)]
    beta_voc_pct_per_c: Annotated[float, Field(lt=0)]
    gamma_pmax_pct_per_c: Annotated[float, Field(lt=0)]
    alpha_isc_pct_per_c: Annotated[float, Field(gt=0)]
    v_system_max_v: Annotated[float, Field(gt=0)]
    max_series_fuse_a: Annotated[float | None, Field(gt=0)] = None
    noct_c: Annotated[float | None, Field(gt=0)] = None
    n_cells_series: Annotated[int | None, Field(gt=0)] = None


class DesignInverterSpecsParams(BaseModel):
    """
    Inverter datasheet inputs for the designer.

    The full-load MPPT window is the sizing window; when a datasheet
    only publishes the wide tracking window, send it in
    ``v_mppt_min_v``/``v_mppt_max_v`` and leave the full-load pair unset.
    """

    p_ac_nom_kw: Annotated[float, Field(gt=0)]
    efficiency_max: Annotated[float | None, Field(gt=0, le=1)] = None
    v_dc_max_v: Annotated[float, Field(gt=0)]
    v_dc_min_v: Annotated[float | None, Field(gt=0)] = None
    v_mppt_min_v: Annotated[float | None, Field(gt=0)] = None
    v_mppt_max_v: Annotated[float | None, Field(gt=0)] = None
    v_mppt_full_load_min_v: Annotated[float | None, Field(gt=0)] = None
    v_mppt_full_load_max_v: Annotated[float | None, Field(gt=0)] = None
    n_mppt_trackers: Annotated[int, Field(ge=1, le=12)] = 1
    i_dc_max_per_mppt_a: Annotated[float, Field(gt=0)]
    i_sc_max_per_mppt_a: Annotated[float, Field(gt=0)]
    max_strings_per_mppt: Annotated[int, Field(ge=1, le=12)] = 1


class DesignSiteParams(BaseModel):
    """Site thermal corners (see ``DesignSite``)."""

    t_min_c: Annotated[float, Field(ge=-60, le=30)]
    t_max_c: Annotated[float, Field(ge=0, le=60)]
    delta_t_cell_c: Annotated[float, Field(ge=0, le=50)] = 30.0


class DesignRequirementsParams(BaseModel):
    """Project requirements + verification parameters."""

    p_ac_required_kw: Annotated[float, Field(gt=0, le=1000)]
    target_dc_ac_ratio: Annotated[float, Field(gt=0.5, le=3.0)] = 1.2
    n_panels_per_string: Annotated[int, Field(ge=1, le=40)]
    safety_factor_isc: Annotated[float, Field(ge=1.0, le=2.0)] = 1.25
    max_cable_loss_fraction: Annotated[float, Field(gt=0, lt=0.2)] = 0.01
    fuse_factor_min: Annotated[float, Field(gt=0)] = 1.5
    fuse_factor_max: Annotated[float, Field(gt=0)] = 2.4


class CableSectionInput(BaseModel):
    """One candidate cable section, optionally priced/rated."""

    section_mm2: Annotated[float, Field(gt=0)]
    price_eur_per_m: Annotated[float | None, Field(ge=0)] = None
    iz_a: Annotated[float | None, Field(gt=0)] = None


class DesignCableParamsInput(BaseModel):
    """Cable-run inputs (defaults mirror the engine defaults)."""

    length_one_way_m: Annotated[float, Field(gt=0, le=500)] = 30.0
    operating_temperature_c: Annotated[float, Field(ge=20, le=120)] = 70.0
    sections: list[CableSectionInput] | None = None


class DesignEvaluateRequest(BaseModel):
    """
    Full design payload for the stateless evaluation endpoint.

    The UI posts this on every input change and repaints all derived
    cells from the response.
    """

    panel: DesignPanelSpecsParams
    inverter: DesignInverterSpecsParams
    site: DesignSiteParams
    requirements: DesignRequirementsParams
    cable: DesignCableParamsInput | None = None


class TemperatureCorrectedOut(BaseModel):
    """Mirror of ``TemperatureCorrectedValues``."""

    t_cell_cold_c: float
    t_cell_hot_c: float
    v_oc_cold_v: float
    v_mp_cold_v: float
    v_mp_hot_v: float
    i_sc_hot_a: float
    i_mp_hot_a: float
    i_sc_design_a: float


class StringBoundsOut(BaseModel):
    """Mirror of ``StringSizingBounds``."""

    v_limit_v: float
    n_max_voc: int
    n_max_mppt: int
    n_min: int
    n_max: int
    feasible: bool


class StringVoltagesOut(BaseModel):
    """Mirror of ``StringVoltageChecks`` (negative margin = check failed)."""

    n_in_range: bool
    v_oc_string_cold_v: float
    v_oc_margin_v: float
    v_mp_string_hot_v: float
    v_mp_hot_margin_v: float
    v_mp_string_cold_v: float
    v_mp_cold_margin_v: float


class PlantSizingOut(BaseModel):
    """Mirror of ``PlantSizing``."""

    p_dc_target_kwp: float
    string_power_kwp: float
    n_strings: int
    total_panels: int
    p_dc_installed_kwp: float
    dc_ac_ratio: float


class CurrentChecksOut(BaseModel):
    """Mirror of ``CurrentChecks`` (negative margin = check failed)."""

    strings_per_mppt: int
    inputs_ok: bool
    i_operating_a: float
    i_operating_margin_a: float
    i_sc_a: float
    i_sc_margin_a: float


class TemperatureMarginsOut(BaseModel):
    """Mirror of ``TemperatureMargins``."""

    t_min_admissible_c: float
    margin_cold_c: float
    t_cell_max_admissible_c: float
    t_amb_max_admissible_c: float
    margin_hot_c: float
    t_min_mppt_tracking_c: float
    robust: bool


class ProtectionOut(BaseModel):
    """Mirror of ``ProtectionSizing``."""

    protection_required: bool
    i_fuse_min_a: float
    i_fuse_max_norm_a: float
    i_fuse_module_max_a: float | None = None
    recommended_fuse_a: float | None = None
    fuse_within_module_limit: bool | None = None
    fuse_within_norm_limit: bool | None = None


class CableRowOut(BaseModel):
    """Mirror of ``CableSectionRow``."""

    section_mm2: float
    resistance_ohm: float
    voltage_drop_v: float
    voltage_drop_fraction: float
    loss_per_string_w: float
    loss_total_kw: float
    loss_fraction_of_dc: float
    loss_ok: bool
    cost_total_eur: float | None = None
    iz_a: float | None = None
    iz_ok: bool | None = None


class CableTableOut(BaseModel):
    """Mirror of ``CableTable``."""

    resistivity_ohm_mm2_per_m: float
    rows: list[CableRowOut]
    recommended_section_mm2: float | None = None


class DesignEvaluateResponse(BaseModel):
    """
    Full design evaluation — every derived block the UI renders.

    Negative margins flag the failing checks; ``all_checks_ok`` is the
    single traffic light at the top of the page.
    """

    corrected: TemperatureCorrectedOut
    bounds: StringBoundsOut
    voltages: StringVoltagesOut
    plant: PlantSizingOut
    currents: CurrentChecksOut
    margins: TemperatureMarginsOut
    protection: ProtectionOut
    cables: CableTableOut
    all_checks_ok: bool


# ---------------------------------------------------------------------------
# Production preview (POST /api/designs/production-preview)
# ---------------------------------------------------------------------------


class ProductionCableInput(BaseModel):
    """Chosen cable run for the hourly loss integration."""

    section_mm2: Annotated[float, Field(gt=0)]
    length_one_way_m: Annotated[float, Field(gt=0, le=500)] = 30.0
    operating_temperature_c: Annotated[float, Field(ge=20, le=120)] = 70.0


class ProductionPreviewRequest(BaseModel):
    """
    Inputs of the designer hourly Monte Carlo production preview.

    Attributes:
        panel: Module datasheet (``noct_c`` and ``n_cells_series``
            required when ``use_electrical_model`` is on).
        inverter: Inverter datasheet.
        n_panels_per_string: Chosen string length.
        n_strings: Number of identical strings.
        solar_profile_id: Site solar profile driving the weather paths.
        climate_profile_id: Optional calibrated climate profile —
            required when ``use_electrical_model`` is on.
        use_electrical_model: Run the MPPT-window + temperature model
            (explicit opt-in; requires the climate profile and the full
            panel datasheet).
        cable: Optional chosen cable run for the I-squared loss
            integration.
        n_paths: Weather paths (default 30, capped at 100).
        seed: Master RNG seed.
    """

    panel: DesignPanelSpecsParams
    inverter: DesignInverterSpecsParams
    n_panels_per_string: Annotated[int, Field(ge=1, le=40)]
    n_strings: Annotated[int, Field(ge=1, le=100)]
    solar_profile_id: int
    climate_profile_id: int | None = None
    use_electrical_model: bool = False
    cable: ProductionCableInput | None = None
    n_paths: Annotated[int, Field(ge=5, le=100)] = 30
    seed: int = 42


class ProductionPreviewResponse(BaseModel):
    """Mirror of ``ProductionPreviewResult`` (annual kWh + loss split)."""

    annual_dc_kwh_mean: float
    annual_ac_kwh_mean: float
    annual_ac_kwh_p05: float
    annual_ac_kwh_p95: float
    clipping_kwh_mean: float
    clipping_fraction: float
    cable_loss_kwh_mean: float
    cable_loss_fraction: float
    electrical_derating_kwh_mean: float
    inverter_efficiency: float
    hours_outside_mppt_per_year_mean: float
    hours_dc_overvoltage_per_year_mean: float
    n_paths: int
