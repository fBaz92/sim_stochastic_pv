"""
One-call orchestrator for the electrical designer.

The web UI behaves like a reactive spreadsheet: every input change posts
the full design to ``/api/designs/evaluate`` and repaints every derived
cell from the returned :class:`DesignEvaluation`. Keeping the
orchestration here (rather than in the route) lets the CLI, the tests
and the future PDF report reuse the identical computation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..electrical import InverterElectricalSpecs, PanelElectricalSpecs
from .cables import CableParams, CableTable, compute_cable_table
from .currents import CurrentChecks, check_mppt_currents
from .inputs import DesignRequirements, DesignSite
from .protections import ProtectionSizing, size_string_protection
from .sizing import (
    PlantSizing,
    StringSizingBounds,
    StringVoltageChecks,
    TemperatureCorrectedValues,
    TemperatureMargins,
    check_string_voltages,
    compute_plant_sizing,
    compute_string_bounds,
    compute_temperature_corrected,
    compute_temperature_margins,
)


@dataclass(frozen=True)
class DesignEvaluation:
    """
    Complete evaluation of one electrical design.

    Bundles every block the UI renders: temperature-corrected values,
    the admissible string range, the voltage checks with margins, the
    plant sizing, the per-MPPT current checks, the temperature margins,
    the string-fuse sizing and the cable comparison table.

    Attributes:
        corrected: Module values at the design temperature corners.
        bounds: Admissible modules-per-string range.
        voltages: String-voltage checks for the chosen N.
        plant: Plant sizing (strings, total power, DC/AC ratio).
        currents: Per-MPPT current checks (worst-loaded tracker).
        margins: Temperature robustness of the chosen design.
        protection: String-fuse sizing (CEI EN 62548).
        cables: DC cable loss table + recommended section.
        all_checks_ok: True when every individual check passes (string
            range, three voltage margins, physical inputs, two current
            margins, both temperature margins, and — when applicable —
            the fuse limits). The cable table is advisory (the user
            picks a section) and does not gate this flag.
    """

    corrected: TemperatureCorrectedValues
    bounds: StringSizingBounds
    voltages: StringVoltageChecks
    plant: PlantSizing
    currents: CurrentChecks
    margins: TemperatureMargins
    protection: ProtectionSizing
    cables: CableTable
    all_checks_ok: bool


def evaluate_design(
    panel: PanelElectricalSpecs,
    inverter: InverterElectricalSpecs,
    site: DesignSite,
    requirements: DesignRequirements,
    cable: CableParams | None = None,
) -> DesignEvaluation:
    """
    Evaluate a full design in one pass.

    Args:
        panel: Module datasheet specs (designer fields required: STC
            voltages/currents, α/β/γ, system voltage; ``power_w`` for
            the plant sizing; ``max_series_fuse_a`` optional).
        inverter: Inverter datasheet specs (DC limit, MPPT window —
            full-load preferred —, per-MPPT current limits, physical
            inputs, AC nameplate).
        site: Thermal corners of the installation site.
        requirements: AC requirement, target DC/AC ratio, chosen
            modules-per-string and verification parameters.
        cable: Cable run parameters; ``None`` uses the defaults
            (30 m, 70 °C, standard section list).

    Returns:
        :class:`DesignEvaluation` with every derived block.

    Raises:
        ValueError: When a required datasheet field is missing — the
            message names the field so the UI can point at it.

    Example:
        ```python
        evaluation = evaluate_design(panel, inverter,
                                     DesignSite(t_min_c=-20, t_max_c=45),
                                     DesignRequirements(p_ac_required_kw=3.0,
                                                        n_panels_per_string=6))
        evaluation.plant.p_dc_installed_kwp   # 6.06 for the reference case
        evaluation.all_checks_ok              # False (currents exceed MPPT)
        ```
    """
    cable = cable or CableParams()

    corrected = compute_temperature_corrected(panel, site, requirements)
    bounds = compute_string_bounds(panel, inverter, corrected)
    voltages = check_string_voltages(
        inverter, corrected, bounds, requirements.n_panels_per_string
    )
    plant = compute_plant_sizing(panel, inverter, requirements)
    currents = check_mppt_currents(inverter, corrected, plant)
    margins = compute_temperature_margins(
        panel, inverter, site, bounds, requirements.n_panels_per_string
    )
    protection = size_string_protection(panel, plant, requirements)
    cables = compute_cable_table(
        panel_i_mp_stc_a=float(panel.i_mpp_stc_a or 0.0),
        corrected=corrected,
        voltages=voltages,
        plant=plant,
        requirements=requirements,
        cable=cable,
    )

    fuse_checks_ok = True
    if protection.protection_required:
        fuse_checks_ok = (
            protection.recommended_fuse_a is not None
            and protection.fuse_within_norm_limit is not False
            and protection.fuse_within_module_limit is not False
        )

    all_ok = (
        voltages.n_in_range
        and voltages.v_oc_margin_v >= 0
        and voltages.v_mp_hot_margin_v >= 0
        and voltages.v_mp_cold_margin_v >= 0
        and currents.inputs_ok
        and currents.i_operating_margin_a >= 0
        and currents.i_sc_margin_a >= 0
        and margins.robust
        and fuse_checks_ok
    )

    return DesignEvaluation(
        corrected=corrected,
        bounds=bounds,
        voltages=voltages,
        plant=plant,
        currents=currents,
        margins=margins,
        protection=protection,
        cables=cables,
        all_checks_ok=all_ok,
    )
