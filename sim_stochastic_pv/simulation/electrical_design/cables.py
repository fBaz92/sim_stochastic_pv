"""
DC string-cable losses per cross-section and recommended section.

The loss model is purely ohmic at the design current, with the copper
resistivity corrected to the cable's operating temperature:

    ρ(T) = ρ₂₀ · (1 + α_Cu · (T − 20))
    R    = 2 · L · ρ(T) / S          (out + return)
    ΔV   = R · I_mp                   (voltage drop at STC current)
    P    = R · I_mp²                  (per-string loss, W)

The drop percentage is computed against the *hot-case* string voltage —
the conservative corner where a given ΔV weighs the most. Beyond the
ohmic criterion the chosen section must also carry the design current
thermally (Iz ≥ 1.25 × I_sc per CEI-UNEL / IEC 60364) — the table
reports the check when the catalogue provides an Iz rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .inputs import DesignRequirements
from .sizing import PlantSizing, StringVoltageChecks, TemperatureCorrectedValues


# Copper resistivity at 20 °C (ohm·mm²/m).
COPPER_RESISTIVITY_20C = 0.0175

# Copper temperature coefficient (1/°C).
COPPER_TEMP_COEFFICIENT = 0.00393

# Typical operating temperature of solar cables in sun-exposed conduit (°C).
DEFAULT_CABLE_OPERATING_T_C = 70.0

# Standard cross-sections offered by the comparison table (mm²).
DEFAULT_SECTIONS_MM2: tuple[float, ...] = (2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 35.0)


@dataclass(frozen=True)
class CableParams:
    """
    Cable run parameters.

    Attributes:
        length_one_way_m: Average string-to-inverter distance (m); the
            loss model doubles it for the return conductor.
        operating_temperature_c: Cable temperature for the resistivity
            correction (°C), default
            :data:`DEFAULT_CABLE_OPERATING_T_C`.
        sections_mm2: Cross-sections to evaluate (mm²), default
            :data:`DEFAULT_SECTIONS_MM2`.
        price_eur_per_m: Optional €/m per section (aligned with
            ``sections_mm2``) — fills the cost column for the
            comparison; ``None`` leaves costs out.
        iz_a: Optional thermal current rating Iz per section (A,
            aligned with ``sections_mm2``) for the carrying-capacity
            check.
    """

    length_one_way_m: float = 30.0
    operating_temperature_c: float = DEFAULT_CABLE_OPERATING_T_C
    sections_mm2: tuple[float, ...] = DEFAULT_SECTIONS_MM2
    price_eur_per_m: tuple[float, ...] | None = None
    iz_a: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.length_one_way_m <= 0:
            raise ValueError("length_one_way_m must be > 0")
        if not self.sections_mm2:
            raise ValueError("sections_mm2 must not be empty")
        for name in ("price_eur_per_m", "iz_a"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.sections_mm2):
                raise ValueError(
                    f"{name} must align with sections_mm2 "
                    f"({len(values)} vs {len(self.sections_mm2)})"
                )


@dataclass(frozen=True)
class CableSectionRow:
    """
    One row of the cable comparison table.

    Attributes:
        section_mm2: Conductor cross-section (mm²).
        resistance_ohm: Out-and-return string resistance (Ω).
        voltage_drop_v: ΔV at the STC operating current (V).
        voltage_drop_fraction: ΔV over the hot-case string voltage.
        loss_per_string_w: Ohmic loss per string (W).
        loss_total_kw: Loss across all strings (kW).
        loss_fraction_of_dc: Total loss over the installed DC power.
        loss_ok: ``loss_fraction_of_dc <= max_cable_loss_fraction``.
        cost_total_eur: Conductor cost (2 × length × strings × €/m),
            ``None`` when no price list was provided.
        iz_a: Thermal rating from the catalogue, when provided.
        iz_ok: ``iz_a >= I_sc design`` — ``None`` when no Iz known.
    """

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


@dataclass(frozen=True)
class CableTable:
    """
    Full cable comparison plus the recommended section.

    Attributes:
        resistivity_ohm_mm2_per_m: ρ at the operating temperature.
        rows: One :class:`CableSectionRow` per section, input order.
        recommended_section_mm2: Smallest section that satisfies both
            the loss threshold and (when known) the Iz check; ``None``
            when no section qualifies.
    """

    resistivity_ohm_mm2_per_m: float
    rows: tuple[CableSectionRow, ...] = field(default_factory=tuple)
    recommended_section_mm2: float | None = None


def compute_cable_table(
    panel_i_mp_stc_a: float,
    corrected: TemperatureCorrectedValues,
    voltages: StringVoltageChecks,
    plant: PlantSizing,
    requirements: DesignRequirements,
    cable: CableParams,
) -> CableTable:
    """
    Build the per-section loss table and pick the recommended section.

    Args:
        panel_i_mp_stc_a: Module I_mp at STC (A) — the loss-table
            current, as in the reference spreadsheet (losses are
            evaluated at nominal operation, not at the safety-factored
            short-circuit corner).
        corrected: Carries the design I_sc for the Iz check.
        voltages: Carries the hot-case string voltage (drop reference).
        plant: String count and installed DC power.
        requirements: Loss-fraction acceptance threshold.
        cable: Run length, operating temperature, sections, prices, Iz.

    Returns:
        :class:`CableTable`.

    Example:
        ```python
        # Reference case (30 m, 70 °C, 2 strings, 6.06 kWp, 0.5 %
        # threshold): the 25 mm² row is the first with loss_ok=True.
        ```
    """
    rho = COPPER_RESISTIVITY_20C * (
        1.0 + COPPER_TEMP_COEFFICIENT * (cable.operating_temperature_c - 20.0)
    )
    i = float(panel_i_mp_stc_a)
    v_ref = voltages.v_mp_string_hot_v

    rows: list[CableSectionRow] = []
    recommended: float | None = None
    for idx, section in enumerate(cable.sections_mm2):
        resistance = 2.0 * cable.length_one_way_m * rho / section
        drop_v = resistance * i
        loss_w = resistance * i * i
        loss_total_kw = loss_w * plant.n_strings / 1000.0
        loss_fraction = (
            loss_total_kw / plant.p_dc_installed_kwp
            if plant.p_dc_installed_kwp > 0
            else 0.0
        )
        loss_ok = loss_fraction <= requirements.max_cable_loss_fraction

        cost = None
        if cable.price_eur_per_m is not None:
            cost = (
                cable.price_eur_per_m[idx]
                * 2.0
                * cable.length_one_way_m
                * plant.n_strings
            )
        iz = cable.iz_a[idx] if cable.iz_a is not None else None
        iz_ok = (iz >= corrected.i_sc_design_a) if iz is not None else None

        rows.append(
            CableSectionRow(
                section_mm2=section,
                resistance_ohm=resistance,
                voltage_drop_v=drop_v,
                voltage_drop_fraction=drop_v / v_ref if v_ref > 0 else 0.0,
                loss_per_string_w=loss_w,
                loss_total_kw=loss_total_kw,
                loss_fraction_of_dc=loss_fraction,
                loss_ok=loss_ok,
                cost_total_eur=cost,
                iz_a=iz,
                iz_ok=iz_ok,
            )
        )
        if recommended is None and loss_ok and iz_ok is not False:
            recommended = section

    return CableTable(
        resistivity_ohm_mm2_per_m=rho,
        rows=tuple(rows),
        recommended_section_mm2=recommended,
    )
