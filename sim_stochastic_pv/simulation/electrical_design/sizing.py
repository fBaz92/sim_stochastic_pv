"""
String sizing: temperature-corrected datasheet values, admissible
modules-per-string range, string-voltage checks, temperature margins and
plant sizing.

All formulas follow the IEC linear temperature model and the standard
designer practice of correcting V_mp with the γ(P_max) coefficient when
the datasheet does not publish a dedicated V_mp coefficient (a slightly
conservative choice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..electrical import InverterElectricalSpecs, PanelElectricalSpecs
from .inputs import DesignRequirements, DesignSite


STC_TEMPERATURE_C = 25.0


def _require(value: float | int | None, name: str) -> float:
    """Return ``value`` as float or raise a designer-grade error."""
    if value is None:
        raise ValueError(
            f"Electrical designer: datasheet field '{name}' is required "
            "for string sizing but is missing from the specs."
        )
    return float(value)


def _full_load_window(inverter: InverterElectricalSpecs) -> tuple[float, float]:
    """
    MPPT window used for string sizing.

    Prefers the *full-load* MPPT range (the stricter window some
    datasheets publish for nominal-power operation); falls back to the
    wide tracking window when absent.
    """
    v_min = (
        inverter.v_mppt_full_load_min_v
        if inverter.v_mppt_full_load_min_v is not None
        else inverter.v_mppt_min_v
    )
    v_max = (
        inverter.v_mppt_full_load_max_v
        if inverter.v_mppt_full_load_max_v is not None
        else inverter.v_mppt_max_v
    )
    return (
        _require(v_min, "v_mppt_full_load_min_v / v_mppt_min_v"),
        _require(v_max, "v_mppt_full_load_max_v / v_mppt_max_v"),
    )


@dataclass(frozen=True)
class TemperatureCorrectedValues:
    """
    Module voltages/currents at the design temperature corners.

    Attributes:
        t_cell_cold_c: Cold-case cell temperature (°C).
        t_cell_hot_c: Hot-case cell temperature (°C).
        v_oc_cold_v: Maximum possible module voltage — V_oc at the cold
            cell (V).
        v_mp_cold_v: V_mp at the cold cell (V), corrected with γ.
        v_mp_hot_v: Minimum operating module voltage — V_mp at the hot
            cell (V).
        i_sc_hot_a: I_sc at the hot cell (A).
        i_mp_hot_a: I_mp at the hot cell (A).
        i_sc_design_a: ``i_sc_hot_a × safety_factor`` — the current used
            to size cables and protections (A).
    """

    t_cell_cold_c: float
    t_cell_hot_c: float
    v_oc_cold_v: float
    v_mp_cold_v: float
    v_mp_hot_v: float
    i_sc_hot_a: float
    i_mp_hot_a: float
    i_sc_design_a: float


def compute_temperature_corrected(
    panel: PanelElectricalSpecs,
    site: DesignSite,
    requirements: DesignRequirements,
) -> TemperatureCorrectedValues:
    """
    Correct the module's STC values to the site's design corners.

    Formulas (linear IEC model, coefficients in %/°C):

        V_oc(T) = V_oc_stc · (1 + β/100 · (T − 25))
        V_mp(T) = V_mp_stc · (1 + γ/100 · (T − 25))   [conservative]
        I_sc(T) = I_sc_stc · (1 + α/100 · (T − 25))
        I_mp(T) = I_mp_stc · (1 + α/100 · (T − 25))

    Args:
        panel: Module datasheet specs (requires ``v_oc_stc_v``,
            ``v_mpp_stc_v``, ``i_sc_stc_a``, ``i_mpp_stc_a``,
            ``beta_voc_pct_per_c``, ``gamma_pmax_pct_per_c``,
            ``alpha_isc_pct_per_c``).
        site: Thermal corners of the site.
        requirements: Carries the I_sc safety factor.

    Returns:
        :class:`TemperatureCorrectedValues`.

    Raises:
        ValueError: When a required datasheet field is missing.

    Example:
        ```python
        # TCL HSM-ND54-DR505 at Tmin -20 / Tmax 45 / ΔT 30:
        # v_oc_cold ≈ 44.656 V, v_mp_hot ≈ 28.984 V, i_sc_hot ≈ 16.237 A
        ```
    """
    v_oc = _require(panel.v_oc_stc_v, "v_oc_stc_v")
    v_mp = _require(panel.v_mpp_stc_v, "v_mpp_stc_v")
    i_sc = _require(panel.i_sc_stc_a, "i_sc_stc_a")
    i_mp = _require(panel.i_mpp_stc_a, "i_mpp_stc_a")
    beta = _require(panel.beta_voc_pct_per_c, "beta_voc_pct_per_c")
    gamma = _require(panel.gamma_pmax_pct_per_c, "gamma_pmax_pct_per_c")
    alpha = _require(panel.alpha_isc_pct_per_c, "alpha_isc_pct_per_c")

    t_cold = site.t_cell_cold_c
    t_hot = site.t_cell_hot_c
    d_cold = t_cold - STC_TEMPERATURE_C
    d_hot = t_hot - STC_TEMPERATURE_C

    i_sc_hot = i_sc * (1.0 + alpha / 100.0 * d_hot)
    return TemperatureCorrectedValues(
        t_cell_cold_c=t_cold,
        t_cell_hot_c=t_hot,
        v_oc_cold_v=v_oc * (1.0 + beta / 100.0 * d_cold),
        v_mp_cold_v=v_mp * (1.0 + gamma / 100.0 * d_cold),
        v_mp_hot_v=v_mp * (1.0 + gamma / 100.0 * d_hot),
        i_sc_hot_a=i_sc_hot,
        i_mp_hot_a=i_mp * (1.0 + alpha / 100.0 * d_hot),
        i_sc_design_a=i_sc_hot * requirements.safety_factor_isc,
    )


@dataclass(frozen=True)
class StringSizingBounds:
    """
    Admissible modules-per-string range for the chosen pairing.

    Attributes:
        v_limit_v: String voltage limit —
            ``min(inverter v_dc_max, module v_system_max)`` (V).
        n_max_voc: Max modules so that the cold-case string V_oc stays
            below ``v_limit_v`` (hard safety bound).
        n_max_mppt: Max modules so that the cold-case string V_mp stays
            inside the MPPT window (exceeding it loses production on
            cold days, no hardware damage).
        n_min: Min modules so that the hot-case string V_mp stays above
            the MPPT lower bound.
        n_max: ``min(n_max_voc, n_max_mppt)`` — the effective upper
            bound shown to the user.
        feasible: ``n_min <= n_max`` — False when this panel+inverter
            pairing admits no valid string length.
    """

    v_limit_v: float
    n_max_voc: int
    n_max_mppt: int
    n_min: int
    n_max: int
    feasible: bool


def compute_string_bounds(
    panel: PanelElectricalSpecs,
    inverter: InverterElectricalSpecs,
    corrected: TemperatureCorrectedValues,
) -> StringSizingBounds:
    """
    Compute the admissible modules-per-string range.

    Args:
        panel: Module specs (requires ``v_system_max_v``).
        inverter: Inverter specs (requires ``v_dc_max_v`` and an MPPT
            window — full-load preferred).
        corrected: Output of :func:`compute_temperature_corrected`.

    Returns:
        :class:`StringSizingBounds`.

    Example:
        ```python
        # DR505 + ZCS 3000: v_limit = 600 V, n_max_voc = 13,
        # n_max_mppt = 13 (520 V window), n_min = 6.
        ```
    """
    v_limit = min(
        _require(inverter.v_dc_max_v, "v_dc_max_v"),
        _require(panel.v_system_max_v, "v_system_max_v"),
    )
    v_fl_min, v_fl_max = _full_load_window(inverter)

    n_max_voc = math.floor(v_limit / corrected.v_oc_cold_v)
    n_max_mppt = math.floor(v_fl_max / corrected.v_mp_cold_v)
    n_min = math.ceil(v_fl_min / corrected.v_mp_hot_v)
    n_max = min(n_max_voc, n_max_mppt)
    return StringSizingBounds(
        v_limit_v=v_limit,
        n_max_voc=n_max_voc,
        n_max_mppt=n_max_mppt,
        n_min=n_min,
        n_max=n_max,
        feasible=n_min <= n_max,
    )


@dataclass(frozen=True)
class StringVoltageChecks:
    """
    String voltages for the chosen modules-per-string, with margins.

    A negative margin means the corresponding check fails; the UI shows
    "FUORI RANGE" with the magnitude. All voltages in V.

    Attributes:
        n_in_range: Chosen N inside ``[n_min, n_max]``.
        v_oc_string_cold_v: Worst-case winter open-circuit voltage.
        v_oc_margin_v: ``v_limit − v_oc_string_cold`` (≥ 0 = OK).
        v_mp_string_hot_v: Worst-case summer operating voltage.
        v_mp_hot_margin_v: ``v_mp_string_hot − v_mppt_min`` (≥ 0 = OK).
        v_mp_string_cold_v: Cold-day operating voltage.
        v_mp_cold_margin_v: ``v_mppt_max − v_mp_string_cold`` (≥ 0 = OK
            for tracking; a breach costs production, not hardware).
    """

    n_in_range: bool
    v_oc_string_cold_v: float
    v_oc_margin_v: float
    v_mp_string_hot_v: float
    v_mp_hot_margin_v: float
    v_mp_string_cold_v: float
    v_mp_cold_margin_v: float


def check_string_voltages(
    inverter: InverterElectricalSpecs,
    corrected: TemperatureCorrectedValues,
    bounds: StringSizingBounds,
    n_panels_per_string: int,
) -> StringVoltageChecks:
    """
    Verify the chosen string length against the voltage windows.

    Args:
        inverter: Inverter specs (MPPT window).
        corrected: Temperature-corrected module values.
        bounds: Admissible range (carries ``v_limit_v``).
        n_panels_per_string: The chosen N.

    Returns:
        :class:`StringVoltageChecks` with one margin per check.
    """
    v_fl_min, v_fl_max = _full_load_window(inverter)
    n = n_panels_per_string
    v_oc_string = n * corrected.v_oc_cold_v
    v_mp_hot_string = n * corrected.v_mp_hot_v
    v_mp_cold_string = n * corrected.v_mp_cold_v
    return StringVoltageChecks(
        n_in_range=bounds.n_min <= n <= bounds.n_max,
        v_oc_string_cold_v=v_oc_string,
        v_oc_margin_v=bounds.v_limit_v - v_oc_string,
        v_mp_string_hot_v=v_mp_hot_string,
        v_mp_hot_margin_v=v_mp_hot_string - v_fl_min,
        v_mp_string_cold_v=v_mp_cold_string,
        v_mp_cold_margin_v=v_fl_max - v_mp_cold_string,
    )


@dataclass(frozen=True)
class TemperatureMargins:
    """
    Temperature robustness of the chosen design.

    Attributes:
        t_min_admissible_c: Cell temperature below which the string
            V_oc exceeds the voltage limit (°C). Often absurdly low for
            short strings — that is the point: huge margin.
        margin_cold_c: ``site t_min − t_min_admissible`` (≥ 0 = OK).
        t_cell_max_admissible_c: Cell temperature above which the
            string V_mp drops below the MPPT lower bound (°C).
        t_amb_max_admissible_c: Same, expressed as ambient temperature
            (cell − ΔT).
        margin_hot_c: ``t_amb_max_admissible − site t_max`` (≥ 0 = OK).
        t_min_mppt_tracking_c: Cell temperature below which the MPP
            leaves the tracking window (production loss only).
        robust: Both margins positive.
    """

    t_min_admissible_c: float
    margin_cold_c: float
    t_cell_max_admissible_c: float
    t_amb_max_admissible_c: float
    margin_hot_c: float
    t_min_mppt_tracking_c: float
    robust: bool


def compute_temperature_margins(
    panel: PanelElectricalSpecs,
    inverter: InverterElectricalSpecs,
    site: DesignSite,
    bounds: StringSizingBounds,
    n_panels_per_string: int,
) -> TemperatureMargins:
    """
    Invert the temperature model to find the admissible site envelope.

    Solving ``n·V(T) = V_limit`` for ``T`` with the linear model gives

        T = 25 + 100 · (V_limit / (n·V_stc) − 1) / coeff

    applied to the three (limit, coefficient) pairs of interest.

    Args:
        panel: Module specs (STC voltages + β, γ).
        inverter: Inverter specs (windows).
        site: Site corners (for the margins and the ΔT shift).
        bounds: Carries ``v_limit_v``.
        n_panels_per_string: The chosen N.

    Returns:
        :class:`TemperatureMargins`.
    """
    v_oc = _require(panel.v_oc_stc_v, "v_oc_stc_v")
    v_mp = _require(panel.v_mpp_stc_v, "v_mpp_stc_v")
    beta = _require(panel.beta_voc_pct_per_c, "beta_voc_pct_per_c")
    gamma = _require(panel.gamma_pmax_pct_per_c, "gamma_pmax_pct_per_c")
    v_fl_min, v_fl_max = _full_load_window(inverter)
    n = n_panels_per_string

    t_min_admissible = STC_TEMPERATURE_C + 100.0 * (
        bounds.v_limit_v / (n * v_oc) - 1.0
    ) / beta
    t_cell_max_admissible = STC_TEMPERATURE_C + 100.0 * (
        v_fl_min / (n * v_mp) - 1.0
    ) / gamma
    t_amb_max_admissible = t_cell_max_admissible - site.delta_t_cell_c
    t_min_tracking = STC_TEMPERATURE_C + 100.0 * (
        v_fl_max / (n * v_mp) - 1.0
    ) / gamma

    margin_cold = site.t_min_c - t_min_admissible
    margin_hot = t_amb_max_admissible - site.t_max_c
    return TemperatureMargins(
        t_min_admissible_c=t_min_admissible,
        margin_cold_c=margin_cold,
        t_cell_max_admissible_c=t_cell_max_admissible,
        t_amb_max_admissible_c=t_amb_max_admissible,
        margin_hot_c=margin_hot,
        t_min_mppt_tracking_c=t_min_tracking,
        robust=margin_cold > 0 and margin_hot > 0,
    )


@dataclass(frozen=True)
class PlantSizing:
    """
    Plant-level sizing for the chosen string length.

    All strings are identical (same module count) — when the string
    count is not a multiple of the MPPT count, the current checks use
    the worst-loaded tracker.

    Attributes:
        p_dc_target_kwp: ``p_ac_required × target ratio`` (kWp).
        string_power_kwp: DC power of one string (kWp).
        n_strings: Number of identical strings (≥ 1, rounded up to
            reach the target).
        total_panels: ``n_strings × n_panels_per_string``.
        p_dc_installed_kwp: Actual installed DC power (kWp).
        dc_ac_ratio: ``p_dc_installed / inverter AC nameplate``.
    """

    p_dc_target_kwp: float
    string_power_kwp: float
    n_strings: int
    total_panels: int
    p_dc_installed_kwp: float
    dc_ac_ratio: float


def compute_plant_sizing(
    panel: PanelElectricalSpecs,
    inverter: InverterElectricalSpecs,
    requirements: DesignRequirements,
) -> PlantSizing:
    """
    Size the plant: number of identical strings to reach the DC target.

    Args:
        panel: Module specs (requires ``power_w``).
        inverter: Inverter specs (requires ``p_ac_nom_kw`` for the
            effective DC/AC ratio).
        requirements: AC requirement + target ratio + chosen N.

    Returns:
        :class:`PlantSizing`.

    Example:
        ```python
        # 3 kW AC × 1.2 → 3.6 kWp target; 6×505 W strings (3.03 kWp)
        # → 2 strings, 12 panels, 6.06 kWp, DC/AC = 2.02 on a 3 kW
        # inverter — the spreadsheet's reference case.
        ```
    """
    power_w = _require(panel.power_w, "power_w")
    p_ac_nom = _require(inverter.p_ac_nom_kw, "p_ac_nom_kw")
    p_dc_target = requirements.p_ac_required_kw * requirements.target_dc_ac_ratio
    string_power = requirements.n_panels_per_string * power_w / 1000.0
    n_strings = max(1, math.ceil(p_dc_target / string_power))
    total_panels = n_strings * requirements.n_panels_per_string
    p_dc_installed = total_panels * power_w / 1000.0
    return PlantSizing(
        p_dc_target_kwp=p_dc_target,
        string_power_kwp=string_power,
        n_strings=n_strings,
        total_panels=total_panels,
        p_dc_installed_kwp=p_dc_installed,
        dc_ac_ratio=p_dc_installed / p_ac_nom,
    )
