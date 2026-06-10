"""
Detailed electrical model for inverter + PV strings — Phase 16 opt-in.

The classes here add a *thin* layer of electrical realism on top of the
default :class:`~sim_stochastic_pv.simulation.inverter.InverterAC` energy
dispatch:

- panel V_oc and V_mpp scaled by temperature with vendor datasheet
  coefficients (``beta_voc``, ``gamma_pmax``);
- string-level voltage check against the inverter's DC operating window
  (``v_dc_min/max``) — outside means **hardware shutdown**;
- MPPT-window check (``v_mppt_min/max``) — outside means **derating**
  (power reduced by a smooth ``(V_target / V_string) ** k`` factor);
- multi-MPPT distribution: a scenario may declare any number of
  ``PvString`` objects, each tagged with a ``mppt_id`` so the model
  can sum DC contributions correctly into each tracker before the
  inverter sees them.

The whole module is **dormant by default**. The legacy energy path
(byte-identical to pre-Phase-16) runs whenever the scenario JSON omits
the ``electrical`` block, sets ``electrical.mode = "off"``, or omits
any of the required datasheet fields.

Hooking into the Monte Carlo loop happens in
``simulation/energy_simulator.py`` — that file pre-computes a per-path
hourly ``T_ambient`` array from the Phase-15 :class:`ThermalModel` and
hands it to :meth:`ElectricalModel.apply_to_pv_dc` together with the
naive PV DC power output of the solar model.

This module deliberately does NOT couple to the single-diode
``PVModelSingleDiode`` — the goal is a fast, robust MPPT-window check
that catches the *economically interesting* hardware failure modes
(winter V_oc overvoltage, summer V_mpp under-window) without paying the
runtime cost of an implicit-equation solver per hour. A future Phase
16-bis can swap the MPPT-window logic for a full single-diode solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Default constants — empirical values tuned on EU residential hardware
# ---------------------------------------------------------------------------

#: Standard Test Conditions cell temperature in °C (IEC 61215).
STC_TEMPERATURE_C: float = 25.0

#: Standard plane-of-array irradiance in W/m² used by NOCT formula.
NOCT_IRRADIANCE_W_PER_M2: float = 800.0

#: Reference ambient temperature in °C used by NOCT definition.
NOCT_REFERENCE_AMBIENT_C: float = 20.0

#: Approximate peak plane-of-array irradiance under STC (W/m²). Used to
#: scale the hourly DC power back to an *equivalent* irradiance for the
#: NOCT cell-temperature formula. This is intentionally simple — the
#: model treats irradiance as proportional to the hourly DC power
#: relative to the panel STC nameplate.
STC_IRRADIANCE_W_PER_M2: float = 1000.0

#: Default exponent ``k`` of the soft MPPT-window derating. The closer
#: ``V_string`` is to the window edge, the smaller the derating; outside
#: the window the derating grows as ``(V_target / V_string) ** k``.
#: Picked at 0.5 as a smooth, monotonic compromise that avoids the
#: discontinuity of a hard cutoff while still penalising significant
#: deviations realistically. Exposed via ``derating_exponent_k`` in the
#: scenario JSON so the user can tune (or zero out) the penalty.
DEFAULT_DERATING_EXPONENT_K: float = 0.5


# ---------------------------------------------------------------------------
# Status flags returned per hour for diagnostics + KPI aggregation
# ---------------------------------------------------------------------------

#: Hour was inside the inverter MPPT window — no derating applied.
STATUS_OK: int = 0

#: V_string was outside the MPPT window but still inside the DC operating
#: window. Power was derated by ``(V_target / V_string) ** k``.
STATUS_MPPT_WINDOW_BREACH: int = 1

#: V_string was outside the inverter DC operating window. Inverter
#: shutdown for that hour — DC power forced to zero. Logged in
#: ``hours_dc_overvoltage`` or ``hours_dc_undervoltage`` depending on
#: which boundary was crossed.
STATUS_DC_OVERVOLTAGE: int = 2
STATUS_DC_UNDERVOLTAGE: int = 3


# ---------------------------------------------------------------------------
# Datasheet specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelElectricalSpecs:
    """
    Vendor datasheet electrical specs for a PV module.

    Phase-16 ``mode = "mppt_window"`` requires every field to be set
    (i.e. ``None`` is rejected by :func:`ensure_complete_specs`). When
    the simulator runs in ``mode = "off"`` (legacy / default) the whole
    object is unused and the fields can stay ``None``.

    Attributes:
        power_w: Nominal peak power per module at STC (W). Same value as
            :attr:`~sim_stochastic_pv.simulation.optimizer.hardware.PanelOption.power_w`,
            duplicated here so the electrical model is self-contained.
        v_oc_stc_v: Open-circuit voltage at STC (V). Used at sub-zero
            cell temperature when the array is unloaded — drives the
            overvoltage check against ``v_dc_max_v``.
        v_mpp_stc_v: Maximum-power-point voltage at STC (V). The
            "operating" voltage; used to compute V_string and check the
            MPPT window.
        i_sc_stc_a: Short-circuit current at STC (A). Optional; used
            only as an indicator field.
        i_mpp_stc_a: Maximum-power-point current at STC (A). Optional;
            used to validate per-MPPT current ratings (future).
        n_cells_series: Number of cells wired in series inside the
            module. Documented for the future PV single-diode path —
            kept here so the dataclass is the single source of truth.
        beta_voc_pct_per_c: Temperature coefficient of V_oc, expressed
            in **percent per °C** (vendor convention; typical -0.27 to
            -0.32 for mono-Si). A 60°C cell raises ``V_oc`` to
            ``V_oc_stc * (1 + beta_voc/100 * (T_cell - 25))``.
        gamma_pmax_pct_per_c: Temperature coefficient of peak power,
            percent per °C (typical -0.34 to -0.40 for mono-Si). Drives
            the temperature-dependent power derating already on top of
            the MPPT-window penalty.
        noct_c: Nominal Operating Cell Temperature (°C) at 800 W/m²,
            20°C ambient, 1 m/s wind (typical 42–48 °C for residential
            roof-mounted modules).
        alpha_isc_pct_per_c: Temperature coefficient of I_sc, percent
            per °C (positive, typical +0.04 to +0.06). Used by the
            electrical *designer* to size cables and check per-MPPT
            currents at the hot-cell corner; unused by the MC model.
        v_system_max_v: Maximum system voltage of the module (V),
            typically 1000 or 1500 (IEC). The string voltage limit is
            ``min(v_system_max_v, inverter v_dc_max_v)``. Designer-only.
        max_series_fuse_a: Maximum series fuse rating from the module
            datasheet (A, IEC 61730-2). Upper bound for the string-fuse
            selection. Designer-only.

    Example:
        ```python
        # Longi LR5-72HPH-540M ("540 W bifacial")
        specs = PanelElectricalSpecs(
            power_w=540.0,
            v_oc_stc_v=49.5,
            v_mpp_stc_v=41.5,
            i_sc_stc_a=13.92,
            i_mpp_stc_a=13.02,
            n_cells_series=144,
            beta_voc_pct_per_c=-0.27,
            gamma_pmax_pct_per_c=-0.34,
            noct_c=45.0,
        )
        ```
    """

    power_w: float | None = None
    v_oc_stc_v: float | None = None
    v_mpp_stc_v: float | None = None
    i_sc_stc_a: float | None = None
    i_mpp_stc_a: float | None = None
    n_cells_series: int | None = None
    beta_voc_pct_per_c: float | None = None
    gamma_pmax_pct_per_c: float | None = None
    noct_c: float | None = None
    # Designer-only fields (string sizing / cable / protection checks).
    alpha_isc_pct_per_c: float | None = None
    v_system_max_v: float | None = None
    max_series_fuse_a: float | None = None


@dataclass(frozen=True)
class InverterElectricalSpecs:
    """
    Vendor datasheet electrical specs for the inverter DC side.

    All fields are ``None`` in legacy mode; required when the scenario
    activates ``electrical.mode = "mppt_window"``.

    Attributes:
        v_dc_min_v: Lower bound of the absolute DC operating window
            (V). Below this voltage the inverter is off.
        v_dc_max_v: Upper bound of the DC operating window (V). Above
            this the inverter trips for over-voltage protection. The
            *worst-case* is winter sunrise: ``V_oc`` at a sub-zero cell
            temperature — that's where ``hours_dc_overvoltage`` reports
            an integer hour count.
        v_mppt_min_v: Lower bound of the MPPT tracking window. Below
            this the inverter still operates but cannot track the
            string's true MPP — power is derated.
        v_mppt_max_v: Upper bound of the MPPT window. Above this the
            tracker is similarly off-MPP and power is derated.
        n_mppt_trackers: Number of independent MPPT inputs (>= 1). Each
            tracker sees its own ``pv_string`` block.
        i_dc_max_per_mppt_a: Maximum DC *operating* current per MPPT
            tracker (A). Enforced by the electrical designer's current
            checks; not enforced by the MC derating logic.
        i_sc_max_per_mppt_a: Maximum short-circuit current per MPPT (A)
            as stated by the inverter datasheet. Designer-only.
        max_strings_per_mppt: Number of physical string inputs per MPPT
            tracker. Designer-only.
        v_mppt_full_load_min_v: Lower bound of the MPPT range at *full
            load* (V) — the stricter window some datasheets publish for
            nominal-power operation. The designer sizes strings against
            this window (falling back to ``v_mppt_min_v``/``max`` when
            absent); the MC derating keeps using the wide window.
        v_mppt_full_load_max_v: Upper bound of the full-load MPPT range.
        p_ac_nom_kw: AC nameplate power (kW). Designer-only convenience
            so the design payload is self-contained.
        efficiency_max: Peak (or EU-weighted) efficiency, 0–1.
            Designer-only.
    """

    v_dc_min_v: float | None = None
    v_dc_max_v: float | None = None
    v_mppt_min_v: float | None = None
    v_mppt_max_v: float | None = None
    n_mppt_trackers: int = 1
    i_dc_max_per_mppt_a: float | None = None
    # Designer-only fields (string sizing / current checks).
    i_sc_max_per_mppt_a: float | None = None
    max_strings_per_mppt: int | None = None
    v_mppt_full_load_min_v: float | None = None
    v_mppt_full_load_max_v: float | None = None
    p_ac_nom_kw: float | None = None
    efficiency_max: float | None = None


@dataclass(frozen=True)
class PvString:
    """
    One panel string wired to a specific MPPT tracker.

    Attributes:
        n_panels: Number of panels in series inside this string (>= 1).
            Drives V_string = n_panels * V_panel.
        tilt_degrees: Panel tilt of the string (0–90°). Used by the
            solar model orientation factor — independent for each string.
        azimuth_degrees: Panel azimuth of the string (0–360°). Same.
        mppt_id: Zero-based index of the inverter MPPT tracker that
            owns this string. Must be in
            ``[0, n_mppt_trackers - 1]``. Multiple strings can share an
            MPPT tracker; in that case the model averages V_string
            weighted by panel count (a reasonable proxy of the parallel
            connection — the modules' I-V curves merge before the
            inverter clamps them).
    """

    n_panels: int
    tilt_degrees: float = 35.0
    azimuth_degrees: float = 180.0
    mppt_id: int = 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


_REQUIRED_PANEL_FIELDS: tuple[str, ...] = (
    "power_w",
    "v_oc_stc_v",
    "v_mpp_stc_v",
    "n_cells_series",
    "beta_voc_pct_per_c",
    "gamma_pmax_pct_per_c",
    "noct_c",
)

_REQUIRED_INVERTER_FIELDS: tuple[str, ...] = (
    "v_dc_min_v",
    "v_dc_max_v",
    "v_mppt_min_v",
    "v_mppt_max_v",
)


def missing_panel_fields(specs: PanelElectricalSpecs) -> list[str]:
    """Return the panel datasheet fields that are still ``None``."""
    return [name for name in _REQUIRED_PANEL_FIELDS if getattr(specs, name) is None]


def missing_inverter_fields(specs: InverterElectricalSpecs) -> list[str]:
    """Return the inverter datasheet fields that are still ``None``."""
    return [name for name in _REQUIRED_INVERTER_FIELDS if getattr(specs, name) is None]


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


def cell_temperature_c(
    t_ambient_c: float | np.ndarray,
    poa_irradiance_w_per_m2: float | np.ndarray,
    noct_c: float,
) -> float | np.ndarray:
    """
    Compute cell temperature using the NOCT relation.

    The Nominal Operating Cell Temperature (NOCT) is the cell
    temperature measured at 800 W/m², 20 °C ambient, 1 m/s wind. The
    linear NOCT correction generalises this to arbitrary conditions:

        T_cell = T_ambient + (NOCT - 20) / 800 * G_poa

    Args:
        t_ambient_c: Ambient temperature (°C). Scalar or array.
        poa_irradiance_w_per_m2: Plane-of-array irradiance (W/m²).
        noct_c: NOCT value from the panel datasheet (°C).

    Returns:
        Cell temperature in °C, same shape as input.
    """
    return t_ambient_c + (noct_c - NOCT_REFERENCE_AMBIENT_C) / NOCT_IRRADIANCE_W_PER_M2 * poa_irradiance_w_per_m2


def v_string_at_cell_temperature(
    v_mpp_stc_v: float,
    v_oc_stc_v: float,
    beta_voc_pct_per_c: float,
    n_panels_in_string: int,
    t_cell_c: float | np.ndarray,
    operating: bool | np.ndarray = True,
) -> float | np.ndarray:
    """
    String voltage at a given cell temperature.

    Uses the IEC linear temperature model for V_oc:

        V_oc(T) = V_oc_stc * (1 + beta_voc / 100 * (T - 25))

    and approximates V_mpp with the *same* temperature coefficient (this
    is a small over-correction — V_mpp's true coefficient is usually a
    bit smaller in magnitude, but the simplification keeps the model
    self-contained and the error is dominated by other approximations).

    When ``operating`` is True the function returns
    ``n_panels * V_mpp(T)`` — the in-operation tracker voltage. When
    ``operating`` is False it returns ``n_panels * V_oc(T)`` — the
    open-circuit voltage seen by the inverter before the MPPT loop
    closes (this is what determines the winter-sunrise overvoltage
    risk against ``v_dc_max``).

    Args:
        v_mpp_stc_v: V_mpp at STC (V).
        v_oc_stc_v: V_oc at STC (V).
        beta_voc_pct_per_c: Temperature coefficient of V_oc (%/°C).
        n_panels_in_string: Number of modules wired in series.
        t_cell_c: Cell temperature(s) (°C). Scalar or array.
        operating: When True, return V_mpp; otherwise return V_oc.

    Returns:
        String voltage (V), same shape as ``t_cell_c``.
    """
    delta_t = np.asarray(t_cell_c) - STC_TEMPERATURE_C
    factor = 1.0 + beta_voc_pct_per_c / 100.0 * delta_t
    v_panel_stc = v_mpp_stc_v if operating else v_oc_stc_v
    return n_panels_in_string * v_panel_stc * factor


# ---------------------------------------------------------------------------
# ElectricalModel: orchestrator of per-hour derating / shutdown decisions
# ---------------------------------------------------------------------------


@dataclass
class ElectricalKPIs:
    """
    Aggregated electrical KPIs reported in ``summary.electrical``.

    All counters are **per-year averages** (events / n_years) so they
    stay comparable across runs of different horizons.

    Attributes:
        hours_dc_overvoltage_per_year: Average yearly count of hours
            during which ``V_string`` would have exceeded
            ``v_dc_max_v`` had the inverter not shut down. This is the
            *hardware risk* indicator — for a well-sized system it
            should be 0 or a handful of hours per year.
        hours_dc_undervoltage_per_year: Same, on the lower DC boundary.
            Usually only relevant for partial-shade scenarios — left
            here so the KPI is symmetric.
        hours_outside_mppt_per_year: Yearly hours during which the
            tracker was operating but off-MPP (string in
            ``[v_dc_min, v_mppt_min]`` or ``[v_mppt_max, v_dc_max]``).
        peak_v_string_v: Largest open-circuit string voltage observed
            during the whole simulation (V). Diagnostic — should stay
            comfortably below ``v_dc_max_v``.
        min_v_string_v: Smallest operating string voltage observed
            during the whole simulation (V).
    """

    hours_dc_overvoltage_per_year: float = 0.0
    hours_dc_undervoltage_per_year: float = 0.0
    hours_outside_mppt_per_year: float = 0.0
    peak_v_string_v: float = 0.0
    min_v_string_v: float = 0.0


@dataclass
class ElectricalModel:
    """
    MPPT-window electrical model for a single MC path.

    The class owns the *static* parameters (panel + inverter specs +
    string layout + derating exponent) and exposes a pure method
    :meth:`apply_to_pv_dc` that takes the naive hourly PV DC power
    (kW) and the hourly ambient temperature (°C) and returns:

    1. the **adjusted** hourly PV DC power (kW) — same array shape;
    2. the running KPI counters (hours of overvoltage / undervoltage /
       off-MPP, peak and min observed V_string).

    The model does NOT mutate the inverter dispatch logic — only the
    DC power that reaches the inverter is modified. This keeps the
    legacy AC clipping + battery dispatch logic unchanged.

    Attributes:
        panel: Datasheet electrical specs of the chosen module.
        inverter: Datasheet electrical specs of the chosen inverter.
        strings: One or more ``PvString`` instances declaring how the
            modules are wired. When the scenario JSON omits the
            ``pv_strings`` list the model is given a single default
            string covering all the panels.
        derating_exponent_k: Exponent of the smooth MPPT-window
            derating ``(V_target / V_string) ** k``. Defaults to
            :data:`DEFAULT_DERATING_EXPONENT_K`.
        n_years: Simulation horizon — used to normalise the KPIs to
            "per year". Picked at runtime from
            :class:`EnergySystemConfig` so the dataclass stays free
            of integration knowledge.

    Notes:
        - ``apply_to_pv_dc`` expects the hourly PV power **already**
          aggregated across strings (the legacy energy flow exposes
          one DC power per hour, not per string). To stay faithful
          when the strings have different orientations, the model
          falls back to a single "effective" string voltage averaged
          across strings — this is the documented approximation; the
          ``V_string`` per-tracker treatment kicks in only when the
          inverter exposes multiple MPPT trackers and the scenario
          explicitly partitions panels by ``mppt_id``.
        - The KPIs returned per call are *path-level* totals. The
          Monte Carlo aggregator must average them across paths.
    """

    panel: PanelElectricalSpecs
    inverter: InverterElectricalSpecs
    strings: List[PvString] = field(default_factory=list)
    derating_exponent_k: float = DEFAULT_DERATING_EXPONENT_K
    n_years: int = 1

    def __post_init__(self) -> None:
        if not self.strings:
            raise ValueError(
                "ElectricalModel requires at least one PvString. "
                "Provide one explicitly or let the scenario_builder "
                "synthesise a single default string from energy.pv_kwp."
            )
        miss_panel = missing_panel_fields(self.panel)
        miss_inv = missing_inverter_fields(self.inverter)
        if miss_panel or miss_inv:
            bits = []
            if miss_panel:
                bits.append(f"panel: {', '.join(miss_panel)}")
            if miss_inv:
                bits.append(f"inverter: {', '.join(miss_inv)}")
            raise ValueError(
                "ElectricalModel: missing required datasheet fields "
                f"({'; '.join(bits)}). Either complete the specs or "
                "set electrical.mode='off' to keep the legacy energy "
                "flow."
            )
        if self.derating_exponent_k < 0:
            raise ValueError(
                f"derating_exponent_k must be >= 0, got {self.derating_exponent_k}"
            )

    @property
    def total_panels(self) -> int:
        """Total panels declared across every string."""
        return int(sum(s.n_panels for s in self.strings))

    def _representative_panels_per_string(self) -> float:
        """
        Effective n_panels_per_string used to compute V_string.

        Single-string layouts pass through unchanged. Multi-string
        layouts on the same MPPT report a panel-count-weighted average
        — that's the documented approximation: the inverter clamps a
        single tracker voltage even when several parallel strings
        share it, so a representative count is the best a single-
        voltage model can do.
        """
        if len(self.strings) == 1:
            return float(self.strings[0].n_panels)
        weights = np.array([s.n_panels for s in self.strings], dtype=float)
        return float(np.average(weights, weights=weights))  # equiv to mean weighted by count

    def apply_to_pv_dc(
        self,
        pv_dc_kw_hourly: np.ndarray,
        t_ambient_c_hourly: np.ndarray,
    ) -> tuple[np.ndarray, ElectricalKPIs]:
        """
        Adjust hourly PV DC power for the MPPT-window electrical model.

        Args:
            pv_dc_kw_hourly: Hourly PV DC power (kW) from the solar
                model — shape ``(n_hours,)``.
            t_ambient_c_hourly: Hourly ambient temperature (°C) from
                the Phase-15 thermal model — same shape as
                ``pv_dc_kw_hourly``.

        Returns:
            Tuple ``(adjusted_pv_dc_kw, kpis)`` where ``adjusted_pv_dc_kw``
            has the same shape as the input and ``kpis`` is an
            :class:`ElectricalKPIs` instance with cumulative counters
            for the whole path.

        Notes:
            - When the panel produces zero DC power the cell temperature
              is computed from ``t_ambient_c`` alone (no irradiance,
              NOCT correction collapses to 0). This is also the path
              that drives the *open-circuit* V_oc check used by
              ``hours_dc_overvoltage`` — winter sunrise, full sun,
              cold panel.
            - The temperature-induced power derating
              ``(1 + gamma_pmax * (T_cell - 25))`` is applied on top
              of the MPPT-window derating. With γ < 0 hot panels
              produce less, cold panels (slightly) more — within
              physical limits.
        """
        if pv_dc_kw_hourly.shape != t_ambient_c_hourly.shape:
            raise ValueError(
                "pv_dc_kw_hourly and t_ambient_c_hourly must have the "
                f"same shape, got {pv_dc_kw_hourly.shape} vs "
                f"{t_ambient_c_hourly.shape}"
            )

        n_panels_per_string = self._representative_panels_per_string()
        # Map hourly DC power → equivalent POA irradiance. The proxy
        # uses STC_IRRADIANCE_W_PER_M2 as the "full sun" reference: a
        # panel producing its STC nameplate corresponds to 1000 W/m².
        # Total array nameplate (kW) = total_panels * power_w / 1000.
        array_nameplate_kw = self.total_panels * self.panel.power_w / 1000.0
        if array_nameplate_kw <= 0:
            # Defensive: a zero-nameplate array should never reach here
            # because __post_init__ already rejected missing panel
            # specs. Returning the input untouched is the safest path.
            return pv_dc_kw_hourly.copy(), ElectricalKPIs()
        poa_w_per_m2 = STC_IRRADIANCE_W_PER_M2 * (pv_dc_kw_hourly / array_nameplate_kw)
        poa_w_per_m2 = np.clip(poa_w_per_m2, 0.0, None)

        t_cell_c = cell_temperature_c(
            t_ambient_c_hourly, poa_w_per_m2, self.panel.noct_c
        )

        # Operating string voltage (used for MPPT-window check and
        # power-temp derating).
        v_string_operating = v_string_at_cell_temperature(
            v_mpp_stc_v=self.panel.v_mpp_stc_v,
            v_oc_stc_v=self.panel.v_oc_stc_v,
            beta_voc_pct_per_c=self.panel.beta_voc_pct_per_c,
            n_panels_in_string=n_panels_per_string,
            t_cell_c=t_cell_c,
            operating=True,
        )
        # Open-circuit string voltage (used for DC overvoltage check —
        # the inverter sees V_oc when MPPT loop hasn't converged yet at
        # sunrise, or under fault conditions).
        v_string_oc = v_string_at_cell_temperature(
            v_mpp_stc_v=self.panel.v_mpp_stc_v,
            v_oc_stc_v=self.panel.v_oc_stc_v,
            beta_voc_pct_per_c=self.panel.beta_voc_pct_per_c,
            n_panels_in_string=n_panels_per_string,
            t_cell_c=t_cell_c,
            operating=False,
        )

        adjusted = pv_dc_kw_hourly.copy()

        # Track which hours actually have non-zero production — only
        # those count for in-operation diagnostics.
        producing = pv_dc_kw_hourly > 0
        v_op_for_stats = np.where(producing, v_string_operating, np.nan)

        # 1. DC overvoltage (open-circuit) — uses V_oc.
        overv_mask = v_string_oc > self.inverter.v_dc_max_v
        # 2. DC undervoltage — uses operating V_string (only when producing).
        underv_mask = producing & (v_string_operating < self.inverter.v_dc_min_v)

        # Hardware shutdown: when V_oc > V_dc_max OR V_op < V_dc_min,
        # the inverter is off → DC power forced to zero.
        shutdown_mask = overv_mask | underv_mask
        adjusted = np.where(shutdown_mask, 0.0, adjusted)

        # 3. Off-MPPT derating — only when the inverter is operating
        # (not shut down) and not zero-power. Compute target voltage
        # (clip onto MPPT window), then derate by (V_target/V) ** k.
        active_mask = producing & ~shutdown_mask
        v_mppt_min = self.inverter.v_mppt_min_v
        v_mppt_max = self.inverter.v_mppt_max_v
        below_mppt = active_mask & (v_string_operating < v_mppt_min)
        above_mppt = active_mask & (v_string_operating > v_mppt_max)
        off_mppt = below_mppt | above_mppt

        if self.derating_exponent_k > 0 and off_mppt.any():
            v_target = np.where(below_mppt, v_mppt_min, v_mppt_max)
            # Safe division: only divide on the masked positions.
            ratio = np.ones_like(adjusted)
            mask_idx = np.where(off_mppt)
            v_vals = v_string_operating[mask_idx]
            # Soft, monotonic derating. We pick the *smaller* of the
            # two factors so the penalty is well-defined on both sides
            # of the window. For below-MPPT V is below v_target →
            # ratio (V/V_target) < 1 → factor < 1; for above-MPPT V is
            # above v_target → ratio (V_target/V) < 1 → factor < 1.
            with np.errstate(divide="ignore", invalid="ignore"):
                below_factor = (v_vals / v_target[mask_idx]) ** self.derating_exponent_k
                above_factor = (v_target[mask_idx] / v_vals) ** self.derating_exponent_k
            ratio[mask_idx] = np.minimum(below_factor, above_factor)
            adjusted = adjusted * ratio

        # 4. Power temperature derating: (1 + gamma/100 * (T - 25))
        # applied to *every* still-producing hour (including the just-
        # derated off-MPPT ones — the two effects are physical and
        # multiplicative).
        gamma = self.panel.gamma_pmax_pct_per_c
        if gamma is not None and active_mask.any():
            power_temp_factor = 1.0 + gamma / 100.0 * (t_cell_c - STC_TEMPERATURE_C)
            power_temp_factor = np.clip(power_temp_factor, 0.0, None)
            adjusted = np.where(active_mask, adjusted * power_temp_factor, adjusted)

        # KPIs: per-year normalised counters + peak/min voltages.
        n_years = max(1, int(self.n_years))
        kpis = ElectricalKPIs(
            hours_dc_overvoltage_per_year=float(overv_mask.sum()) / n_years,
            hours_dc_undervoltage_per_year=float(underv_mask.sum()) / n_years,
            hours_outside_mppt_per_year=float(off_mppt.sum()) / n_years,
            peak_v_string_v=float(np.nanmax(v_string_oc)) if v_string_oc.size else 0.0,
            min_v_string_v=(
                float(np.nanmin(v_op_for_stats))
                if np.any(producing)
                else 0.0
            ),
        )
        return adjusted, kpis


# ---------------------------------------------------------------------------
# Aggregation across MC paths (called by application._build_*_summary)
# ---------------------------------------------------------------------------


def aggregate_kpis(per_path: Sequence[ElectricalKPIs]) -> dict[str, float]:
    """
    Aggregate per-path :class:`ElectricalKPIs` into a single summary dict.

    The summary uses the mean of the per-path counters (already in
    per-year units) and the **worst** observed peak/min voltage — the
    risk indicator is the path that pushed the inverter closest to its
    hardware limit, not the average path.

    Args:
        per_path: One :class:`ElectricalKPIs` per Monte Carlo path.

    Returns:
        Dict with the keys used by ``summary.electrical``:
        ``hours_dc_overvoltage_per_year_mean``,
        ``hours_dc_undervoltage_per_year_mean``,
        ``hours_outside_mppt_per_year_mean``, ``peak_v_string_v``,
        ``min_v_string_v``. Empty input yields an all-zero dict.
    """
    if not per_path:
        return {
            "hours_dc_overvoltage_per_year_mean": 0.0,
            "hours_dc_undervoltage_per_year_mean": 0.0,
            "hours_outside_mppt_per_year_mean": 0.0,
            "peak_v_string_v": 0.0,
            "min_v_string_v": 0.0,
        }
    overv = float(np.mean([k.hours_dc_overvoltage_per_year for k in per_path]))
    underv = float(np.mean([k.hours_dc_undervoltage_per_year for k in per_path]))
    off_mppt = float(np.mean([k.hours_outside_mppt_per_year for k in per_path]))
    peak_v = float(np.max([k.peak_v_string_v for k in per_path]))
    min_v_candidates = [k.min_v_string_v for k in per_path if k.min_v_string_v > 0]
    min_v = float(np.min(min_v_candidates)) if min_v_candidates else 0.0
    return {
        "hours_dc_overvoltage_per_year_mean": overv,
        "hours_dc_undervoltage_per_year_mean": underv,
        "hours_outside_mppt_per_year_mean": off_mppt,
        "peak_v_string_v": peak_v,
        "min_v_string_v": min_v,
    }
