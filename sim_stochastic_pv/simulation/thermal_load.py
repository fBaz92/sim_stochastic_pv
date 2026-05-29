"""
Phase 17 — Single-zone RC house model + heat pump for HVAC load coupling.

Adds a *physical* electric load whose magnitude depends on the
ambient temperature provided by the Phase-15 :class:`ThermalModel`. The
model is deliberately the simplest meaningful one — a 1st-order RC
house with a single heat-pump unit — so the user has 3-5 knobs (preset
insulation, floor area, COP, max power, setpoints) instead of 30
calibration parameters.

Output is **additive** on top of the baseline load profile: the user's
HVAC kW is summed with the appliance kW returned by
:meth:`LoadProfile.get_hourly_load_kw`.

Modes (selected by ``ThermalLoadConfig.dynamic``):
    - Steady-state (default): `P_thermal_req(h) = UA * (T_set - T_out(h))`
      → `P_elec(h) = |P_thermal_req(h)| / COP`. Capped at
      ``p_elec_max_kw``; over-cap hours are counted in the
      ``comfort_breach_hours`` KPI. The indoor temperature is the setpoint
      by assumption (no trajectory).
    - Dynamic RC (Phase 18, ``dynamic=True``): integrates
      ``C·dT_in/dt = Q_HVAC + Q_int - UA·(T_in - T_out)`` with implicit
      Euler at a 1-hour step and a deadbeat controller. Produces the real
      indoor-temperature trajectory (cached on
      ``HvacController.last_indoor_temp_c`` and summarised by the
      ``t_in_min_c`` / ``t_in_max_c`` KPIs) — the house drops below the
      heating setpoint when the heat pump saturates. With
      ``internal_gains_kw = 0`` and no capping it reduces exactly to the
      steady-state energy.

Out of scope (intentional for v1): multi-zone, domestic hot water,
demand response, hour-of-day tariff optimisation (the per-hour setpoint
arrays are schedule-ready for Phase 19), COP(T_out) curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Insulation presets — W/°C/m² (envelope total transmittance)
# ---------------------------------------------------------------------------

#: Pre-1980 buildings without retrofit. Typical Italian houses from
#: the 60s–70s with thin walls + single-pane windows.
PRESET_POOR_W_PER_C_PER_M2: float = 2.5

#: 1990s-2000s buildings: double-glazing + cavity-fill insulation.
PRESET_STANDARD_W_PER_C_PER_M2: float = 1.5

#: NZEB / Class-A buildings with continuous insulation, triple-glazing,
#: thermal bridge breaks.
PRESET_GOOD_W_PER_C_PER_M2: float = 0.8

INSULATION_PRESETS: dict[str, float] = {
    "poor": PRESET_POOR_W_PER_C_PER_M2,
    "standard": PRESET_STANDARD_W_PER_C_PER_M2,
    "good": PRESET_GOOD_W_PER_C_PER_M2,
}

#: Default thermal capacitance per square metre. Rough envelope-weighted
#: value for an inhabited apartment: ~0.05 kWh/°C/m² (heavy walls + light
#: furnishings). Exposed as an advanced parameter; the steady-state
#: model does not actually use it, but the future dynamic RC path will.
DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2: float = 0.05


# ---------------------------------------------------------------------------
# Datasheet specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HouseThermalConfig:
    """
    Single-zone RC house thermal config.

    Attributes:
        floor_area_m2: Building floor area (m²). Default 100. Drives
            both UA (via the preset W/°C/m²) and the future capacitance C.
        insulation_preset: One of ``"poor"``, ``"standard"``, ``"good"``
            (case-insensitive). Determines ``UA`` via
            :data:`INSULATION_PRESETS`. Custom value passes through
            ``ua_w_per_c_per_m2`` (if provided) overriding the preset.
        ua_w_per_c_per_m2: Override the preset envelope U-value
            (W/°C/m²) when the user has a specific number from an
            energy certificate. ``None`` means use the preset.
        capacitance_kwh_per_c_per_m2: Thermal capacitance per m². Used
            by the **dynamic RC mode** (Phase 18) to set the building's
            thermal inertia; unused by the steady-state path. Defaults to
            :data:`DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2`.
        internal_gains_kw: Constant free heat gains (occupants +
            appliances + solar through windows), in kW. Used **only** by
            the dynamic RC mode as an additive thermal source; the
            steady-state path ignores it. Defaults to ``0.0`` so the
            dynamic mode reduces *exactly* to the steady-state energy when
            the heat pump is not capped (no hidden offset). Typical
            inhabited-home value: 0.2–0.5 kW.

    Notes:
        - The total UA used by the model is
          ``floor_area_m2 * (ua_w_per_c_per_m2 OR preset value)`` divided
          by 1000 to convert from W/°C to kW/°C — keeping all energy
          flows in kW for consistency with the rest of the simulator.
        - The dataclass is frozen so it can be hashed / shared safely
          across paths.
    """

    floor_area_m2: float = 100.0
    insulation_preset: str = "standard"
    ua_w_per_c_per_m2: Optional[float] = None
    capacitance_kwh_per_c_per_m2: float = DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2
    internal_gains_kw: float = 0.0

    def __post_init__(self) -> None:
        if self.floor_area_m2 <= 0:
            raise ValueError(
                f"floor_area_m2 must be > 0, got {self.floor_area_m2}"
            )
        if self.capacitance_kwh_per_c_per_m2 < 0:
            raise ValueError(
                "capacitance_kwh_per_c_per_m2 must be >= 0, "
                f"got {self.capacitance_kwh_per_c_per_m2}"
            )
        if self.internal_gains_kw < 0:
            raise ValueError(
                f"internal_gains_kw must be >= 0, got {self.internal_gains_kw}"
            )
        preset_key = self.insulation_preset.lower()
        if self.ua_w_per_c_per_m2 is None and preset_key not in INSULATION_PRESETS:
            raise ValueError(
                f"insulation_preset={self.insulation_preset!r} not recognised. "
                f"Valid presets: {sorted(INSULATION_PRESETS)}."
            )
        if self.ua_w_per_c_per_m2 is not None and self.ua_w_per_c_per_m2 < 0:
            raise ValueError(
                "ua_w_per_c_per_m2 must be >= 0, "
                f"got {self.ua_w_per_c_per_m2}"
            )

    @property
    def ua_kw_per_c(self) -> float:
        """Total envelope UA in kW/°C (= W/°C / 1000)."""
        ua = self.ua_w_per_c_per_m2
        if ua is None:
            ua = INSULATION_PRESETS[self.insulation_preset.lower()]
        return ua * self.floor_area_m2 / 1000.0

    @property
    def capacitance_kwh_per_c(self) -> float:
        """
        Total building thermal capacitance in kWh/°C.

        Equals ``capacitance_kwh_per_c_per_m2 * floor_area_m2``. Drives the
        thermal inertia of the dynamic RC mode (Phase 18): the building's
        time constant is ``tau = C / UA`` (hours), so a poorly insulated
        100 m² home (UA ≈ 0.25 kW/°C, C ≈ 5 kWh/°C) has ``tau ≈ 20 h``.
        Unused by the steady-state path.
        """
        return self.capacitance_kwh_per_c_per_m2 * self.floor_area_m2


@dataclass(frozen=True)
class HeatPumpConfig:
    """
    Heat-pump electric/thermal characterisation.

    Attributes:
        cop_heating: Constant heating COP (default 3.5). The future
            optional ``cop_heating_curve`` will let the user supply a
            piecewise-linear lookup of ``[(T_out, COP), ...]`` for
            cold-weather realism; for v1 we stay with a constant.
        cop_cooling: Constant cooling COP (default 3.0).
        p_elec_max_kw: Maximum electrical power the heat pump draws
            (kW). The model caps the per-hour P_elec at this value
            and records over-cap hours as comfort breaches.
    """

    cop_heating: float = 3.5
    cop_cooling: float = 3.0
    p_elec_max_kw: float = 3.0

    def __post_init__(self) -> None:
        if self.cop_heating <= 0 or self.cop_cooling <= 0:
            raise ValueError(
                "cop_heating and cop_cooling must be > 0, got "
                f"{self.cop_heating=}, {self.cop_cooling=}"
            )
        if self.p_elec_max_kw <= 0:
            raise ValueError(
                f"p_elec_max_kw must be > 0, got {self.p_elec_max_kw}"
            )


@dataclass(frozen=True)
class SetpointConfig:
    """
    Comfort setpoints for the HVAC controller.

    Two modes coexist:

    - **Single setpoint** (the default): one ``t_setpoint_heating_c`` and one
      ``t_setpoint_cooling_c`` held all day. This is what every pre-Phase-19
      scenario uses and the JSON round-trip stays byte-identical when the
      schedules are absent.
    - **Time-of-day schedule** (Phase 19): two optional 24-entry arrays
      ``heating_schedule_c`` / ``cooling_schedule_c`` giving the setpoint for
      each hour of the day (index 0 = 00:00–01:00, … index 23 = 23:00–24:00).
      When present they override the scalar for the *home* hours; the scalars
      remain the representative single-number summary (and the fallback for
      whichever side is left ``None``). This is the seam
      :meth:`HvacController._build_setpoint_arrays` was prepared for in
      Phase 18 — e.g. a night setback that drops the heating target to 17 °C
      from 23:00 to 06:00 to save energy.

    Attributes:
        t_setpoint_heating_c: Indoor temperature target during heating
            mode (°C). Default 20. Whenever ``T_out < t_setpoint_heating_c``
            the heat pump runs in heating mode. Acts as the home-hours
            default when ``heating_schedule_c is None``.
        t_setpoint_cooling_c: Indoor temperature target during cooling
            mode (°C). Default 26. Whenever ``T_out > t_setpoint_cooling_c``
            the heat pump runs in cooling mode. Between heating and
            cooling setpoints is the dead-band: HVAC is off.
        t_setpoint_away_c: Optional fallback setpoint when the user is
            not at home. ``None`` (default) → HVAC off during away
            hours. A finite value (e.g. 16 in winter) keeps the house
            at a reduced setback so it does not freeze. The away policy
            takes precedence over the time-of-day schedule.
        heating_schedule_c: Optional 24-entry hour-of-day heating
            setpoints (°C). ``None`` (default) → use ``t_setpoint_heating_c``
            for every hour. Coerced to a ``tuple`` so the dataclass stays
            hashable.
        cooling_schedule_c: Optional 24-entry hour-of-day cooling
            setpoints (°C). ``None`` (default) → use ``t_setpoint_cooling_c``
            for every hour.

    Raises:
        ValueError: If a schedule does not have exactly 24 entries, or if
            at any hour the effective heating setpoint is not strictly
            below the cooling one (the dead-band invariant must hold
            hour-by-hour, not just for the scalar summary).
    """

    t_setpoint_heating_c: float = 20.0
    t_setpoint_cooling_c: float = 26.0
    t_setpoint_away_c: Optional[float] = None
    heating_schedule_c: Optional[tuple[float, ...]] = None
    cooling_schedule_c: Optional[tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if self.t_setpoint_heating_c >= self.t_setpoint_cooling_c:
            raise ValueError(
                "t_setpoint_heating_c must be strictly < "
                "t_setpoint_cooling_c to leave room for a dead-band, "
                f"got heating={self.t_setpoint_heating_c}, "
                f"cooling={self.t_setpoint_cooling_c}"
            )
        # Coerce any provided schedule to a 24-tuple of floats (keeps the
        # frozen dataclass hashable and validates the length at the boundary).
        for field_name in ("heating_schedule_c", "cooling_schedule_c"):
            sched = getattr(self, field_name)
            if sched is None:
                continue
            coerced = tuple(float(x) for x in sched)
            if len(coerced) != 24:
                raise ValueError(
                    f"{field_name} must have exactly 24 hour-of-day entries, "
                    f"got {len(coerced)}"
                )
            object.__setattr__(self, field_name, coerced)
        # Per-hour dead-band invariant, mixing schedules with the scalar
        # fallback for whichever side is None.
        if self.heating_schedule_c is not None or self.cooling_schedule_c is not None:
            heat = self.heating_schedule_c or ((self.t_setpoint_heating_c,) * 24)
            cool = self.cooling_schedule_c or ((self.t_setpoint_cooling_c,) * 24)
            for hour in range(24):
                if heat[hour] >= cool[hour]:
                    raise ValueError(
                        "heating setpoint must be strictly < cooling setpoint "
                        f"at hour {hour}, got heating={heat[hour]}, "
                        f"cooling={cool[hour]}"
                    )


@dataclass(frozen=True)
class ThermalLoadConfig:
    """
    Top-level config for the Phase-17 HVAC additive load.

    Attributes:
        enabled: Toggle. When False the simulator must not even build
            the controller — the additive HVAC kW path stays at zero.
        house: :class:`HouseThermalConfig` — envelope thermal
            characterisation.
        heat_pump: :class:`HeatPumpConfig` — COP + p_elec_max.
        setpoint: :class:`SetpointConfig` — comfort targets.
        dynamic: Reserved flag for the future dynamic RC integration.
            For now the simulator always runs the steady-state path
            even when ``dynamic=True``; we keep the flag so the JSON
            round-trip stays symmetric.
    """

    enabled: bool = False
    house: HouseThermalConfig = field(default_factory=HouseThermalConfig)
    heat_pump: HeatPumpConfig = field(default_factory=HeatPumpConfig)
    setpoint: SetpointConfig = field(default_factory=SetpointConfig)
    dynamic: bool = False


# ---------------------------------------------------------------------------
# Per-path KPIs (aggregated across paths in :func:`aggregate_thermal_kpis`)
# ---------------------------------------------------------------------------


@dataclass
class ThermalLoadKPIs:
    """
    Path-level HVAC KPIs collected by the simulator.

    All counters are normalised to *per year* — comparable across runs
    of different horizons.

    Attributes:
        hvac_kwh_annual: Yearly HVAC electrical energy (kWh/yr).
        hvac_share_of_total_load_pct: Fraction of the total household
            load that came from the HVAC (heating + cooling), expressed
            as a percentage (0–100).
        comfort_breach_hours_per_year: Yearly hours during which the
            heat pump was capped at ``p_elec_max_kw`` and therefore
            unable to maintain the setpoint at steady state. KPI of
            **dimensioning adequacy**; >100 indicates an undersized
            heat pump for the building envelope.
        p_elec_hvac_peak_kw: Largest instantaneous HVAC electric draw
            seen in this path (kW). Useful for contract sizing (kW
            tariff slabs).
        t_in_min_c: Minimum indoor temperature reached in this path (°C).
            In **steady-state** mode the indoor temperature is pinned at
            the setpoint by assumption, so this is set to
            ``t_setpoint_heating_c``. In **dynamic RC** mode it is the
            true minimum of the integrated indoor-temperature series — it
            drops below the heating setpoint when the heat pump saturates.
        t_in_max_c: Maximum indoor temperature reached in this path (°C).
            In steady-state mode set to ``t_setpoint_cooling_c``; in
            dynamic mode the true maximum (rises above the cooling setpoint
            when cooling capacity is insufficient).
    """

    hvac_kwh_annual: float = 0.0
    hvac_share_of_total_load_pct: float = 0.0
    comfort_breach_hours_per_year: float = 0.0
    p_elec_hvac_peak_kw: float = 0.0
    t_in_min_c: float = 0.0
    t_in_max_c: float = 0.0


# ---------------------------------------------------------------------------
# HVAC controller (steady-state + dynamic RC)
# ---------------------------------------------------------------------------

#: Integration step of the dynamic RC mode, in hours. The simulator works
#: on hourly series, so the implicit-Euler step is fixed at 1 hour. Kept as
#: a named constant (not an inline ``1.0``) so the unit is explicit in the
#: ``C/dt`` term of :meth:`HvacController._compute_dynamic`.
_HOUR_STEP_H: float = 1.0


class HvacController:
    """
    HVAC controller (steady-state or dynamic RC) — one per Monte Carlo path.

    Reads the hourly ambient temperature array, the path's per-hour
    occupancy mask (the user is at home YES/NO), and the configured
    house/heat-pump/setpoint specs, then computes the **additive
    electric load** ``P_elec_HVAC(h)`` for every hour.

    Two modes, selected by :attr:`ThermalLoadConfig.dynamic`:

    - **Steady-state** (``dynamic=False``, default): the inverse of the RC
      balance assuming the house is held *at* the setpoint —
      ``P_th = UA * (T_set - T_out)`` → ``P_elec = |P_th| / COP``, capped at
      ``p_elec_max_kw``. The indoor temperature is the setpoint by
      assumption (no trajectory).
    - **Dynamic RC** (``dynamic=True``, Phase 18): integrates the 1st-order
      RC balance ``C·dT_in/dt = Q_HVAC + Q_int - UA·(T_in - T_out)`` with
      **implicit Euler** at a 1-hour step, using a *deadbeat* controller
      (drive to setpoint in one step when possible, otherwise saturate at
      ``p_elec_max·COP``). This produces the real indoor-temperature
      trajectory — it drops below the heating setpoint when the heat pump
      can't keep up with the losses of a poorly insulated house.

    The two modes are reconciled by design: with ``internal_gains_kw=0`` and
    the heat pump never capped, the dynamic mode's electric energy is
    **identical** to the steady-state result (the deadbeat controller holds
    the setpoint exactly). They diverge only when the pump saturates or the
    thermal mass buffers a transient.

    The controller does not own the calendar — it expects pre-aligned
    hourly arrays. The :class:`EnergySystemSimulator` is responsible for
    building those arrays from ``ThermalModel.simulate_daily_means(...)``
    and the :class:`LoadProfile`'s home/away day picks.

    Attributes:
        config: :class:`ThermalLoadConfig` — the full spec set.
        ua_kw_per_c: Derived envelope U·A in kW/°C (= W/°C/1000).
        last_indoor_temp_c: The indoor-temperature series (°C) from the
            most recent :meth:`compute_hourly_p_elec_kw` call. Populated
            only in **dynamic** mode (``None`` after a steady-state call).
            Cached for the future timeseries-preview endpoint.
    """

    def __init__(self, config: ThermalLoadConfig) -> None:
        if not config.enabled:
            raise ValueError(
                "HvacController should not be instantiated when "
                "ThermalLoadConfig.enabled is False. The scenario_builder "
                "is responsible for skipping the instantiation."
            )
        self.config = config
        self.ua_kw_per_c = config.house.ua_kw_per_c
        self.last_indoor_temp_c: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public entry point — dispatches on config.dynamic
    # ------------------------------------------------------------------

    def compute_hourly_p_elec_kw(
        self,
        t_ambient_hourly_c: np.ndarray,
        at_home_hourly: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ThermalLoadKPIs]:
        """
        Compute the hourly electric HVAC draw for the entire path.

        Dispatches to the steady-state or the dynamic RC integrator based
        on :attr:`ThermalLoadConfig.dynamic`. Both return the same
        ``(p_elec_kw_hourly, kpis)`` shape so the simulator call-site is
        mode-agnostic.

        Args:
            t_ambient_hourly_c: Ambient temperature (°C) per hour for
                the simulation horizon. Shape ``(n_hours,)``.
            at_home_hourly: Optional boolean array of the same shape
                marking the hours the user is at home. ``None`` (the
                default) treats all hours as occupied — useful for
                unit tests and for profiles that don't model occupancy.

        Returns:
            ``(p_elec_kw_hourly, kpis)``. ``p_elec_kw_hourly`` has the
            same shape as the input; ``kpis`` is a fresh
            :class:`ThermalLoadKPIs` instance with the path-level
            counters. The simulator finalises ``hvac_share_of_total_load_pct``
            (left at 0.0 here) against the baseline load it owns.

        Notes:
            - When the user is away and ``t_setpoint_away_c is None`` the
              HVAC is off (P_elec = 0) for that hour in both modes.
            - When the user is away and ``t_setpoint_away_c`` is set, that
              single setback setpoint is used for both heating and cooling.
        """
        n_hours = t_ambient_hourly_c.shape[0]
        if at_home_hourly is None:
            at_home_hourly = np.ones(n_hours, dtype=bool)
        if at_home_hourly.shape != t_ambient_hourly_c.shape:
            raise ValueError(
                "at_home_hourly and t_ambient_hourly_c must have the "
                f"same shape, got {at_home_hourly.shape} vs "
                f"{t_ambient_hourly_c.shape}"
            )

        t_set_heating, t_set_cooling = self._build_setpoint_arrays(at_home_hourly)

        if self.config.dynamic:
            return self._compute_dynamic(
                t_ambient_hourly_c, t_set_heating, t_set_cooling
            )
        return self._compute_steady_state(
            t_ambient_hourly_c, t_set_heating, t_set_cooling
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def setpoint_arrays(
        self,
        at_home_hourly: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Public accessor for the per-hour heating/cooling setpoint arrays.

        Same payload :meth:`compute_hourly_p_elec_kw` consumes internally,
        exposed so previews and the thermal lab (Phase 19) can draw the
        "setpoint vs indoor temperature" line without re-deriving the
        occupancy/schedule logic.

        Args:
            at_home_hourly: Boolean occupancy mask, shape ``(n_hours,)``.

        Returns:
            ``(t_set_heating, t_set_cooling)`` each shape ``(n_hours,)`` in °C
            (with ``±inf`` on away hours that have no setback setpoint).
        """
        return self._build_setpoint_arrays(at_home_hourly)

    def _build_setpoint_arrays(
        self,
        at_home_hourly: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the per-hour heating/cooling setpoint arrays.

        Encodes occupancy, the optional time-of-day schedule (Phase 19) and
        the away policy as plain float arrays so both modes can consume them
        identically (the dynamic loop reads them scalar-by-scalar, the
        steady-state path vectorises over them):

        - HOME → the hour-of-day schedule value if a schedule is set, else
          the scalar ``(t_setpoint_heating_c, t_setpoint_cooling_c)``. The
          hour-of-day is ``index % 24`` because the simulator's hourly arrays
          start at hour 0 of day 0 (see :meth:`ThermalModel.to_hourly`).
        - AWAY with ``t_setpoint_away_c`` set → ``(away, away)`` (single
          setback target for both modes); the away policy overrides the
          schedule.
        - AWAY with ``t_setpoint_away_c is None`` → ``(-inf, +inf)``, i.e.
          the whole temperature range is dead-band ⇒ HVAC off for that hour.

        Returns:
            ``(t_set_heating, t_set_cooling)`` each shape ``(n_hours,)``.
        """
        sp = self.config.setpoint
        n_hours = at_home_hourly.shape[0]
        hour_of_day = np.arange(n_hours) % 24

        if sp.heating_schedule_c is not None:
            home_heating = np.asarray(sp.heating_schedule_c, dtype=float)[hour_of_day]
        else:
            home_heating = np.full(n_hours, sp.t_setpoint_heating_c, dtype=float)
        if sp.cooling_schedule_c is not None:
            home_cooling = np.asarray(sp.cooling_schedule_c, dtype=float)[hour_of_day]
        else:
            home_cooling = np.full(n_hours, sp.t_setpoint_cooling_c, dtype=float)

        if sp.t_setpoint_away_c is None:
            away_heating: float | np.ndarray = -np.inf
            away_cooling: float | np.ndarray = np.inf
        else:
            away_heating = away_cooling = float(sp.t_setpoint_away_c)

        t_set_heating = np.where(at_home_hourly, home_heating, away_heating)
        t_set_cooling = np.where(at_home_hourly, home_cooling, away_cooling)
        return t_set_heating.astype(float), t_set_cooling.astype(float)

    @staticmethod
    def _n_years_from_hours(n_hours: int) -> int:
        """Number of whole years represented by ``n_hours`` (>= 1)."""
        return max(1, int(round(n_hours / (365.0 * 24.0))))

    # ------------------------------------------------------------------
    # Steady-state path
    # ------------------------------------------------------------------

    def _compute_steady_state(
        self,
        t_ambient_hourly_c: np.ndarray,
        t_set_heating: np.ndarray,
        t_set_cooling: np.ndarray,
    ) -> tuple[np.ndarray, ThermalLoadKPIs]:
        """
        Steady-state HVAC draw: instantaneous inverse of the RC balance.

        ``P_th = UA·(T_set - T_out)`` (heating, positive) /
        ``UA·(T_out - T_set)`` (cooling, positive), ``P_elec = P_th / COP``,
        capped at ``p_elec_max_kw`` with a comfort-breach count. The indoor
        temperature is the setpoint by assumption, so the KPI exposes the
        configured heating/cooling setpoints as ``t_in_min_c`` / ``t_in_max_c``.
        """
        hp = self.config.heat_pump
        sp = self.config.setpoint
        ua = self.ua_kw_per_c
        n_hours = t_ambient_hourly_c.shape[0]

        heating_mask = t_ambient_hourly_c < t_set_heating
        cooling_mask = t_ambient_hourly_c > t_set_cooling

        p_thermal_heating = ua * np.maximum(0.0, t_set_heating - t_ambient_hourly_c)
        p_thermal_cooling = ua * np.maximum(0.0, t_ambient_hourly_c - t_set_cooling)

        p_elec_heating = np.where(heating_mask, p_thermal_heating / hp.cop_heating, 0.0)
        p_elec_cooling = np.where(cooling_mask, p_thermal_cooling / hp.cop_cooling, 0.0)
        p_elec_uncapped = p_elec_heating + p_elec_cooling

        breach_mask = p_elec_uncapped > hp.p_elec_max_kw
        p_elec_kw_hourly = np.minimum(p_elec_uncapped, hp.p_elec_max_kw)

        self.last_indoor_temp_c = None  # no trajectory in steady-state
        n_years = self._n_years_from_hours(n_hours)
        kpis = ThermalLoadKPIs(
            hvac_kwh_annual=float(p_elec_kw_hourly.sum()) / n_years,
            hvac_share_of_total_load_pct=0.0,  # finalised by the simulator
            comfort_breach_hours_per_year=float(breach_mask.sum()) / n_years,
            p_elec_hvac_peak_kw=float(p_elec_kw_hourly.max()) if n_hours else 0.0,
            t_in_min_c=sp.t_setpoint_heating_c,
            t_in_max_c=sp.t_setpoint_cooling_c,
        )
        return p_elec_kw_hourly, kpis

    # ------------------------------------------------------------------
    # Dynamic RC path (Phase 18)
    # ------------------------------------------------------------------

    def _compute_dynamic(
        self,
        t_ambient_hourly_c: np.ndarray,
        t_set_heating: np.ndarray,
        t_set_cooling: np.ndarray,
    ) -> tuple[np.ndarray, ThermalLoadKPIs]:
        """
        Dynamic single-zone RC integration with implicit Euler.

        Per hour the implicit-Euler update of ``C·dT/dt = Q + Q_int - UA·(T - T_out)``
        with step ``dt = 1 h`` reads (``a = C/dt + UA``)::

            T_free = (C/dt·T_prev + Q_int + UA·T_out) / a     # HVAC off
            heating if T_free < T_set_heat ; cooling if T_free > T_set_cool
            Q_need(heat) = a·T_set_heat - (C/dt·T_prev + Q_int + UA·T_out)
            Q = clip(Q_need, 0, p_elec_max·COP)               # thermal kW
            T_new = (C/dt·T_prev + Q_int + UA·T_out ± Q) / a

        The controller is *deadbeat* (drives to setpoint in one step when
        the pump can deliver). When ``Q_need`` exceeds the cap the hour is
        a comfort breach and ``T_new`` falls short of (heating) / overshoots
        (cooling) the setpoint — the visible "the house gets cold" effect.

        The indoor series is cached on :attr:`last_indoor_temp_c`. The
        integration starts from the home heating setpoint (a comfortable,
        neutral initial condition); the choice washes out within a few
        time constants.

        Notes:
            - Implicit Euler is unconditionally stable, so the short time
              constant of a poorly insulated home (``tau = C/UA`` can be a
              handful of hours) never causes numerical oscillation.
            - This path is a sequential per-hour loop (the state couples
              consecutive hours) — comparable in cost to the AR(1) loop of
              :meth:`ThermalModel.simulate_daily_means` but at hourly
              resolution. It is opt-in (``dynamic=True``).
        """
        house = self.config.house
        hp = self.config.heat_pump
        sp = self.config.setpoint
        ua = self.ua_kw_per_c
        n_hours = t_ambient_hourly_c.shape[0]

        c_over_dt = house.capacitance_kwh_per_c / _HOUR_STEP_H  # kW/°C
        if c_over_dt <= 0.0:
            # Zero thermal mass degenerates to the steady-state response;
            # delegate to keep a single, well-tested code path.
            return self._compute_steady_state(
                t_ambient_hourly_c, t_set_heating, t_set_cooling
            )
        q_int = house.internal_gains_kw
        a = c_over_dt + ua
        q_max_heat = hp.p_elec_max_kw * hp.cop_heating
        q_max_cool = hp.p_elec_max_kw * hp.cop_cooling
        cop_h = hp.cop_heating
        cop_c = hp.cop_cooling

        t_in = np.empty(n_hours, dtype=float)
        p_elec = np.empty(n_hours, dtype=float)
        breach = 0
        t_prev = float(sp.t_setpoint_heating_c)

        for h in range(n_hours):
            t_out = t_ambient_hourly_c[h]
            base = c_over_dt * t_prev + q_int + ua * t_out  # kW (no HVAC)
            t_free = base / a
            if t_free < t_set_heating[h]:  # heating
                q_need = a * t_set_heating[h] - base
                if q_need > q_max_heat:
                    q = q_max_heat
                    breach += 1
                else:
                    q = q_need
                p_elec[h] = q / cop_h
                t_new = (base + q) / a
            elif t_free > t_set_cooling[h]:  # cooling
                q_need = base - a * t_set_cooling[h]
                if q_need > q_max_cool:
                    q = q_max_cool
                    breach += 1
                else:
                    q = q_need
                p_elec[h] = q / cop_c
                t_new = (base - q) / a
            else:  # dead-band → HVAC off, free run
                p_elec[h] = 0.0
                t_new = t_free
            t_in[h] = t_new
            t_prev = t_new

        self.last_indoor_temp_c = t_in
        n_years = self._n_years_from_hours(n_hours)
        kpis = ThermalLoadKPIs(
            hvac_kwh_annual=float(p_elec.sum()) / n_years,
            hvac_share_of_total_load_pct=0.0,  # finalised by the simulator
            comfort_breach_hours_per_year=float(breach) / n_years,
            p_elec_hvac_peak_kw=float(p_elec.max()) if n_hours else 0.0,
            t_in_min_c=float(t_in.min()) if n_hours else sp.t_setpoint_heating_c,
            t_in_max_c=float(t_in.max()) if n_hours else sp.t_setpoint_cooling_c,
        )
        return p_elec, kpis


# ---------------------------------------------------------------------------
# Aggregation across MC paths
# ---------------------------------------------------------------------------


def aggregate_thermal_kpis(per_path: list) -> dict:
    """
    Aggregate per-path :class:`ThermalLoadKPIs` into a summary dict.

    Uses **mean** for the kWh + share + peak (these are well-behaved
    averages) and **mean** for the comfort-breach counter (already
    per-year). For the indoor-temperature KPIs it takes the **worst case**
    across paths — the coldest ``t_in_min_c`` and the hottest ``t_in_max_c``
    — mirroring the peak-voltage convention of
    :func:`electrical.aggregate_kpis`: the risk is the worst path, not the
    average one. These are only meaningful when the dynamic RC mode is
    active; in steady-state every path reports its setpoints, so the
    aggregate collapses to the setpoints. Empty input → all-zero dict so
    the API can serialise a stable schema even when the user disables the
    feature.
    """
    if not per_path:
        return {
            "hvac_kwh_annual_mean": 0.0,
            "hvac_share_of_total_load_pct_mean": 0.0,
            "comfort_breach_hours_per_year_mean": 0.0,
            "p_elec_hvac_peak_kw_mean": 0.0,
            "t_in_min_c": 0.0,
            "t_in_max_c": 0.0,
        }
    return {
        "hvac_kwh_annual_mean": float(
            np.mean([k.hvac_kwh_annual for k in per_path])
        ),
        "hvac_share_of_total_load_pct_mean": float(
            np.mean([k.hvac_share_of_total_load_pct for k in per_path])
        ),
        "comfort_breach_hours_per_year_mean": float(
            np.mean([k.comfort_breach_hours_per_year for k in per_path])
        ),
        "p_elec_hvac_peak_kw_mean": float(
            np.mean([k.p_elec_hvac_peak_kw for k in per_path])
        ),
        "t_in_min_c": float(np.min([k.t_in_min_c for k in per_path])),
        "t_in_max_c": float(np.max([k.t_in_max_c for k in per_path])),
    }
