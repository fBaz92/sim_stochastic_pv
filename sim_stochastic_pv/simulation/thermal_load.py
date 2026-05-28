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

Modes:
    - Steady-state (default): `P_thermal_req(h) = UA * (T_set - T_out(h))`
      → `P_elec(h) = |P_thermal_req(h)| / COP`. Capped at
      ``p_elec_max_kw``; over-cap hours are counted in the
      ``comfort_breach_hours`` KPI.
    - Dynamic RC (Phase 17.x, scheduled but not implemented yet): solve
      the ODE with implicit Euler over hourly steps. The wiring is
      documented in the docstrings but the implementation is a stub —
      users get the steady-state result in the meantime.

Out of scope (intentional for v1): multi-zone, domestic hot water,
demand response, hour-of-day tariff optimisation, set-point ramping.
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
            by the future dynamic RC mode; unused by the steady-state
            path. Defaults to :data:`DEFAULT_CAPACITANCE_KWH_PER_C_PER_M2`.

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

    Attributes:
        t_setpoint_heating_c: Indoor temperature target during heating
            mode (°C). Default 20. Whenever ``T_out < t_setpoint_heating_c``
            the heat pump runs in heating mode.
        t_setpoint_cooling_c: Indoor temperature target during cooling
            mode (°C). Default 26. Whenever ``T_out > t_setpoint_cooling_c``
            the heat pump runs in cooling mode. Between heating and
            cooling setpoints is the dead-band: HVAC is off.
        t_setpoint_away_c: Optional fallback setpoint when the user is
            not at home. ``None`` (default) → HVAC off during away
            hours. A finite value (e.g. 16 in winter) keeps the house
            at a reduced setback so it does not freeze.
    """

    t_setpoint_heating_c: float = 20.0
    t_setpoint_cooling_c: float = 26.0
    t_setpoint_away_c: Optional[float] = None

    def __post_init__(self) -> None:
        if self.t_setpoint_heating_c >= self.t_setpoint_cooling_c:
            raise ValueError(
                "t_setpoint_heating_c must be strictly < "
                "t_setpoint_cooling_c to leave room for a dead-band, "
                f"got heating={self.t_setpoint_heating_c}, "
                f"cooling={self.t_setpoint_cooling_c}"
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
    """

    hvac_kwh_annual: float = 0.0
    hvac_share_of_total_load_pct: float = 0.0
    comfort_breach_hours_per_year: float = 0.0
    p_elec_hvac_peak_kw: float = 0.0


# ---------------------------------------------------------------------------
# Steady-state controller
# ---------------------------------------------------------------------------


class HvacController:
    """
    Steady-state HVAC controller — one instance per Monte Carlo path.

    Reads the hourly ambient temperature array, the path's per-hour
    occupancy mask (the user is at home YES/NO), and the configured
    house/heat-pump/setpoint specs, then computes the **additive
    electric load** ``P_elec_HVAC(h)`` for every hour.

    The controller does not own the calendar — it expects pre-aligned
    hourly arrays. The :class:`EnergySystemSimulator` is responsible for
    building those arrays from ``ThermalModel.simulate_daily_means(...)``
    and the :class:`LoadProfile`'s home/away day picks.

    Attributes:
        config: :class:`ThermalLoadConfig` — the full v1 spec set.
        ua_kw_per_c: Derived envelope U·A in kW/°C (= W/°C/1000).
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

    # ------------------------------------------------------------------
    # Hourly arrays
    # ------------------------------------------------------------------

    def compute_hourly_p_elec_kw(
        self,
        t_ambient_hourly_c: np.ndarray,
        at_home_hourly: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ThermalLoadKPIs]:
        """
        Compute the hourly electric HVAC draw for the entire path.

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
            counters. The simulator is responsible for normalising
            ``hvac_share_of_total_load_pct`` against the baseline load
            (which the controller doesn't see). The KPI returned here
            therefore leaves that field at 0.0 — the
            :class:`EnergySystemSimulator` finalises it.

        Notes:
            - Steady-state formula:
              ``P_th = UA * (T_set - T_out)`` (heating, positive),
              ``P_th = UA * (T_out - T_set)`` (cooling, positive).
            - ``P_elec = P_th / COP``, then capped at ``p_elec_max_kw``
              with a comfort-breach count.
            - When the user is away and ``t_setpoint_away_c is None`` the
              HVAC is off (P_elec = 0) regardless of the outdoor
              temperature — the heat pump simply doesn't run.
            - When the user is away and ``t_setpoint_away_c`` is set,
              the controller uses that (single) setpoint for both
              heating and cooling — the simplification is intentional
              for v1 (a real setback would have a separate cooling
              setpoint, usually higher).
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

        cfg = self.config
        sp = cfg.setpoint
        hp = cfg.heat_pump
        ua = self.ua_kw_per_c

        # 1) When the user is HOME → standard heating / cooling setpoints.
        # 2) When AWAY:
        #    - if t_setpoint_away_c is None → HVAC off (target = T_out so
        #      the steady-state demand is zero);
        #    - else → use the away setpoint for both modes.
        t_set_heating = np.where(
            at_home_hourly,
            sp.t_setpoint_heating_c,
            (
                t_ambient_hourly_c
                if sp.t_setpoint_away_c is None
                else sp.t_setpoint_away_c
            ),
        )
        t_set_cooling = np.where(
            at_home_hourly,
            sp.t_setpoint_cooling_c,
            (
                t_ambient_hourly_c
                if sp.t_setpoint_away_c is None
                else sp.t_setpoint_away_c
            ),
        )

        heating_mask = t_ambient_hourly_c < t_set_heating
        cooling_mask = t_ambient_hourly_c > t_set_cooling

        # Steady-state thermal power demand (positive in both modes).
        p_thermal_heating = ua * np.maximum(0.0, t_set_heating - t_ambient_hourly_c)
        p_thermal_cooling = ua * np.maximum(0.0, t_ambient_hourly_c - t_set_cooling)

        p_elec_heating = np.where(heating_mask, p_thermal_heating / hp.cop_heating, 0.0)
        p_elec_cooling = np.where(cooling_mask, p_thermal_cooling / hp.cop_cooling, 0.0)
        p_elec_uncapped = p_elec_heating + p_elec_cooling

        # Cap at p_elec_max_kw; count comfort breaches.
        breach_mask = p_elec_uncapped > hp.p_elec_max_kw
        p_elec_kw_hourly = np.minimum(p_elec_uncapped, hp.p_elec_max_kw)

        # Per-year KPIs.
        n_years = max(1, int(round(n_hours / (365.0 * 24.0))))
        kpis = ThermalLoadKPIs(
            hvac_kwh_annual=float(p_elec_kw_hourly.sum()) / n_years,
            hvac_share_of_total_load_pct=0.0,  # finalised by the simulator
            comfort_breach_hours_per_year=float(breach_mask.sum()) / n_years,
            p_elec_hvac_peak_kw=float(p_elec_kw_hourly.max()) if n_hours else 0.0,
        )
        return p_elec_kw_hourly, kpis


# ---------------------------------------------------------------------------
# Aggregation across MC paths
# ---------------------------------------------------------------------------


def aggregate_thermal_kpis(per_path: list) -> dict:
    """
    Aggregate per-path :class:`ThermalLoadKPIs` into a summary dict.

    Uses **mean** for the kWh + share + peak (these are well-behaved
    averages) and **mean** for the comfort-breach counter (already
    per-year). Empty input → all-zero dict so the API can serialise
    a stable schema even when the user disables the feature.
    """
    if not per_path:
        return {
            "hvac_kwh_annual_mean": 0.0,
            "hvac_share_of_total_load_pct_mean": 0.0,
            "comfort_breach_hours_per_year_mean": 0.0,
            "p_elec_hvac_peak_kw_mean": 0.0,
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
    }
