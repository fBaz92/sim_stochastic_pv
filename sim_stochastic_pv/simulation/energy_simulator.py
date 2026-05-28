"""
Hourly PV + battery + load simulation engine.

The :class:`EnergySystemSimulator` generates one Monte Carlo path by coupling
solar production, household load, inverter dispatch, and the battery bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..calendar_utils import build_calendar
from .battery import BatteryBank, BatterySpecs
from .electrical import ElectricalKPIs, ElectricalModel
from .inverter import InverterAC
from .load_profiles import LoadProfile, StochasticLoadConfig, StochasticLoadProfile
from .solar import SolarModel
from .thermal import ThermalModel
from .thermal_load import HvacController, ThermalLoadConfig, ThermalLoadKPIs


@dataclass
class EnergySystemConfig:
    """
    Configuration parameters for the energy system simulation.
    
    Attributes:
        n_years: Number of years to simulate.
        pv_kwp: PV system capacity in kWp.
        battery_specs: Battery specifications (capacity and cycle life).
        n_batteries: Number of batteries in the bank.
        eta_charge: Battery charging efficiency (0-1).
        eta_discharge: Battery discharging efficiency (0-1).
        inverter_p_ac_max_kw: Maximum AC power output of the inverter in kW.
        battery_max_charge_kw: Maximum charging power per battery in kW (None for unlimited).
        battery_max_discharge_kw: Maximum discharging power per battery in kW (None for unlimited).
        dt_hours: Time step duration in hours.
        calendar_start_weekday: Starting weekday for the calendar (0=Monday, 6=Sunday).
    """
    n_years: int = 20
    pv_kwp: float = 2.0
    battery_specs: BatterySpecs = field(default_factory=lambda: BatterySpecs(capacity_kwh=1.92, cycles_life=6000))
    n_batteries: int = 2
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    inverter_p_ac_max_kw: float = 0.8
    inverter_p_dc_max_kw: float | None = None
    battery_max_charge_kw: float | None = 2.0
    battery_max_discharge_kw: float | None = None
    dt_hours: float = 1.0
    calendar_start_weekday: int = 0
    # Phase 16 — opt-in detailed electrical model. When ``electrical_model``
    # is None (default) the simulator runs the legacy energy path
    # byte-identical to pre-Phase-16. When provided, the simulator also
    # requires a ``thermal_model`` so it can compute hourly cell
    # temperatures and apply the MPPT-window derating + DC overvoltage
    # shutdown logic to the PV power before it reaches the inverter.
    electrical_model: ElectricalModel | None = None
    thermal_model: ThermalModel | None = None
    # Phase 17 — opt-in stochastic load decorator. When ``enabled=False``
    # (default) the wrapper is skipped and the baseline LoadProfile
    # behaves byte-identically to pre-Phase-17.
    stochastic_load_config: StochasticLoadConfig | None = None
    # Phase 17 — opt-in thermal (HVAC) additive load. Requires a
    # thermal_model to source hourly ambient temperatures. ``None`` or
    # ``enabled=False`` keeps the legacy load path intact.
    thermal_load_config: ThermalLoadConfig | None = None


class EnergySystemSimulator:
    """
    Simulates one Monte Carlo path of the PV + battery + load system with hourly resolution.
    """

    def __init__(
        self,
        config: EnergySystemConfig,
        solar_model: SolarModel,
        load_profile: LoadProfile,
    ) -> None:
        """
        Initialize the energy system simulator.
        
        Args:
            config: Energy system configuration parameters.
            solar_model: Solar PV production model.
            load_profile: Load profile model for electricity consumption.
        """
        self.config = config
        self.solar_model = solar_model
        # Phase 17 — wrap the base LoadProfile with the stochastic
        # decorator when the scenario enables it. The wrapper preserves
        # the long-run mean by construction (LogN with Itō correction)
        # so the wrapped profile keeps the user's energy budget.
        stoch_cfg = config.stochastic_load_config
        if stoch_cfg is not None and stoch_cfg.enabled and stoch_cfg.sigma_log > 0:
            self.load_profile = StochasticLoadProfile(load_profile, stoch_cfg)
        else:
            self.load_profile = load_profile
        self.inverter = InverterAC(
            p_ac_max_kw=config.inverter_p_ac_max_kw,
            p_dc_max_kw=config.inverter_p_dc_max_kw,
        )
        # Phase 16: pull the optional detailed-electrical components from
        # the config. Both must be set together; the scenario_builder
        # enforces this contract upstream.
        self.electrical_model: ElectricalModel | None = config.electrical_model
        self.thermal_model: ThermalModel | None = config.thermal_model
        if self.electrical_model is not None and self.thermal_model is None:
            raise ValueError(
                "EnergySystemConfig.electrical_model requires a "
                "thermal_model to source ambient temperatures (T_cell "
                "depends on T_ambient via the NOCT relation)."
            )
        # The last per-path :class:`ElectricalKPIs` is cached here so
        # ``MonteCarloSimulator`` can collect it after each call to
        # ``run_one_path``. None when the electrical model is disabled.
        self.last_electrical_kpis: ElectricalKPIs | None = None
        # Phase 17 — opt-in HVAC controller. When the user enables
        # ``thermal_load_config`` we instantiate the controller once and
        # use it on every path. The controller is stateless across paths;
        # the per-hour T_ambient + at_home arrays are passed at run time.
        self.hvac_controller: HvacController | None = None
        if config.thermal_load_config is not None and config.thermal_load_config.enabled:
            if self.thermal_model is None:
                raise ValueError(
                    "thermal_load_config.enabled=True requires a thermal_model "
                    "to source hourly T_ambient. Wire a climate_profile_id in "
                    "the scenario JSON or disable the HVAC block."
                )
            self.hvac_controller = HvacController(config.thermal_load_config)
        # Phase 17 — cached per-path KPIs picked up by the MC orchestrator.
        self.last_thermal_kpis: ThermalLoadKPIs | None = None

        self.battery_bank = BatteryBank(
            specs=config.battery_specs,
            n_batteries=config.n_batteries,
            eta_charge=config.eta_charge,
            eta_discharge=config.eta_discharge,
            max_charge_kw=config.battery_max_charge_kw,
            max_discharge_kw=config.battery_max_discharge_kw,
            dt_hours=config.dt_hours,
        )

        (
            self.month_index_for_day,
            self.month_in_year_for_day,
            self.year_index_for_day,
            self.day_in_month_for_day,
            self.weekday_for_day,
        ) = build_calendar(
            config.n_years,
            start_weekday=config.calendar_start_weekday,
        )

    def run_one_path(
        self,
        rng: np.random.Generator,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray
    ]:
        """
        Run one Monte Carlo simulation path with hourly resolution.
        
        Simulates the energy system over the configured time period, tracking
        PV production, battery operation, load consumption, and grid interactions.
        
        Args:
            rng: Random number generator for stochastic processes.
        
        Returns:
            Tuple containing (in order):
            - monthly_pv_prod_kwh: Monthly PV production in kWh (array of length n_months).
            - monthly_pv_direct_kwh: Monthly PV energy directly used by load in kWh.
            - monthly_batt_to_load_kwh: Monthly battery discharge to load in kWh.
            - monthly_grid_import_kwh: Monthly grid import in kWh.
            - monthly_load_kwh: Monthly total load consumption in kWh.
            - soh_end_of_month: Battery state of health at end of each month (0-1).
            - soc_profile_first_year: Average state of charge profile for first year,
              shape (12, 24) representing [month, hour].
        """
        n_days = len(self.month_index_for_day)
        n_months = self.config.n_years * 12

        self.battery_bank.reset(soc_init=0.5)
        self.load_profile.reset_for_run(rng=rng, n_years=self.config.n_years)

        pv_daily_kwh = self.solar_model.simulate_daily_energy(
            n_years=self.config.n_years,
            month_in_year_for_day=self.month_in_year_for_day,
            year_index_for_day=self.year_index_for_day,
            rng=rng,
        )

        # Phase 16 — pre-compute the entire hourly PV DC array, apply the
        # electrical model (T_cell, V_string, MPPT-window derating, DC
        # shutdown) in one vectorised pass, then feed the *adjusted*
        # hourly PV into the existing inverter dispatch loop. This keeps
        # the legacy code path byte-identical when ``electrical_model``
        # is None (the conditional simply skips the adjustment step).
        pv_hourly_kw_path = np.zeros(n_days * 24)
        for d in range(n_days):
            pv_hourly_kw_path[d * 24:(d + 1) * 24] = self.solar_model.daily_profile_kwh(
                pv_daily_kwh[d]
            )

        # Phase 17 — pre-compute the hourly T_ambient when either the
        # electrical OR thermal-load (HVAC) model is active. Both
        # features consume the same path-level temperature realisation,
        # so we share the array between them.
        t_ambient_hourly: np.ndarray | None = None
        if self.thermal_model is not None and (
            self.electrical_model is not None or self.hvac_controller is not None
        ):
            daily_means = self.thermal_model.simulate_daily_means(n_days, rng)
            t_ambient_hourly = self.thermal_model.to_hourly(daily_means)

        electrical_kpis_path: ElectricalKPIs | None = None
        if self.electrical_model is not None and t_ambient_hourly is not None:
            pv_hourly_kw_path, electrical_kpis_path = self.electrical_model.apply_to_pv_dc(
                pv_hourly_kw_path, t_ambient_hourly
            )
        self.last_electrical_kpis = electrical_kpis_path

        # Phase 17 — pre-compute the hourly additive HVAC load. The
        # per-hour occupancy mask uses the load profile's home/away
        # decision when available; otherwise we treat every hour as
        # occupied (the steady-state controller produces the same
        # numbers as a continuously-occupied home).
        hvac_p_elec_hourly: np.ndarray | None = None
        thermal_kpis_path: ThermalLoadKPIs | None = None
        if self.hvac_controller is not None and t_ambient_hourly is not None:
            at_home_hourly = self._compute_at_home_hourly(n_days)
            hvac_p_elec_hourly, thermal_kpis_path = (
                self.hvac_controller.compute_hourly_p_elec_kw(
                    t_ambient_hourly, at_home_hourly
                )
            )
        self.last_thermal_kpis = thermal_kpis_path

        monthly_pv_prod_kwh = np.zeros(n_months)
        monthly_pv_direct_kwh = np.zeros(n_months)
        monthly_batt_to_load_kwh = np.zeros(n_months)
        monthly_grid_import_kwh = np.zeros(n_months)
        monthly_load_kwh = np.zeros(n_months)
        soh_end_of_month = np.zeros(n_months)

        soc_accum = np.zeros((12, 24))
        soc_count = np.zeros((12, 24), dtype=int)

        for d in range(n_days):
            month_idx = self.month_index_for_day[d]
            month_in_year = self.month_in_year_for_day[d]
            day_in_month = self.day_in_month_for_day[d]
            year_idx = self.year_index_for_day[d]
            weekday = self.weekday_for_day[d]

            pv_hourly_kw = pv_hourly_kw_path[d * 24:(d + 1) * 24]

            for h in range(24):
                p_load_kw = self.load_profile.get_hourly_load_kw(
                    year_index=year_idx,
                    month_in_year=month_in_year,
                    day_in_month=day_in_month,
                    hour_in_day=h,
                    weekday=weekday,
                )
                # Phase 17 — add the HVAC additive load (heating/cooling)
                # on top of the appliance baseline before the inverter
                # dispatch. When HVAC is disabled the array is None and
                # the legacy path stays untouched.
                if hvac_p_elec_hourly is not None:
                    p_load_kw = p_load_kw + float(hvac_p_elec_hourly[d * 24 + h])
                p_pv_kw = pv_hourly_kw[h]

                (
                    e_pv_prod,
                    e_pv_direct,
                    e_batt_discharge_to_load,
                    e_grid_to_load,
                    _e_pv_to_batt,
                ) = self.inverter.dispatch(
                    p_pv_dc_kw=p_pv_kw,
                    p_load_kw=p_load_kw,
                    battery=self.battery_bank,
                )

                e_load = p_load_kw * 1.0

                monthly_pv_prod_kwh[month_idx] += e_pv_prod
                monthly_pv_direct_kwh[month_idx] += e_pv_direct
                monthly_batt_to_load_kwh[month_idx] += e_batt_discharge_to_load
                monthly_grid_import_kwh[month_idx] += e_grid_to_load
                monthly_load_kwh[month_idx] += e_load

                if year_idx == 0:
                    soc = self.battery_bank.soc_fraction()
                    soc_accum[month_in_year, h] += soc
                    soc_count[month_in_year, h] += 1

            soh_end_of_month[month_idx] = self.battery_bank.soh

        soc_profile_first_year = np.zeros_like(soc_accum)
        mask = soc_count > 0
        soc_profile_first_year[mask] = soc_accum[mask] / soc_count[mask]

        # Phase 17 — finalise the HVAC share-of-total-load KPI now that
        # the simulator knows the full monthly_load_kwh aggregate. Total
        # load includes both the baseline (already inside
        # ``monthly_load_kwh``) and the HVAC additive contribution.
        if thermal_kpis_path is not None and monthly_load_kwh.sum() > 0:
            total_load_kwh = float(monthly_load_kwh.sum())
            # ``hvac_kwh_annual`` is per year; total_load_kwh covers the
            # whole horizon. Convert HVAC back to total then take the
            # ratio so the share is a pure dimensionless number.
            n_years = max(1, int(self.config.n_years))
            hvac_total_kwh = thermal_kpis_path.hvac_kwh_annual * n_years
            thermal_kpis_path.hvac_share_of_total_load_pct = (
                100.0 * hvac_total_kwh / total_load_kwh
            )

        return (
            monthly_pv_prod_kwh,
            monthly_pv_direct_kwh,
            monthly_batt_to_load_kwh,
            monthly_grid_import_kwh,
            monthly_load_kwh,
            soh_end_of_month,
            soc_profile_first_year,
        )

    def _compute_at_home_hourly(self, n_days: int) -> np.ndarray:
        """
        Build an hourly occupancy boolean mask for the whole path.

        Currently uses the calendar plus a flat "always at home" default
        — the home/away mask of :class:`HomeAwayLoadProfile` lives
        inside that profile and is not exposed cleanly. A future
        refinement could read it via ``self.load_profile.is_at_home(...)``
        once the interface is added. For now we mark all hours as
        occupied, which is conservative for the HVAC sizing (the heat
        pump runs whenever T_out is outside the dead-band) and keeps
        ``t_setpoint_away_c`` as an opt-in advanced knob.
        """
        return np.ones(n_days * 24, dtype=bool)
