from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .battery import BatteryBank, BatterySpecs
from .calendar_utils import build_calendar
from .inverter import InverterAC
from .load_profiles import LoadProfile
from .solar import SolarModel


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
        self.load_profile = load_profile
        self.inverter = InverterAC(
            p_ac_max_kw=config.inverter_p_ac_max_kw,
            p_dc_max_kw=config.inverter_p_dc_max_kw,
        )

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

            daily_pv_profile_kwh = self.solar_model.daily_profile_kwh(pv_daily_kwh[d])
            pv_hourly_kw = daily_pv_profile_kwh

            for h in range(24):
                p_load_kw = self.load_profile.get_hourly_load_kw(
                    year_index=year_idx,
                    month_in_year=month_in_year,
                    day_in_month=day_in_month,
                    hour_in_day=h,
                    weekday=weekday,
                )
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

        return (
            monthly_pv_prod_kwh,
            monthly_pv_direct_kwh,
            monthly_batt_to_load_kwh,
            monthly_grid_import_kwh,
            monthly_load_kwh,
            soh_end_of_month,
            soc_profile_first_year,
        )
