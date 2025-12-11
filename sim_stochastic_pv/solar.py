from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SolarMonthParams:
    """
    Parameters for stochastic solar PV production model for one month.
    
    Attributes:
        avg_daily_kwh_per_kwp: Average daily energy production per kWp (kWh/kWp/day).
        p_sunny: Probability of a sunny day (0-1).
        sunny_factor: Multiplier for sunny days (typically > 1.0).
        cloudy_factor: Multiplier for cloudy days (typically < 1.0).
    """
    avg_daily_kwh_per_kwp: float
    p_sunny: float
    sunny_factor: float
    cloudy_factor: float


class SolarModel:
    """
    Stochastic daily PV production model (kWh/day) plus a fixed normalized hourly shape.
    """

    def __init__(
        self,
        pv_kwp: float,
        month_params: List[SolarMonthParams],
        degradation_per_year: float = 0.007,
    ) -> None:
        """
        Initialize a stochastic solar PV production model.
        
        Args:
            pv_kwp: PV system capacity in kWp.
            month_params: Monthly parameters (12 entries, one per month).
            degradation_per_year: Annual degradation rate (e.g., 0.007 = 0.7% per year).
        """
        if len(month_params) != 12:
            raise ValueError("month_params must contain 12 entries")
        self.pv_kwp = pv_kwp
        self.month_params = month_params
        self.degradation_per_year = degradation_per_year

        hours = np.arange(24)
        daylight_mask = (hours >= 6) & (hours <= 18)
        x = hours[daylight_mask] - 12.0
        shape = np.exp(-(x ** 2) / (2 * 3.0 ** 2))
        shape /= shape.sum()
        self.hourly_shape = np.zeros(24)
        self.hourly_shape[daylight_mask] = shape

    def simulate_daily_energy(
        self,
        n_years: int,
        month_in_year_for_day: np.ndarray,
        year_index_for_day: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Returns [n_days] array with PV energy (kWh/day).
        """
        n_days = len(month_in_year_for_day)
        pv_daily = np.zeros(n_days, dtype=float)

        for i in range(n_days):
            m = month_in_year_for_day[i]
            y = year_index_for_day[i]
            params = self.month_params[m]

            base_daily = params.avg_daily_kwh_per_kwp * self.pv_kwp

            is_sunny = rng.random() < params.p_sunny
            factor = params.sunny_factor if is_sunny else params.cloudy_factor

            degradation_factor = (1.0 - self.degradation_per_year) ** y

            pv_daily[i] = base_daily * factor * degradation_factor

        return pv_daily

    def daily_profile_kwh(self, daily_energy_kwh: float) -> np.ndarray:
        """
        Split daily energy into 24 hourly energy values using self.hourly_shape.
        """
        return daily_energy_kwh * self.hourly_shape


def make_default_solar_params_for_pavullo() -> List[SolarMonthParams]:
    """
    Approximate ~1250 kWh/kWp/year with a reasonable monthly distribution.
    Values are kWh/kWp/day.
    """
    avg_daily_kwh_per_kwp = [
        1.46,  # Jan
        2.27,  # Feb
        3.18,  # Mar
        4.09,  # Apr
        5.00,  # May
        5.46,  # Jun
        5.46,  # Jul
        4.55,  # Aug
        3.64,  # Sep
        2.73,  # Oct
        1.82,  # Nov
        1.36,  # Dec
    ]

    p_sunny = [
        0.4, 0.45, 0.5, 0.55, 0.6, 0.7,
        0.7, 0.65, 0.55, 0.5, 0.45, 0.4,
    ]

    params = []
    for i in range(12):
        params.append(
            SolarMonthParams(
                avg_daily_kwh_per_kwp=avg_daily_kwh_per_kwp[i],
                p_sunny=p_sunny[i],
                sunny_factor=1.2,
                cloudy_factor=0.3,
            )
        )
    return params
