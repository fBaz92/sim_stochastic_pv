"""
Time grid and load profile models.

Provides the temporal backbone (:class:`TimeGrid`) that maps each quarter-hour
index to calendar metadata (month, hour, day-of-week), and a multiplicative
load model (:class:`LoadProfile`) that composes monthly, hourly, weekday,
holiday, and stochastic factors into a full-year demand profile.
"""

from __future__ import annotations

import numpy as np

from sim_stochastic_pv.market.config import (
    QUARTERS_PER_YEAR, QUARTERS_PER_DAY, QUARTERS_PER_HOUR,
    MONTHLY_LOAD_FACTORS, HOURLY_LOAD_FACTORS,
)


class TimeGrid:
    """Backbone temporal vector: 35 040 quarter-hours with calendar metadata.

    Pre-computes arrays mapping each quarter-hour index to its month, hour,
    day-of-year, and day-of-week. Used by all availability models and the
    load profile to look up time-dependent factors via vectorized indexing.

    Attributes:
        n (int): Total number of quarter-hour intervals (35 040).
        quarter_index (np.ndarray): Array ``[0, 1, ..., 35039]``.
        day_of_year (np.ndarray): Day of year (0-364) for each quarter-hour.
        hour (np.ndarray): Hour of day (0-23) for each quarter-hour.
        month (np.ndarray): Month (1-12) for each quarter-hour.
        day_of_week (np.ndarray): Day of week (0=Monday, 6=Sunday) for each
            quarter-hour. Year starts on an arbitrary Monday.
        is_holiday (np.ndarray): Boolean mask, ``True`` for holiday quarter-hours.
    """

    def __init__(self) -> None:
        """Initialize the time grid with 35 040 quarter-hour intervals."""
        self.n: int = QUARTERS_PER_YEAR

        self.quarter_index: np.ndarray = np.arange(self.n)
        self.day_of_year: np.ndarray = self.quarter_index // QUARTERS_PER_DAY

        quarter_in_day = self.quarter_index % QUARTERS_PER_DAY
        self.hour: np.ndarray = quarter_in_day // QUARTERS_PER_HOUR

        month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        cum_days = np.cumsum(month_days)
        self.month: np.ndarray = np.searchsorted(cum_days, self.day_of_year, side='right') + 1

        self.day_of_week: np.ndarray = self.day_of_year % 7
        self.is_holiday: np.ndarray = np.zeros(self.n, dtype=bool)

    def set_holiday_calendar(self, holiday_days: list[int]) -> None:
        """Mark specific days of year as holidays.

        Args:
            holiday_days: List of day-of-year indices (0-364) to mark as holidays.
        """
        self.is_holiday[:] = False
        for d in holiday_days:
            mask = self.day_of_year == d
            self.is_holiday[mask] = True


class LoadProfile:
    """Multiplicative load model: ``P_peak * k_month * k_hour * [k_weekday] * [k_holiday] * [noise]``.

    Builds a quarter-hourly load profile by multiplying a peak power value
    with monthly and hourly shaping factors, plus optional weekday, holiday,
    and stochastic noise adjustments.

    Attributes:
        tg (TimeGrid): Reference to the time grid.
        p_peak_pu (float): Peak load in per-unit of system base.
        monthly_factors (np.ndarray): Array of length 13 (index 0 unused,
            1-12 = months) with monthly load multipliers.
        hourly_factors (np.ndarray): Array of length 24 (index 0-23 = hours)
            with hourly load multipliers.
        weekday_factors (np.ndarray | None): Optional array of length 7
            (0=Monday, 6=Sunday) with weekday multipliers.
        holiday_factor (float | None): Optional multiplier applied to
            quarter-hours flagged as holidays.
    """

    def __init__(self, time_grid: TimeGrid, p_peak_pu: float = 1.0) -> None:
        """Initialize the load profile with default monthly and hourly factors.

        Args:
            time_grid: The temporal backbone providing calendar metadata.
            p_peak_pu: Peak load in per-unit of system base. Defaults to 1.0.
        """
        self.tg = time_grid
        self.p_peak_pu = p_peak_pu
        self.monthly_factors = np.ones(13)  # index 1-12
        self.hourly_factors = np.ones(24)
        self.weekday_factors = None
        self.holiday_factor = None
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Load default monthly and hourly factors from config."""
        for m, k in MONTHLY_LOAD_FACTORS.items():
            self.monthly_factors[m] = k
        for h, k in HOURLY_LOAD_FACTORS.items():
            self.hourly_factors[h] = k

    def set_monthly_factors(self, factors: dict[int, float]) -> None:
        """Override monthly load factors.

        Args:
            factors: Mapping from month (1-12) to load multiplier.
        """
        for m, k in factors.items():
            self.monthly_factors[m] = k

    def set_hourly_factors(self, factors: dict[int, float]) -> None:
        """Override hourly load factors.

        Args:
            factors: Mapping from hour (0-23) to load multiplier.
        """
        for h, k in factors.items():
            self.hourly_factors[h] = k

    def set_weekday_factors(self, factors: dict[int, float]) -> None:
        """Set weekday-dependent load multipliers.

        Args:
            factors: Mapping from day-of-week (0=Monday, 6=Sunday) to multiplier.
        """
        self.weekday_factors = np.ones(7)
        for d, k in factors.items():
            self.weekday_factors[d] = k

    def set_holiday_factor(self, factor: float) -> None:
        """Set the load multiplier for holiday quarter-hours.

        Args:
            factor: Multiplier applied on top of other factors during holidays
                (e.g. 0.85 for 15% load reduction).
        """
        self.holiday_factor = factor

    def generate(self, rng: np.random.Generator = None,
                 noise_sigma: float = 0.0) -> np.ndarray:
        """Generate the full-year load profile.

        Computes ``P_peak * k_month * k_hour`` for each quarter-hour, then
        optionally applies weekday factors, holiday factor, and Gaussian
        multiplicative noise.

        Args:
            rng: NumPy random generator for stochastic noise. Required only
                if ``noise_sigma > 0``.
            noise_sigma: Standard deviation of multiplicative Gaussian noise
                (mean=1.0). Set to 0.0 for a deterministic profile.

        Returns:
            np.ndarray: Load profile array of shape ``(35040,)`` in per-unit
                of system base.
        """
        k_m = self.monthly_factors[self.tg.month]
        k_h = self.hourly_factors[self.tg.hour]
        load = self.p_peak_pu * k_m * k_h

        if self.weekday_factors is not None:
            load *= self.weekday_factors[self.tg.day_of_week]

        if self.holiday_factor is not None:
            load[self.tg.is_holiday] *= self.holiday_factor

        if noise_sigma > 0 and rng is not None:
            noise = rng.normal(1.0, noise_sigma, size=self.tg.n)
            noise = np.clip(noise, 0.5, 1.5)
            load *= noise

        return load
