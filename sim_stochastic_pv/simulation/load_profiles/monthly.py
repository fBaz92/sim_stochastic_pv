"""
Monthly average load profile implementation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import LoadProfile
from ...calendar_utils import MONTH_LENGTHS


class MonthlyAverageLoadProfile(LoadProfile):
    """
    Deterministic load profile with fixed 24-hour pattern per month.

    Models electricity consumption using monthly-averaged diurnal patterns,
    where each month has a fixed 24-hour profile that repeats every day.
    This is the simplest realistic load model, capturing:
    - Seasonal variation: Different consumption between winter and summer
    - Diurnal patterns: Higher load during day, lower at night
    - No day-to-day variation: Deterministic (same pattern daily)

    Use cases:
    - Quick simulations without stochastic load complexity
    - Baseline/reference scenarios
    - Away-mode consumption (minimal variability expected)
    - Validation against monthly utility bills

    Attributes:
        monthly_profiles_kw: Array of shape (12, 24) with hourly consumption
            in kW for each month (rows) and hour (columns).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile

        # Create simple profile: 100W at night, 200W during day
        profiles_w = np.zeros((12, 24))
        for month in range(12):
            for hour in range(24):
                if 8 <= hour <= 20:  # Daytime
                    profiles_w[month, hour] = 200.0
                else:  # Nighttime
                    profiles_w[month, hour] = 100.0

        load_model = MonthlyAverageLoadProfile(profiles_w)

        # Query consumption
        load_night = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=0,
            hour_in_day=3, weekday=0
        )  # 0.1 kW (night)

        load_day = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=15,
            hour_in_day=12, weekday=2
        )  # 0.2 kW (day)

        # Same hour, different day → same load
        load_day2 = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=25,
            hour_in_day=12, weekday=5
        )  # 0.2 kW (identical)
        ```

    Notes:
        - Fully deterministic: No stochastic variation
        - No weekday/weekend distinction (unless encoded in input profiles)
        - Day-independent: day_in_month parameter ignored
        - Year-independent: year_index parameter ignored
        - Efficient: O(1) lookup, no caching needed
        - Typical use: 12 months × 24 hours = 288 unique values
    """

    def __init__(
        self,
        monthly_profiles_w: Optional[np.ndarray] = None,
        *,
        monthly_avg_kwh: Optional[List[float]] = None,
    ) -> None:
        """
        Initialize monthly average load profile with fixed diurnal patterns.

        Accepts two mutually exclusive interfaces:

        1. ``monthly_profiles_w`` (detailed): a (12, 24) Watts array giving the
           exact hourly load for each month.  Full control over the diurnal shape.
        2. ``monthly_avg_kwh`` (simple): a list of 12 total kWh values — one per
           month.  Each month is automatically converted to a *flat* (constant)
           24-hour profile using the actual number of days in that month.

        Args:
            monthly_profiles_w: Hourly load profiles in Watts (shape: 12 × 24).
                Rows: months (Jan=0 … Dec=11). Columns: hours (0…23).
                Mutually exclusive with *monthly_avg_kwh*.
            monthly_avg_kwh: Total energy consumed in each calendar month (kWh).
                List of 12 non-negative floats (Jan first). The hourly load is
                derived as ``kWh * 1000 / (days_in_month * 24)`` Watts, spread
                uniformly across all 24 hours and all days in the month.
                Mutually exclusive with *monthly_profiles_w*.

        Raises:
            ValueError: If neither or both arguments are provided.
            ValueError: If monthly_profiles_w is not shape (12, 24).
            ValueError: If monthly_avg_kwh does not have exactly 12 elements.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Simple interface – 300 kWh/month flat profile
            model_simple = MonthlyAverageLoadProfile(
                monthly_avg_kwh=[300.0] * 12
            )

            # Detailed interface – constant 150 W all day every month
            profiles_flat = make_flat_monthly_load_profiles(base_load_w=150.0)
            model_flat = MonthlyAverageLoadProfile(profiles_flat)
            ```

        Notes:
            - Profiles stored internally in kW (converted from input Watts).
            - When monthly_avg_kwh is used, the resulting flat per-hour load in
              Watts is ``monthly_avg_kwh[m] * 1000 / (MONTH_LENGTHS[m] * 24)``.
            - The ``monthly_avg_kwh`` attribute is set only when that interface
              is used; it is not back-calculated from ``monthly_profiles_w``.
        """
        if monthly_avg_kwh is not None and monthly_profiles_w is not None:
            raise ValueError(
                "Provide either monthly_profiles_w or monthly_avg_kwh, not both."
            )
        if monthly_avg_kwh is not None:
            if len(monthly_avg_kwh) != 12:
                raise ValueError("monthly_avg_kwh must have exactly 12 elements.")
            profiles_w = np.zeros((12, 24), dtype=float)
            for m, kwh in enumerate(monthly_avg_kwh):
                hours_in_month = MONTH_LENGTHS[m] * 24
                profiles_w[m, :] = kwh * 1000.0 / hours_in_month
            self.monthly_avg_kwh: Optional[List[float]] = list(monthly_avg_kwh)
            self.monthly_profiles_kw = profiles_w / 1000.0
        elif monthly_profiles_w is not None:
            if monthly_profiles_w.shape != (12, 24):
                raise ValueError("monthly_profiles_w must have shape (12, 24)")
            self.monthly_avg_kwh = None
            self.monthly_profiles_kw = monthly_profiles_w / 1000.0
        else:
            raise ValueError(
                "One of monthly_profiles_w or monthly_avg_kwh must be provided."
            )

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load from monthly average profile.

        Returns the consumption for the specified hour, looking up the
        corresponding month and hour in the stored profiles. The returned
        value is identical for all days within the same month at the same hour.

        Args:
            year_index: Ignored (deterministic profile, no year variation).
            month_in_year: Month index (0-11) to select row from profiles.
            day_in_month: Ignored (same pattern all days in month).
            hour_in_day: Hour index (0-23) to select column from profiles.
            weekday: Ignored (no weekday/weekend distinction).

        Returns:
            float: Electricity consumption in kW for the specified month and hour.
                Direct lookup: monthly_profiles_kw[month_in_year, hour_in_day].

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # 200W constant load
            model = MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(200.0)
            )

            # All queries for same month/hour return identical value
            load1 = model.get_hourly_load_kw(0, 5, 10, 14, 2)  # 0.2 kW
            load2 = model.get_hourly_load_kw(10, 5, 25, 14, 5)  # 0.2 kW (same)
            # (year=0 vs 10, day=10 vs 25, weekday=2 vs 5 → all ignored)
            ```

        Notes:
            - O(1) lookup: Direct array indexing, very fast
            - Deterministic: Same inputs always return same output
            - Ignores day_in_month, year_index, weekday parameters
            - No bounds checking: Invalid indices will raise IndexError
        """
        return float(self.monthly_profiles_kw[month_in_year, hour_in_day])
