"""
Monthly average load profile implementation.
"""

from __future__ import annotations

import numpy as np

from .base import LoadProfile


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

    def __init__(self, monthly_profiles_w: np.ndarray) -> None:
        """
        Initialize monthly average load profile with fixed diurnal patterns.

        Args:
            monthly_profiles_w: Hourly load profiles in Watts (shape: 12 × 24).
                Array structure:
                - Rows (12): Months (Jan=0, Feb=1, ..., Dec=11)
                - Columns (24): Hours (0=midnight-1am, ..., 23=11pm-midnight)
                - Values: Load in Watts (non-negative floats)

                Typical residential values: 50-500W per hour.
                Examples:
                - 100W: Standby consumption (refrigerator, routers)
                - 250W: Light usage (lighting, small appliances)
                - 500W+: Active usage (cooking, heating, AC)

        Raises:
            ValueError: If monthly_profiles_w is not shape (12, 24).

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Option 1: Constant load (150W)
            profiles_flat = make_flat_monthly_load_profiles(base_load_w=150.0)
            model_flat = MonthlyAverageLoadProfile(profiles_flat)

            # Option 2: Custom seasonal pattern
            profiles_custom = np.zeros((12, 24))
            # Winter months (higher heating load)
            for month in [0, 1, 11]:  # Jan, Feb, Dec
                profiles_custom[month, :] = 300.0
            # Summer months (moderate)
            for month in [5, 6, 7]:  # Jun, Jul, Aug
                profiles_custom[month, :] = 150.0
            # Spring/Fall (lower)
            for month in [2, 3, 4, 8, 9, 10]:
                profiles_custom[month, :] = 200.0

            model_seasonal = MonthlyAverageLoadProfile(profiles_custom)

            # Option 3: Load from external source (utility data)
            # profiles_measured = np.load('measured_profiles.npy')
            # model_measured = MonthlyAverageLoadProfile(profiles_measured)
            ```

        Notes:
            - Profiles stored internally in kW (converted from input Watts)
            - No validation of value ranges (negative loads allowed but illogical)
            - Profile array is NOT copied (stores reference for efficiency)
        """
        if monthly_profiles_w.shape != (12, 24):
            raise ValueError("monthly_profiles_w must have shape (12, 24)")
        self.monthly_profiles_kw = monthly_profiles_w / 1000.0

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
