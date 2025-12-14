"""
Stochastic home/away load profile implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .base import LoadProfile, BL_TABLE
from ...calendar_utils import MONTH_LENGTHS

class HomeAwayLoadProfile(LoadProfile):
    """
    Stochastic load profile combining home and away consumption patterns.

    Models realistic residential consumption by switching between two load
    profiles based on random occupancy schedules. This captures the significant
    consumption difference between occupied days (home profile: cooking, heating,
    appliances) and unoccupied days (away profile: standby loads only).

    Use cases:
    - Vacation homes with variable occupancy
    - Primary residences with frequent travel
    - Second homes used part-time
    - Modeling uncertainty in household presence patterns

    Each month, the model randomly selects N days to use the home profile
    (where N is uniformly sampled from [min_days_home, max_days_home]), with
    remaining days using the away profile. The specific days are chosen randomly
    without replacement, ensuring realistic spread throughout the month.

    Attributes:
        home_profile: LoadProfile used for occupied days (high consumption).
        away_profile: LoadProfile used for unoccupied days (low consumption).
        min_days_home: Minimum home days per month (12-element array).
        max_days_home: Maximum home days per month (12-element array).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            HomeAwayLoadProfile,
            MonthlyAverageLoadProfile,
            make_flat_monthly_load_profiles
        )

        # Create home and away profiles
        home_load = MonthlyAverageLoadProfile(
            make_flat_monthly_load_profiles(300.0)  # 300W when home
        )
        away_load = MonthlyAverageLoadProfile(
            make_flat_monthly_load_profiles(80.0)   # 80W when away (standby)
        )

        # Define occupancy bounds (home 15-25 days/month)
        min_days = [15] * 12  # At least 15 days home each month
        max_days = [25] * 12  # At most 25 days home each month

        # Create stochastic home/away model
        load_model = HomeAwayLoadProfile(
            home_profile=home_load,
            away_profile=away_load,
            min_days_home=min_days,
            max_days_home=max_days
        )

        # Reset for Monte Carlo path
        rng = np.random.default_rng(seed=42)
        load_model.reset_for_run(rng=rng, n_years=20)

        # Query load (internally selects home or away based on schedule)
        load_day5 = load_model.get_hourly_load_kw(0, 0, 5, 12, 1)
        # Returns 0.3 kW if day 5 is a home day, 0.08 kW if away day
        ```

    Notes:
        - Stochastic: Each Monte Carlo path generates different occupancy schedule
        - Month-level granularity: Schedule generated once per (year, month)
        - Day-level switching: Different profile used for each day
        - Supports seasonal patterns: Different occupancy ranges per month
        - Nested profiles: Both home and away profiles can be stochastic too
        - Cache efficient: Schedule computed once per month, reused for all hours
    """

    def __init__(
        self,
        home_profile: LoadProfile,
        away_profile: LoadProfile,
        min_days_home: List[int],
        max_days_home: List[int],
    ) -> None:
        """
        Initialize home/away load profile with occupancy bounds.

        Args:
            home_profile: Load profile for occupied days.
                Can be any LoadProfile subclass. Typically higher consumption.
                Example: 200-500W average for active household usage.
            away_profile: Load profile for unoccupied days.
                Can be any LoadProfile subclass. Typically minimal standby loads.
                Example: 50-150W average for refrigerator, routers, etc.
            min_days_home: Minimum home days per month (12 integers).
                Array of length 12, one value per month (Jan=0 ... Dec=11).
                Values: 0 to days_in_month (e.g., 0-31).
                Example: [20]*12 means at least 20 days home each month.
            max_days_home: Maximum home days per month (12 integers).
                Array of length 12, one value per month.
                Must satisfy: min_days_home[m] ≤ max_days_home[m] ≤ days_in_month
                Example: [25]*12 means at most 25 days home each month.

        Raises:
            ValueError: If min_days_home or max_days_home not length 12.

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                HomeAwayLoadProfile,
                AreraLoadProfile,
                make_flat_monthly_load_profiles,
                MonthlyAverageLoadProfile
            )

            # Scenario: Vacation home used mostly in summer
            home = AreraLoadProfile()  # Full residential when occupied
            away = MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(70.0)  # Minimal standby
            )

            # High occupancy in summer (Jun-Aug), low in winter
            min_days = [5, 5, 5, 5, 5, 20, 25, 25, 5, 5, 5, 5]  # More in Jun-Aug
            max_days = [10, 10, 10, 10, 15, 30, 31, 31, 15, 10, 10, 10]

            model = HomeAwayLoadProfile(home, away, min_days, max_days)

            # Scenario: Primary residence with frequent travel
            min_days_travel = [18]*12  # Home at least 18 days/month
            max_days_travel = [28]*12  # Away up to 13 days/month
            model_travel = HomeAwayLoadProfile(home, away, min_days_travel, max_days_travel)
            ```

        Notes:
            - Both profiles are stored by reference (not copied)
            - min_days and max_days are clipped to [0, days_in_month] at runtime
            - If min > max after clipping, max is set to min (all deterministic)
            - Profiles are reset together during reset_for_run()
        """
        if len(min_days_home) != 12 or len(max_days_home) != 12:
            raise ValueError("min_days_home and max_days_home must have length 12")

        self.home_profile = home_profile
        self.away_profile = away_profile
        self.min_days_home = np.array(min_days_home, dtype=int)
        self.max_days_home = np.array(max_days_home, dtype=int)

        self._home_days_schedule: Dict[Tuple[int, int], np.ndarray] = {}
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Reset for new Monte Carlo simulation path.

        Clears the cached occupancy schedule from the previous path and
        propagates the reset to both nested profiles. This ensures each
        Monte Carlo path gets an independent random occupancy pattern.

        Args:
            rng: Random number generator for stochastic schedule generation.
                Stored for use during get_hourly_load_kw() calls.
            n_years: Number of simulation years (passed to nested profiles).

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile

            # ... (create model as in __init__ example)

            # Monte Carlo loop
            for mc_path in range(1000):
                rng = np.random.default_rng(seed=100 + mc_path)

                # Reset for this path (clears old schedule, new RNG)
                model.reset_for_run(rng=rng, n_years=25)

                # Now queries will use fresh random schedule for this path
                # ...
            ```

        Notes:
            - Clears _home_days_schedule cache (forces regeneration)
            - Stores rng for lazy schedule generation during simulation
            - Resets both home_profile and away_profile with same rng
            - Called once per Monte Carlo path before year loop
        """
        self._home_days_schedule.clear()
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng
        self.home_profile.reset_for_run(rng=rng, n_years=n_years)
        self.away_profile.reset_for_run(rng=rng, n_years=n_years)

    def _rng_for_month(self) -> np.random.Generator:
        """
        Get random number generator for stochastic schedule generation.

        Returns the stored RNG or fallback if not initialized. This ensures
        a valid RNG is always available for generating occupancy schedules.

        Returns:
            np.random.Generator: Active random number generator instance.

        Notes:
            - Internal helper method (not part of public API)
            - Automatically initializes fallback if _rng not set
            - Used by _get_month_schedule() for random day selection
        """
        if self._rng is None:
            self._rng = self._fallback_rng
        return self._rng

    def _get_month_schedule(
        self,
        year_index: int,
        month_in_year: int,
    ) -> np.ndarray:
        """
        Get or lazily generate occupancy schedule for specified month.

        Checks cache for existing schedule, otherwise generates a new random
        schedule determining which days are "home" vs "away" for this month.
        The schedule is cached for reuse across all hours in the month.

        Algorithm:
        1. Check cache for (year, month) key
        2. If miss: Sample n_home_days ~ Uniform[min_days, max_days]
        3. Randomly select n_home_days distinct days (without replacement)
        4. Mark selected days as True (home), others False (away)
        5. Cache and return boolean mask

        Args:
            year_index: Simulation year (0-based) for cache key.
            month_in_year: Month index (0-11) for cache key and bounds.

        Returns:
            np.ndarray: Boolean array of length days_in_month.
                True = home day (use home_profile)
                False = away day (use away_profile)
                Example for 31-day month with 20 home days:
                [True, False, True, True, False, ..., True]

        Example:
            ```python
            # Internal usage (called automatically by get_hourly_load_kw)
            schedule_jan = model._get_month_schedule(year_index=0, month_in_year=0)
            # Returns cached schedule if already generated for (0, 0)
            # Otherwise generates new random schedule

            # January has 31 days, min_days=[15], max_days=[25]
            # schedule_jan.shape = (31,)
            # sum(schedule_jan) = random value in [15, 25]
            # True days are randomly distributed throughout month
            ```

        Notes:
            - Cached by (year, month) key for efficiency
            - Lazy generation: Only computed when first queried
            - Clipped to valid range: [0, days_in_month]
            - Deterministic count if min_days == max_days
            - Fully random day selection (uniform distribution over days)
            - February: 28 days (no leap year handling)
        """
        key = (year_index, month_in_year)
        schedule = self._home_days_schedule.get(key)
        if schedule is not None:
            return schedule

        days_in_month = MONTH_LENGTHS[month_in_year]
        min_days = int(np.clip(self.min_days_home[month_in_year], 0, days_in_month))
        max_days = int(np.clip(self.max_days_home[month_in_year], 0, days_in_month))
        if max_days < min_days:
            max_days = min_days

        rng = self._rng_for_month()
        if max_days == min_days:
            n_home_days = min_days
        else:
            n_home_days = int(rng.integers(min_days, max_days + 1))

        mask = np.zeros(days_in_month, dtype=bool)
        if n_home_days > 0:
            day_indices = rng.choice(days_in_month, size=n_home_days, replace=False)
            mask[day_indices] = True

        self._home_days_schedule[key] = mask
        return mask

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load based on stochastic home/away occupancy schedule.

        Determines whether the specified day is a "home" or "away" day
        based on the random occupancy schedule, then delegates to the
        corresponding profile to get the actual consumption value.

        Args:
            year_index: Simulation year (0-based) for schedule lookup.
            month_in_year: Month (0-11) for schedule lookup.
            day_in_month: Day (0-based) to check in schedule.
            hour_in_day: Hour (0-23) passed to nested profile.
            weekday: Weekday (0-6) passed to nested profile.

        Returns:
            float: Electricity consumption in kW.
                Returns home_profile.get_hourly_load_kw(...) if home day,
                otherwise away_profile.get_hourly_load_kw(...).

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                HomeAwayLoadProfile,
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Setup (from class example)
            home = MonthlyAverageLoadProfile(make_flat_monthly_load_profiles(300.0))
            away = MonthlyAverageLoadProfile(make_flat_monthly_load_profiles(80.0))
            model = HomeAwayLoadProfile(home, away, [15]*12, [25]*12)
            model.reset_for_run(rng=np.random.default_rng(42), n_years=1)

            # Query January 15th at noon
            load = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=0,   # January
                day_in_month=14,   # 15th day (0-indexed)
                hour_in_day=12,    # Noon
                weekday=2          # Wednesday
            )

            # If day 14 is in home schedule: load ≈ 0.300 kW
            # If day 14 is in away schedule: load ≈ 0.080 kW
            # (Specific value depends on random seed)
            ```

        Notes:
            - Stochastic: Different paths return different values
            - Day-level switching: All hours in a day use same profile
            - Schedule lazy-generated: First call triggers _get_month_schedule()
            - Subsequent calls for same month use cached schedule (fast)
            - Bounds checking: Invalid day_in_month defaults to away profile
            - All parameters forwarded to nested profile for final calculation
        """
        schedule = self._get_month_schedule(year_index, month_in_year)
        if 0 <= day_in_month < schedule.size and schedule[day_in_month]:
            return self.home_profile.get_hourly_load_kw(
                year_index=year_index,
                month_in_year=month_in_year,
                day_in_month=day_in_month,
                hour_in_day=hour_in_day,
                weekday=weekday,
            )

        return self.away_profile.get_hourly_load_kw(
            year_index=year_index,
            month_in_year=month_in_year,
            day_in_month=day_in_month,
            hour_in_day=hour_in_day,
            weekday=weekday,
        )


Z_VALUE_95 = 1.6448536269514722


