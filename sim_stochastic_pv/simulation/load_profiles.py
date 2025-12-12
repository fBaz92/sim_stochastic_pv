"""
Synthetic electricity consumption profile generators.

Implements deterministic and stochastic load models used by the energy
simulation (ARERA baseline, home/away, variable occupancy, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..calendar_utils import MONTH_LENGTHS

# Rows represent months (0=January..11=December)
# Columns represent the base load in Watt for ARERA tariff bands [F1, F2, F3].
# The table is used to build deterministic load profiles for the "away" periods.
BL_TABLE: np.ndarray = np.array([
    [110.6719368,  83.79888268, 157.0512821 ],  # Gen
    [ 68.18181818, 73.17073171,  72.91666667],  # Feb
    [ 99.56709957,113.5135135,   94.51219512],  # Mar
    [144.6280992, 120.6896552,  138.1578947 ],  # Apr
    [ 90.90909091, 89.47368421, 102.5641026 ],  # Mag
    [116.8831169, 130.1775148,  134.375     ],  # Giu
    [130.4347826, 145.2513966,  125.        ],  # Lug
    [251.0822511, 286.4864865,  280.4878049],  # Ago
    [ 95.04132231, 97.70114943,  92.10526316],  # Set
    [169.9604743, 111.7318436, 118.5897436 ],  # Ott
    [131.8181818,  83.33333333, 168.75     ],  # Nov
    [185.770751,  229.0502793,  211.5384615],  # Dic
])


class LoadProfile:
    """
    Abstract base class for hourly electricity consumption models.

    Defines the interface for load profile implementations used in energy
    system simulations. Load profiles provide hourly electricity consumption
    (in kW) as a function of time, enabling realistic modeling of household
    or building energy demand patterns.

    The interface supports both deterministic and stochastic load profiles:
    - Deterministic: Fixed consumption patterns (e.g., monthly averages)
    - Stochastic: Random daily variations and occupancy patterns (Monte Carlo)

    Subclasses must implement:
    - get_hourly_load_kw(): Return consumption for specific hour
    - reset_for_run(): Optional initialization before each Monte Carlo path

    Example subclasses:
    - MonthlyAverageLoadProfile: Fixed 24h pattern per month
    - AreraLoadProfile: Italian ARERA tariff-based consumption
    - HomeAwayLoadProfile: Stochastic presence/absence patterns
    - VariableLoadProfile: Daily stochastic multipliers

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
        import numpy as np

        # Create simple constant load profile
        monthly_profiles_w = np.full((12, 24), 150.0)  # 150W constant
        load_model = MonthlyAverageLoadProfile(monthly_profiles_w)

        # Get consumption for first hour of first year
        load_kw = load_model.get_hourly_load_kw(
            year_index=0,
            month_in_year=0,
            day_in_month=0,
            hour_in_day=0,
            weekday=0
        )
        print(f"Load: {load_kw:.3f} kW")  # 0.150 kW

        # Reset for Monte Carlo path
        rng = np.random.default_rng(seed=42)
        load_model.reset_for_run(rng=rng, n_years=20)
        ```

    Notes:
        - All time indices are 0-based
        - year_index: 0 = first year, 1 = second year, etc.
        - month_in_year: 0 = Jan, 11 = Dec
        - day_in_month: 0 = first day of month
        - hour_in_day: 0 = midnight-1am, 23 = 11pm-midnight
        - weekday: 0 = Monday, 6 = Sunday
        - Consumption returned in kW (power, not energy)
    """

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Prepare load profile for new simulation run (optional hook).

        Called before each Monte Carlo simulation path to initialize or
        reset any stochastic state. The default implementation is a no-op,
        suitable for deterministic load profiles.

        Stochastic models should override this to:
        1. Store the provided random number generator
        2. Clear any cached stochastic schedules from previous runs
        3. Reset wrapped/nested profiles

        Args:
            rng: Random number generator for stochastic load variation.
                If None, model should use internal/default generator.
                Provided by Monte Carlo simulator to ensure reproducibility.
            n_years: Number of simulation years for preparation.
                If None, model may skip precomputation.

        Notes:
            - Default implementation does nothing (for deterministic models)
            - Stochastic models must store rng for get_hourly_load_kw() calls
            - Called once per Monte Carlo path
        """

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get electricity consumption for specific hour.

        Returns the load demand (in kW) at the specified time point. This is
        the core method called repeatedly during simulation to determine hourly
        consumption for energy system optimization and economic calculations.

        Args:
            year_index: Simulation year (0-based integer).
                0 = first year, 1 = second year, etc.
                Allows modeling consumption changes over time (rare).
            month_in_year: Month within year (0-based integer, 0-11).
                0 = January, 1 = February, ..., 11 = December.
                Used for seasonal consumption patterns.
            day_in_month: Day within month (0-based integer).
                0 = first day, 1 = second day, etc.
                Range: 0-27 (Feb) to 0-30 (31-day months).
                Used for daily stochastic variations.
            hour_in_day: Hour within day (0-based integer, 0-23).
                0 = midnight-1am, 12 = noon-1pm, 23 = 11pm-midnight.
                Core dimension for load profiles (diurnal patterns).
            weekday: Day of week (0-based integer, 0-6).
                0 = Monday, 1 = Tuesday, ..., 6 = Sunday.
                Used for weekday/weekend patterns.

        Returns:
            float: Electricity consumption in kW (kilowatts, power not energy).
                Must be non-negative. Typical residential values: 0.05-5.0 kW.
                Example: 0.150 kW = 150W standby, 3.0 kW = cooking + appliances.

        Raises:
            NotImplementedError: If subclass doesn't implement this method.

        Example:
            ```python
            # Implementing in a subclass
            class ConstantLoad(LoadProfile):
                def __init__(self, constant_kw):
                    self.constant_kw = constant_kw

                def get_hourly_load_kw(self, year_index, month_in_year,
                                       day_in_month, hour_in_day, weekday):
                    return self.constant_kw

            # Usage
            load = ConstantLoad(constant_kw=0.2)
            consumption = load.get_hourly_load_kw(0, 0, 0, 12, 0)
            print(f"{consumption} kW")  # 0.2 kW
            ```

        Notes:
            - Called very frequently: n_mc × n_years × 8760 times per MC simulation
            - Should be fast (avoid heavy computation; precompute in reset_for_run)
            - Return value used for energy system optimization (battery, grid)
            - Must be deterministic for given inputs within a single run
            - Stochastic variation comes from reset_for_run(), not here
        """
        raise NotImplementedError


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


class AreraLoadProfile(LoadProfile):
    """
    Italian ARERA tariff-based load profile (F1/F2/F3 bands).

    Models electricity consumption following the Italian ARERA (Autorità di
    Regolazione per Energia Reti e Ambiente) time-of-use tariff structure.
    The ARERA system divides hours into three price bands based on time and
    day of week, reflecting grid demand patterns:

    - F1 (Peak): Weekday business hours (Mon-Fri, 8am-7pm) - highest rates
    - F2 (Mid-peak): Weekday mornings/evenings + Saturday daytime - medium rates
    - F3 (Off-peak): Nights, Sundays, holidays - lowest rates

    This profile assigns different base consumption levels to each tariff band,
    enabling realistic modeling of Italian residential consumption patterns.
    The base load table (BL_TABLE) contains empirically derived consumption
    values for each month and tariff band.

    Attributes:
        bl_table: Base load table (12 × 3 array) in Watts.
            Rows: Months (Jan=0 ... Dec=11)
            Columns: Tariff bands (F1, F2, F3)
            Default: BL_TABLE (Italian residential averages)

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import AreraLoadProfile, BL_TABLE

        # Use default Italian consumption pattern
        load_model = AreraLoadProfile()

        # Weekday peak hour (Monday 10am, January)
        load_f1 = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=0,
            hour_in_day=10, weekday=0  # Mon 10am
        )  # Returns F1 band value for January

        # Sunday night (off-peak)
        load_f3 = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=6,
            hour_in_day=22, weekday=6  # Sun 10pm
        )  # Returns F3 band value (lower)

        # Custom base load table (e.g., commercial profile)
        bl_custom = np.array([
            # F1,   F2,   F3 (per month)
            [500.0, 300.0, 100.0],  # Jan (higher winter usage)
            [450.0, 280.0,  90.0],  # Feb
            # ... (12 rows total)
        ])
        load_commercial = AreraLoadProfile(bl_table=bl_custom)
        ```

    Notes:
        - Designed for Italian electricity market simulations
        - Deterministic: No stochastic variation (combine with VariableLoadProfile)
        - Weekday-aware: Distinguishes Mon-Fri, Saturday, Sunday
        - Hour-aware: Different bands throughout the day
        - Default BL_TABLE based on typical Italian residential consumption
    """

    def __init__(self, bl_table: np.ndarray = BL_TABLE) -> None:
        """
        Initialize ARERA load profile with tariff band consumption table.

        Args:
            bl_table: Base load table in Watts (shape: 12 × 3).
                Array structure:
                - Rows (12): Months (Jan=0, Feb=1, ..., Dec=11)
                - Columns (3): ARERA tariff bands [F1, F2, F3]
                  - Column 0: F1 (peak) consumption in Watts
                  - Column 1: F2 (mid-peak) consumption in Watts
                  - Column 2: F3 (off-peak) consumption in Watts
                - Values: Hourly load in Watts (non-negative floats)

                Default: BL_TABLE (Italian residential average from empirical data).
                Typical values: 70-300W depending on season and tariff band.

        Raises:
            ValueError: If bl_table is not shape (12, 3).

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                AreraLoadProfile,
                BL_TABLE
            )

            # Option 1: Use default (Italian residential average)
            model_default = AreraLoadProfile()

            # Option 2: Scaled version (50% higher consumption)
            bl_scaled = BL_TABLE * 1.5
            model_scaled = AreraLoadProfile(bl_table=bl_scaled)

            # Option 3: Custom pattern (high peak, low off-peak)
            bl_custom = np.zeros((12, 3))
            for month in range(12):
                bl_custom[month, 0] = 400.0  # F1 peak
                bl_custom[month, 1] = 200.0  # F2 mid
                bl_custom[month, 2] = 80.0   # F3 off-peak
            model_custom = AreraLoadProfile(bl_table=bl_custom)
            ```

        Notes:
            - Default BL_TABLE reflects Italian residential consumption patterns
            - Table values vary by month (seasonal effects) and band (time-of-use)
            - Higher values in winter months (heating), summer (AC), August (variable)
            - F1 typically higher than F2, F2 higher than F3 (demand-based)
        """
        if bl_table.shape != (12, 3):
            raise ValueError("bl_table must have shape (12, 3)")
        self.bl_table = bl_table

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load based on ARERA tariff bands and calendar position.

        Determines the appropriate ARERA tariff band (F1/F2/F3) based on the
        weekday and hour, then returns the corresponding base load value from
        the stored table for the specified month.

        Args:
            year_index: Ignored (no year variation).
            month_in_year: Month index (0-11) to select table row.
            day_in_month: Day within month (0-based) for weekday calculation.
            hour_in_day: Hour of day (0-23) for tariff band determination.
            weekday: Day of week (0=Mon, 6=Sun) for tariff band determination.

        Returns:
            float: Electricity consumption in kW for the specified time.
                Lookup: bl_table[month, band] where band ∈ {F1, F2, F3}
                determined by weekday and hour combination.

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import AreraLoadProfile

            model = AreraLoadProfile()

            # Monday 10am in January (F1 peak band)
            load_peak = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=0,   # January
                day_in_month=0,
                hour_in_day=10,    # 10am
                weekday=0          # Monday
            )  # Returns bl_table[0, 0] / 1000  (Jan F1)

            # Sunday 3am in January (F3 off-peak)
            load_offpeak = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=0,   # January
                day_in_month=6,
                hour_in_day=3,     # 3am
                weekday=6          # Sunday
            )  # Returns bl_table[0, 2] / 1000  (Jan F3)

            # Saturday 8pm in August (F2 mid-peak)
            load_mid = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=7,   # August
                day_in_month=5,
                hour_in_day=20,    # 8pm
                weekday=5          # Saturday
            )  # Returns bl_table[7, 1] / 1000  (Aug F2)
            ```

        Notes:
            - Tariff band assignment (see _get_band_arera for logic):
              - F1: Mon-Fri 8am-7pm (08:00-18:59)
              - F2: Mon-Fri 7am+19-22pm, Sat 7am-10pm
              - F3: All other hours (nights, Sundays)
            - Deterministic: Same weekday+hour+month always returns same value
            - Typical F1 > F2 > F3 (reflecting demand and pricing)
        """
        hour_in_month = day_in_month * 24 + hour_in_day
        first_weekday = (weekday - day_in_month) % 7
        load_w = get_load_w(
            month_index=month_in_year,
            hour_in_month=hour_in_month,
            first_weekday_of_month=first_weekday,
            bl_table=self.bl_table,
        )
        return load_w / 1000.0


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


class VariableLoadProfile(LoadProfile):
    """
    Stochastic decorator applying daily random multipliers to base load profile.

    Wraps any LoadProfile and adds day-to-day consumption variability by
    applying random multipliers sampled from a truncated normal distribution.
    This models realistic household consumption uncertainty: some days use more
    electricity (guests, cooking, heating), others less (away, mild weather).

    Each calendar day gets a unique random multiplier sampled once and cached.
    All 24 hours in that day use the same multiplier, preserving the base
    profile's diurnal shape while scaling the overall daily consumption.

    Multiplier Distribution:
        multiplier ~ 1 + TruncatedNormal(0, σ, p05_delta, p95_delta)
        where σ = max(|p05|, |p95|) / 1.6448536 (95th percentile z-score)

    Use cases:
    - Add realism to deterministic profiles (ARERA, monthly averages)
    - Model consumption uncertainty in Monte Carlo simulations
    - Represent unpredictable household behavior (occupancy, appliance use)
    - Create realistic load distributions for risk analysis

    Attributes:
        base_profile: Wrapped LoadProfile providing reference consumption.
        p05_delta: 5th percentile variation bound (negative, e.g., -0.10 = -10%).
        p95_delta: 95th percentile variation bound (positive, e.g., +0.10 = +10%).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            VariableLoadProfile,
            MonthlyAverageLoadProfile,
            make_flat_monthly_load_profiles
        )

        # Start with deterministic profile
        base = MonthlyAverageLoadProfile(
            make_flat_monthly_load_profiles(200.0)  # 200W constant
        )

        # Add ±10% daily variation
        variable_model = VariableLoadProfile(
            base_profile=base,
            p05_delta=-0.10,  # 5% of days: 10% below (180W)
            p95_delta=0.10    # 5% of days: 10% above (220W)
        )

        # Reset for Monte Carlo path
        rng = np.random.default_rng(seed=42)
        variable_model.reset_for_run(rng=rng, n_years=20)

        # Day 1: multiplier might be 0.95 → all hours: 200W × 0.95 = 190W
        load_day1 = variable_model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 0.19 kW

        # Day 2: different multiplier, e.g., 1.07 → all hours: 200W × 1.07 = 214W
        load_day2 = variable_model.get_hourly_load_kw(0, 0, 1, 12, 1)  # 0.214 kW

        # Conservative variation (±5%)
        model_conservative = VariableLoadProfile(base, -0.05, 0.05)

        # High variation (±20%, e.g., vacation home uncertainty)
        model_high_var = VariableLoadProfile(base, -0.20, 0.20)
        ```

    Notes:
        - Stochastic: Each day gets random multiplier from truncated normal
        - Day-level granularity: Same multiplier for all 24 hours in a day
        - Preserves shape: Diurnal pattern maintained, only scaled
        - Decorator pattern: Can wrap any LoadProfile subclass
        - Cascadable: Can wrap other stochastic profiles (e.g., HomeAwayLoadProfile)
        - Cache efficient: One sample per day, reused for all hours
        - Non-negative: Multipliers clipped to [0, ∞) to prevent negative loads
    """

    def __init__(
        self,
        base_profile: LoadProfile,
        p05_delta: float = -0.1,
        p95_delta: float = 0.1,
    ) -> None:
        """
        Initialize variable load profile with stochastic daily multipliers.

        Args:
            base_profile: Base LoadProfile to wrap and add variation to.
                Can be any LoadProfile subclass (deterministic or stochastic).
                The wrapped profile's consumption is multiplied by random factors.
            p05_delta: 5th percentile variation (must be negative).
                Lower bound for daily multipliers. Example: -0.10 means 5% of
                days have ≤10% reduction from base consumption.
                Typical values: -0.05 to -0.20 (conservative to high uncertainty).
            p95_delta: 95th percentile variation (must be positive).
                Upper bound for daily multipliers. Example: +0.10 means 5% of
                days have ≥10% increase from base consumption.
                Typical values: +0.05 to +0.20 (conservative to high uncertainty).

        Raises:
            ValueError: If p05_delta ≥ 0 (must be negative).
            ValueError: If p95_delta ≤ 0 (must be positive).

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                VariableLoadProfile,
                AreraLoadProfile
            )

            # Conservative residential (±8% variation)
            base = AreraLoadProfile()
            model = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.08,
                p95_delta=0.08
            )

            # High uncertainty commercial (±15% variation)
            model_commercial = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.15,
                p95_delta=0.15
            )

            # Asymmetric variation (more likely to exceed than undershoot)
            model_asymmetric = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.05,  # Rarely much less
                p95_delta=0.15    # Often significantly more
            )
            ```

        Notes:
            - Standard deviation σ derived from percentiles: σ = max(|p05|, |p95|) / 1.645
            - Symmetric bounds (|p05| = p95) produce symmetric multiplier distribution
            - Asymmetric bounds allow modeling skewed consumption patterns
            - Zero sigma (p05=p95=0) → no variation (passes through base profile)
            - Multiplier mean = 1.0 (on average, matches base profile)
        """
        if p05_delta >= 0:
            raise ValueError("p05_delta must be negative")
        if p95_delta <= 0:
            raise ValueError("p95_delta must be positive")
        self.base_profile = base_profile
        self.p05_delta = float(p05_delta)
        self.p95_delta = float(p95_delta)
        max_delta = max(abs(self.p05_delta), abs(self.p95_delta))
        self._sigma = max_delta / Z_VALUE_95 if max_delta > 0 else 0.0
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()
        self._daily_multipliers: Dict[Tuple[int, int, int], float] = {}

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Reset wrapped profile and clear daily multipliers for new Monte Carlo path.

        Clears cached multipliers from previous path and propagates reset to
        the wrapped base profile. This ensures each Monte Carlo path gets
        independent random daily variations.

        Args:
            rng: Random number generator for stochastic multiplier sampling.
                Stored for use during get_hourly_load_kw() calls.
            n_years: Number of simulation years (passed to base_profile).
                Unused by this class but forwarded to wrapped profile.

        Notes:
            - Clears _daily_multipliers cache (forces regeneration)
            - Stores rng for lazy multiplier sampling during simulation
            - Resets base_profile with same rng for consistency
            - Called once per Monte Carlo path before year loop
        """
        self.base_profile.reset_for_run(rng=rng, n_years=n_years)
        self._daily_multipliers.clear()
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

    def _rng_for_variation(self) -> np.random.Generator:
        """
        Get random number generator for multiplier sampling.

        Returns the stored RNG or fallback if not initialized. Internal helper
        ensuring valid RNG is always available.

        Returns:
            np.random.Generator: Active random number generator instance.

        Notes:
            - Internal helper method (not part of public API)
            - Automatically initializes fallback if _rng not set
            - Used by _sample_multiplier() for random variation generation
        """
        if self._rng is None:
            self._rng = self._fallback_rng
        return self._rng

    def _sample_multiplier(self) -> float:
        """
        Sample random daily multiplier from truncated normal distribution.

        Generates a single multiplier value representing the day's consumption
        scaling factor. Sampled from normal distribution centered at 0 with
        standard deviation derived from percentile bounds, then clipped and
        shifted to produce multipliers around 1.0.

        Algorithm:
        1. Sample variation ~ Normal(0, σ)
        2. Clip to [p05_delta, p95_delta]
        3. Compute multiplier = 1.0 + variation
        4. Clip to [0, ∞) to prevent negative loads

        Returns:
            float: Daily consumption multiplier (non-negative).
                Typical range: 0.8-1.2 for ±10% variation bounds.
                Mean ≈ 1.0 (on average matches base profile).
                Examples:
                - 0.92: Day with 8% reduction
                - 1.0: Day matching base profile
                - 1.15: Day with 15% increase

        Notes:
            - Returns 1.0 if σ = 0 (no variation configured)
            - Truncated normal ensures realistic bounds (no extreme outliers)
            - Non-negative guarantee: max(0.0, multiplier) prevents negative loads
            - Each call samples independently (for different days)
        """
        if self._sigma <= 0.0:
            return 1.0
        rng = self._rng_for_variation()
        variation = rng.normal(loc=0.0, scale=self._sigma)
        variation = np.clip(variation, self.p05_delta, self.p95_delta)
        multiplier = 1.0 + variation
        return max(0.0, multiplier)

    def _get_multiplier(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
    ) -> float:
        """
        Get or lazily sample daily multiplier for specified calendar day.

        Checks cache for existing multiplier, otherwise samples new random
        value and caches it. This ensures all hours in the same day use the
        same multiplier while different days get independent random values.

        Args:
            year_index: Simulation year (0-based) for cache key.
            month_in_year: Month index (0-11) for cache key.
            day_in_month: Day index (0-based) for cache key.

        Returns:
            float: Daily consumption multiplier for this specific day.
                Cached value if previously sampled, new random value otherwise.

        Notes:
            - Cached by (year, month, day) key for efficiency
            - Lazy sampling: Only computed when first queried
            - Persistent within run: Same day always returns same multiplier
            - Independent across days: Different days get different multipliers
            - Cleared by reset_for_run(): Fresh multipliers each MC path
        """
        key = (year_index, month_in_year, day_in_month)
        multiplier = self._daily_multipliers.get(key)
        if multiplier is not None:
            return multiplier
        multiplier = self._sample_multiplier()
        self._daily_multipliers[key] = multiplier
        return multiplier

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get stochastic hourly load with daily random variation.

        Queries the base profile for the reference consumption, then applies
        the day's random multiplier to introduce realistic day-to-day variability.
        All hours in the same day use the same multiplier.

        Args:
            year_index: Simulation year (0-based).
            month_in_year: Month (0-11).
            day_in_month: Day (0-based) for multiplier lookup.
            hour_in_day: Hour (0-23) passed to base profile.
            weekday: Weekday (0-6) passed to base profile.

        Returns:
            float: Stochastic electricity consumption in kW.
                Formula: base_load × daily_multiplier
                where daily_multiplier is constant for all hours in the day.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                VariableLoadProfile,
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Base: 200W constant
            base = MonthlyAverageLoadProfile(make_flat_monthly_load_profiles(200.0))

            # Add ±15% variation
            model = VariableLoadProfile(base, -0.15, 0.15)
            model.reset_for_run(rng=np.random.default_rng(42), n_years=1)

            # Day 0: multiplier might be 0.92
            load_d0_h0 = model.get_hourly_load_kw(0, 0, 0, 0, 0)   # 200W × 0.92 = 184W
            load_d0_h12 = model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 200W × 0.92 = 184W (same)

            # Day 1: different multiplier, e.g., 1.08
            load_d1_h0 = model.get_hourly_load_kw(0, 0, 1, 0, 1)   # 200W × 1.08 = 216W
            load_d1_h12 = model.get_hourly_load_kw(0, 0, 1, 12, 1)  # 200W × 1.08 = 216W (same)
            ```

        Notes:
            - Stochastic: Different days return different scaled values
            - Day-consistent: All hours in same day use same multiplier
            - Multiplier lazy-sampled: First query triggers sampling and caching
            - Shape-preserving: Diurnal pattern maintained, only amplitude varies
            - Two-stage lookup: base_profile.get_hourly_load_kw() then multiply
        """
        base_kw = self.base_profile.get_hourly_load_kw(
            year_index=year_index,
            month_in_year=month_in_year,
            day_in_month=day_in_month,
            hour_in_day=hour_in_day,
            weekday=weekday,
        )
        multiplier = self._get_multiplier(year_index, month_in_year, day_in_month)
        return base_kw * multiplier


def get_load_w(
    month_index: int,
    hour_in_month: int,
    first_weekday_of_month: int = 0,
    bl_table: np.ndarray = BL_TABLE,
) -> float:
    """
    Calculate ARERA-based load consumption for specific hour in month.

    Determines the appropriate Italian ARERA tariff band (F1/F2/F3) for the
    given hour based on day of week and time of day, then returns the
    corresponding base load value from the lookup table.

    This function is the core of AreraLoadProfile, converting calendar position
    (month, hour) into consumption values using the Italian time-of-use tariff
    structure.

    Args:
        month_index: Month index (0-based integer, 0-11).
            0 = January, 1 = February, ..., 11 = December.
        hour_in_month: Absolute hour within the month (0-based integer).
            0 = first hour of month (day 0, hour 0).
            Range: 0 to (24 × days_in_month - 1).
            Example: hour 25 = day 1, hour 1 (second day, second hour).
        first_weekday_of_month: Weekday of first day of month (0-based integer).
            0 = Monday, 1 = Tuesday, ..., 6 = Sunday.
            Used to compute weekday for any hour_in_month.
            Default: 0 (month starts on Monday).
        bl_table: Base load lookup table in Watts (shape: 12 × 3).
            Rows: 12 months (Jan=0 ... Dec=11).
            Columns: 3 ARERA bands (F1, F2, F3).
            Default: BL_TABLE (Italian residential empirical data).

    Returns:
        float: Base electricity load in Watts for the specified hour.
            Looked up from bl_table[month, band] where band is determined
            by ARERA tariff rules.

    Raises:
        ValueError: If month_index not in [0, 11].
        ValueError: If hour_in_month not in valid range for the month.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            get_load_w,
            BL_TABLE
        )

        # January (month 0), first day (hours 0-23), starts on Monday (0)
        # Hour 10 = day 0, hour 10 = Monday 10am → F1 (peak)
        load_peak = get_load_w(
            month_index=0,
            hour_in_month=10,
            first_weekday_of_month=0,
            bl_table=BL_TABLE
        )  # Returns BL_TABLE[0, 0] ≈ 110.67W (Jan F1)

        # Hour 50 = day 2 (50÷24=2), hour 2 (50%24=2) = Wednesday 2am → F3 (off-peak)
        load_offpeak = get_load_w(
            month_index=0,
            hour_in_month=50,
            first_weekday_of_month=0
        )  # Returns BL_TABLE[0, 2] ≈ 157.05W (Jan F3)

        # Custom base load table
        bl_custom = np.array([
            [400.0, 250.0, 120.0],  # Jan: F1=400W, F2=250W, F3=120W
            # ... (11 more months)
        ])
        load_custom = get_load_w(0, 10, 0, bl_custom)  # 400W
        ```

    Notes:
        - Used internally by AreraLoadProfile.get_hourly_load_kw()
        - ARERA bands (see _get_band_arera for details):
          - F1 (peak): Mon-Fri 8am-7pm
          - F2 (mid): Mon-Fri 7am+19-22pm, Sat 7am-10pm
          - F3 (off): Nights, Sundays
        - February assumed 28 days (no leap year support)
        - Error messages in Italian for historical reasons
    """
    if not (0 <= month_index < 12):
        raise ValueError("month_index deve essere tra 0 e 11")

    days_in_month = MONTH_LENGTHS[month_index]
    max_hours = 24 * days_in_month
    if not (0 <= hour_in_month < max_hours):
        raise ValueError(
            f"hour_in_month deve essere tra 0 e {max_hours-1} per il mese {month_index}"
        )

    day_in_month = hour_in_month // 24
    hour = hour_in_month % 24
    weekday = (first_weekday_of_month + day_in_month) % 7
    band = _get_band_arera(weekday, hour)

    if band == "F1":
        bl = bl_table[month_index, 0]
    elif band == "F2":
        bl = bl_table[month_index, 1]
    else:
        bl = bl_table[month_index, 2]

    return bl


def _get_band_arera(weekday: int, hour: int) -> str:
    """
    Determine Italian ARERA tariff band for given weekday and hour.

    Implements the Italian ARERA (Autorità di Regolazione per Energia Reti e
    Ambiente) time-of-use tariff structure, which divides hours into three
    bands based on typical grid demand patterns.

    ARERA Band Rules:
    - F1 (Peak): Monday-Friday, 8am-7pm (08:00-18:59)
      Highest rates during weekday business hours (peak demand).
    - F2 (Mid-peak): Monday-Friday 7am + 7-10pm, Saturday 7am-10pm
      Intermediate rates during shoulder hours and Saturdays.
    - F3 (Off-peak): All other hours (nights, Sundays, holidays)
      Lowest rates during low-demand periods.

    Args:
        weekday: Day of week (0-based integer, 0-6).
            0 = Monday, 1 = Tuesday, ..., 5 = Saturday, 6 = Sunday.
        hour: Hour of day (0-based integer, 0-23).
            0 = midnight-1am, 8 = 8am-9am, 18 = 6pm-7pm, 23 = 11pm-midnight.

    Returns:
        str: ARERA tariff band identifier.
            - "F1": Peak hours (weekday daytime)
            - "F2": Mid-peak hours (weekday shoulder + Saturday daytime)
            - "F3": Off-peak hours (nights, Sundays)

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import _get_band_arera

        # Monday 10am → F1 (peak)
        band1 = _get_band_arera(weekday=0, hour=10)  # "F1"

        # Monday 7am → F2 (shoulder)
        band2 = _get_band_arera(weekday=0, hour=7)   # "F2"

        # Monday 3am → F3 (night)
        band3 = _get_band_arera(weekday=0, hour=3)   # "F3"

        # Saturday 2pm → F2 (Saturday daytime)
        band4 = _get_band_arera(weekday=5, hour=14)  # "F2"

        # Sunday 2pm → F3 (Sunday = always off-peak)
        band5 = _get_band_arera(weekday=6, hour=14)  # "F3"
        ```

    Detailed Rules:
        Sunday (weekday=6):
            All hours → F3

        Monday-Friday (weekday=0-4):
            Hour 8-18 (8am-7pm) → F1
            Hour 7, 19-22 (7am, 7pm-11pm) → F2
            Hour 0-6, 23 (midnight-7am, 11pm-midnight) → F3

        Saturday (weekday=5):
            Hour 7-22 (7am-11pm) → F2
            Hour 0-6, 23 (midnight-7am, 11pm-midnight) → F3

    Notes:
        - Designed for Italian electricity market
        - Based on ARERA regulatory framework
        - Holidays treated as Sundays (not implemented here)
        - Hour ranges are inclusive: 8 <= hour <= 18 includes hours 8-18
    """
    if weekday == 6:
        return "F3"

    if weekday in range(0, 5):
        if 8 <= hour <= 18:
            return "F1"
        elif hour == 7 or (19 <= hour <= 22):
            return "F2"
        else:
            return "F3"
    else:
        if 7 <= hour <= 22:
            return "F2"
        else:
            return "F3"


def make_flat_monthly_load_profiles(base_load_w: float = 140.0) -> np.ndarray:
    """
    Create constant load profile array (same value for all months and hours).

    Generates a 12×24 array filled with a single load value, suitable for
    initializing MonthlyAverageLoadProfile with constant consumption. Useful
    for simple baseline scenarios, standby loads, or testing.

    Args:
        base_load_w: Constant hourly load in Watts (non-negative float).
            Applied to all months and all hours uniformly.
            Default: 140.0W (typical Italian residential standby consumption).
            Examples:
            - 80W: Minimal standby (fridge, routers)
            - 150W: Light residential baseline
            - 300W: Active household consumption

    Returns:
        np.ndarray: Array of shape (12, 24) with dtype=float.
            All elements set to base_load_w.
            Rows: 12 months (Jan=0 ... Dec=11).
            Columns: 24 hours (0=midnight-1am ... 23=11pm-midnight).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            make_flat_monthly_load_profiles,
            MonthlyAverageLoadProfile
        )

        # Create constant 200W profile
        profiles = make_flat_monthly_load_profiles(base_load_w=200.0)
        print(profiles.shape)  # (12, 24)
        print(profiles[0, 0])  # 200.0
        print(profiles[6, 12]) # 200.0 (all values identical)

        # Use with MonthlyAverageLoadProfile
        load_model = MonthlyAverageLoadProfile(profiles)
        load = load_model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 0.2 kW

        # Different constant values for different scenarios
        standby = make_flat_monthly_load_profiles(80.0)   # Away mode
        moderate = make_flat_monthly_load_profiles(200.0) # Light usage
        active = make_flat_monthly_load_profiles(400.0)   # High usage
        ```

    Notes:
        - Simplest possible load profile (no variation)
        - No seasonal or diurnal patterns
        - Primarily used as building block for more complex profiles
        - Can serve as base for VariableLoadProfile decoration
        - Efficient: Uses np.full (no loops)
    """
    return np.full((12, 24), base_load_w, dtype=float)


@dataclass
class LoadScenarioBlueprint:
    """
    Configuration blueprint for building complex load profile compositions.

    Provides a declarative way to assemble load profiles with optional
    home/away patterns and stochastic variations. Acts as a factory pattern
    for creating composable LoadProfile hierarchies.

    This class simplifies construction of common load profile scenarios:
    - Single profile (optionally with variation)
    - Home/away dual-profile (with optional variations on each)
    - Vacation home patterns (seasonal occupancy with stochastic variation)

    Build Logic:
    - If both factories provided → HomeAwayLoadProfile(home, away)
    - If only home_factory → home profile (possibly with variation)
    - If only away_factory → away profile (possibly with variation)
    - If neither factory → ValueError

    Each factory is wrapped with VariableLoadProfile if percentiles provided.

    Attributes:
        home_profile_factory: Factory function creating home/occupied profile.
        away_profile_factory: Factory function creating away/standby profile.
        min_days_home: Minimum home days per month (12 values) for home/away.
        max_days_home: Maximum home days per month (12 values) for home/away.
        home_variation_percentiles: Optional (p05, p95) for home profile variation.
        away_variation_percentiles: Optional (p05, p95) for away profile variation.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import (
            LoadScenarioBlueprint,
            AreraLoadProfile,
            MonthlyAverageLoadProfile,
            make_flat_monthly_load_profiles
        )

        # Scenario 1: Single profile with variation
        blueprint_simple = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            home_variation_percentiles=(-0.10, 0.10)  # ±10% variation
        )
        load_simple = blueprint_simple.build_load_profile()
        # Returns: VariableLoadProfile(AreraLoadProfile())

        # Scenario 2: Home/Away without variation
        blueprint_homeaway = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            away_profile_factory=lambda: MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(80.0)
            ),
            min_days_home=[20]*12,  # Home 20-25 days/month
            max_days_home=[25]*12
        )
        load_homeaway = blueprint_homeaway.build_load_profile()
        # Returns: HomeAwayLoadProfile(home=ARERA, away=80W constant)

        # Scenario 3: Vacation home with variations
        blueprint_vacation = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            away_profile_factory=lambda: MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(70.0)
            ),
            min_days_home=[5, 5, 5, 5, 5, 20, 25, 25, 5, 5, 5, 5],  # Summer use
            max_days_home=[10, 10, 10, 10, 15, 30, 31, 31, 15, 10, 10, 10],
            home_variation_percentiles=(-0.15, 0.15),  # ±15% when home
            away_variation_percentiles=(-0.05, 0.05)   # ±5% when away (minimal)
        )
        load_vacation = blueprint_vacation.build_load_profile()
        # Returns: HomeAwayLoadProfile(
        #     home=VariableLoadProfile(ARERA, ±15%),
        #     away=VariableLoadProfile(70W constant, ±5%)
        # )
        ```

    Notes:
        - Factory pattern: Defers profile creation until build_load_profile() called
        - Variation decoration automatic: Percentiles trigger VariableLoadProfile wrapping
        - Validation: Checks min/max_days_home present for home/away scenarios
        - Flexible: Supports any LoadProfile subclass via factories
    """

    home_profile_factory: Callable[[], LoadProfile] | None = None
    """Factory creating home/occupied LoadProfile. Called by build_load_profile()."""

    away_profile_factory: Callable[[], LoadProfile] | None = None
    """Factory creating away/standby LoadProfile. Called by build_load_profile()."""

    min_days_home: List[int] | None = None
    """Minimum home days per month (12 ints). Required if both factories provided."""

    max_days_home: List[int] | None = None
    """Maximum home days per month (12 ints). Required if both factories provided."""

    home_variation_percentiles: Tuple[float, float] | None = None
    """Optional (p05_delta, p95_delta) for VariableLoadProfile wrapping home profile."""

    away_variation_percentiles: Tuple[float, float] | None = None
    """Optional (p05_delta, p95_delta) for VariableLoadProfile wrapping away profile."""

    @staticmethod
    def _apply_variation(
        profile: LoadProfile,
        percentiles: Tuple[float, float] | None,
    ) -> LoadProfile:
        """
        Optionally wrap profile with VariableLoadProfile for stochastic variation.

        If percentiles provided, wraps the base profile with VariableLoadProfile
        using the specified variation bounds. If percentiles is None, returns
        profile unchanged.

        Args:
            profile: Base LoadProfile to optionally decorate.
                Can be any LoadProfile subclass.
            percentiles: Optional (p05_delta, p95_delta) tuple for variation bounds.
                p05_delta: 5th percentile (must be negative, e.g., -0.10).
                p95_delta: 95th percentile (must be positive, e.g., +0.10).
                If None, no wrapping applied (deterministic profile returned).

        Returns:
            LoadProfile: Either the original profile (if percentiles=None)
                or VariableLoadProfile(profile, p05, p95) decorator.

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                LoadScenarioBlueprint,
                AreraLoadProfile
            )

            base = AreraLoadProfile()

            # No variation
            profile1 = LoadScenarioBlueprint._apply_variation(base, None)
            # Returns: base (unchanged)

            # With variation
            profile2 = LoadScenarioBlueprint._apply_variation(base, (-0.08, 0.08))
            # Returns: VariableLoadProfile(base, -0.08, 0.08)
            ```

        Notes:
            - Internal helper method (static, no self needed)
            - Decorator pattern: Wraps profile without modifying it
            - Validation performed by VariableLoadProfile constructor
        """
        if percentiles is None:
            return profile
        p05, p95 = percentiles
        return VariableLoadProfile(
            base_profile=profile,
            p05_delta=p05,
            p95_delta=p95,
        )

    def build_load_profile(self) -> LoadProfile:
        """
        Construct LoadProfile from blueprint configuration.

        Instantiates and assembles the configured load profile hierarchy
        by calling factories and applying decorators. The returned profile
        is ready for use in energy simulations.

        Build Algorithm:
        1. If both factories provided:
           - Call home_factory() and away_factory()
           - Apply variations if percentiles specified
           - Validate min/max_days_home present
           - Return HomeAwayLoadProfile(home, away, min_days, max_days)
        2. Else if home_factory only:
           - Call home_factory()
           - Apply variation if home_variation_percentiles specified
           - Return (possibly wrapped) home profile
        3. Else if away_factory only:
           - Call away_factory()
           - Apply variation if away_variation_percentiles specified
           - Return (possibly wrapped) away profile
        4. Else: Raise ValueError (no factory provided)

        Returns:
            LoadProfile: Fully constructed load profile ready for simulation.
                Type depends on configuration:
                - HomeAwayLoadProfile: If both factories provided
                - VariableLoadProfile: If single factory + percentiles
                - Concrete LoadProfile subclass: If single factory, no variation

        Raises:
            ValueError: If no factory provided (both None).
            ValueError: If both factories provided but min/max_days_home missing.
            Any exceptions from factory functions (e.g., invalid arguments).

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                LoadScenarioBlueprint,
                AreraLoadProfile,
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Build scenario 1: Simple ARERA with variation
            blueprint1 = LoadScenarioBlueprint(
                home_profile_factory=lambda: AreraLoadProfile(),
                home_variation_percentiles=(-0.10, 0.10)
            )
            profile1 = blueprint1.build_load_profile()
            # Type: VariableLoadProfile wrapping AreraLoadProfile

            # Build scenario 2: Home/away vacation home
            blueprint2 = LoadScenarioBlueprint(
                home_profile_factory=lambda: AreraLoadProfile(),
                away_profile_factory=lambda: MonthlyAverageLoadProfile(
                    make_flat_monthly_load_profiles(75.0)
                ),
                min_days_home=[5]*6 + [25]*2 + [5]*4,  # Summer occupancy
                max_days_home=[10]*6 + [31]*2 + [10]*4,
                home_variation_percentiles=(-0.12, 0.12)
            )
            profile2 = blueprint2.build_load_profile()
            # Type: HomeAwayLoadProfile with VariableLoadProfile-wrapped home

            # Use in simulation
            import numpy as np
            rng = np.random.default_rng(42)
            profile2.reset_for_run(rng=rng, n_years=20)
            load = profile2.get_hourly_load_kw(0, 6, 15, 12, 0)  # July 16, noon
            ```

        Notes:
            - Factories called during build (lazy instantiation)
            - Decorators applied automatically based on percentiles
            - Validation ensures required parameters present
            - Each call creates fresh profile instances (factories re-executed)
            - Use in API/config systems for declarative load profile definition
        """
        if self.home_profile_factory and self.away_profile_factory:
            if self.min_days_home is None or self.max_days_home is None:
                raise ValueError("min_days_home and max_days_home are required for home/away scenarios")
            home_profile = self._apply_variation(
                self.home_profile_factory(),
                self.home_variation_percentiles,
            )
            away_profile = self._apply_variation(
                self.away_profile_factory(),
                self.away_variation_percentiles,
            )
            return HomeAwayLoadProfile(
                home_profile=home_profile,
                away_profile=away_profile,
                min_days_home=self.min_days_home,
                max_days_home=self.max_days_home,
            )

        if self.home_profile_factory:
            return self._apply_variation(
                self.home_profile_factory(),
                self.home_variation_percentiles,
            )

        if self.away_profile_factory:
            return self._apply_variation(
                self.away_profile_factory(),
                self.away_variation_percentiles,
            )

        raise ValueError("At least one profile factory must be provided")
